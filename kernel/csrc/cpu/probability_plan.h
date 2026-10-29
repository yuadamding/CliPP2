#pragma once
#include "../common/emission_math.h"
#include "../common/validation.h"
#include <cstring>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace clipp2 {
inline std::uint64_t probability_bits(double value) {
    std::uint64_t result; std::memcpy(&result, &value, sizeof(result)); return result;
}
using ProbabilityFamilyKey = std::tuple<std::uint64_t, std::uint8_t>;
using ProbabilityShapeKey = std::tuple<std::int64_t, std::int64_t, std::uint64_t,
                                       std::uint64_t, std::uint64_t>;
// Model-scoped exact identities. Payloads remain authoritative in ObservationSoA;
// these immutable arrays intern equivalence only, not bounds or read reductions.
struct ProbabilityMetadata {
    std::vector<std::int64_t> families, shapes;
    std::int64_t bytes;
    explicit ProbabilityMetadata(const ObservationSoA& model)
        : families(checked_product(model.mutations, model.regions), -1),
          shapes(families.size(), -1), bytes(checked_product(families.size(), 2 * sizeof(std::int64_t))) {
        std::map<ProbabilityFamilyKey, std::int64_t> family_ids;
        std::map<ProbabilityShapeKey, std::int64_t> shape_ids;
        for (std::size_t cell=0; cell<families.size(); ++cell) {
            const auto o=observation(model,cell);
            if (!(o.flags & observed_flag)) continue;
            const ProbabilityFamilyKey key{probability_bits(o.scaling), static_cast<std::uint8_t>(o.flags & mixed_flag)};
            const auto family=family_ids.emplace(key,family_ids.size()).first->second;
            const ProbabilityShapeKey shape{family,o.support,probability_bits(o.alt),
                                            probability_bits(o.ref),probability_bits(o.log_constant)};
            families[cell]=family; shapes[cell]=shape_ids.emplace(shape,shape_ids.size()).first->second;
        }
    }
};
struct ProbabilityWork {
    std::int64_t reserved = 0, plans = 0, components = 0, shapes = 0, rows = 0;
    void add(std::int64_t basis, std::int64_t shape_count, std::int64_t row_count,
             std::int64_t evaluations) {
        const auto accumulate = [&](std::int64_t& total, std::int64_t per_call) {
            const auto amount = checked_product(per_call, evaluations);
            require(total <= std::numeric_limits<std::int64_t>::max() - amount,
                    "Probability reuse counter overflow");
            total += amount;
        };
        accumulate(components, basis); accumulate(shapes, shape_count); accumulate(rows, row_count);
    }
};
// Shared-center arithmetic only. Families use exact floating-point bits; shape
// values are expanded back into original member order before any reduction.
class ProbabilityPlan {
    struct Family { std::int64_t cell, support, offset; };
    struct Shape { std::int64_t cell, family; };
    ObservationSoA model_{};
    std::vector<Family> families_;
    std::vector<Shape> shapes_;
    std::vector<std::int64_t> row_shapes_;
    std::int64_t components_ = 0, bytes_ = 0;
public:
    struct Workspace {
        std::vector<double> log_p, log_q;
        std::vector<CellTerms> terms;
        Workspace() = default;
        explicit Workspace(const ProbabilityPlan& plan)
            : log_p(plan.components_), log_q(plan.components_), terms(plan.shapes_.size()) {}
    };
    std::int64_t bytes() const { return bytes_; }
    std::int64_t components() const { return components_; }
    std::int64_t shapes() const { return static_cast<std::int64_t>(shapes_.size()); }
    std::int64_t scratch_bytes() const {
        return components_ * 16 + static_cast<std::int64_t>(shapes_.size()) * sizeof(CellTerms);
    }
    ProbabilityPlan(const ObservationSoA& model, std::int64_t region,
                    const std::int64_t* rows, std::int64_t count,
                    const ProbabilityMetadata& catalog) : model_(model) {
        if (!count) return;
        std::map<std::int64_t, std::int64_t> active_families, active_shapes;
        families_.reserve(count); shapes_.reserve(count); row_shapes_.reserve(count);
        for (std::int64_t index = 0; index < count; ++index) {
            const auto cell = region * model.mutations + (rows ? rows[index] : index);
            const auto o = observation(model, cell);
            if (!(o.flags & observed_flag)) { row_shapes_.push_back(-1); continue; }
            const auto found=active_families.emplace(catalog.families[cell],families_.size());
            const auto family=found.first->second;
            const auto found_shape=active_shapes.emplace(catalog.shapes[cell],shapes_.size());
            const auto shape=found_shape.first->second;
            if (found.second) families_.push_back({cell, o.support, 0});
            else families_[family].support = std::max(families_[family].support, o.support);
            if (found_shape.second) shapes_.push_back({cell, family});
            row_shapes_.push_back(shape);
        }
        for (auto& family : families_) {
            const auto support = family.support;
            require(support <= std::numeric_limits<std::int64_t>::max() - components_,
                    "Probability support size overflow");
            family.offset = components_; components_ += support;
        }
        checked_product(components_, 2 * sizeof(double));
        bytes_ = checked_product(families_.capacity(), sizeof(Family))
               + checked_product(shapes_.capacity(), sizeof(Shape))
               + checked_product(row_shapes_.capacity(), sizeof(std::int64_t));
    }
    void evaluate(double center, Workspace& workspace) const {
        if (!finite_number(center) || center < 0 || center > 1) {
            for (auto& term : workspace.terms)
                term = {positive_infinity, 0, 1, EmitStatus::InvalidInput};
            return;
        }
        for (const auto& family : families_) {
            const auto o = observation(model_, family.cell);
            for (std::int64_t m = 0; m < family.support; ++m) {
                const auto p = component_probability(o, center, m + 1);
                workspace.log_p[family.offset + m] = ::log(p);
                workspace.log_q[family.offset + m] = ::log1p(-p);
            }
        }
        for (std::size_t index = 0; index < shapes_.size(); ++index) {
            const auto& shape = shapes_[index]; const auto o = observation(model_, shape.cell);
            const auto offset = families_[shape.family].offset;
            const auto mass = [&](std::int64_t m) {
                return (o.alt == 0 ? 0 : o.alt * workspace.log_p[offset + m])
                     + (o.ref == 0 ? 0 : o.ref * workspace.log_q[offset + m]) + o.log_constant;
            };
            double maximum = -positive_infinity;
            for (std::int64_t m = 0; m < o.support; ++m) {
                const auto value = mass(m); if (value > maximum) maximum = value;
            }
            if (maximum == -positive_infinity)
                workspace.terms[index] = {positive_infinity, 0, 1, EmitStatus::ZeroLikelihood};
            else if (!finite_number(maximum))
                workspace.terms[index] = {positive_infinity, 0, 1, EmitStatus::NumericalFailure};
            else {
                double total = 0;
                for (std::int64_t m = 0; m < o.support; ++m) total += ::exp(mass(m) - maximum);
                workspace.terms[index] = {-(maximum + ::log(total)), 0, 0, EmitStatus::Ok};
            }
        }
    }
    CellTerms term(const Workspace& workspace, std::int64_t row) const {
        const auto shape = row_shapes_[row];
        return shape < 0 ? CellTerms{0, 0, 0, EmitStatus::Ok} : workspace.terms[shape];
    }
};

// Context-owned immutable plans. Exact ordered row identities are deliberately
// separate from assignment bounds: these plans cache only read likelihoods.
class ProbabilityCache {
    using Key = std::vector<std::int64_t>;
    std::map<Key, std::shared_ptr<const ProbabilityPlan>> entries_;
    std::unique_ptr<ProbabilityMetadata> metadata_;
    std::int64_t bytes_ = 0;
public:
    std::int64_t hits = 0, misses = 0;
    std::int64_t bytes() const { return bytes_; }
    std::int64_t metadata_bytes() const { return metadata_ ? metadata_->bytes : 0; }
    void clear() {
        entries_.clear(); metadata_.reset(); bytes_ = 0;
    }
    std::shared_ptr<const ProbabilityPlan> get(const ObservationSoA& model, std::int64_t region,
            const std::int64_t* rows, std::int64_t count) {
        if (!count) return {};
        if (!metadata_) {
            metadata_=std::make_unique<ProbabilityMetadata>(model);
            bytes_+=metadata_->bytes;
        }
        Key key; key.reserve(count + 1); key.push_back(region);
        for (std::int64_t i = 0; i < count; ++i) key.push_back(rows ? rows[i] : i);
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            ++hits;
            return found->second;
        }
        ++misses;
        auto plan = std::make_shared<ProbabilityPlan>(model, region, rows, count,
                                                     *metadata_);
        const auto reservation = checked_product(key.capacity(), sizeof(std::int64_t)) + plan->bytes();
        entries_.emplace(std::move(key), plan); bytes_ += reservation;
        return plan;
    }
};

// One exclusive scratch lease per OpenMP worker, allocated before entering its
// team. Parallel width follows the configured CPU thread count.
class ProbabilityTasks {
    struct Task {
        std::shared_ptr<const ProbabilityPlan> plan;
        ProbabilityPlan::Workspace workspace;
        std::int64_t evaluations = 0;
        explicit Task(std::shared_ptr<const ProbabilityPlan> value)
            : plan(std::move(value)), workspace(plan ? Workspace(*plan) : Workspace()) {}
        using Workspace = ProbabilityPlan::Workspace;
    };
    std::vector<Task> tasks_;
    std::int64_t regions_, rows_, reserved_ = 0;
public:
    int workers = 1;
    ProbabilityTasks(const ObservationSoA& model, std::int64_t first_region,
                     std::int64_t regions, const std::int64_t* rows, std::int64_t count,
                     std::int64_t queries, ProbabilityCache& cache) : regions_(regions), rows_(count) {
#if defined(_OPENMP)
        workers = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(1, queries), omp_get_max_threads()));
#else
        (void)queries;
#endif
        if (!count || !queries || !regions) return;
        const auto tasks = checked_product(workers, regions);
        const auto objects = checked_product(tasks, sizeof(Task));
        reserved_ = objects; tasks_.reserve(tasks);
        for (int worker = 0; worker < workers; ++worker)
            for (std::int64_t region = 0; region < regions; ++region) {
                tasks_.emplace_back(cache.get(model, first_region + region, rows, count));
                if (tasks_.back().plan) reserved_ += tasks_.back().plan->scratch_bytes();
            }
    }
    static int worker() {
#if defined(_OPENMP)
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
    void evaluate(int worker, std::int64_t region, double center) {
        if (tasks_.empty()) return;
        auto& task = tasks_[worker * regions_ + region];
        task.plan->evaluate(center, task.workspace);
        if (finite_number(center) && center >= 0 && center <= 1) ++task.evaluations;
    }
    CellTerms term(int worker, std::int64_t region, std::int64_t row) const {
        const auto& task = tasks_[worker * regions_ + region];
        return task.plan->term(task.workspace, row);
    }
    ProbabilityWork work() const {
        ProbabilityWork result; result.reserved = reserved_;
        for (const auto& task : tasks_) if (task.plan) {
            ++result.plans;
            result.add(task.plan->components(), task.plan->shapes(), rows_, task.evaluations);
        }
        return result;
    }
};
}
