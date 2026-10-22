#pragma once
#include "../common/emission_math.h"
#include "../common/validation.h"
#include <cstring>
#include <map>
#include <tuple>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace clipp2 {
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
    struct Family { Observation observation; std::int64_t offset; };
    struct Shape { Observation observation; std::int64_t family; };
    std::vector<Family> families_;
    std::vector<Shape> shapes_;
    std::vector<std::int64_t> row_shapes_;
    std::int64_t components_ = 0, bytes_ = 0;
    static std::uint64_t bits(double value) {
        std::uint64_t result; std::memcpy(&result, &value, sizeof(result)); return result;
    }
public:
    struct Workspace {
        std::vector<double> log_p, log_q;
        std::vector<CellTerms> terms;
        Workspace() = default;
        explicit Workspace(const ProbabilityPlan& plan)
            : log_p(plan.components_), log_q(plan.components_), terms(plan.shapes_.size()) {}
    };
    bool enabled() const { return !row_shapes_.empty(); }
    std::int64_t bytes() const { return bytes_; }
    std::int64_t components() const { return components_; }
    std::int64_t shapes() const { return static_cast<std::int64_t>(shapes_.size()); }
    ProbabilityPlan() = default;
    // The conservative reservation includes temporary map nodes, persistent
    // metadata, one scratch lease and vector capacity. Insufficient optional
    // workspace retains the original scalar arithmetic; it is not rejection.
    ProbabilityPlan(const ObservationSoA& model, std::int64_t region,
                    const std::int64_t* rows, std::int64_t count, std::int64_t budget) {
        if (!count || budget < 0 || count > budget / 512) return;
        const auto metadata = count * 512;
        using FamilyKey = std::tuple<std::uint64_t, std::uint8_t>;
        using ShapeKey = std::tuple<std::int64_t, std::int64_t, std::uint64_t,
                                    std::uint64_t, std::uint64_t>;
        std::map<FamilyKey, std::int64_t> family_ids;
        std::map<ShapeKey, std::int64_t> shape_ids;
        families_.reserve(count); shapes_.reserve(count); row_shapes_.reserve(count);
        for (std::int64_t index = 0; index < count; ++index) {
            const auto o = observation(model, region * model.mutations + (rows ? rows[index] : index));
            if (!(o.flags & observed_flag)) { row_shapes_.push_back(-1); continue; }
            const FamilyKey family_key{bits(o.scaling), static_cast<std::uint8_t>(o.flags & mixed_flag)};
            const auto found = family_ids.emplace(family_key, families_.size());
            const auto family = found.first->second;
            if (found.second) families_.push_back({o, 0});
            else families_[family].observation.support = std::max(families_[family].observation.support, o.support);
            const ShapeKey shape_key{family, o.support, bits(o.alt), bits(o.ref), bits(o.log_constant)};
            const auto shape = shape_ids.emplace(shape_key, shapes_.size());
            if (shape.second) shapes_.push_back({o, family});
            row_shapes_.push_back(shape.first->second);
        }
        for (auto& family : families_) {
            const auto support = family.observation.support;
            if (support > (budget - metadata) / 16 - components_) {
                // Release the optional plan before returning to direct work.
                std::vector<Family>().swap(families_); std::vector<Shape>().swap(shapes_);
                std::vector<std::int64_t>().swap(row_shapes_); components_ = 0; return;
            }
            family.offset = components_; components_ += support;
        }
        bytes_ = metadata + components_ * 16;
    }
    void evaluate(double center, Workspace& workspace) const {
        if (!finite_number(center) || center < 0 || center > 1) {
            for (auto& term : workspace.terms)
                term = {positive_infinity, 0, 1, EmitStatus::InvalidInput};
            return;
        }
        for (const auto& family : families_) {
            for (std::int64_t m = 0; m < family.observation.support; ++m) {
                const auto p = component_probability(family.observation, center, m + 1);
                workspace.log_p[family.offset + m] = ::log(p);
                workspace.log_q[family.offset + m] = ::log1p(-p);
            }
        }
        for (std::size_t index = 0; index < shapes_.size(); ++index) {
            const auto& shape = shapes_[index]; const auto& o = shape.observation;
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

// One exclusive scratch lease per OpenMP worker, allocated before entering its
// team. Parallel width is reduced when the optional reuse budget is small.
class ProbabilityTasks {
    struct Task {
        ProbabilityPlan plan;
        ProbabilityPlan::Workspace workspace;
        std::int64_t evaluations = 0;
        Task(const ObservationSoA& model, std::int64_t region, const std::int64_t* rows,
             std::int64_t count, std::int64_t budget)
            : plan(model, region, rows, count, budget), workspace(plan) {}
    };
    std::vector<Task> tasks_;
    std::int64_t regions_, rows_, reserved_ = 0;
public:
    int workers = 1;
    ProbabilityTasks(const ObservationSoA& model, std::int64_t first_region,
                     std::int64_t regions, const std::int64_t* rows, std::int64_t count,
                     std::int64_t queries, std::int64_t budget) : regions_(regions), rows_(count) {
#if defined(_OPENMP)
        workers = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(1, queries), omp_get_max_threads()));
#else
        (void)queries;
#endif
        if (!count || !queries || !regions || budget < static_cast<std::int64_t>(sizeof(Task))) return;
        const auto per_worker = checked_product(regions, sizeof(Task) + checked_product(count, 512));
        workers = static_cast<int>(std::min<std::int64_t>(workers, std::max<std::int64_t>(1, budget / per_worker)));
        const auto tasks = checked_product(workers, regions);
        const auto objects = checked_product(tasks, sizeof(Task));
        if (objects > budget) return;
        reserved_ = objects; tasks_.reserve(tasks);
        for (int worker = 0; worker < workers; ++worker)
            for (std::int64_t region = 0; region < regions; ++region) {
                tasks_.emplace_back(model, first_region + region, rows, count, budget - reserved_);
                reserved_ += tasks_.back().plan.bytes();
            }
    }
    static int worker() {
#if defined(_OPENMP)
        return omp_get_thread_num();
#else
        return 0;
#endif
    }
    bool evaluate(int worker, std::int64_t region, double center) {
        if (tasks_.empty()) return false;
        auto& task = tasks_[worker * regions_ + region];
        if (!task.plan.enabled()) return false;
        task.plan.evaluate(center, task.workspace);
        if (finite_number(center) && center >= 0 && center <= 1) ++task.evaluations;
        return true;
    }
    bool available(int worker, std::int64_t region) const {
        return !tasks_.empty() && tasks_[worker * regions_ + region].plan.enabled();
    }
    CellTerms term(int worker, std::int64_t region, std::int64_t row) const {
        const auto& task = tasks_[worker * regions_ + region];
        return task.plan.term(task.workspace, row);
    }
    ProbabilityWork work() const {
        ProbabilityWork result; result.reserved = reserved_;
        for (const auto& task : tasks_) if (task.plan.enabled()) {
            ++result.plans;
            result.add(task.plan.components(), task.plan.shapes(), rows_, task.evaluations);
        }
        return result;
    }
};
}
