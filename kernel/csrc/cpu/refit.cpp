#include "../common/backend.h"
#include "../common/emission_math.h"
#include "../common/refit_validation.h"
#include "../common/scalar_search.h"
#include "probability_plan.h"
#include <vector>

namespace clipp2 {
namespace {
struct Emissions {
    const ObservationSoA& model;
    const std::int64_t* rows;
    std::int64_t region;
    double center;
    const ProbabilityPlan* plan;
    const ProbabilityPlan::Workspace* workspace;
    double operator()(std::int64_t member) const {
        if (plan->enabled()) return -plan->term(*workspace, member).loss;
        return -evaluate_observation<EmitMode::LossOnly>(
            observation(model, region * model.mutations + rows[member]), center).loss;
    }
};
struct Objective {
    const ObservationSoA& model;
    const std::int64_t* rows;
    std::int64_t count, region;
    const ProbabilityPlan& plan;
    ProbabilityPlan::Workspace& workspace;
    double operator()(double center) const {
        if (plan.enabled()) plan.evaluate(center, workspace);
        return pairwise_sum(Emissions{model, rows, region, center, &plan, &workspace}, 0, count);
    }
    void store_grid(double* destination, double value) const { *destination = value; }
    void finish_grid() const {}
};

}

void refit_centers(const ObservationSoA& model, const RefitBatch& batch, RefitResult* output,
                   std::int64_t budget, ProbabilityWork& work, ProbabilityCache* cache) {
    validate_refit_batch(model, batch, output);
    // Allocate once before the worker team: allocation failure reaches the C
    // boundary, and no task allocates from an OpenMP worker or objective loop.
    std::vector<std::int64_t> observed(batch.members), offsets(batch.tasks + 1);
    std::vector<double> values(batch.grid_points);
    std::int64_t used = 0;
    for (std::int64_t task = 0; task < batch.tasks; ++task) {
        offsets[task] = used;
        const auto region = batch.regions[task];
        for (auto index = batch.member_offsets[task]; index < batch.member_offsets[task + 1]; ++index) {
            const auto row = batch.member_rows[index];
            if (model.flags[region * model.mutations + row] & observed_flag) observed[used++] = row;
        }
    }
    offsets[batch.tasks] = used;
    std::vector<std::shared_ptr<const ProbabilityPlan>> plans;
    std::vector<ProbabilityPlan::Workspace> scratch;
    const auto objects = checked_product(batch.tasks, sizeof(ProbabilityPlan) + sizeof(ProbabilityPlan::Workspace));
    const bool planned = objects <= budget;
    ProbabilityPlan direct; ProbabilityPlan::Workspace direct_scratch;
    if (planned) {
        work.reserved = objects;
        plans.reserve(batch.tasks); scratch.reserve(batch.tasks);
    }
    for (std::int64_t task = 0; planned && task < batch.tasks; ++task) {
        auto plan = cache ? cache->get(model, batch.regions[task], observed.data() + offsets[task],
                                      offsets[task + 1] - offsets[task], budget - work.reserved)
                          : std::make_shared<ProbabilityPlan>(model, batch.regions[task], observed.data() + offsets[task],
                                      offsets[task + 1] - offsets[task], budget - work.reserved);
        if (!plan || !plan->enabled()) plan.reset();
        work.reserved += plan ? (cache ? plan->scratch_bytes() : plan->bytes()) : 0;
        work.plans += bool(plan); plans.push_back(plan);
        scratch.emplace_back(plan ? *plan : direct);
    }
    // Independent tasks share no mutable state. The serial path is complete;
    // builds may enable one optional OpenMP team, never nested reductions.
    CLIPP_PARALLEL_FOR
    for (std::int64_t task = 0; task < batch.tasks; ++task) {
        const auto region = batch.regions[task];
        const auto count = offsets[task + 1] - offsets[task];
        if (!count) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            output[task] = {nan, nan, nan, nan, 0, static_cast<std::int32_t>(RefitStatus::Unsupported)};
            continue;
        }
        output[task] = refit_search(batch, task, values.data() + batch.grid_offsets[task],
            Objective{model, observed.data() + offsets[task], count, region,
                      planned && plans[task] ? *plans[task] : direct, planned ? scratch[task] : direct_scratch});
    }
    for (std::int64_t task = 0; planned && task < batch.tasks; ++task)
        if (plans[task]) work.add(plans[task]->components(), plans[task]->shapes(),
            offsets[task + 1] - offsets[task], output[task].evaluations);
}

void conditional_values(const ObservationSoA& model, const ConditionalBatch& batch,
                         double* output, std::uint8_t* status, std::int64_t budget, ProbabilityWork& work,
                         ProbabilityCache* cache) {
    validate_conditional_batch(model, batch, output, status);
    std::vector<std::int64_t> observed(batch.members), offsets(batch.tasks + 1);
    std::int64_t used = 0;
    for (std::int64_t task = 0; task < batch.tasks; ++task) {
        offsets[task] = used;
        const auto region = batch.regions[task];
        auto state = EmitStatus::Ok;
        const double center = batch.centers[task];
        if (!std::isfinite(center) || center < 0 || center > 1) state = EmitStatus::InvalidInput;
        for (auto index = batch.member_offsets[task]; index < batch.member_offsets[task + 1]; ++index) {
            const auto row = batch.member_rows[index], cell = region * model.mutations + row;
            if (model.flags[cell] & observed_flag) observed[used++] = row;
            if (state == EmitStatus::Ok && (center < model.lower[cell] || center > model.upper[cell]))
                state = EmitStatus::ZeroLikelihood;
        }
        if (used == offsets[task]) state = EmitStatus::InvalidInput;
        status[task] = static_cast<std::uint8_t>(state);
    }
    offsets[batch.tasks] = used;
    std::vector<std::shared_ptr<const ProbabilityPlan>> plans;
    std::vector<ProbabilityPlan::Workspace> scratch;
    const auto objects = checked_product(batch.tasks, sizeof(ProbabilityPlan) + sizeof(ProbabilityPlan::Workspace));
    const bool planned = objects <= budget;
    ProbabilityPlan direct; ProbabilityPlan::Workspace direct_scratch;
    if (planned) {
        work.reserved = objects;
        plans.reserve(batch.tasks); scratch.reserve(batch.tasks);
    }
    for (std::int64_t task = 0; planned && task < batch.tasks; ++task) {
        const auto count = status[task] ? 0 : offsets[task + 1] - offsets[task];
        auto plan = cache ? cache->get(model, batch.regions[task], observed.data() + offsets[task],
                                      count, budget - work.reserved)
                          : std::make_shared<ProbabilityPlan>(model, batch.regions[task], observed.data() + offsets[task],
                                      count, budget - work.reserved);
        if (!plan || !plan->enabled()) plan.reset();
        work.reserved += plan ? (cache ? plan->scratch_bytes() : plan->bytes()) : 0;
        work.plans += bool(plan); plans.push_back(plan);
        scratch.emplace_back(plan ? *plan : direct);
    }
    CLIPP_PARALLEL_FOR
    for (std::int64_t task = 0; task < batch.tasks; ++task) {
        output[task] = -std::numeric_limits<double>::infinity();
        if (status[task]) continue;
        output[task] = Objective{model, observed.data() + offsets[task],
                                offsets[task + 1] - offsets[task], batch.regions[task],
                                planned && plans[task] ? *plans[task] : direct,
                                planned ? scratch[task] : direct_scratch}(batch.centers[task]);
        if (output[task] == -std::numeric_limits<double>::infinity())
            status[task] = static_cast<std::uint8_t>(EmitStatus::ZeroLikelihood);
        else if (!std::isfinite(output[task]))
            status[task] = static_cast<std::uint8_t>(EmitStatus::NumericalFailure);
    }
    for (std::int64_t task = 0; planned && task < batch.tasks; ++task)
        if (plans[task]) work.add(plans[task]->components(), plans[task]->shapes(),
            offsets[task + 1] - offsets[task], 1);
}
}
