#include "../common/backend.h"
#include "../common/emission_math.h"
#include "../common/validation.h"
#include "../common/refit_validation.h"
#include "probability_plan.h"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef CLIPP2_BACKEND_BUILD_ID
#error "CLIPP2_BACKEND_BUILD_ID must bind the compiled backend source"
#endif

namespace {
thread_local char last_error[1024] = "";
using clipp2::EmitMode;
using clipp2::require;
std::int64_t product(std::int64_t a, std::int64_t b) { return clipp2::checked_product(a, b); }
template<typename Function> int guarded(Function function) {
    last_error[0] = '\0';
    try { function(); return 0; }
    catch (const std::exception& error) {
        std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "Unknown native CPU backend failure");
    }
    return 1;
}

struct Context {
    std::vector<double> alt, ref, scaling, constants, lower, upper;
    std::vector<std::int64_t> support;
    std::vector<std::uint8_t> flags;
    clipp2::ObservationSoA view{};
    CliPP2BackendDiagnostics diagnostics{};
    clipp2::ProbabilityWork probability_work{};
    clipp2::ProbabilityCache probability_cache;
    int cpu_threads = 0;

    Context(const clipp2::ObservationSoA& input, std::int64_t budget) : probability_cache(budget / 2) {
        const auto cells = clipp2::validate_observations(input, budget);
        alt.assign(input.alt, input.alt + cells); ref.assign(input.ref, input.ref + cells);
        scaling.assign(input.scaling, input.scaling + cells);
        constants.assign(input.log_constant, input.log_constant + cells);
        lower.assign(input.lower, input.lower + cells); upper.assign(input.upper, input.upper + cells);
        support.assign(input.support, input.support + cells); flags.assign(input.flags, input.flags + cells);
        view = {input.mutations, input.regions, alt.data(), ref.data(), scaling.data(),
                constants.data(), lower.data(), upper.data(), support.data(), flags.data()};
        diagnostics.prepared_bytes = cells * 57;
        diagnostics.workspace_limit = budget;
    }
};
Context& context(void* value) {
    require(value != nullptr, "Null backend context");
    auto& result = *static_cast<Context*>(value);
#if defined(_OPENMP)
    if (result.cpu_threads) { omp_set_num_threads(result.cpu_threads); omp_set_max_active_levels(1); }
#endif
    return result;
}
void record_evaluations(Context& ctx, std::int64_t count) {
    require(count >= 0 && ctx.diagnostics.cell_evaluations
            <= std::numeric_limits<std::int64_t>::max() - count,
            "Cell evaluation counter overflow");
    ctx.diagnostics.cell_evaluations += count;
}
void record_probability_work(Context& ctx, const clipp2::ProbabilityWork& work) {
    ctx.probability_work.plans += work.plans;
    ctx.probability_work.add(work.components, work.shapes, work.rows, 1);
    ctx.probability_work.reserved = std::max(ctx.probability_work.reserved, work.reserved);
}
void validate_outputs(std::int64_t count, const double* phi, int mode,
                      double* loss, double* gradient, double* curvature, std::uint8_t* status) {
    require(mode == 0 || mode == 1, "Unknown emission mode");
    require(!count || (phi && loss && status && (!mode || (gradient && curvature))),
            "Null emission argument");
}
template<EmitMode Mode>
void store(const clipp2::Observation& observation, double phi, std::int64_t index,
           double* loss, double* gradient, double* curvature, std::uint8_t* status) {
    const auto terms = clipp2::evaluate_observation<Mode>(observation, phi);
    loss[index] = terms.loss;
    status[index] = static_cast<std::uint8_t>(terms.status);
    if constexpr (Mode == EmitMode::LossGradientCurvature) {
        gradient[index] = terms.gradient; curvature[index] = terms.curvature;
    }
}
template<EmitMode Mode>
void nodes(Context& ctx, std::int64_t count, const double* phi, double* loss,
           double* gradient, double* curvature, std::uint8_t* status) {
    const auto cells = ctx.view.mutations * ctx.view.regions;
    CLIPP_PARALLEL_FOR
    for (std::int64_t index = 0; index < count; ++index)
        store<Mode>(clipp2::observation(ctx.view, index % cells), phi[index], index,
                    loss, gradient, curvature, status);
}
}

extern "C" int CliPP2BackendABI() { return clipp2::backend_abi; }
extern "C" const char* CliPP2BackendBuildId() { return CLIPP2_BACKEND_BUILD_ID; }
extern "C" const char* CliPP2BackendNumericalPolicy() { return clipp2::numerical_policy; }
extern "C" const char* CliPP2BackendLastError() { return last_error; }
extern "C" int CliPP2BackendCreate(const clipp2::ObservationSoA* input, std::int64_t budget,
                                    int device_index, void** output) {
    if (output) *output = nullptr;
    return guarded([&] {
        require(input && output, "Null model creation argument");
        require(device_index == -1, "CPU backend device index must be -1");
        *output = new Context(*input, budget);
    });
}
extern "C" void CliPP2BackendDestroy(void* ctx) { delete static_cast<Context*>(ctx); }
extern "C" int CliPP2BackendGetDiagnostics(void* value, CliPP2BackendDiagnostics* output) {
    return guarded([&] {
        require(output, "Null diagnostics output"); auto& ctx = context(value);
        ctx.diagnostics.cache_bytes = ctx.probability_cache.bytes(); *output = ctx.diagnostics;
    });
}
extern "C" int CliPP2ConfigureCPUExecution(void* value, std::int64_t host_limit, int threads) {
    return guarded([&] {
        auto& ctx = context(value); require(threads > 0, "CPU threads must be positive");
        ctx.cpu_threads = threads;
        require(host_limit >= ctx.diagnostics.prepared_bytes, "Native host cap cannot hold observations");
        const auto remaining = host_limit - ctx.diagnostics.prepared_bytes;
        const auto scratch = std::min(ctx.diagnostics.workspace_limit, remaining);
        ctx.probability_cache.set_limit(std::min(ctx.diagnostics.workspace_limit / 2, remaining - scratch));
        ctx.diagnostics.workspace_limit = scratch;
#if defined(_OPENMP)
        omp_set_num_threads(threads); omp_set_max_active_levels(1);
#endif
    });
}
extern "C" int CliPP2ProbabilityCacheDiagnostics(void* value, std::int64_t* output) {
    return guarded([&] {
        require(output, "Null probability cache diagnostics output");
        const auto& cache = context(value).probability_cache;
        output[0] = cache.hits; output[1] = cache.misses; output[2] = cache.bytes();
        output[3] = cache.metadata_bytes();
    });
}
extern "C" int CliPP2ReleaseProbabilityCache(void* value) {
    return guarded([&] { context(value).probability_cache.clear(); });
}
// Borrowed immutable view, valid only while this CPU context remains alive.
// Continuation owns its mutable states but need not copy the prepared 57NR bytes.
extern "C" int CliPP2BackendObservationView(void* value, clipp2::ObservationSoA* output) {
    return guarded([&] {
        require(output, "Null observation view output");
        *output = context(value).view;
    });
}
// Optional CPU telemetry; the shared backend ABI and numerical policy are unchanged.
extern "C" int CliPP2ProbabilityReuseDiagnostics(void* value, std::int64_t* output) {
    return guarded([&] {
        require(output, "Null probability diagnostics output");
        const auto& work = context(value).probability_work;
        output[0] = work.plans; output[1] = work.components; output[2] = work.shapes;
        output[3] = work.rows; output[4] = work.reserved;
    });
}
extern "C" int CliPP2EvaluateNodes(void* value, std::int64_t batches, const double* phi,
                                    int mode, double* loss, double* gradient,
                                    double* curvature, std::uint8_t* status) {
    return guarded([&] {
        auto& ctx = context(value);
        const auto count = product(batches, ctx.view.mutations * ctx.view.regions);
        validate_outputs(count, phi, mode, loss, gradient, curvature, status);
        if (mode) nodes<EmitMode::LossGradientCurvature>(ctx, count, phi, loss, gradient, curvature, status);
        else nodes<EmitMode::LossOnly>(ctx, count, phi, loss, gradient, curvature, status);
        ++ctx.diagnostics.calls; ctx.diagnostics.cell_evaluations += count;
    });
}
extern "C" int CliPP2EvaluateCells(void* value, std::int64_t count, const std::int64_t* cells,
                                    const double* phi, int mode, double* loss, double* gradient,
                                    double* curvature, std::uint8_t* status) {
    return guarded([&] {
        auto& ctx = context(value); product(count, 1);
        validate_outputs(count, phi, mode, loss, gradient, curvature, status);
        require(!count || cells, "Null cell indices");
        for (std::int64_t index = 0; index < count; ++index)
            require(cells[index] >= 0 && cells[index] < ctx.view.mutations * ctx.view.regions,
                    "Cell index out of range");
        CLIPP_PARALLEL_FOR
        for (std::int64_t index = 0; index < count; ++index) {
            const auto observation = clipp2::observation(ctx.view, cells[index]);
            if (mode) store<EmitMode::LossGradientCurvature>(observation, phi[index], index, loss, gradient, curvature, status);
            else store<EmitMode::LossOnly>(observation, phi[index], index, loss, gradient, curvature, status);
        }
        ++ctx.diagnostics.calls; ctx.diagnostics.cell_evaluations += count;
    });
}
extern "C" int CliPP2MultiplicityMAP(void* value, const double* phi, double* output,
                                      std::uint8_t* status) {
    return guarded([&] {
        auto& ctx = context(value);
        require(phi && output && status, "Null multiplicity argument");
        const auto count = ctx.view.mutations * ctx.view.regions;
        CLIPP_PARALLEL_FOR
        for (std::int64_t cell = 0; cell < count; ++cell) {
            auto state = clipp2::EmitStatus::Ok;
            output[cell] = clipp2::multiplicity_map(clipp2::observation(ctx.view, cell), phi[cell], state);
            status[cell] = static_cast<std::uint8_t>(state);
        }
        ++ctx.diagnostics.calls; ctx.diagnostics.cell_evaluations += count;
    });
}
extern "C" int CliPP2EvaluateRegionalGrid(void* value, std::int64_t region,
    std::int64_t row_count, const std::int64_t* rows, std::int64_t grid_count,
    const double* grid, double* output, std::uint8_t* status) {
    return guarded([&] {
        auto& ctx = context(value); const auto count = product(row_count, grid_count);
        require(region >= 0 && region < ctx.view.regions, "Region index out of range");
        require(!row_count || rows, "Null grid row indices");
        require(!count || (grid && output && status), "Null grid argument");
        for (std::int64_t row = 0; row < row_count; ++row)
            require(rows[row] >= 0 && rows[row] < ctx.view.mutations, "Grid row index out of range");
        clipp2::ProbabilityTasks plans(ctx.view, region, 1, rows, row_count, grid_count,
                                      ctx.diagnostics.workspace_limit, ctx.probability_cache);
        const auto workers = plans.workers;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(workers)
#else
        (void)workers;
#endif
        for (std::int64_t point = 0; point < grid_count; ++point) {
            const auto worker = clipp2::ProbabilityTasks::worker();
            const bool planned = plans.evaluate(worker, 0, grid[point]);
            for (std::int64_t row = 0; row < row_count; ++row) {
                const auto terms = planned ? plans.term(worker, 0, row) : clipp2::evaluate_observation<EmitMode::LossOnly>(
                    clipp2::observation(ctx.view, region * ctx.view.mutations + rows[row]), grid[point]);
                output[point * row_count + row] = -terms.loss;
                status[point * row_count + row] = static_cast<std::uint8_t>(terms.status);
            }
        }
        const auto work = plans.work(); record_probability_work(ctx, work);
        ctx.diagnostics.workspace_bytes = std::max(ctx.diagnostics.workspace_bytes, work.reserved);
        ++ctx.diagnostics.calls; ctx.diagnostics.cell_evaluations += count;
    });
}
extern "C" int CliPP2EvaluateJointColumns(void* value, std::int64_t count,
                                          const double* centers, double* output,
                                          std::uint8_t* status) {
    return guarded([&] {
        auto& ctx = context(value);
        const auto cells = product(count, ctx.view.mutations); product(count, ctx.view.regions);
        require(!cells || (centers && output && status), "Null joint column argument");
        clipp2::ProbabilityTasks plans(ctx.view, 0, ctx.view.regions, nullptr, ctx.view.mutations,
                                      count, ctx.diagnostics.workspace_limit, ctx.probability_cache);
        const auto workers = plans.workers;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(workers)
#else
        (void)workers;
#endif
        for (std::int64_t cluster = 0; cluster < count; ++cluster) {
            const auto worker = clipp2::ProbabilityTasks::worker();
            // Evaluate each regional plan once, then keep the original row and
            // region traversal, bound checks and ordered joint additions.
            for (std::int64_t region = 0; region < ctx.view.regions; ++region)
                plans.evaluate(worker, region, centers[cluster * ctx.view.regions + region]);
            for (std::int64_t row = 0; row < ctx.view.mutations; ++row) {
                double ll = 0;
                auto state = clipp2::EmitStatus::Ok;
                for (std::int64_t region = 0; region < ctx.view.regions; ++region) {
                    const auto cell = region * ctx.view.mutations + row;
                    const double phi = centers[cluster * ctx.view.regions + region];
                    if (!std::isfinite(phi)) {
                        state = clipp2::EmitStatus::InvalidInput;
                        ll = -std::numeric_limits<double>::infinity(); break;
                    }
                    if (phi < ctx.view.lower[cell] || phi > ctx.view.upper[cell]) {
                        state = clipp2::EmitStatus::ZeroLikelihood;
                        ll = -std::numeric_limits<double>::infinity(); break;
                    }
                    const auto terms = plans.available(worker, region) ? plans.term(worker, region, row) :
                        clipp2::evaluate_observation<EmitMode::LossOnly>(clipp2::observation(ctx.view, cell), phi);
                    ll -= terms.loss;
                    state = static_cast<clipp2::EmitStatus>(std::max(static_cast<int>(state), static_cast<int>(terms.status)));
                }
                output[cluster * ctx.view.mutations + row] = ll;
                status[cluster * ctx.view.mutations + row] = static_cast<std::uint8_t>(state);
            }
        }
        const auto work = plans.work(); record_probability_work(ctx, work);
        ctx.diagnostics.workspace_bytes = std::max(ctx.diagnostics.workspace_bytes, work.reserved);
        ++ctx.diagnostics.calls; ctx.diagnostics.cell_evaluations += product(cells, ctx.view.regions);
    });
}
extern "C" int CliPP2RefitCentersBatch(void* value, const clipp2::RefitBatch* tasks,
                                       clipp2::RefitResult* output) {
    return guarded([&] {
        auto& ctx = context(value);
        require(tasks && (!tasks->tasks || output), "Null refit argument");
        clipp2::validate_refit_batch(ctx.view, *tasks, output);
        const auto scratch = static_cast<std::uint64_t>(product(tasks->members, 8)) +
            static_cast<std::uint64_t>(product(tasks->grid_points, 8)) +
            static_cast<std::uint64_t>(product(tasks->tasks, 8)) + 8;
        require(scratch <= static_cast<std::uint64_t>(ctx.diagnostics.workspace_limit),
                "Conditional-refit scratch exceeds workspace budget");
        clipp2::ProbabilityWork probability_work;
        clipp2::refit_centers(ctx.view, *tasks, output,
                            ctx.diagnostics.workspace_limit - scratch, probability_work, &ctx.probability_cache);
        record_probability_work(ctx, probability_work);
        for (std::int64_t task = 0; task < tasks->tasks; ++task) {
            std::int64_t observed = 0;
            const auto region = tasks->regions[task];
            for (auto member = tasks->member_offsets[task]; member < tasks->member_offsets[task + 1]; ++member)
                observed += (ctx.view.flags[region * ctx.view.mutations + tasks->member_rows[member]]
                             & clipp2::observed_flag) != 0;
            // A required interval can fail after real work. Its returned
            // evaluation count includes that work, just as the CUDA counter.
            record_evaluations(ctx, product(observed, output[task].evaluations));
        }
        ctx.diagnostics.workspace_bytes = std::max(ctx.diagnostics.workspace_bytes,
                                                   static_cast<std::int64_t>(scratch) + probability_work.reserved);
        ++ctx.diagnostics.calls;
    });
}
extern "C" int CliPP2EvaluateConditionalValues(void* value, const clipp2::ConditionalBatch* tasks,
                                                double* output, std::uint8_t* status) {
    return guarded([&] {
        auto& ctx = context(value);
        require(tasks, "Null conditional-value task argument");
        clipp2::validate_conditional_batch(ctx.view, *tasks, output, status);
        const auto scratch = static_cast<std::uint64_t>(product(tasks->members, 8)) +
            static_cast<std::uint64_t>(product(tasks->tasks, 8)) + 8;
        require(scratch <= static_cast<std::uint64_t>(ctx.diagnostics.workspace_limit),
                "Conditional-value scratch exceeds workspace budget");
        clipp2::ProbabilityWork probability_work;
        clipp2::conditional_values(ctx.view, *tasks, output, status,
                                 ctx.diagnostics.workspace_limit - scratch, probability_work, &ctx.probability_cache);
        record_probability_work(ctx, probability_work);
        for (std::int64_t task = 0; task < tasks->tasks; ++task) {
            const double center = tasks->centers[task];
            bool evaluated = std::isfinite(center) && center >= 0 && center <= 1;
            std::int64_t observed = 0;
            for (auto member = tasks->member_offsets[task]; member < tasks->member_offsets[task + 1]; ++member) {
                const auto cell = tasks->regions[task] * ctx.view.mutations + tasks->member_rows[member];
                evaluated = evaluated && center >= ctx.view.lower[cell] && center <= ctx.view.upper[cell];
                observed += (ctx.view.flags[cell] & clipp2::observed_flag) != 0;
            }
            if (evaluated) record_evaluations(ctx, observed);
        }
        ctx.diagnostics.workspace_bytes = std::max(ctx.diagnostics.workspace_bytes,
                                                   static_cast<std::int64_t>(scratch) + probability_work.reserved);
        ++ctx.diagnostics.calls;
    });
}
