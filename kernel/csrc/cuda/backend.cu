// Optional native CUDA backend. Prepared observations and each numerical phase
// stay resident; the C ABI makes the requested host-result boundary explicit.
#include "../common/backend.h"
#include "../common/emission_math.h"
#include "../common/scalar_search.h"
#include "../common/validation.h"
#include "../common/refit_validation.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CLIPP2_BACKEND_BUILD_ID
#error "CLIPP2_BACKEND_BUILD_ID must bind the compiled backend source"
#endif

namespace {
using namespace clipp2;
using i64 = std::int64_t;
constexpr unsigned threads = 128;
thread_local char last_error[1024] = "";

i64 multiply(i64 a, i64 b) {
    if (a < 0 || b < 0 || (b && a > std::numeric_limits<i64>::max() / b))
        throw std::overflow_error("Native CUDA dimension or byte-size overflow");
    return a * b;
}
i64 add(i64 a, i64 b) {
    if (a < 0 || b < 0 || a > std::numeric_limits<i64>::max() - b)
        throw std::overflow_error("Native CUDA workspace-size overflow");
    return a + b;
}
void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
template<class Function> int guarded(Function function) {
    last_error[0] = '\0';
    try { function(); return 0; }
    catch (const std::exception& error) {
        std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "Unknown native CUDA failure");
    }
    return 1;
}

struct Layout {
    i64 bytes = 0;
    template<class T> i64 take(i64 count) {
        const i64 alignment = alignof(T);
        const i64 offset = add(bytes, (alignment - bytes % alignment) % alignment);
        bytes = add(offset, multiply(count, sizeof(T)));
        return offset;
    }
};

struct Context {
    ObservationSoA data{};
    int device = 0;
    cudaStream_t stream = nullptr;
    unsigned char *prepared = nullptr, *workspace = nullptr;
    CliPP2BackendDiagnostics diagnostics{};
    i64 host_staging_limit = 0;
    ~Context() {
        // Destruction cannot report errors through the C ABI. Normal operations
        // synchronize and report failures before reaching this cleanup path.
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (workspace) cudaFree(workspace);
        if (prepared) cudaFree(prepared);
        if (stream) cudaStreamDestroy(stream);
    }
    void reserve(i64 bytes) {
        require(bytes <= diagnostics.workspace_limit,
                "Native CUDA call exceeds workspace budget; reduce its batch");
        if (bytes <= diagnostics.workspace_bytes) return;
        // No allocations occur inside kernels or bounded-search iterations.
        if (workspace) {
            cuda_check(cudaStreamSynchronize(stream), "workspace synchronization");
            ++diagnostics.synchronizations;
            cuda_check(cudaFree(workspace), "release old workspace");
            workspace = nullptr;
            diagnostics.workspace_bytes = 0;
        }
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&workspace), bytes), "allocate workspace");
        diagnostics.workspace_bytes = bytes;
    }
    template<class T> T* pointer(i64 offset) {
        return reinterpret_cast<T*>(workspace + offset);
    }
    template<class T> void upload(T* target, const T* source, i64 count) {
        const i64 bytes = multiply(count, sizeof(T));
        if (!bytes) return;
        cuda_check(cudaMemcpyAsync(target, source, bytes, cudaMemcpyHostToDevice, stream),
                   "upload phase input");
        diagnostics.host_to_device_bytes = add(diagnostics.host_to_device_bytes, bytes);
    }
    template<class T> void download(T* target, const T* source, i64 count) {
        const i64 bytes = multiply(count, sizeof(T));
        if (!bytes) return;
        cuda_check(cudaMemcpyAsync(target, source, bytes, cudaMemcpyDeviceToHost, stream),
                   "download phase result");
        diagnostics.device_to_host_bytes = add(diagnostics.device_to_host_bytes, bytes);
    }
    void launched() {
        cuda_check(cudaGetLastError(), "launch native CUDA kernel");
        ++diagnostics.kernel_launches;
    }
    void finish() {
        cuda_check(cudaStreamSynchronize(stream), "complete native CUDA phase");
        ++diagnostics.synchronizations;
        ++diagnostics.calls;
    }
};

struct HostCounts {
    Context& context;
    std::vector<i64> values;
    HostCounts(Context& owner,i64 count):context(owner) {
        const auto bytes=multiply(count,sizeof(i64));
        require(bytes<=owner.host_staging_limit,"CUDA host staging exceeds native host budget");
        values.resize(static_cast<std::size_t>(count));
        context.diagnostics.staging_bytes=bytes;
    }
    ~HostCounts() {
        // Failed transfers must stop referencing the host buffer before release.
        if(std::uncaught_exceptions()) cudaStreamSynchronize(context.stream);
        context.diagnostics.staging_bytes=0;
    }
};

Context& context(void* value) {
    require(value != nullptr, "Null native CUDA context");
    auto& result = *static_cast<Context*>(value);
    cuda_check(cudaSetDevice(result.device), "select CUDA context device");
    return result;
}
unsigned blocks(i64 count) {
    return static_cast<unsigned>(std::min<i64>(65535, count / threads + (count % threads != 0)));
}
void output_check(i64 count, int mode, const double* loss, const double* gradient,
                  const double* curvature, const std::uint8_t* status) {
    require(mode == 0 || mode == 1, "Unknown native emission mode");
    require(count >= 0, "Negative native emission count");
    require(!count || (loss && status && (!mode || (gradient && curvature))),
            "Missing native emission result buffer");
}

template<EmitMode Mode> __device__ void store_terms(
        CellTerms terms, i64 index, double* loss, double* gradient,
        double* curvature, std::uint8_t* status) {
    loss[index] = terms.loss;
    status[index] = static_cast<std::uint8_t>(terms.status);
    if constexpr (Mode == EmitMode::LossGradientCurvature) {
        gradient[index] = terms.gradient;
        curvature[index] = terms.curvature;
    }
}
template<EmitMode Mode> __global__ void node_kernel(
        ObservationSoA data, i64 count, const double* phi, double* loss,
        double* gradient, double* curvature, std::uint8_t* status) {
    const i64 cells = data.mutations * data.regions;
    for (i64 index = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<i64>(blockDim.x) * gridDim.x)
        store_terms<Mode>(evaluate_observation<Mode>(observation(data, index % cells), phi[index]),
                          index, loss, gradient, curvature, status);
}
template<EmitMode Mode> __global__ void cell_kernel(
        ObservationSoA data, i64 count, const i64* cells, const double* phi,
        double* loss, double* gradient, double* curvature, std::uint8_t* status) {
    for (i64 index = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<i64>(blockDim.x) * gridDim.x)
        store_terms<Mode>(evaluate_observation<Mode>(observation(data, cells[index]), phi[index]),
                          index, loss, gradient, curvature, status);
}
__global__ void multiplicity_kernel(ObservationSoA data, i64 first, i64 count, const double* phi,
                                   double* output, std::uint8_t* status) {
    for (i64 index = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<i64>(blockDim.x) * gridDim.x) {
        EmitStatus state;
        output[index] = multiplicity_map(observation(data, first + index), phi[index], state);
        status[index] = static_cast<std::uint8_t>(state);
    }
}
__global__ void grid_kernel(ObservationSoA data, i64 region, i64 row_count,
        const i64* rows, i64 count, const double* grid, double* ll,
        std::uint8_t* status) {
    for (i64 index = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<i64>(blockDim.x) * gridDim.x) {
        const i64 cell = region * data.mutations + rows[index % row_count];
        const auto terms = evaluate_observation<EmitMode::LossOnly>(
            observation(data, cell), grid[index / row_count]);
        ll[index] = -terms.loss;
        status[index] = static_cast<std::uint8_t>(terms.status);
    }
}
__global__ void joint_kernel(ObservationSoA data, i64 first_row, i64 rows, i64 count, const double* centers,
                            double* ll, std::uint8_t* status) {
    for (i64 index = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<i64>(blockDim.x) * gridDim.x) {
        const i64 cluster = index / rows, row = first_row + index % rows;
        double total = 0;
        EmitStatus state = EmitStatus::Ok;
        for (i64 region = 0; region < data.regions; ++region) {
            const i64 cell = region * data.mutations + row;
            const double phi = centers[cluster * data.regions + region];
            // Assignment boxes apply even when this observation is missing.
            if (phi < data.lower[cell] || phi > data.upper[cell]) {
                total = -positive_infinity;
                state = EmitStatus::ZeroLikelihood;
                break;
            }
            const auto terms = evaluate_observation<EmitMode::LossOnly>(observation(data, cell), phi);
            if (static_cast<int>(terms.status) > static_cast<int>(state)) state = terms.status;
            total -= terms.loss;
        }
        ll[index] = total;
        status[index] = static_cast<std::uint8_t>(state);
    }
}

struct ArrayValues {
    const double* values;
    __device__ double operator()(i64 index) const { return values[index]; }
};
struct RefitObjective {
    ObservationSoA data;
    const i64* rows;
    i64 count, region;
    double *values, *total;
    __device__ double operator()(double center) const {
        for (i64 i = threadIdx.x; i < count; i += blockDim.x)
            values[i] = -evaluate_observation<EmitMode::LossOnly>(
                observation(data, region * data.mutations + rows[i]), center).loss;
        __syncthreads();
        // The exact logical reduction topology is shared with CPU; it does not
        // depend on block size, scheduling, cache budgets, or atomic order.
        if (!threadIdx.x) *total = pairwise_sum(ArrayValues{values}, 0, count);
        __syncthreads();
        return *total;
    }
    __device__ void store_grid(double* destination, double value) const {
        if (!threadIdx.x) *destination = value;
    }
    __device__ void finish_grid() const { __syncthreads(); }
};

#include "refit.cuh"

__global__ void conditional_kernel(ObservationSoA data, ConditionalBatch batch,
        i64* observed_rows, i64* observed_counts, double* member_values,
        double* output, std::uint8_t* status) {
    __shared__ i64 observed_count;
    __shared__ int state;
    __shared__ double total;
    for (i64 task = blockIdx.x; task < batch.tasks; task += gridDim.x) {
        const i64 begin = batch.member_offsets[task], end = batch.member_offsets[task + 1];
        const i64 region = batch.regions[task];
        const double center = batch.centers[task];
        if (!threadIdx.x) {
            observed_count = 0;
            state = finite_number(center) && center >= 0 && center <= 1
                ? static_cast<int>(EmitStatus::Ok) : static_cast<int>(EmitStatus::InvalidInput);
            for (i64 member = begin; member < end; ++member) {
                const i64 row = batch.member_rows[member], cell = region * data.mutations + row;
                if (state != static_cast<int>(EmitStatus::InvalidInput) &&
                    (center < data.lower[cell] || center > data.upper[cell]))
                    state = static_cast<int>(EmitStatus::ZeroLikelihood);
                if (data.flags[cell] & observed_flag)
                    observed_rows[begin + observed_count++] = row;
            }
            if (!observed_count) state = static_cast<int>(EmitStatus::InvalidInput);
            observed_counts[task] = state ? 0 : observed_count;
        }
        __syncthreads();
        if (!state) {
            const double value = RefitObjective{data, observed_rows + begin, observed_count,
                                               region, member_values + begin, &total}(center);
            if (!threadIdx.x) {
                output[task] = value;
                status[task] = static_cast<std::uint8_t>(finite_number(value) ? EmitStatus::Ok
                    : (value == -positive_infinity
                       ? EmitStatus::ZeroLikelihood : EmitStatus::NumericalFailure));
            }
        } else if (!threadIdx.x) {
            output[task] = -positive_infinity;
            status[task] = static_cast<std::uint8_t>(state);
        }
        __syncthreads();
    }
}

// A full phase owns one workspace. Slot offsets are planned before allocation,
// including all argument, result, and reduction buffers (no hidden N*R*M array).
struct EmissionBuffers {
    i64 phi, loss, gradient, curvature, status, cells, bytes;
    EmissionBuffers(i64 count, int mode, bool indexed) {
        Layout layout;
        phi = layout.take<double>(count);
        loss = layout.take<double>(count);
        gradient = layout.take<double>(mode ? count : 0);
        curvature = layout.take<double>(mode ? count : 0);
        status = layout.take<std::uint8_t>(count);
        cells = layout.take<i64>(indexed ? count : 0);
        bytes = layout.bytes;
    }
};
void emission_results(Context& c, const EmissionBuffers& b, i64 count, int mode,
                      double* loss, double* gradient, double* curvature, std::uint8_t* status) {
    c.download(loss, c.pointer<double>(b.loss), count);
    if (mode) {
        c.download(gradient, c.pointer<double>(b.gradient), count);
        c.download(curvature, c.pointer<double>(b.curvature), count);
    }
    c.download(status, c.pointer<std::uint8_t>(b.status), count);
    c.diagnostics.cell_evaluations = add(c.diagnostics.cell_evaluations, count);
    c.finish();
}
}  // namespace

extern "C" int CliPP2BackendABI() { return clipp2::backend_abi; }
extern "C" const char* CliPP2BackendBuildId() { return CLIPP2_BACKEND_BUILD_ID; }
extern "C" const char* CliPP2BackendNumericalPolicy() {
    return clipp2::numerical_policy;
}
extern "C" const char* CliPP2BackendLastError() { return last_error; }

// Native-only headroom probe: no Python framework/runtime import is required.
// Preserve the caller's current device, including after cudaMemGetInfo failure.
extern "C" int CliPP2DeviceMemoryInfo(int device, i64* free_bytes, i64* total_bytes) {
    return guarded([&] {
        require(device>=0 && free_bytes && total_bytes,"Invalid device-memory query");
        int previous=0;
        cuda_check(cudaGetDevice(&previous),"query current CUDA device");
        cuda_check(cudaSetDevice(device),"select CUDA memory-query device");
        std::size_t available=0,total=0;
        const auto status=cudaMemGetInfo(&available,&total);
        const auto restored=cudaSetDevice(previous);
        cuda_check(status,"query available CUDA memory");
        cuda_check(restored,"restore CUDA memory-query device");
        require(available<=static_cast<std::uint64_t>(INT64_MAX) &&
                total<=static_cast<std::uint64_t>(INT64_MAX),"CUDA byte count overflow");
        *free_bytes=static_cast<i64>(available); *total_bytes=static_cast<i64>(total);
    });
}

extern "C" int CliPP2BackendCreate(const ObservationSoA* input, i64 workspace_bytes,
                                    int device_index, void** result) {
    return guarded([&] {
        require(result != nullptr, "Missing native CUDA context destination");
        *result = nullptr;
        require(input, "Missing prepared CUDA observations");
        const i64 count = clipp2::validate_observations(*input, workspace_bytes);
        require(device_index >= 0, "CUDA device index must be nonnegative");
        auto c = std::make_unique<Context>();
        c->device = device_index;
        cuda_check(cudaSetDevice(device_index), "select CUDA backend device");
        cuda_check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "create CUDA stream");
        c->diagnostics.workspace_limit = workspace_bytes;
        c->host_staging_limit = workspace_bytes;
        Layout layout;
        const i64 alt = layout.take<double>(count), ref = layout.take<double>(count),
                  scaling = layout.take<double>(count), constant = layout.take<double>(count),
                  lower = layout.take<double>(count), upper = layout.take<double>(count),
                  support = layout.take<i64>(count), flags = layout.take<std::uint8_t>(count);
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&c->prepared), layout.bytes), "allocate prepared CUDA data");
        auto d = [&](i64 offset) { return reinterpret_cast<double*>(c->prepared + offset); };
        c->data = {input->mutations, input->regions, d(alt), d(ref), d(scaling), d(constant),
                   d(lower), d(upper), reinterpret_cast<i64*>(c->prepared + support), c->prepared + flags};
        c->upload(d(alt), input->alt, count); c->upload(d(ref), input->ref, count);
        c->upload(d(scaling), input->scaling, count); c->upload(d(constant), input->log_constant, count);
        c->upload(d(lower), input->lower, count); c->upload(d(upper), input->upper, count);
        c->upload(reinterpret_cast<i64*>(c->prepared + support), input->support, count);
        c->upload(c->prepared + flags, input->flags, count);
        cuda_check(cudaStreamSynchronize(c->stream), "complete prepared CUDA upload");
        c->diagnostics.prepared_bytes = layout.bytes;
        c->diagnostics.synchronizations = 1;
        *result = c.release();
    });
}
extern "C" void CliPP2BackendDestroy(void* value) { delete static_cast<Context*>(value); }
extern "C" int CliPP2BackendGetDiagnostics(void* value, CliPP2BackendDiagnostics* output) {
    return guarded([&] { require(output, "Missing CUDA diagnostic destination"); *output = context(value).diagnostics; });
}
extern "C" int CliPP2ConfigureCUDAExecution(void* value, i64 host_staging_limit) {
    return guarded([&] {
        require(host_staging_limit>=0,"CUDA host staging limit must be nonnegative");
        auto& c=context(value);
        require(c.diagnostics.calls==0,"CUDA execution budgets are fixed before first use");
        c.host_staging_limit=host_staging_limit;
    });
}
extern "C" int CliPP2EvaluateNodes(void* value, i64 batches, const double* phi, int mode,
        double* loss, double* gradient, double* curvature, std::uint8_t* status) {
    return guarded([&] {
        auto& c = context(value);
        const i64 count = multiply(batches, multiply(c.data.mutations, c.data.regions));
        output_check(count, mode, loss, gradient, curvature, status);
        require(!count || phi, "Missing CUDA node states");
        if (!count) return;
        const EmissionBuffers b(count, mode, false); c.reserve(b.bytes);
        c.upload(c.pointer<double>(b.phi), phi, count);
        if (mode) node_kernel<EmitMode::LossGradientCurvature><<<blocks(count), threads, 0, c.stream>>>(
            c.data, count, c.pointer<double>(b.phi), c.pointer<double>(b.loss),
            c.pointer<double>(b.gradient), c.pointer<double>(b.curvature), c.pointer<std::uint8_t>(b.status));
        else node_kernel<EmitMode::LossOnly><<<blocks(count), threads, 0, c.stream>>>(
            c.data, count, c.pointer<double>(b.phi), c.pointer<double>(b.loss), nullptr, nullptr,
            c.pointer<std::uint8_t>(b.status));
        c.launched(); emission_results(c, b, count, mode, loss, gradient, curvature, status);
    });
}
extern "C" int CliPP2EvaluateCells(void* value, i64 count, const i64* cells, const double* phi,
        int mode, double* loss, double* gradient, double* curvature, std::uint8_t* status) {
    return guarded([&] {
        auto& c = context(value);
        output_check(count, mode, loss, gradient, curvature, status);
        require(!count || (cells && phi), "Missing CUDA indexed states");
        if (!count) return;
        const i64 prepared_count = multiply(c.data.mutations, c.data.regions);
        for (i64 i = 0; i < count; ++i)
            require(cells[i] >= 0 && cells[i] < prepared_count, "CUDA observation index out of range");
        const EmissionBuffers b(count, mode, true); c.reserve(b.bytes);
        c.upload(c.pointer<double>(b.phi), phi, count); c.upload(c.pointer<i64>(b.cells), cells, count);
        if (mode) cell_kernel<EmitMode::LossGradientCurvature><<<blocks(count), threads, 0, c.stream>>>(
            c.data, count, c.pointer<i64>(b.cells), c.pointer<double>(b.phi), c.pointer<double>(b.loss),
            c.pointer<double>(b.gradient), c.pointer<double>(b.curvature), c.pointer<std::uint8_t>(b.status));
        else cell_kernel<EmitMode::LossOnly><<<blocks(count), threads, 0, c.stream>>>(
            c.data, count, c.pointer<i64>(b.cells), c.pointer<double>(b.phi), c.pointer<double>(b.loss),
            nullptr, nullptr, c.pointer<std::uint8_t>(b.status));
        c.launched(); emission_results(c, b, count, mode, loss, gradient, curvature, status);
    });
}
extern "C" int CliPP2MultiplicityMAP(void* value, const double* phi, double* output,
                                      std::uint8_t* status) {
    return guarded([&] {
        auto& c = context(value);
        const i64 count = multiply(c.data.mutations, c.data.regions);
        output_check(count, 0, output, nullptr, nullptr, status);
        require(phi, "Missing CUDA multiplicity CCFs");
        const i64 tile = std::min(count, (c.diagnostics.workspace_limit - 8) / 17);
        require(tile > 0, "CUDA multiplicity workspace cannot hold one cell");
        for (i64 first = 0; first < count; first += tile) {
            const i64 size = std::min(tile, count - first);
            const EmissionBuffers b(size, 0, false); c.reserve(b.bytes);
            c.upload(c.pointer<double>(b.phi), phi + first, size);
            multiplicity_kernel<<<blocks(size), threads, 0, c.stream>>>(c.data, first, size,
                c.pointer<double>(b.phi), c.pointer<double>(b.loss), c.pointer<std::uint8_t>(b.status));
            c.launched(); emission_results(c, b, size, 0, output + first, nullptr, nullptr, status + first);
        }
    });
}
extern "C" int CliPP2EvaluateRegionalGrid(void* value, i64 region, i64 row_count,
        const i64* rows, i64 grid_count, const double* grid, double* ll, std::uint8_t* status) {
    return guarded([&] {
        auto& c = context(value);
        const i64 count = multiply(row_count, grid_count);
        output_check(count, 0, ll, nullptr, nullptr, status);
        require(region >= 0 && region < c.data.regions, "CUDA grid region out of range");
        require(!count || (rows && grid), "Missing CUDA regional grid input");
        if (!count) return;
        for (i64 i = 0; i < row_count; ++i)
            require(rows[i] >= 0 && rows[i] < c.data.mutations, "CUDA grid row out of range");
        Layout layout;
        const auto dr = layout.take<i64>(row_count), dg = layout.take<double>(grid_count),
                   dl = layout.take<double>(count), ds = layout.take<std::uint8_t>(count);
        c.reserve(layout.bytes); c.upload(c.pointer<i64>(dr), rows, row_count);
        c.upload(c.pointer<double>(dg), grid, grid_count);
        grid_kernel<<<blocks(count), threads, 0, c.stream>>>(c.data, region, row_count,
            c.pointer<i64>(dr), count, c.pointer<double>(dg), c.pointer<double>(dl), c.pointer<std::uint8_t>(ds));
        c.launched(); c.download(ll, c.pointer<double>(dl), count);
        c.download(status, c.pointer<std::uint8_t>(ds), count);
        c.diagnostics.cell_evaluations = add(c.diagnostics.cell_evaluations, count); c.finish();
    });
}
extern "C" int CliPP2EvaluateJointColumns(void* value, i64 clusters, const double* centers,
                                           double* ll, std::uint8_t* status) {
    return guarded([&] {
        auto& c = context(value);
        const i64 count = multiply(clusters, c.data.mutations), coordinates = multiply(clusters, c.data.regions);
        output_check(count, 0, ll, nullptr, nullptr, status);
        require(!count || centers, "Missing CUDA cluster centers");
        if (!count) return;
        for (i64 i = 0; i < coordinates; ++i)
            require(std::isfinite(centers[i]), "CUDA centers must be finite");
        // Finite centers outside [0,1] violate every assignment box, so the
        // kernel returns -inf just as CPU does, including for missing reads.
        const i64 budget = c.diagnostics.workspace_limit;
        const i64 one_column = add(multiply(c.data.regions, 8), multiply(c.data.mutations, 9));
        const i64 cluster_tile = std::min(clusters, std::max<i64>(1, (budget - 8) / one_column));
        const i64 row_tile = std::min(c.data.mutations,
            (budget - 8 - multiply(multiply(cluster_tile, c.data.regions), 8)) / multiply(cluster_tile, 9));
        require(row_tile > 0, "CUDA column workspace cannot hold one center and observation");
        for (i64 first_cluster = 0; first_cluster < clusters; first_cluster += cluster_tile) {
            const i64 width = std::min(cluster_tile, clusters - first_cluster);
            for (i64 first_row = 0; first_row < c.data.mutations; first_row += row_tile) {
                const i64 rows = std::min(row_tile, c.data.mutations - first_row);
                const i64 tile_count = multiply(width, rows), tile_coordinates = multiply(width, c.data.regions);
                Layout layout;
                const auto dc = layout.take<double>(tile_coordinates), dl = layout.take<double>(tile_count),
                           ds = layout.take<std::uint8_t>(tile_count);
                c.reserve(layout.bytes);
                c.upload(c.pointer<double>(dc), centers + first_cluster * c.data.regions, tile_coordinates);
                joint_kernel<<<blocks(tile_count), threads, 0, c.stream>>>(c.data, first_row, rows, tile_count,
                    c.pointer<double>(dc), c.pointer<double>(dl), c.pointer<std::uint8_t>(ds));
                c.launched();
                for (i64 cluster = 0; cluster < width; ++cluster) {
                    const i64 destination = (first_cluster + cluster) * c.data.mutations + first_row;
                    c.download(ll + destination, c.pointer<double>(dl) + cluster * rows, rows);
                    c.download(status + destination, c.pointer<std::uint8_t>(ds) + cluster * rows, rows);
                }
                c.diagnostics.cell_evaluations = add(c.diagnostics.cell_evaluations,
                    multiply(tile_count, c.data.regions));
                c.finish();
            }
        }
    });
}

extern "C" int CliPP2RefitCentersBatch(void* value, const RefitBatch* batch, RefitResult* output) {
    return guarded([&] {
        auto& c = context(value);
        require(batch, "Missing CUDA refit batch");
        validate_refit_batch(c.data, *batch, output, false);
        if (!batch->tasks) return;
        refit_cuda(c, *batch, output);
    });
}

extern "C" int CliPP2EvaluateConditionalValues(void* value, const ConditionalBatch* batch,
                                                double* output, std::uint8_t* status) {
    return guarded([&] {
        auto& c = context(value);
        require(batch, "Missing CUDA conditional batch");
        validate_conditional_batch(c.data, *batch, output, status);
        if (!batch->tasks) return;
        HostCounts observed_counts(c,batch->tasks);
        Layout layout;
        const i64 regions = layout.take<i64>(batch->tasks),
                  offsets = layout.take<i64>(add(batch->tasks, 1)),
                  rows = layout.take<i64>(batch->members), centers = layout.take<double>(batch->tasks),
                  compacted = layout.take<i64>(batch->members), counts = layout.take<i64>(batch->tasks),
                  member_values = layout.take<double>(batch->members),
                  results = layout.take<double>(batch->tasks), states = layout.take<std::uint8_t>(batch->tasks);
        c.reserve(layout.bytes);
        ConditionalBatch device_batch{batch->tasks, batch->members, c.pointer<i64>(regions),
            c.pointer<i64>(offsets), c.pointer<i64>(rows), c.pointer<double>(centers)};
        c.upload(c.pointer<i64>(regions), batch->regions, batch->tasks);
        c.upload(c.pointer<i64>(offsets), batch->member_offsets, add(batch->tasks, 1));
        c.upload(c.pointer<i64>(rows), batch->member_rows, batch->members);
        c.upload(c.pointer<double>(centers), batch->centers, batch->tasks);
        const unsigned task_blocks = static_cast<unsigned>(std::min<i64>(65535, batch->tasks));
        conditional_kernel<<<task_blocks, threads, 0, c.stream>>>(c.data, device_batch,
            c.pointer<i64>(compacted), c.pointer<i64>(counts), c.pointer<double>(member_values),
            c.pointer<double>(results), c.pointer<std::uint8_t>(states));
        c.launched();
        c.download(output, c.pointer<double>(results), batch->tasks);
        c.download(status, c.pointer<std::uint8_t>(states), batch->tasks);
        c.download(observed_counts.values.data(), c.pointer<i64>(counts), batch->tasks);
        c.finish();
        for (const auto count : observed_counts.values)
            c.diagnostics.cell_evaluations = add(c.diagnostics.cell_evaluations, count);
    });
}
#include "../common/refit_validation.h"
