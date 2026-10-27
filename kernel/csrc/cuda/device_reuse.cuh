// Optional exact read-emission tables. This deliberately does not cache box
// masks: missing members still constrain assignments in their existing callers.
constexpr i64 regular_grid_points = 513;
constexpr i64 grid_state_host_charge = 128;
constexpr i64 grid_entry_host_charge = 256;

CLIPP_HD inline int regular_grid_index(double value) {
    if (!(value >= 0 && value <= 1) || std::signbit(value)) return -1;
    const double scaled = value * 512.0;
    const int index = static_cast<int>(scaled);
    return scaled == static_cast<double>(index) ? index : -1;
}

struct RegularGridTable {
    Context& owner;
    double* values;
    i64 bytes, region;
    RegularGridTable(Context& c, double* pointer, i64 size, i64 coordinate)
        : owner(c), values(pointer), bytes(size), region(coordinate) {}
    ~RegularGridTable() {
        owner.release_resident(values, bytes);
        owner.grid_cache_host_bytes -= grid_entry_host_charge;
    }
};
struct GridCacheState {
    std::map<i64, std::shared_ptr<RegularGridTable>> entries;
};

void release_optional_grid_cache(Context& c) {
    if (!c.grid_state) return;
    auto& cache = *static_cast<GridCacheState*>(c.grid_state);
    for (const auto& entry : cache.entries) c.drop_optional(entry.second->bytes);
    cache.entries.clear();  // A live refit lease keeps its table resident.
    delete static_cast<GridCacheState*>(c.grid_state);
    c.grid_state = nullptr;
    c.grid_cache_host_bytes -= grid_state_host_charge;
}
void destroy_grid_cache(Context& c) noexcept {
    release_optional_grid_cache(c);
}

__global__ void regular_grid_kernel(ObservationSoA data, i64 region, i64 count,
                                    double* values) {
    for (i64 index = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<i64>(blockDim.x) * gridDim.x) {
        const i64 row = index % data.mutations;
        const double center = static_cast<double>(index / data.mutations) / 512.0;
        values[index] = -evaluate_observation<EmitMode::LossOnly>(
            observation(data, region * data.mutations + row), center).loss;
    }
}

std::shared_ptr<RegularGridTable> regular_grid(Context& c, i64 region) {
    if (c.grid_state) {
        auto& entries = static_cast<GridCacheState*>(c.grid_state)->entries;
        const auto found = entries.find(region);
        if (found != entries.end()) { ++c.grid_cache_hits; return found->second; }
    }
    ++c.grid_cache_misses;
    const i64 count = multiply(c.data.mutations, regular_grid_points);
    const i64 bytes = multiply(count, sizeof(double));
    // Optional misses always retain the direct CUDA path, including very small
    // device/host budgets. Mandatory scoring allocations have priority.
    if (bytes > c.optional_cache_limit || bytes > c.resident_limit) return {};
    const i64 host_bytes = grid_entry_host_charge + (c.grid_state ? 0 : grid_state_host_charge);
    if (host_bytes > c.host_staging_limit - c.diagnostics.staging_bytes - c.grid_cache_host_bytes
            - scoring_host_bytes(c))
        return {};
    if (bytes > c.resident_limit - c.resident_bytes) {
        release_optional_grid_cache(c);
        // Do not evict scoring-owned optional data merely to populate a grid.
        if (bytes > c.resident_limit - c.resident_bytes) return {};
    }
    if (!c.retain_optional(bytes)) return {};
    const i64 retained_host = grid_entry_host_charge + (c.grid_state ? 0 : grid_state_host_charge);
    if (retained_host > c.host_staging_limit - c.diagnostics.staging_bytes - c.grid_cache_host_bytes
            - scoring_host_bytes(c)) {
        c.drop_optional(bytes);
        return {};
    }
    unsigned char* allocation = nullptr;
    try {
        allocation = c.allocate_resident(bytes);
        if (!c.grid_state) {
            c.grid_state = new GridCacheState;
            c.grid_cache_host_bytes += grid_state_host_charge;
        }
        auto table = std::make_shared<RegularGridTable>(c, reinterpret_cast<double*>(allocation), bytes, region);
        c.grid_cache_host_bytes += grid_entry_host_charge;
        allocation = nullptr;  // The immutable table now owns the allocation.
        regular_grid_kernel<<<blocks(count), threads, 0, c.stream>>>(c.data, region, count, table->values);
        c.launched();
        c.grid_fresh_cells = add(c.grid_fresh_cells, count);
        static_cast<GridCacheState*>(c.grid_state)->entries.emplace(region, table);
        return table;
    } catch (...) {
        if (allocation) c.release_resident(allocation, bytes);
        c.drop_optional(bytes);
        throw;
    }
}

// The mandatory eight-byte-per-task host counter buffer doubles as pointer
// staging. Extra shared_ptr leases are optional and explicitly host-budgeted.
struct GridLeases {
    Context& owner;
    std::vector<std::shared_ptr<RegularGridTable>> tables;
    i64 host_bytes = 0;
    GridLeases(Context& c, i64 tasks) : owner(c) {
        const i64 bytes = multiply(tasks, sizeof(std::shared_ptr<RegularGridTable>));
        if (!c.optional_cache_limit || bytes > c.host_staging_limit
                - c.diagnostics.staging_bytes - c.grid_cache_host_bytes - scoring_host_bytes(c)) return;
        tables.resize(static_cast<std::size_t>(tasks));
        host_bytes = bytes;
        c.diagnostics.staging_bytes += bytes;
    }
    std::shared_ptr<RegularGridTable> get(i64 region) {
        // Eviction removes only retention, not an in-flight slice's leases.
        // Reuse those leases too, rather than allocating a second region table.
        for (const auto& table : tables)
            if (table && table->region == region) {
                ++owner.grid_cache_hits;
                return table;
            }
        return regular_grid(owner, region);
    }
    ~GridLeases() {
        if (std::uncaught_exceptions()) cudaStreamSynchronize(owner.stream);
        tables.clear();
        owner.diagnostics.staging_bytes -= host_bytes;
    }
};
