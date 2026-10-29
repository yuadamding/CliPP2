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
    std::vector<i64> demand;
    explicit GridCacheState(i64 regions) : demand(static_cast<std::size_t>(regions), 0) {}
};

void release_optional_grid_cache(Context& c) {
    if (!c.grid_state) return;
    auto& cache = *static_cast<GridCacheState*>(c.grid_state);
    for (const auto& entry : cache.entries) c.drop_optional(entry.second->bytes);
    delete static_cast<GridCacheState*>(c.grid_state);
    c.grid_state = nullptr;
    c.grid_cache_host_bytes -= grid_state_host_charge + multiply(c.data.regions, sizeof(i64));
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

void record_grid_demand(Context& c, i64 region, i64 direct_cells) {
    if (!c.grid_state) return;
    auto& state = *static_cast<GridCacheState*>(c.grid_state);
    if (state.entries.count(region)) return;
    const i64 count = multiply(c.data.mutations, regular_grid_points);
    auto& demand = state.demand[region];
    demand += std::min(count - demand, direct_cells);
}

std::shared_ptr<RegularGridTable> regular_grid(Context& c, i64 region) {
    if (c.grid_state) {
        auto& entries = static_cast<GridCacheState*>(c.grid_state)->entries;
        const auto found = entries.find(region);
        if (found != entries.end()) {
            ++c.grid_cache_hits;
            return found->second;
        }
    }
    ++c.grid_cache_misses;
    const i64 count = multiply(c.data.mutations, regular_grid_points);
    const i64 bytes = multiply(count, sizeof(double));
    const i64 state_charge = add(grid_state_host_charge, multiply(c.data.regions, sizeof(i64)));
    if (!c.grid_state) {
        c.grid_state = new GridCacheState(c.data.regions);
        c.grid_cache_host_bytes += state_charge;
    }
    auto& demand = static_cast<GridCacheState*>(c.grid_state)->demand[region];
    // Build only after prior successful direct refits performed a table's worth
    // of observed exact-grid cells. Missing rows and invalid boxes earn nothing;
    // sparse requests must demonstrate reuse before paying 513*N fresh cells.
    if (demand < count) { ++c.grid_demand_bypasses; return {}; }
    unsigned char* allocation = nullptr;
    try {
        allocation = c.allocate_resident(bytes);
        auto table = std::make_shared<RegularGridTable>(c, reinterpret_cast<double*>(allocation), bytes, region);
        c.grid_cache_host_bytes += grid_entry_host_charge;
        allocation = nullptr;  // The immutable table now owns the allocation.
        regular_grid_kernel<<<blocks(count), threads, 0, c.stream>>>(c.data, region, count, table->values);
        c.launched();
        c.grid_fresh_cells = add(c.grid_fresh_cells, count);
        c.grid_build_cells = add(c.grid_build_cells, count);
        static_cast<GridCacheState*>(c.grid_state)->entries.emplace(region, table);
        c.optional_cache_bytes = add(c.optional_cache_bytes, bytes);
        demand = 0;
        return table;
    } catch (...) {
        if (allocation) c.release_resident(allocation, bytes);
        throw;
    }
}

// Keep table ownership through every in-flight slice and transfer.
struct GridLeases {
    Context& owner;
    std::vector<std::shared_ptr<RegularGridTable>> tables;
    i64 host_bytes = 0;
    GridLeases(Context& c, i64 tasks) : owner(c) {
        const i64 bytes = multiply(tasks, sizeof(std::shared_ptr<RegularGridTable>));
        tables.resize(static_cast<std::size_t>(tasks));
        host_bytes = bytes;
        c.diagnostics.staging_bytes += bytes;
    }
    std::shared_ptr<RegularGridTable> get(i64 region) {
        // Reuse in-flight leases rather than looking up the same region again.
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
