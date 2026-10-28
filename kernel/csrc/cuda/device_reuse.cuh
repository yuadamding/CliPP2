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
    i64 bytes, region, last_use = 0;
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

bool evict_optional_grid_cache(Context& c) {
    if (!c.grid_state) return false;
    auto& cache = *static_cast<GridCacheState*>(c.grid_state);
    if (!cache.entries.empty()) {
        auto cold = cache.entries.begin();
        for (auto entry = std::next(cold); entry != cache.entries.end(); ++entry) {
            const bool free = entry->second.use_count() == 1;
            const bool cold_free = cold->second.use_count() == 1;
            if ((free && !cold_free) || (free == cold_free &&
                    entry->second->last_use < cold->second->last_use)) cold = entry;
        }
        c.drop_optional(cold->second->bytes);
        c.grid_retained_bytes -= cold->second->bytes;
        cache.entries.erase(cold);  // In-flight leases remain valid and charged.
        ++c.grid_evictions;
    } else {
        c.grid_cache_host_bytes -= grid_state_host_charge + multiply(c.data.regions, sizeof(i64));
        delete static_cast<GridCacheState*>(c.grid_state);
        c.grid_state = nullptr;
    }
    return true;
}
void release_optional_grid_cache(Context& c) {
    while (evict_optional_grid_cache(c)) {}
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
            found->second->last_use = c.cache_clock = add(c.cache_clock, 1);
            return found->second;
        }
    }
    ++c.grid_cache_misses;
    const i64 count = multiply(c.data.mutations, regular_grid_points);
    const i64 bytes = multiply(count, sizeof(double));
    // Optional misses always retain the direct CUDA path, including very small
    // device/host budgets. Mandatory scoring allocations have priority.
    if (bytes > c.optional_cache_limit || bytes > c.resident_limit) return {};
    const i64 state_charge = add(grid_state_host_charge, multiply(c.data.regions, sizeof(i64)));
    const i64 host_bytes = grid_entry_host_charge + (c.grid_state ? 0 : state_charge);
    if (host_bytes > c.host_staging_limit - c.diagnostics.staging_bytes - c.grid_cache_host_bytes
            - c.probability_host_bytes - scoring_host_bytes(c))
        return {};
    if (!c.grid_state) {
        c.grid_state = new GridCacheState(c.data.regions);
        c.grid_cache_host_bytes += state_charge;
    }
    auto& demand = static_cast<GridCacheState*>(c.grid_state)->demand[region];
    // Build only after prior successful direct refits performed a table's worth
    // of observed exact-grid cells. Missing rows and invalid boxes earn nothing;
    // sparse requests must demonstrate reuse before paying 513*N fresh cells.
    if (demand < count) { ++c.grid_admission_bypasses; return {}; }
    if (bytes > c.optional_cache_limit - c.grid_retained_bytes) {
        ++c.grid_admission_bypasses;
        return {};
    }
    // Do not cycle through regional tables if the working set cannot fit.
    // Only cold scoring retention may yield to a newly justified grid table.
    while (bytes > c.resident_limit - c.resident_bytes && evict_optional_scoring_cache(c)) {}
    if (bytes > c.resident_limit - c.resident_bytes || !c.retain_optional(bytes)) {
        ++c.grid_admission_bypasses;
        return {};
    }
    const i64 retained_host = grid_entry_host_charge;
    if (retained_host > c.host_staging_limit - c.diagnostics.staging_bytes - c.grid_cache_host_bytes
            - c.probability_host_bytes - scoring_host_bytes(c)) {
        c.drop_optional(bytes);
        return {};
    }
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
        table->last_use = c.cache_clock = add(c.cache_clock, 1);
        static_cast<GridCacheState*>(c.grid_state)->entries.emplace(region, table);
        c.grid_retained_bytes += bytes;
        demand = 0;
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
                - c.diagnostics.staging_bytes - c.grid_cache_host_bytes
                - c.probability_host_bytes - scoring_host_bytes(c)) return;
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
