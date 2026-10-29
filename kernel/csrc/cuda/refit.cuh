// Internal staged refit; included by the single CUDA translation unit.
// Batches retain absolute CSR offsets, rebased only in device views. This avoids
// host packing arrays; exact shape reuse keeps its maps in the same arena.
struct RefitLayout {
    Layout layout;
    i64 regions, offsets, rows, grid_offsets, grid, lower, upper, xatol,
        incumbent, compacted, counts, grid_values, results, grid_tables, probability_flags,
        shape_counts, shape_ids, row_shapes, shape_hash_keys, shape_hash_values;
    RefitLayout(i64 tasks, i64 members, i64 points, bool has_incumbent, bool probability) {
        regions = layout.take<i64>(tasks);
        offsets = layout.take<i64>(add(tasks, 1));
        rows = layout.take<i64>(members);
        grid_offsets = layout.take<i64>(add(tasks, 1));
        grid = layout.take<double>(points);
        lower = layout.take<double>(tasks);
        upper = layout.take<double>(tasks);
        xatol = layout.take<double>(tasks);
        incumbent = layout.take<double>(has_incumbent ? tasks : 0);
        compacted = layout.take<i64>(members);
        counts = layout.take<i64>(tasks);
        grid_values = layout.take<double>(points);
        results = layout.take<RefitResult>(tasks);
        grid_tables = layout.take<i64>(tasks);
        probability_flags = layout.take<i64>(probability ? tasks : 0);
        shape_counts = layout.take<i64>(probability ? tasks : 0);
        shape_ids = layout.take<i64>(probability ? members : 0);
        row_shapes = layout.take<i64>(probability ? members : 0);
        // Each task owns a power-of-two table below four times its members.
        // Fixed 4*member CSR spans avoid extra host offsets or allocations.
        shape_hash_keys = layout.take<i64>(probability ? multiply(members,4) : 0);
        shape_hash_values = layout.take<i64>(probability ? multiply(members,4) : 0);
    }
};

struct RefitPlan {
    i64 members, points, max_rows, stride, wave, scratch, bytes;
};

RefitPlan plan_refit_batch(const RefitBatch& batch, i64 wave_size, i64 probability_components) {
    const bool probability=probability_components>0;
    i64 maximum=1;
    for(i64 task=0;task<batch.tasks;++task)
        maximum=std::max(maximum,batch.member_offsets[task+1]-batch.member_offsets[task]);
    const i64 members=batch.member_offsets[batch.tasks]-batch.member_offsets[0];
    const i64 points=batch.grid_offsets[batch.tasks]-batch.grid_offsets[0];
    RefitLayout buffers(batch.tasks,members,points,batch.incumbent!=nullptr,probability);
    // Grid points are independent blocks. Wave width follows hardware block
    // concurrency, not a byte ceiling; adaptive objectives reuse this arena.
    const i64 stride=add(multiply(maximum,probability?2:1),multiply(probability_components,2));
    const i64 adaptive=probability?multiply(batch.tasks,stride):members;
    const i64 wave=std::min<i64>({points,65535,wave_size});
    const i64 scratch=buffers.layout.take<double>(std::max(adaptive,multiply(wave,stride)));
    return {members,points,maximum,stride,wave,scratch,buffers.layout.bytes};
}

__global__ void prepare_refit_kernel(ObservationSoA data, RefitBatch batch,
        i64* observed_rows, i64* observed_counts, RefitResult* output,
        CUDAProbabilityView probability,i64* probability_flags,
        i64* shape_counts,i64* shape_ids,i64* row_shapes,i64* hash_keys,i64* hash_values) {
    for (i64 task = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < batch.tasks; task += static_cast<i64>(blockDim.x) * gridDim.x) {
        const i64 begin = batch.member_offsets[task] - batch.member_offsets[0];
        const i64 end = batch.member_offsets[task + 1] - batch.member_offsets[0];
        const i64 region = batch.regions[task];
        i64 count = 0, components=0;
        double lower = 0, upper = 1;
        for (i64 member = begin; member < end; ++member) {
            const i64 row = batch.member_rows[member], cell = region * data.mutations + row;
            if (data.lower[cell] > lower) lower = data.lower[cell];
            if (data.upper[cell] < upper) upper = data.upper[cell];
            if (data.flags[cell] & observed_flag) {
                observed_rows[begin + count++] = row;
                if(probability.regions) components=data.support[cell]>INT64_MAX-components
                    ?INT64_MAX:components+data.support[cell];
            }
        }
        int status = 0;
        // Includes the scalar search's box prerequisite, before any grid work.
        if (!(finite_number(batch.lower[task]) && finite_number(batch.upper[task]) &&
              0 <= batch.lower[task] && batch.lower[task] <= batch.upper[task] &&
              batch.upper[task] <= 1) || batch.lower[task] < lower || batch.upper[task] > upper)
            status = static_cast<int>(RefitStatus::InvalidBox);
        else if (!count) status = static_cast<int>(RefitStatus::Unsupported);
        observed_counts[task] = count;
        output[task] = {quiet_nan, quiet_nan, quiet_nan, quiet_nan, 0, status};
        if(probability_flags) probability_flags[task]=!status && probability.regions && count>=32 &&
            probability.regions[region].components<components/2;
        if(shape_counts) {
            shape_counts[task]=0;
            // Catalog cardinality is a cheap conservative heuristic. A task
            // can decline reuse without changing its existing basis/row path.
            if(probability_flags[task] && probability.regions[region].shapes<count) {
                for(i64 i=0;i<count;++i)
                    row_shapes[begin+i]=probability.row_shapes[region*data.mutations+observed_rows[begin+i]];
                const auto slots=probability_shape_slots(end-begin);
                const auto unique=probability_task_shapes(count,slots,row_shapes+begin,shape_ids+begin,
                    hash_keys+4*begin,hash_values+4*begin);
                // Expansion/storage must amortize the map and extra barrier.
                if(unique>0 && unique<=count-count/4) shape_counts[task]=unique;
            }
        }
    }
}

__global__ void refit_grid_kernel(ObservationSoA data, RefitBatch batch,
        const i64* observed_rows, const i64* observed_counts, const RefitResult* state,
        i64 first, i64 jobs, i64 row_stride, i64 max_rows, double* scratch, double* grid_values,
        const i64* grid_tables,CUDAProbabilityView probability,const i64* probability_flags,
        const i64* shape_counts,const i64* shape_ids,const i64* row_shapes) {
    __shared__ double total;
    __shared__ double partials[128];
    const i64 job = blockIdx.x;
    if (job >= jobs) return;
    const i64 point = first + job, absolute = point + batch.grid_offsets[0];
    i64 lo = 0, hi = batch.tasks;
    while (lo + 1 < hi) {
        const i64 middle = lo + (hi - lo) / 2;
        if (batch.grid_offsets[middle] <= absolute) lo = middle;
        else hi = middle;
    }
    const i64 task = lo;
    if (state[task].status) return;  // Block-uniform; no skipped barrier participant.
    const i64 begin = batch.member_offsets[task] - batch.member_offsets[0];
    double* values = scratch + job * row_stride;
    const int grid_index = regular_grid_index(batch.grid[point]);
    const double* table = grid_tables ? reinterpret_cast<const double*>(grid_tables[task]) : nullptr;
    double value;
    if (table && grid_index >= 0) {
        for (i64 i = threadIdx.x; i < observed_counts[task]; i += blockDim.x)
            values[i] = table[static_cast<i64>(grid_index) * data.mutations + observed_rows[begin + i]];
        value = ordered_block_sum(values, observed_counts[task], partials, &total);
    } else {
        value = RefitObjective{data, observed_rows + begin, observed_counts[task],
            batch.regions[task], values, partials, &total,probability,
            probability_flags && probability_flags[task]?values+max_rows:nullptr,
            shape_ids?shape_ids+begin:nullptr,row_shapes?row_shapes+begin:nullptr,
            shape_counts?shape_counts[task]:0,
            shape_counts && probability.regions
                ?values+max_rows+2*probability.regions[batch.regions[task]].components:nullptr}(batch.grid[point]);
    }
    if (!threadIdx.x) grid_values[point] = value;
}

__global__ void finish_refit_kernel(ObservationSoA data, RefitBatch batch,
        const i64* observed_rows, const i64* observed_counts, double* scratch,
        const double* grid_values, RefitResult* output,i64 row_stride,i64 max_rows,
        CUDAProbabilityView probability,const i64* probability_flags,
        const i64* shape_counts,const i64* shape_ids,const i64* row_shapes) {
    __shared__ double total;
    __shared__ double partials[128];
    __shared__ int initial_status;
    for (i64 task = blockIdx.x; task < batch.tasks; task += gridDim.x) {
        if (!threadIdx.x) initial_status = output[task].status;
        __syncthreads();
        if (!initial_status) {
            const i64 begin = batch.member_offsets[task] - batch.member_offsets[0];
            const i64 point = batch.grid_offsets[task] - batch.grid_offsets[0];
            // A one-task view lets the shared scalar helper consume normalized
            // offsets without allocating/repacking CSR arrays on the host.
            const i64 offsets[2] = {0, batch.grid_offsets[task + 1] - batch.grid_offsets[task]};
            RefitBatch local = batch;
            local.grid = batch.grid + point;
            local.grid_offsets = offsets;
            local.lower = batch.lower + task; local.upper = batch.upper + task;
            local.xatol = batch.xatol + task;
            local.incumbent = batch.incumbent ? batch.incumbent + task : nullptr;
            auto* values=scratch+(probability_flags?task*row_stride:begin);
            const auto result = refit_search_from_grid(local, 0, grid_values + point,
                RefitObjective{data, observed_rows + begin, observed_counts[task],
                    batch.regions[task], values, partials, &total,probability,
                    probability_flags && probability_flags[task]?values+max_rows:nullptr,
                    shape_ids?shape_ids+begin:nullptr,row_shapes?row_shapes+begin:nullptr,
                    shape_counts?shape_counts[task]:0,
                    shape_counts && probability.regions
                        ?values+max_rows+2*probability.regions[batch.regions[task]].components:nullptr});
            if (!threadIdx.x) output[task] = result;
        }
        __syncthreads();
    }
}

void refit_cuda(Context& c, const RefitBatch& batch, RefitResult* output) {
    i64 probability_components=0;
    if(c.probability_state) {
        const auto& state=*static_cast<CUDAProbabilityState*>(c.probability_state);
        for(const auto& region:state.regions)
            probability_components=std::max(probability_components,region.components);
    }
    const bool probability=probability_components>0;
    int multiprocessors=0;
    cuda_check(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,c.device),
               "query refit block concurrency");
    const auto plan=plan_refit_batch(batch,multiply(std::max(1,multiprocessors),2),probability_components);
    const i64 tasks = batch.tasks;
    HostCounts observed_counts(c, multiply(tasks,probability?3:1));
    CUDAProbabilityLease probability_lease(c,probability);
    const auto probability_view=probability_lease.view();
    GridLeases cached(c, tasks);
    const RefitLayout b(tasks, plan.members, plan.points, batch.incumbent != nullptr,probability);
    c.reserve(plan.bytes);
    for (i64 task = 0; task < tasks; ++task) {
        if (batch.lower[task] == 0 && batch.upper[task] == 1)
            cached.tables[task] = cached.get(batch.regions[task]);
        observed_counts.values[task] = !cached.tables[task] ? 0
            : reinterpret_cast<i64>(cached.tables[task]->values);
    }
    c.upload(c.pointer<i64>(b.grid_tables), observed_counts.values.data(), tasks);
    RefitBatch device{tasks, plan.members, plan.points,
        c.pointer<i64>(b.regions), c.pointer<i64>(b.offsets), c.pointer<i64>(b.rows),
        c.pointer<i64>(b.grid_offsets), c.pointer<double>(b.grid), c.pointer<double>(b.lower),
        c.pointer<double>(b.upper), c.pointer<double>(b.xatol),
        batch.incumbent ? c.pointer<double>(b.incumbent) : nullptr};
    c.upload(c.pointer<i64>(b.regions), batch.regions, tasks);
    c.upload(c.pointer<i64>(b.offsets), batch.member_offsets, add(tasks, 1));
    c.upload(c.pointer<i64>(b.rows), batch.member_rows + batch.member_offsets[0], plan.members);
    c.upload(c.pointer<i64>(b.grid_offsets), batch.grid_offsets, add(tasks, 1));
    c.upload(c.pointer<double>(b.grid), batch.grid + batch.grid_offsets[0], plan.points);
    c.upload(c.pointer<double>(b.lower), batch.lower, tasks);
    c.upload(c.pointer<double>(b.upper), batch.upper, tasks);
    c.upload(c.pointer<double>(b.xatol), batch.xatol, tasks);
    if (batch.incumbent) c.upload(c.pointer<double>(b.incumbent), batch.incumbent, tasks);
    prepare_refit_kernel<<<blocks(tasks), threads, 0, c.stream>>>(c.data, device,
        c.pointer<i64>(b.compacted), c.pointer<i64>(b.counts), c.pointer<RefitResult>(b.results),
        probability_view,probability?c.pointer<i64>(b.probability_flags):nullptr,
        probability?c.pointer<i64>(b.shape_counts):nullptr,
        probability?c.pointer<i64>(b.shape_ids):nullptr,
        probability?c.pointer<i64>(b.row_shapes):nullptr,
        probability?c.pointer<i64>(b.shape_hash_keys):nullptr,
        probability?c.pointer<i64>(b.shape_hash_values):nullptr);
    c.launched();
    for (i64 point = 0; point < plan.points;) {
        const i64 jobs = std::min(plan.wave, plan.points - point);
        refit_grid_kernel<<<static_cast<unsigned>(jobs), threads, 0, c.stream>>>(c.data, device,
            c.pointer<i64>(b.compacted), c.pointer<i64>(b.counts), c.pointer<RefitResult>(b.results),
            point, jobs, plan.stride,plan.max_rows,c.pointer<double>(plan.scratch), c.pointer<double>(b.grid_values),
            c.pointer<i64>(b.grid_tables),probability_view,
            probability?c.pointer<i64>(b.probability_flags):nullptr,
            probability?c.pointer<i64>(b.shape_counts):nullptr,
            probability?c.pointer<i64>(b.shape_ids):nullptr,
            probability?c.pointer<i64>(b.row_shapes):nullptr);
        c.launched(); point += jobs;
    }
    const unsigned task_blocks = static_cast<unsigned>(std::min<i64>(65535, tasks));
    finish_refit_kernel<<<task_blocks, threads, 0, c.stream>>>(c.data, device,
        c.pointer<i64>(b.compacted), c.pointer<i64>(b.counts), c.pointer<double>(plan.scratch),
        c.pointer<double>(b.grid_values), c.pointer<RefitResult>(b.results),plan.stride,plan.max_rows,
        probability_view,probability?c.pointer<i64>(b.probability_flags):nullptr,
        probability?c.pointer<i64>(b.shape_counts):nullptr,
        probability?c.pointer<i64>(b.shape_ids):nullptr,
        probability?c.pointer<i64>(b.row_shapes):nullptr);
    c.launched();
    c.download(output, c.pointer<RefitResult>(b.results), tasks);
    c.download(observed_counts.values.data(), c.pointer<i64>(b.counts), tasks);
    if(probability) c.download(observed_counts.values.data()+tasks,c.pointer<i64>(b.probability_flags),tasks);
    if(probability) c.download(observed_counts.values.data()+2*tasks,c.pointer<i64>(b.shape_counts),tasks);
    c.finish();
    for (i64 task = 0; task < tasks; ++task) {
        i64 regular_points = 0;
        if (batch.lower[task] == 0 && batch.upper[task] == 1
                && output[task].status != static_cast<int>(RefitStatus::InvalidBox)
                && output[task].status != static_cast<int>(RefitStatus::Unsupported)) {
            for (i64 point = batch.grid_offsets[task];
                 point < batch.grid_offsets[task + 1]; ++point)
                regular_points += regular_grid_index(batch.grid[point]) >= 0;
        }
        const i64 reused_points = cached.tables[task] ? regular_points : 0;
        const i64 reused = multiply(observed_counts.values[task], reused_points);
        if (output[task].status == static_cast<int>(RefitStatus::Ok))
            record_grid_demand(c, batch.regions[task],
                multiply(observed_counts.values[task], regular_points - reused_points));
        c.grid_reused_cells = add(c.grid_reused_cells, reused);
        const i64 logical = multiply(observed_counts.values[task], output[task].evaluations);
        if(probability && observed_counts.values[tasks+task]) {
            const auto evaluations=output[task].evaluations-reused_points;
            const auto components=probability_lease.state->regions[batch.regions[task]].components;
            ++c.probability_refit_counters[0];
            c.probability_refit_counters[1]=add(c.probability_refit_counters[1],evaluations);
            c.probability_refit_counters[2]=add(c.probability_refit_counters[2],multiply(components,evaluations));
            c.probability_refit_counters[3]=add(c.probability_refit_counters[3],logical-reused);
            if(observed_counts.values[2*tasks+task]>0) {
                ++c.probability_shape_refit_counters[0];
                c.probability_shape_refit_counters[1]=add(c.probability_shape_refit_counters[1],evaluations);
                c.probability_shape_refit_counters[2]=add(c.probability_shape_refit_counters[2],
                    multiply(observed_counts.values[2*tasks+task],evaluations));
                c.probability_shape_refit_counters[3]=add(c.probability_shape_refit_counters[3],logical-reused);
            }
        }
        c.grid_fresh_cells = add(c.grid_fresh_cells, logical - reused);
        c.grid_direct_cells = add(c.grid_direct_cells, logical - reused);
        c.diagnostics.cell_evaluations = add(c.diagnostics.cell_evaluations,
            logical);
    }
}
