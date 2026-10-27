// Internal staged refit; included by the single CUDA translation unit.
// Slices retain absolute CSR offsets, rebased only in device views. This avoids
// host packing arrays and preserves the eight-byte-per-task staging contract.
struct RefitLayout {
    Layout layout;
    i64 regions, offsets, rows, grid_offsets, grid, lower, upper, xatol,
        incumbent, compacted, counts, grid_values, results;
    RefitLayout(i64 tasks, i64 members, i64 points, bool has_incumbent) {
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
    }
};

struct RefitPlan {
    i64 last, members, points, max_rows, wave, scratch, bytes;
};

RefitPlan plan_refit_slice(const RefitBatch& batch, i64 first, i64 budget, i64 host_limit) {
    const i64 task_limit = host_limit / sizeof(i64);
    require(task_limit > 0, "CUDA refit host staging requires eight bytes per task");
    RefitPlan plan{first, 0, 0, 1, 0, 0, 0};
    while (plan.last < batch.tasks && plan.last - first < task_limit) {
        const i64 end = plan.last + 1;
        const i64 members = batch.member_offsets[end] - batch.member_offsets[first];
        const i64 points = batch.grid_offsets[end] - batch.grid_offsets[first];
        const i64 maximum = std::max(plan.max_rows,
            batch.member_offsets[end] - batch.member_offsets[plan.last]);
        RefitLayout buffers(end - first, members, points, batch.incumbent != nullptr);
        const i64 fixed = buffers.layout.bytes;
        // One arena is reused: grid-wave row values, then all adaptive tasks'
        // row values. Never reserve/grow after installing the slice pointers.
        const i64 minimum = multiply(std::max(members, maximum), sizeof(double));
        if (fixed > budget || minimum > budget - fixed) break;
        const i64 wave = std::min<i64>({points, 65535,
            (budget - fixed) / multiply(maximum, sizeof(double))});
        const i64 scratch = buffers.layout.take<double>(std::max(members, multiply(wave, maximum)));
        plan = {end, members, points, maximum, wave, scratch, buffers.layout.bytes};
    }
    require(plan.last > first,
            "Native CUDA refit workspace cannot hold one task and one grid evaluation");
    return plan;
}

__global__ void prepare_refit_kernel(ObservationSoA data, RefitBatch batch,
        i64* observed_rows, i64* observed_counts, RefitResult* output) {
    for (i64 task = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < batch.tasks; task += static_cast<i64>(blockDim.x) * gridDim.x) {
        const i64 begin = batch.member_offsets[task] - batch.member_offsets[0];
        const i64 end = batch.member_offsets[task + 1] - batch.member_offsets[0];
        const i64 region = batch.regions[task];
        i64 count = 0;
        double lower = 0, upper = 1;
        for (i64 member = begin; member < end; ++member) {
            const i64 row = batch.member_rows[member], cell = region * data.mutations + row;
            if (data.lower[cell] > lower) lower = data.lower[cell];
            if (data.upper[cell] < upper) upper = data.upper[cell];
            if (data.flags[cell] & observed_flag) observed_rows[begin + count++] = row;
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
    }
}

__global__ void refit_grid_kernel(ObservationSoA data, RefitBatch batch,
        const i64* observed_rows, const i64* observed_counts, const RefitResult* state,
        i64 first, i64 jobs, i64 row_stride, double* scratch, double* grid_values) {
    __shared__ double total;
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
    const double value = RefitObjective{data, observed_rows + begin, observed_counts[task],
        batch.regions[task], scratch + job * row_stride, &total}(batch.grid[point]);
    if (!threadIdx.x) grid_values[point] = value;
}

__global__ void finish_refit_kernel(ObservationSoA data, RefitBatch batch,
        const i64* observed_rows, const i64* observed_counts, double* scratch,
        const double* grid_values, RefitResult* output) {
    __shared__ double total;
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
            const auto result = refit_search_from_grid(local, 0, grid_values + point,
                RefitObjective{data, observed_rows + begin, observed_counts[task],
                    batch.regions[task], scratch + begin, &total});
            if (!threadIdx.x) output[task] = result;
        }
        __syncthreads();
    }
}

void refit_cuda(Context& c, const RefitBatch& batch, RefitResult* output) {
    for (i64 first = 0; first < batch.tasks;) {
        const auto plan = plan_refit_slice(batch, first, c.diagnostics.workspace_limit, c.host_staging_limit);
        const i64 tasks = plan.last - first;
        HostCounts observed_counts(c, tasks);
        const RefitLayout b(tasks, plan.members, plan.points, batch.incumbent != nullptr);
        c.reserve(plan.bytes);
        RefitBatch device{tasks, plan.members, plan.points,
            c.pointer<i64>(b.regions), c.pointer<i64>(b.offsets), c.pointer<i64>(b.rows),
            c.pointer<i64>(b.grid_offsets), c.pointer<double>(b.grid), c.pointer<double>(b.lower),
            c.pointer<double>(b.upper), c.pointer<double>(b.xatol),
            batch.incumbent ? c.pointer<double>(b.incumbent) : nullptr};
        c.upload(c.pointer<i64>(b.regions), batch.regions + first, tasks);
        c.upload(c.pointer<i64>(b.offsets), batch.member_offsets + first, add(tasks, 1));
        c.upload(c.pointer<i64>(b.rows), batch.member_rows + batch.member_offsets[first], plan.members);
        c.upload(c.pointer<i64>(b.grid_offsets), batch.grid_offsets + first, add(tasks, 1));
        c.upload(c.pointer<double>(b.grid), batch.grid + batch.grid_offsets[first], plan.points);
        c.upload(c.pointer<double>(b.lower), batch.lower + first, tasks);
        c.upload(c.pointer<double>(b.upper), batch.upper + first, tasks);
        c.upload(c.pointer<double>(b.xatol), batch.xatol + first, tasks);
        if (batch.incumbent) c.upload(c.pointer<double>(b.incumbent), batch.incumbent + first, tasks);
        prepare_refit_kernel<<<blocks(tasks), threads, 0, c.stream>>>(c.data, device,
            c.pointer<i64>(b.compacted), c.pointer<i64>(b.counts), c.pointer<RefitResult>(b.results));
        c.launched();
        for (i64 point = 0; point < plan.points;) {
            const i64 jobs = std::min(plan.wave, plan.points - point);
            refit_grid_kernel<<<static_cast<unsigned>(jobs), threads, 0, c.stream>>>(c.data, device,
                c.pointer<i64>(b.compacted), c.pointer<i64>(b.counts), c.pointer<RefitResult>(b.results),
                point, jobs, plan.max_rows, c.pointer<double>(plan.scratch), c.pointer<double>(b.grid_values));
            c.launched(); point += jobs;
        }
        const unsigned task_blocks = static_cast<unsigned>(std::min<i64>(65535, tasks));
        finish_refit_kernel<<<task_blocks, threads, 0, c.stream>>>(c.data, device,
            c.pointer<i64>(b.compacted), c.pointer<i64>(b.counts), c.pointer<double>(plan.scratch),
            c.pointer<double>(b.grid_values), c.pointer<RefitResult>(b.results));
        c.launched();
        c.download(output + first, c.pointer<RefitResult>(b.results), tasks);
        c.download(observed_counts.values.data(), c.pointer<i64>(b.counts), tasks);
        c.finish();
        for (i64 task = 0; task < tasks; ++task)
            c.diagnostics.cell_evaluations = add(c.diagnostics.cell_evaluations,
                multiply(observed_counts.values[task], output[first + task].evaluations));
        first = plan.last;
    }
}
