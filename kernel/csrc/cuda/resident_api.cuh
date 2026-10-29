// Explicit experimental complete-weight ABI; production dispatch stays off.
// Include at global scope after scoring.cuh. Uses ordinary fixed CUDA graphs,
// not conditional nodes. There is no host numerical or CPU fallback.
#pragma once
#include "resident_weight_graph.cuh"
#include <chrono>

namespace {
struct ResidentWeightPlan {
    Layout layout;
    i64 descriptor_offset,state_offset,initial_offset,flags_offset,arena_start;
    i64 total_q=0,max_q=0,host_charge,graph_charge;
    bool graph;
    template<class Sizes>
    ResidentWeightPlan(i64 n,i64 count,Sizes size,bool force_graph=false):graph(force_graph||n>small_mixture_rows) {
        if(count<1)throw std::domain_error("Resident admission requires a nonempty batch");
        if(graph&&count>65535)throw std::domain_error("Resident batch exceeds CUDA grid.y capacity");
        for(i64 index=0;index<count;++index) {
            const i64 q=size(index);
            if(!ResidentWeightState::valid_dimensions(n,q))
                throw std::domain_error("Resident SLSQP exceeds its qualified LP64 native-state range");
            graph=graph||q>small_mixture_columns;
            max_q=std::max(max_q,q);total_q=add(total_q,q);
        }
        if(graph&&count>65535)throw std::domain_error("Resident batch exceeds CUDA grid.y capacity");
        // Native vector owners/capacities and fixed frame overhead are charged
        // conservatively; graph internals are a separate labelled reservation.
        host_charge=add(2048,multiply(count,add(sizeof(ResidentWeightItem)+sizeof(ResidentWeightState),112)));
        graph_charge=graph?add(65536,multiply(resident_chunk_nodes,4096)):0;
        descriptor_offset=layout.take<ResidentWeightItem>(count);
        state_offset=layout.take<ResidentWeightState>(count);
        initial_offset=layout.take<double>(total_q);
        flags_offset=layout.take<int>(graph?count:0);
        arena_start=layout.bytes;
        for(i64 index=0;index<count;++index)take_problem(layout,size(index));
    }
    static std::pair<i64,i64> take_problem(Layout& layout,i64 q) {
        const i64 arena=layout.take<double>(ResidentWeightState::doubles(static_cast<int>(q)));
        const i64 indices=layout.take<int>(add(multiply(2,q),3));
        return {arena,indices};
    }
};
struct ResidentGraphOwner {
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    ~ResidentGraphOwner() { if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph); }
    void close() {
        if(executable) { auto value=executable;executable=nullptr;cuda_check(cudaGraphExecDestroy(value),"resident graph exec release"); }
        if(graph) { auto value=graph;graph=nullptr;cuda_check(cudaGraphDestroy(value),"resident graph release"); }
    }
};

// One stable allocation and graph for an exact ordered geometry. Descriptor
// contents are replaced for every solve; graph nodes never own mixture handles.
// A different geometry discards the idle workspace before creating its replacement.
struct ResidentExecutionWorkspace {
    Context& context;
    ResidentWeightPlan plan;
    i64 n,count,host_charge=0,graph_reservation=0,optional_charge=0,retention_charge=0;
    std::vector<i64> sizes;
    std::vector<ResidentWeightItem> items;
    std::vector<ResidentWeightState> completed;
    std::vector<int> terminal_flags;
    ScoringAllocation allocation;
    ResidentGraphOwner graph;

    template<class Sizes>
    ResidentExecutionWorkspace(Context& c,const ResidentWeightPlan& geometry,i64 rows,i64 tasks,Sizes size):
        context(c),plan(geometry),n(rows),count(tasks),sizes(static_cast<std::size_t>(tasks)),
        items(static_cast<std::size_t>(tasks)),completed(static_cast<std::size_t>(tasks)),
        terminal_flags(static_cast<std::size_t>(plan.graph?tasks:0)),allocation(c,plan.layout.bytes) {
        for(i64 index=0;index<count;++index)sizes[index]=size(index);
        // Opaque graph storage is a labelled admission reservation, not an
        // assertion about physical CUDA allocation size. Charge it in both
        // native host and tracked device budgets; qualification measures peaks.
        while(plan.graph_charge>c.resident_limit-c.resident_bytes &&
              (evict_optional_scoring_cache(c)||evict_optional_grid_cache(c)||evict_optional_probability_cache(c))) {}
        cuda_check(c.resident_release_error,"release storage before resident graph reservation");
        require(plan.graph_charge<=c.resident_limit-c.resident_bytes,
                "Resident graph reservation exceeds device budget");
        graph_reservation=plan.graph_charge;
        c.resident_bytes=add(c.resident_bytes,graph_reservation);
        c.resident_peak_bytes=std::max(c.resident_peak_bytes,c.resident_bytes);
        c.diagnostics.cache_bytes=c.resident_bytes;
    }
    ~ResidentExecutionWorkspace() {
        bool released=true;
        try { graph.close(); }
        catch(...) {
            released=false;
            if(context.resident_release_error==cudaSuccess)context.resident_release_error=cudaErrorUnknown;
        }
        // The graph is destroyed before any referenced arena is freed.
        if(released) {
            context.resident_bytes-=graph_reservation;
            context.diagnostics.cache_bytes=context.resident_bytes;
        }
        if(host_charge)static_cast<ScoringState*>(context.scoring_state)->host_bytes-=host_charge;
    }
    template<class Sizes>
    bool compatible(const ResidentWeightPlan& geometry,i64 rows,i64 tasks,Sizes size) const {
        if(n!=rows||count!=tasks||plan.graph!=geometry.graph||plan.layout.bytes!=geometry.layout.bytes)return false;
        for(i64 index=0;index<count;++index)if(sizes[index]!=size(index))return false;
        return true;
    }
};

bool release_resident_execution_cache(Context& c,ScoringState& state) noexcept {
    if(!state.resident_execution_cache)return false;
    c.drop_optional(state.resident_execution_cache->optional_charge);
    state.resident_execution_cache->optional_charge=0;
    state.resident_execution_cache.reset();
    ++state.resident_execution_counters[5];
    return true;
}

struct ResidentExecutionLease {
    Context& context;
    std::shared_ptr<ResidentExecutionWorkspace> workspace;
    bool complete=false;
    template<class Sizes>
    ResidentExecutionLease(Context& c,const ResidentWeightPlan& plan,i64 count,Sizes size):context(c) {
        auto& state=scoring_state(c);
        if(state.resident_execution_busy.exchange(true))
            throw std::domain_error("A resident execution workspace already has a submission owner");
        try {
            if(state.resident_execution_cache &&
                    state.resident_execution_cache->compatible(plan,c.data.mutations,count,size)) {
                c.drop_optional(state.resident_execution_cache->optional_charge);
                state.resident_execution_cache->optional_charge=0;
                workspace=std::move(state.resident_execution_cache);
                ++state.resident_execution_counters[1];
                if(plan.graph)++state.resident_execution_counters[3];
            } else {
                release_resident_execution_cache(c,state);
                const i64 host_bytes=add(plan.host_charge,plan.graph_charge);
                ScoringHostScope host(c,host_bytes);
                workspace=std::make_shared<ResidentExecutionWorkspace>(c,plan,c.data.mutations,count,size);
                workspace->retention_charge=add(add(workspace->allocation.bytes,workspace->graph_reservation),host_bytes);
                const i64 persistent=add(state.host_bytes,host_bytes);
                workspace->host_charge=host_bytes;
                state.host_bytes=persistent;
                host.release();
                ++state.resident_execution_counters[0];
            }
        } catch(...) {
            state.resident_execution_busy=false;
            throw;
        }
    }
    ~ResidentExecutionLease() noexcept {
        auto& state=*static_cast<ScoringState*>(context.scoring_state);
        if(!complete)++state.resident_execution_counters[6];
        // Completed scratch has lower retention priority than computed data.
        // Do not evict columns or grids merely to preserve graph setup work.
        const i64 charge=workspace->retention_charge;
        if(complete && charge<=context.optional_cache_limit-context.optional_cache_bytes &&
                context.retain_optional(charge)) {
            workspace->optional_charge=charge;
            state.resident_execution_cache=std::move(workspace);
        }
        workspace.reset();
        state.resident_execution_busy=false;
    }
};

int experimental_resident_weights(void* value,i64 count,const i64* handles,
        const double* starts,double* weights,double* likelihood,double* optimality,
        double* active,int* statuses,i64* counters,bool force_graph) {
    return scoring_guarded([&] {
        auto& c=context(value);
        if(count<1||!handles||!starts||!weights||!likelihood||!optimality||!active||!statuses||!counters)
            throw std::domain_error("Missing resident weight request fields");
        const ResidentWeightPlan plan(c.data.mutations,count,[&](i64 index) {
            return scoring_problem(c,handles[index]).q;
        },force_graph);
        i64 start_offset=0;
        for(i64 index=0;index<count;++index) {
            auto& p=scoring_problem(c,handles[index]);
            for(i64 prior=0;prior<index;++prior)if(handles[index]==handles[prior])
                throw std::domain_error("Resident optimizers require independent mixture states");
            validate_scoring_weights(starts+start_offset,p.q);start_offset=add(start_offset,p.q);
        }
        ResidentExecutionLease lease(c,plan,count,[&](i64 index) { return scoring_problem(c,handles[index]).q; });
        auto& workspace=*lease.workspace;
        auto& allocation=workspace.allocation;auto& graph=workspace.graph;
        auto& items=workspace.items;auto& completed=workspace.completed;
        auto& terminal_flags=workspace.terminal_flags;
        ScoringStreamFence fence(c);
        // Reset cached endpoints, optimizer/arena/index storage and every
        // descriptor slot before uploading this solve's independent inputs.
        cuda_check(cudaMemsetAsync(allocation.data,0,static_cast<std::size_t>(allocation.bytes),c.stream),
                   "reset resident execution workspace");
        std::fill(items.begin(),items.end(),ResidentWeightItem{});
        std::fill(terminal_flags.begin(),terminal_flags.end(),0);
        ++scoring_state(c).resident_execution_counters[7];
        i64 cursor=0;
        Layout arena_layout;arena_layout.bytes=plan.arena_start;
        for(i64 index=0;index<count;++index) {
            auto& p=scoring_problem(c,handles[index]);auto& item=items[index];auto& target=item.problem;
            target.columns=p.at<const double*>(p.pointers);target.offsets=p.at<double>(p.offset);
            target.kernel=p.at<double>(p.kernel);target.logs=p.at<double>(p.log_weights);
            target.other=p.at<double>(p.other);target.grow_log=p.at<double>(p.grow_log);
            target.shrink_log=p.at<double>(p.shrink_log);target.numerator=p.at<double>(p.numerator);
            target.sign=p.at<double>(p.sign);target.n=p.n;target.q=p.q;
            item.full_chunks=p.reduction.full_chunks;item.tail_count=p.reduction.tail_count;
            item.leaf_count=p.reduction.leaf_count;item.chunks=p.chunks;
            item.tail=p.at<ScoringLeaf>(p.leaf_metadata);item.leaf_partials=p.at<double>(p.leaf_partials);
            item.chunk_partials=p.at<double>(p.partials);
            const auto offsets=ResidentWeightPlan::take_problem(arena_layout,p.q);
            item.state=allocation.at<ResidentWeightState>(plan.state_offset)+index;
            item.arena=allocation.at<double>(offsets.first);
            item.indices=allocation.at<int>(offsets.second);
            item.initial=allocation.at<double>(plan.initial_offset)+cursor;
            cursor+=p.q;p.pair_ready=false;
        }
        c.upload(allocation.at<double>(plan.initial_offset),starts,plan.total_q);
        c.upload(allocation.at<ResidentWeightItem>(plan.descriptor_offset),items.data(),count);
        if(plan.graph) {
            auto* device_items=allocation.at<ResidentWeightItem>(plan.descriptor_offset);
            if(!graph.executable) {
                const auto started=std::chrono::steady_clock::now();
                cuda_check(cudaGraphCreate(&graph.graph,0),"resident graph create");
                const auto& first=items.front();
                add_resident_weight_graph(graph.graph,device_items,count,first.problem.n,plan.max_q,first.leaf_count,first.chunks);
                cuda_check(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0),"resident graph instantiate");
                ++scoring_state(c).resident_execution_counters[2];
                scoring_state(c).resident_execution_counters[8]=add(scoring_state(c).resident_execution_counters[8],
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count());
            }
            resident_graph_initialize<<<static_cast<unsigned>(count),threads,0,c.stream>>>(device_items);
            c.launched();
            for(;;) {
                cuda_check(cudaGraphLaunch(graph.executable,c.stream),"resident chunk launch");
                c.diagnostics.kernel_launches=add(c.diagnostics.kernel_launches,resident_chunk_nodes);
                resident_graph_terminal_flags<<<blocks(count),threads,0,c.stream>>>(
                    device_items,count,allocation.at<int>(plan.flags_offset));
                c.launched();
                c.download(terminal_flags.data(),allocation.at<int>(plan.flags_offset),count);
                c.finish();
                ++scoring_state(c).resident_execution_counters[4];
                bool done=true;
                for(int flag:terminal_flags) {
                    if(flag!=0&&flag!=1)throw std::runtime_error("Resident graph returned an invalid terminal flag");
                    done=done&&flag==1;
                }
                if(done)break;
            }
        } else {
            resident_weight_small_kernel<<<static_cast<unsigned>(std::min<i64>(65535,count)),threads,0,c.stream>>>(
                allocation.at<ResidentWeightItem>(plan.descriptor_offset),count);
            c.launched();
        }
        c.download(completed.data(),allocation.at<ResidentWeightState>(plan.state_offset),count);
        cursor=0;
        for(i64 index=0;index<count;++index) {
            auto& p=scoring_problem(c,handles[index]);
            c.download(weights+cursor,items[index].arena,p.q);cursor+=p.q;
        }
        c.finish();
        // Check all execution failures before making scientific result scalars
        // available. Genuine final stationarity failures retain per-case status.
        for(i64 index=0;index<count;++index) {
            const auto& state=completed[index];
            if(state.operation==ResidentWeightOperation::Error&&state.error!=ResidentWeightError::Convergence)
                throw std::runtime_error("Resident weight execution failed with code "+std::to_string(static_cast<int>(state.error)));
            if(state.operation!=ResidentWeightOperation::Complete&&state.operation!=ResidentWeightOperation::Error)
                throw std::runtime_error("Resident optimizer did not return a terminal state");
        }
        for(i64 index=0;index<count;++index) {
            const auto& state=completed[index];
            statuses[index]=static_cast<int>(state.error);
            likelihood[index]=state.cache_likelihood;optimality[index]=state.optimality_gap;active[index]=state.active_gap;
            counters[index*6]=state.evaluations;counters[index*6+1]=state.cache_hits;
            counters[index*6+2]=state.preparations;counters[index*6+3]=state.trials;
            counters[index*6+4]=state.slsqp.iter;counters[index*6+5]=state.repair_steps;
        }
        lease.complete=true;
    });
}

} // namespace

// Read-only extra-storage query for the auto route. Existing mixture leases,
// columns and caller-owned exports are separate. No cache is evicted or created.
extern "C" int CliPP2ExperimentalResidentAdmission(void* value,i64 count,const i64* sizes,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value);
        if(!sizes||!output)throw std::domain_error("Missing resident admission fields");
        const ResidentWeightPlan plan(c.data.mutations,count,[&](i64 index) { return sizes[index]; });
        output[0]=add(plan.layout.bytes,plan.graph_charge);
        // This query is read-only. The execution lease will either reuse or
        // discard its own idle workspace before requesting replacement storage.
        const auto cached=c.scoring_state?static_cast<ScoringState*>(c.scoring_state)->resident_execution_cache:nullptr;
        output[1]=add(c.resident_limit-c.resident_bytes,cached?add(cached->allocation.bytes,cached->graph_reservation):0);
        output[2]=add(plan.host_charge,plan.graph_charge);
        output[3]=c.host_staging_limit-scoring_host_bytes(c)-c.grid_cache_host_bytes
                  -c.probability_host_bytes-c.diagnostics.staging_bytes+(cached?cached->host_charge:0);
    });
}

extern "C" int CliPP2ResidentExecutionDiagnostics(void* value,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value);
        if(!output)throw std::domain_error("Missing resident execution diagnostic fields");
        if(!c.scoring_state) { std::fill(output,output+13,0);return; }
        const auto& state=*static_cast<ScoringState*>(c.scoring_state);
        std::copy(state.resident_execution_counters,state.resident_execution_counters+9,output);
        const auto& cached=state.resident_execution_cache;
        output[9]=cached?cached->allocation.bytes:0;
        output[10]=cached?cached->host_charge:0;
        output[11]=cached?cached->graph_reservation:0;
        output[12]=state.resident_execution_busy?1:0;
    });
}

extern "C" int CliPP2ExperimentalResidentWeights(void* value,i64 count,const i64* handles,
        const double* starts,double* weights,double* likelihood,double* optimality,
        double* active,int* statuses,i64* counters) {
    return experimental_resident_weights(value,count,handles,starts,weights,likelihood,optimality,active,statuses,counters,false);
}
// Private qualification hook: compare schedules on identical small inputs.
// Ordinary fitting has one automatic dimension-based schedule and no override.
extern "C" int CliPP2QualifyResidentGraphWeights(void* value,i64 count,const i64* handles,
        const double* starts,double* weights,double* likelihood,double* optimality,
        double* active,int* statuses,i64* counters) {
    return experimental_resident_weights(value,count,handles,starts,weights,likelihood,optimality,active,statuses,counters,true);
}
