// Explicit experimental complete-weight ABI; production dispatch stays off.
// Include at global scope after scoring.cuh. Uses ordinary fixed CUDA graphs,
// not conditional nodes. There is no host numerical or CPU fallback.
#pragma once
#include "resident_weight_graph.cuh"

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
        host_charge=add(1024,multiply(count,add(sizeof(ResidentWeightItem)+sizeof(ResidentWeightState),96)));
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
int experimental_resident_weights(void* value,i64 count,const i64* handles,
        const double* starts,double* weights,double* likelihood,double* optimality,
        double* active,int* statuses,i64* counters,bool force_graph,
        i64* profile=nullptr) {
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
        ScoringHostScope host_scope(c,plan.host_charge);
        // Runtime graph storage is opaque to CUDA's public size API. This is
        // an explicit conservative admission reservation, not measured RSS or
        // device bytes; source-bound qualification must record physical peaks.
        ScoringHostScope graph_scope(c,plan.graph_charge);
        std::vector<ResidentWeightItem> items(static_cast<std::size_t>(count));
        std::vector<ResidentWeightState> completed(static_cast<std::size_t>(count));
        std::vector<int> terminal_flags(static_cast<std::size_t>(plan.graph?count:0));
        ScoringAllocation allocation(c,plan.layout.bytes);
        ResidentGraphOwner graph;
        ScoringStreamFence fence(c);
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
        i64 chunk_launches=0,terminal_bytes=0;
        if(plan.graph) {
            cuda_check(cudaGraphCreate(&graph.graph,0),"resident graph create");
            auto* device_items=allocation.at<ResidentWeightItem>(plan.descriptor_offset);
            const auto& first=items.front();
            add_resident_weight_graph(graph.graph,device_items,count,first.problem.n,plan.max_q,first.leaf_count,first.chunks);
            cuda_check(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0),"resident graph instantiate");
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
                chunk_launches=add(chunk_launches,1);terminal_bytes=add(terminal_bytes,multiply(count,sizeof(int)));
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
        graph.close();
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
            if(profile) {
                profile[index*6]=state.controller_cycles;
                profile[index*6+1]=state.slsqp_cycles;
                profile[index*6+2]=state.controller_calls;
                profile[index*6+3]=chunk_launches;
                profile[index*6+4]=terminal_bytes;
                profile[index*6+5]=chunk_launches;
            }
        }
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
        output[0]=plan.layout.bytes;
        output[1]=c.resident_limit-c.resident_bytes;
        output[2]=add(plan.host_charge,plan.graph_charge);
        output[3]=c.host_staging_limit-scoring_host_bytes(c)-c.grid_cache_host_bytes
                  -c.probability_host_bytes-c.diagnostics.staging_bytes;
    });
}

extern "C" int CliPP2ExperimentalResidentWeights(void* value,i64 count,const i64* handles,
        const double* starts,double* weights,double* likelihood,double* optimality,
        double* active,int* statuses,i64* counters) {
    return experimental_resident_weights(value,count,handles,starts,weights,likelihood,optimality,active,statuses,counters,false);
}
extern "C" int CliPP2ExperimentalResidentGraphWeights(void* value,i64 count,const i64* handles,
        const double* starts,double* weights,double* likelihood,double* optimality,
        double* active,int* statuses,i64* counters) {
    return experimental_resident_weights(value,count,handles,starts,weights,likelihood,optimality,active,statuses,counters,true);
}
