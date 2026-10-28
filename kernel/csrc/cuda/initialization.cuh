// Opt-in resident initializer oracle. Its deterministic FP64 reduction policy
// is separately qualified against the existing NumPy/OpenBLAS controller.
namespace {
struct InitializationProblem {
    Context& owner;
    i64 region,n,q,block_rows,host_charge=0;
    double denominator;
    bool mask_bounds,standard,cached=false,full_valid=false,pair_ready=false;
    ScoringReductionPlan full,tail,global;
    std::unique_ptr<ScoringAllocation> storage;
    i64 rows,grid,frequency,offset,mass,direction,pilot,weights,positions,matrix,
        leaves,partials,sums,invalid,full_tail,last_tail,global_tail;
    i64 counters[5]{};
    InitializationProblem(Context& c,i64 region_,i64 n_,i64 q_,i64 block_,double denominator_,
            bool mask_,bool standard_):owner(c),region(region_),n(n_),q(q_),block_rows(block_),
        denominator(denominator_),mask_bounds(mask_),standard(standard_),full(block_),
        tail(n_%block_?n_%block_:block_),global(n_) {}
    ~InitializationProblem() {
        if(host_charge && owner.scoring_state)
            static_cast<ScoringState*>(owner.scoring_state)->host_bytes-=host_charge;
    }
    template<class T> T* at(i64 offset_) { return storage->at<T>(offset_); }
};
struct InitializationState {
    std::map<i64,std::unique_ptr<InitializationProblem>> problems;
};
void destroy_initialization_state(Context& c) noexcept {
    if(!c.initialization_state) return;
    delete static_cast<InitializationState*>(c.initialization_state);
    c.initialization_state=nullptr;
    if(c.scoring_state) static_cast<ScoringState*>(c.scoring_state)->host_bytes-=sizeof(InitializationState)+128;
}
InitializationState& initialization_state(Context& c) {
    if(!c.initialization_state) {
        auto& scoring=scoring_state(c);
        check_scoring_host(c,sizeof(InitializationState)+128);
        auto state=std::make_unique<InitializationState>();
        scoring.host_bytes+=sizeof(InitializationState)+128;
        c.initialization_state=state.release();
    }
    return *static_cast<InitializationState*>(c.initialization_state);
}
InitializationProblem& initialization_problem(Context& c,i64 handle) {
    if(c.initialization_state) {
        auto& map=static_cast<InitializationState*>(c.initialization_state)->problems;
        const auto found=map.find(handle);
        if(found!=map.end()) return *found->second;
    }
    throw std::domain_error("Unknown or released CUDA initialization handle");
}

__global__ void initialization_grid_kernel(ObservationSoA data,i64 region,const i64* rows,
        i64 first,i64 count,i64 width,const double* grid,const i64* positions,bool masked,
        double* matrix,int* invalid) {
    const auto cells=count*width;
    for(i64 index=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;index<cells;
            index+=static_cast<i64>(blockDim.x)*gridDim.x) {
        const auto row=index%count, column=index/count;
        const auto cell=region*data.mutations+rows[first+row];
        const double center=grid[positions?positions[column]:column];
        if(masked && (center<data.lower[cell] || center>data.upper[cell])) matrix[index]=-positive_infinity;
        else {
            const auto value=evaluate_observation<EmitMode::LossOnly>(observation(data,cell),center);
            matrix[index]=-value.loss;
            if(value.status>EmitStatus::ZeroLikelihood) atomicMax(invalid,1);
        }
    }
}
__global__ void initialization_scale_kernel(i64 first,i64 count,i64 width,double* matrix,
        double* offset,int* invalid) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<count;
            row+=static_cast<i64>(blockDim.x)*gridDim.x) {
        double maximum=-positive_infinity;
        for(i64 k=0;k<width;++k) if(matrix[k*count+row]>maximum) maximum=matrix[k*count+row];
        offset[first+row]=maximum;
        if(!finite_number(maximum)) atomicMax(invalid,1);
        for(i64 k=0;k<width;++k) matrix[k*count+row]=::exp(matrix[k*count+row]-maximum);
    }
}
__global__ void initialization_rows_kernel(i64 first,i64 count,i64 stride,i64 width,
        const double* matrix,const i64* positions,bool selected_matrix,const double* weights,
        const double* grid,double* mass,double* pilot,bool collect,int* invalid) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<count;
            row+=static_cast<i64>(blockDim.x)*gridDim.x) {
        double sum=0,numerator=0;
        for(i64 k=0;k<width;++k) {
            const auto column=positions?positions[k]:k;
            const auto value=matrix[(selected_matrix?k:column)*stride+row];
            sum+=value*weights[k];
            if(collect) numerator+=value*(weights[k]*grid[column]);
        }
        mass[first+row]=sum;
        if(!(sum>0) || !finite_number(sum)) atomicMax(invalid,1);
        if(collect) pilot[first+row]=numerator/sum;
    }
}
__global__ void initialization_score_leaves(i64 first,i64 count,i64 stride,i64 width,
        const double* matrix,const i64* positions,bool selected_matrix,const double* frequency,
        const double* mass,const double* offset,i64 full_chunks,const ScoringLeaf* tail,
        i64 leaves,double* partials) {
    __shared__ double values[scoring_plane_tile*128],chains[128];
    const auto row=threadIdx.x;
    for(i64 leaf=blockIdx.x;leaf<leaves;leaf+=gridDim.x) {
        const auto range=scoring_leaf(leaf,full_chunks,tail);
        for(i64 start=0;start<width+1;start+=scoring_plane_tile) {
            const auto planes=width+1-start<scoring_plane_tile?width+1-start:scoring_plane_tile;
            if(row<range.count) {
                const auto local=range.first+row,source=first+local;
                for(i64 plane=0;plane<planes;++plane) {
                    const auto k=start+plane;
                    if(k==width) values[plane*128+row]=frequency[source]*(::log(mass[source])+offset[source]);
                    else {
                        const auto column=selected_matrix?k:(positions?positions[k]:k);
                        values[plane*128+row]=(frequency[source]/mass[source])*matrix[column*stride+local];
                    }
                }
            }
            __syncthreads();
            scoring_leaf_reduce(values,range.count,start,planes,leaf,leaves,chains,partials);
        }
    }
}
__global__ void initialization_fold(i64 planes,i64 chunks,const double* partials,double* sums) {
    for(i64 plane=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;plane<planes;
            plane+=static_cast<i64>(blockDim.x)*gridDim.x) {
        double sum=-0.;
        for(i64 chunk=0;chunk<chunks;++chunk) sum+=partials[plane*chunks+chunk];
        sums[plane]+=sum; // Preserve original block order, including the +0 accumulator.
    }
}
__global__ void initialization_direction_kernel(i64 first,i64 count,i64 stride,const double* matrix,
        i64 grow,i64 shrink,double amount,double* direction) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<count;
            row+=static_cast<i64>(blockDim.x)*gridDim.x)
        direction[first+row]=amount*(matrix[grow*stride+row]-matrix[shrink*stride+row]);
}
__global__ void initialization_trial_leaves(i64 n,const double* mass,const double* direction,
        const double* frequency,double normalization,bool standard,double fraction,
        i64 full_chunks,const ScoringLeaf* tail,i64 leaves,double* partials) {
    __shared__ double values[128],chains[128];
    const auto row=threadIdx.x;
    for(i64 leaf=blockIdx.x;leaf<leaves;leaf+=gridDim.x) {
        const auto range=scoring_leaf(leaf,full_chunks,tail);
        if(row<range.count) {
            const auto index=range.first+row;
            const auto sum=mass[index]+fraction*direction[index];
            const auto denominator=sum<DBL_MIN?DBL_MIN:sum;
            values[row]=standard?(frequency[index]/normalization*direction[index]/denominator)
                                :(direction[index]/denominator);
        }
        __syncthreads();
        scoring_leaf_reduce(values,range.count,0,1,leaf,leaves,chains,partials);
    }
}
void initialization_grid(InitializationProblem& p,i64 first,i64 rows,i64 width,const i64* positions) {
    auto& c=p.owner;
    initialization_grid_kernel<<<blocks(multiply(rows,width)),threads,0,c.stream>>>(c.data,p.region,
        p.at<i64>(p.rows),first,rows,width,p.at<double>(p.grid),positions,p.mask_bounds,
        p.at<double>(p.matrix),p.at<int>(p.invalid));
    c.launched();
    initialization_scale_kernel<<<blocks(rows),threads,0,c.stream>>>(first,rows,width,
        p.at<double>(p.matrix),p.at<double>(p.offset),p.at<int>(p.invalid));
    c.launched(); ++p.counters[0];
    p.counters[1]=add(p.counters[1],multiply(rows,width));
    c.diagnostics.cell_evaluations=add(c.diagnostics.cell_evaluations,multiply(rows,width));
}
std::pair<const double*,i64> initialization_matrix(InitializationProblem& p,i64 first,i64 rows,
                                                i64 width,const i64* positions,bool rescale) {
    if(p.cached && !rescale) {
        if(!p.full_valid) { initialization_grid(p,0,p.n,p.q,nullptr); p.full_valid=true; }
        return {p.at<double>(p.matrix)+first,p.n};
    }
    p.full_valid=false;
    initialization_grid(p,first,rows,rescale?width:p.q,rescale?positions:nullptr);
    return {p.at<double>(p.matrix),rows};
}
void initialization_combine(InitializationProblem& p,i64 count,i64 planes,
        const ScoringReductionPlan& plan,i64 tail) {
    auto& c=p.owner;
    const auto chunks=count/8192+(count%8192!=0);
    mixture_combine_leaves_kernel<<<static_cast<unsigned>(std::min<i64>(65535,multiply(planes,chunks))),threads,0,c.stream>>>(
        p.at<double>(p.leaves),count,planes,chunks,plan.full_chunks,p.at<ScoringLeaf>(tail),
        plan.tail_count,plan.leaf_count,p.at<double>(p.partials));
    c.launched();
    initialization_fold<<<blocks(planes),threads,0,c.stream>>>(planes,chunks,
        p.at<double>(p.partials),p.at<double>(p.sums));
    c.launched();
}
void initialization_weights(InitializationProblem& p,i64 width,const double* weights,const i64* positions) {
    if(width<=0 || width>p.q || !weights || (width!=p.q && !positions))
        throw std::domain_error("Invalid CUDA initialization weights or positions");
    for(i64 k=0;k<width;++k) {
        if(!finite_number(weights[k]) || weights[k]<0 || (positions &&
                (positions[k]<0 || positions[k]>=p.q || (k && positions[k]<=positions[k-1]))))
            throw std::domain_error("Invalid CUDA initialization weight or active position");
    }
    p.owner.upload(p.at<double>(p.weights),weights,width);
    if(positions) p.owner.upload(p.at<i64>(p.positions),positions,width);
}
} // namespace

extern "C" int CliPP2CreateInitialization(void* value,i64 region,i64 n,const i64* rows,
        i64 q,const double* grid,const double* frequency,double normalization,i64 block_rows,
        int mask_bounds,int standard,i64 cache_bytes,i64* handle) {
    return scoring_guarded([&] {
        auto& c=context(value);
        if(!handle || !rows || !grid || !frequency || n<=0 || q<=0 || block_rows<=0 ||
                region<0 || region>=c.data.regions || !(normalization>0) || !finite_number(normalization)
                || cache_bytes<0 || (mask_bounds!=0 && mask_bounds!=1) || (standard!=0 && standard!=1))
            throw std::domain_error("Invalid CUDA initialization preparation");
        *handle=0; block_rows=std::min(block_rows,n);
        for(i64 row=0;row<n;++row)
            if(rows[row]<0 || rows[row]>=c.data.mutations || !(frequency[row]>0) || !finite_number(frequency[row]))
                throw std::domain_error("Invalid CUDA initialization rows or frequencies");
        for(i64 k=0;k<q;++k)
            if(!finite_number(grid[k]) || grid[k]<0 || grid[k]>1 || (k && grid[k]<=grid[k-1]))
                throw std::domain_error("CUDA initialization grid must increase within [0,1]");
        auto& registry=initialization_state(c);
        auto p=std::make_unique<InitializationProblem>(c,region,n,q,block_rows,normalization,mask_bounds,standard);
        const auto host=sizeof(InitializationProblem)+256;
        check_scoring_host(c,host);
        p->host_charge=host; scoring_state(c).host_bytes+=host;
        Layout layout;
        p->rows=layout.take<i64>(n); p->grid=layout.take<double>(q);
        p->frequency=layout.take<double>(n); p->offset=layout.take<double>(n);
        p->mass=layout.take<double>(n); p->direction=layout.take<double>(n); p->pilot=layout.take<double>(n);
        p->weights=layout.take<double>(q); p->positions=layout.take<i64>(q);
        p->full_tail=layout.take<ScoringLeaf>(p->full.tail_count);
        p->last_tail=layout.take<ScoringLeaf>(p->tail.tail_count);
        p->global_tail=layout.take<ScoringLeaf>(p->global.tail_count);
        p->leaves=layout.take<double>(std::max(multiply(q+1,std::max(p->full.leaf_count,p->tail.leaf_count)),p->global.leaf_count));
        p->partials=layout.take<double>(std::max(multiply(q+1,block_rows/8192+(block_rows%8192!=0)),n/8192+(n%8192!=0)));
        p->sums=layout.take<double>(q+1); p->invalid=layout.take<int>(1);
        const auto full_bytes=multiply(multiply(n,q),sizeof(double));
        p->cached=full_bytes<=cache_bytes && full_bytes<=c.resident_limit-c.resident_bytes-layout.bytes;
        p->matrix=layout.take<double>(multiply(p->cached?n:block_rows,q));
        // Optional oracle admission returns no handle; a live oracle is bounded
        // mandatory working state until its initializer finishes and closes it.
        if(layout.bytes>c.resident_limit-c.resident_bytes) return;
        p->storage=std::make_unique<ScoringAllocation>(c,layout.bytes);
        int invalid=0; ScoringStreamFence fence(c);
        c.upload(p->at<i64>(p->rows),rows,n); c.upload(p->at<double>(p->grid),grid,q);
        c.upload(p->at<double>(p->frequency),frequency,n);
        c.upload(p->at<ScoringLeaf>(p->full_tail),p->full.tail,p->full.tail_count);
        c.upload(p->at<ScoringLeaf>(p->last_tail),p->tail.tail,p->tail.tail_count);
        c.upload(p->at<ScoringLeaf>(p->global_tail),p->global.tail,p->global.tail_count);
        cuda_check(cudaMemsetAsync(p->at<int>(p->invalid),0,sizeof(int),c.stream),"initialize grid status");
        // The standard controller validates its first expectation lazily. The
        // mixed adapter validates all supported grid rows during construction.
        // Preserve that boundary without a discarded standard streaming pass.
        if(p->cached || !p->standard)
            for(i64 first=0;first<n;first+=block_rows)
                initialization_matrix(*p,first,std::min(block_rows,n-first),q,nullptr,false);
        c.download(&invalid,p->at<int>(p->invalid),1); c.finish();
        if(invalid) throw std::domain_error("No finite supported grid likelihood for CUDA initialization");
        const auto id=scoring_state(c).id();
        registry.problems.emplace(id,std::move(p)); *handle=id;
    });
}
extern "C" int CliPP2ReleaseInitialization(void* value,i64 handle) {
    return scoring_guarded([&] {
        auto& c=context(value); initialization_problem(c,handle);
        static_cast<InitializationState*>(c.initialization_state)->problems.erase(handle);
    });
}
extern "C" int CliPP2EvaluateInitialization(void* value,i64 handle,i64 width,const double* weights,
        const i64* positions,int rescale,int collect,double* result,double* pilot,int* invalid) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=initialization_problem(c,handle);
        if(!result || !invalid || (collect && !pilot) || (rescale && !positions) ||
                (rescale!=0 && rescale!=1) || (collect!=0 && collect!=1))
            throw std::domain_error("Invalid CUDA initialization result request");
        ScoringStreamFence fence(c); initialization_weights(p,width,weights,positions);
        p.pair_ready=false; // This request overwrites mass; an older direction no longer matches it.
        cuda_check(cudaMemsetAsync(p.at<int>(p.invalid),0,sizeof(int),c.stream),"reset initializer status");
        cuda_check(cudaMemsetAsync(p.at<double>(p.sums),0,multiply(width+1,8),c.stream),"reset initializer sums");
        const auto* selected=positions?p.at<i64>(p.positions):nullptr;
        for(i64 first=0;first<p.n;first+=p.block_rows) {
            const auto count=std::min(p.block_rows,p.n-first);
            const auto matrix=initialization_matrix(p,first,count,width,selected,rescale);
            initialization_rows_kernel<<<blocks(count),threads,0,c.stream>>>(first,count,matrix.second,width,
                matrix.first,selected,rescale,p.at<double>(p.weights),p.at<double>(p.grid),
                p.at<double>(p.mass),p.at<double>(p.pilot),collect,p.at<int>(p.invalid)); c.launched();
            const auto& plan=count==p.block_rows?p.full:p.tail;
            const auto tail=count==p.block_rows?p.full_tail:p.last_tail;
            initialization_score_leaves<<<static_cast<unsigned>(std::min<i64>(65535,plan.leaf_count)),threads,0,c.stream>>>(
                first,count,matrix.second,width,matrix.first,selected,rescale,p.at<double>(p.frequency),
                p.at<double>(p.mass),p.at<double>(p.offset),plan.full_chunks,p.at<ScoringLeaf>(tail),
                plan.leaf_count,p.at<double>(p.leaves)); c.launched();
            initialization_combine(p,count,width+1,plan,tail);
        }
        c.download(result,p.at<double>(p.sums),width+1); c.download(invalid,p.at<int>(p.invalid),1);
        if(collect) c.download(pilot,p.at<double>(p.pilot),p.n);
        c.finish(); ++p.counters[2];
    });
}
extern "C" int CliPP2PrepareInitializationTransfer(void* value,i64 handle,const double* weights,
        i64 grow,i64 shrink) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=initialization_problem(c,handle);
        if(grow<0 || shrink<0 || grow>=p.q || shrink>=p.q || grow==shrink)
            throw std::domain_error("Invalid CUDA initializer transfer pair");
        int invalid=0; ScoringStreamFence fence(c); initialization_weights(p,p.q,weights,nullptr); p.pair_ready=false;
        cuda_check(cudaMemsetAsync(p.at<int>(p.invalid),0,sizeof(int),c.stream),"reset initializer transfer status");
        for(i64 first=0;first<p.n;first+=p.block_rows) {
            const auto count=std::min(p.block_rows,p.n-first);
            const auto matrix=initialization_matrix(p,first,count,p.q,nullptr,false);
            initialization_rows_kernel<<<blocks(count),threads,0,c.stream>>>(first,count,matrix.second,p.q,
                matrix.first,nullptr,false,p.at<double>(p.weights),p.at<double>(p.grid),
                p.at<double>(p.mass),p.at<double>(p.pilot),false,p.at<int>(p.invalid)); c.launched();
            initialization_direction_kernel<<<blocks(count),threads,0,c.stream>>>(first,count,matrix.second,
                matrix.first,grow,shrink,weights[shrink],p.at<double>(p.direction)); c.launched();
        }
        c.download(&invalid,p.at<int>(p.invalid),1); c.finish();
        if(invalid) throw std::domain_error("Nonfinite CUDA initialization transfer mass");
        p.pair_ready=true; ++p.counters[3];
    });
}
extern "C" int CliPP2InitializationTransferDerivative(void* value,i64 handle,double fraction,double* result) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=initialization_problem(c,handle);
        if(!p.pair_ready || !result || !finite_number(fraction) || fraction<0 || fraction>1)
            throw std::domain_error("Invalid or unprepared CUDA initializer transfer trial");
        const auto& plan=p.global;
        cuda_check(cudaMemsetAsync(p.at<double>(p.sums),0,sizeof(double),c.stream),"reset initializer derivative");
        initialization_trial_leaves<<<static_cast<unsigned>(std::min<i64>(65535,plan.leaf_count)),threads,0,c.stream>>>(
            p.n,p.at<double>(p.mass),p.at<double>(p.direction),p.at<double>(p.frequency),p.denominator,p.standard,
            fraction,plan.full_chunks,p.at<ScoringLeaf>(p.global_tail),plan.leaf_count,p.at<double>(p.leaves));
        c.launched(); initialization_combine(p,p.n,1,plan,p.global_tail);
        c.download(result,p.at<double>(p.sums),1); c.finish(); ++p.counters[4];
    });
}
extern "C" int CliPP2InitializationDiagnostics(void* value,i64 handle,i64* output) {
    return scoring_guarded([&] {
        auto& p=initialization_problem(context(value),handle);
        if(!output) throw std::domain_error("Missing CUDA initialization diagnostic destination");
        std::copy(p.counters,p.counters+5,output);
        output[5]=p.cached; output[6]=p.storage->bytes; output[7]=p.host_charge;
    });
}
