// Context-owned immutable likelihood columns and resident mixture arithmetic.
// Included after backend.cu's kernels, Context and ordered reduction helper.
#include <list>
#include <map>
#include <cfloat>
#include <atomic>

namespace {
#include "scoring_reduce.cuh"

struct ScoringAllocation {
    Context& context;
    unsigned char* data = nullptr;
    i64 bytes = 0;
    ScoringAllocation(Context& owner, i64 size):context(owner),bytes(size) {
        data = context.allocate_resident(size);
    }
    ~ScoringAllocation() { if(data) context.release_resident(data,bytes); }
    ScoringAllocation(const ScoringAllocation&) = delete;
    ScoringAllocation& operator=(const ScoringAllocation&) = delete;
    template<class T> T* at(i64 offset=0) { return reinterpret_cast<T*>(data+offset); }
};
struct ScoringStreamFence {
    Context& context;
    int exceptions=std::uncaught_exceptions();
    explicit ScoringStreamFence(Context& owner):context(owner) {}
    ~ScoringStreamFence() noexcept {
        // Declare after all local host transfer buffers so unwind drains before
        // those buffers die. Keep the originating execution error intact.
        if(std::uncaught_exceptions()>exceptions) cudaStreamSynchronize(context.stream);
    }
};
struct ResidentColumn {
    ScoringAllocation values;
    ResidentColumn(Context& context):values(context,multiply(context.data.mutations,8)) {}
    double* data() { return values.at<double>(); }
};
struct MixtureLayout {
    i64 n,q,chunks,bytes=0,pointers=0,weights=0,log_weights=0,offset=0,kernel=0;
    i64 leaf_metadata=0,leaf_partials=0,partials=0,sums=0;
    i64 other=0,grow_log=0,shrink_log=0,numerator=0,sign=0,invalid=0;
    ScoringReductionPlan reduction;
    MixtureLayout(i64 rows,i64 count):n(rows),q(count),chunks(rows/8192+(rows%8192!=0)),reduction(rows) {
        if(n<=0 || q<=0) throw std::domain_error("CUDA mixture dimensions must be positive");
        Layout layout;
        pointers=layout.take<const double*>(q); weights=layout.take<double>(q);
        log_weights=layout.take<double>(q);
        offset=layout.take<double>(n); kernel=layout.take<double>(multiply(n,q));
        leaf_metadata=layout.take<ScoringLeaf>(reduction.tail_count);
        leaf_partials=layout.take<double>(multiply(reduction.leaf_count,add(q,2)));
        partials=layout.take<double>(multiply(chunks,add(q,2)));
        sums=layout.take<double>(add(q,2)); other=layout.take<double>(n);
        grow_log=layout.take<double>(n); shrink_log=layout.take<double>(n);
        numerator=layout.take<double>(n); sign=layout.take<double>(n);
        invalid=layout.take<int>(1); bytes=layout.bytes;
    }
};
struct MixtureProblem:MixtureLayout {
    Context& context;
    std::vector<std::shared_ptr<ResidentColumn>> columns;
    std::unique_ptr<ScoringAllocation> allocation;
    double grow_weight=0,amount=0;
    bool pair_ready=false;
    i64 host_charge=0;
    MixtureProblem(Context& owner,i64 count):MixtureLayout(owner.data.mutations,count),context(owner) {}
    template<class T> T* at(i64 position) { return allocation->at<T>(position); }
};
struct ScoringCacheEntry {
    std::shared_ptr<ResidentColumn> column;
    std::list<std::string>::iterator order;
    i64 charge=0;
    i64 host_charge=0;
};
struct ScoringRequestStorage {
    i64 capacity,host_charge,optional_charge;
    std::unique_ptr<const double*[]> pointers;
    std::unique_ptr<ScoringAllocation> header;
    static i64 host_bytes(i64 count) { return add(256,multiply(count,sizeof(const double*))); }
    explicit ScoringRequestStorage(i64 count):capacity(count),host_charge(host_bytes(count)),
            optional_charge(host_charge),pointers(std::make_unique<const double*[]>(count)) {}
};
struct ScoringState {
    std::map<i64,std::shared_ptr<ResidentColumn>> handles;
    std::map<i64,std::unique_ptr<MixtureProblem>> problems;
    std::map<std::string,ScoringCacheEntry> cache;
    std::list<std::string> order;
    // Only idle scratch is retained. Live problems always own distinct arenas.
    std::unique_ptr<ScoringAllocation> arena_cache;
    std::unique_ptr<ScoringRequestStorage> request_cache;
    i64 arena_allocations=0,arena_reuses=0;
    i64 request_allocations=0,request_reuses=0,request_header_allocations=0;
    i64 request_live_objects=0,request_live_host_bytes=0,request_live_device_bytes=0;
    i64 host_bytes=0,staging_bytes=0;
    // evaluations, cache hits, exports, gathers, weight evaluations,
    // transfer preparations, O(N) trials, live handles, live problems.
    i64 counters[9]{};
    i64 proposal_counters[4]{};
    i64 id() {
        static std::atomic<i64> next{1};
        const i64 result=next.fetch_add(1);
        require(result>0 && result<INT64_MAX,"CUDA scoring handle space exhausted");
        return result;
    }
};
ScoringState& scoring_state(Context& c) {
    if(!c.scoring_state) {
        const i64 charge=add(sizeof(ScoringState),128);
        while(charge>c.host_staging_limit-add(add(c.grid_cache_host_bytes,c.probability_host_bytes),c.diagnostics.staging_bytes)
                && (evict_optional_grid_cache(c)||evict_optional_probability_cache(c))) {}
        require(charge<=c.host_staging_limit-add(add(c.grid_cache_host_bytes,c.probability_host_bytes),c.diagnostics.staging_bytes),
                "CUDA scoring registry exceeds native host budget");
        auto state=std::make_unique<ScoringState>(); state->host_bytes=charge;
        c.scoring_state=state.release();
    }
    return *static_cast<ScoringState*>(c.scoring_state);
}
i64 scoring_host_bytes(Context& c) noexcept {
    if(!c.scoring_state) return 0;
    const auto& s=*static_cast<ScoringState*>(c.scoring_state);
    return s.host_bytes+s.staging_bytes;
}
i64 scoring_persistent_host_bytes(Context& c) noexcept {
    if(!c.scoring_state) return 0;
    return static_cast<ScoringState*>(c.scoring_state)->host_bytes;
}
void release_request_cache(Context& c,ScoringState& s) noexcept {
    if(!s.request_cache) return;
    c.drop_optional(s.request_cache->optional_charge);
    s.host_bytes-=s.request_cache->host_charge;
    s.request_cache.reset();
}
bool evict_optional_scoring_cache(Context& c) {
    if(!c.scoring_state) return false;
    auto& s=*static_cast<ScoringState*>(c.scoring_state);
    // Request storage saves allocation only; computed columns/grids have priority.
    if(s.request_cache) { release_request_cache(c,s); return true; }
    // Reusing an arena saves allocation, whereas a column saves computation.
    // Give the latter priority; mandatory live leases are never invalidated.
    if(s.arena_cache) {
        c.drop_optional(s.arena_cache->bytes);
        s.arena_cache.reset();
        return true;
    }
    if(s.order.empty()) return false;
    const auto cold=s.cache.find(s.order.front());
    c.drop_optional(cold->second.charge);
    s.host_bytes-=cold->second.host_charge;
    s.cache.erase(cold);
    s.order.pop_front();
    return true;
}
void release_optional_scoring_cache(Context& c) {
    while(evict_optional_scoring_cache(c)) {}
}
void destroy_scoring_state(Context& c) noexcept {
    if(!c.scoring_state) return;
    release_optional_scoring_cache(c);
    delete static_cast<ScoringState*>(c.scoring_state);
    c.scoring_state=nullptr;
}
std::shared_ptr<ResidentColumn> scoring_column(ScoringState& s,i64 handle) {
    const auto found=s.handles.find(handle);
    if(found==s.handles.end()) throw std::domain_error("Unknown or released CUDA column handle");
    return found->second;
}
MixtureProblem& scoring_problem(Context& c,i64 handle) {
    auto& s=scoring_state(c);
    const auto found=s.problems.find(handle);
    if(found==s.problems.end()) throw std::domain_error("Unknown or released CUDA mixture handle");
    return *found->second;
}
template<class Function> int scoring_guarded(Function function) {
    last_error[0]='\0';
    try { function(); return 0; }
    catch(const std::domain_error& error) {
        std::snprintf(last_error,sizeof(last_error),"%s",error.what()); return 2;
    } catch(const std::exception& error) {
        std::snprintf(last_error,sizeof(last_error),"%s",error.what());
    } catch(...) {
        std::snprintf(last_error,sizeof(last_error),"Unknown native CUDA scoring failure");
    }
    return 1;
}
void check_scoring_host(Context& c,i64 extra) {
    auto used=add(scoring_host_bytes(c),add(add(c.grid_cache_host_bytes,c.probability_host_bytes),c.diagnostics.staging_bytes));
    while(extra>c.host_staging_limit-used &&
            (evict_optional_scoring_cache(c) || evict_optional_grid_cache(c) || evict_optional_probability_cache(c))) {
        used=add(scoring_host_bytes(c),add(add(c.grid_cache_host_bytes,c.probability_host_bytes),c.diagnostics.staging_bytes));
    }
    require(extra<=c.host_staging_limit-used,"CUDA scoring host metadata/staging exceeds native host budget");
}
struct ScoringHostScope {
    Context& context;
    i64 bytes;
    ScoringHostScope(Context& c,i64 size):context(c),bytes(size) {
        check_scoring_host(c,size); scoring_state(c).staging_bytes+=size;
    }
    void release() noexcept {
        if(bytes) { static_cast<ScoringState*>(context.scoring_state)->staging_bytes-=bytes; bytes=0; }
    }
    ~ScoringHostScope() { release(); }
};
struct ScoringRequestLease {
    Context& context;
    std::unique_ptr<ScoringRequestStorage> storage;
    bool complete=false;
    int exceptions=std::uncaught_exceptions();
    ScoringRequestLease(Context& c,i64 count,bool device_header):context(c) {
        if(count<=0) throw std::domain_error("CUDA request storage must be nonempty");
        auto& s=scoring_state(c);
        try {
            if(s.request_cache && s.request_cache->capacity>=count) {
                c.drop_optional(s.request_cache->optional_charge);
                storage=std::move(s.request_cache); ++s.request_reuses;
            } else {
                release_request_cache(c,s);
                check_scoring_host(c,ScoringRequestStorage::host_bytes(count));
                storage=std::make_unique<ScoringRequestStorage>(count);
                s.host_bytes=add(s.host_bytes,storage->host_charge); ++s.request_allocations;
            }
            if(device_header && !storage->header) {
                storage->header=std::make_unique<ScoringAllocation>(c,
                    multiply(storage->capacity,sizeof(const double*)));
                storage->optional_charge=add(storage->host_charge,storage->header->bytes);
                ++s.request_header_allocations;
            }
        } catch(...) {
            if(storage) { s.host_bytes-=storage->host_charge; storage.reset(); }
            throw;
        }
        ++s.request_live_objects;
        s.request_live_host_bytes+=storage->host_charge;
        s.request_live_device_bytes+=storage->header?storage->header->bytes:0;
    }
    const double** pointers() { return storage->pointers.get(); }
    const double** device_pointers() { return storage->header->at<const double*>(); }
    ~ScoringRequestLease() noexcept {
        auto& s=*static_cast<ScoringState*>(context.scoring_state);
        const bool succeeded=complete && std::uncaught_exceptions()==exceptions;
        // Unsuccessful requests cannot leave queued transfers pointing into
        // recycled host/device storage. Preserve the originating error.
        if(!succeeded) cudaStreamSynchronize(context.stream);
        --s.request_live_objects;
        s.request_live_host_bytes-=storage->host_charge;
        s.request_live_device_bytes-=storage->header?storage->header->bytes:0;
        const auto old=s.request_cache.get();
        const i64 free_optional=context.optional_cache_limit-context.optional_cache_bytes+
            (old?old->optional_charge:0);
        // Only unused optional capacity may retain scratch. Never evict a
        // computed column/grid merely to retain this completed request.
        if(succeeded && storage->optional_charge<=free_optional &&
                (!old || old->capacity<storage->capacity ||
                 (old->capacity==storage->capacity && !old->header && storage->header))) {
            release_request_cache(context,s);
            if(context.retain_optional(storage->optional_charge)) s.request_cache=std::move(storage);
        }
        if(storage) s.host_bytes-=storage->host_charge;
    }
};
__global__ void resident_joint_kernel(ObservationSoA data,i64 count,const double* centers,
        double* const* columns,int* maximum) {
    for(i64 index=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
        index+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const i64 cluster=index/data.mutations,row=index%data.mutations;
        double total=0.; EmitStatus state=EmitStatus::Ok;
        for(i64 region=0;region<data.regions;++region) {
            const i64 cell=region*data.mutations+row;
            const double phi=centers[cluster*data.regions+region];
            if(phi<data.lower[cell]||phi>data.upper[cell]) {
                total=-positive_infinity; state=EmitStatus::ZeroLikelihood; break;
            }
            const auto terms=evaluate_observation<EmitMode::LossOnly>(observation(data,cell),phi);
            if(static_cast<int>(terms.status)>static_cast<int>(state)) state=terms.status;
            total-=terms.loss;
        }
        columns[cluster][row]=total;
        if(state>EmitStatus::ZeroLikelihood) atomicMax(maximum,static_cast<int>(state));
    }
}
void retain_scoring_column(Context& c,const std::string& key,const std::shared_ptr<ResidentColumn>& column) {
    auto& s=scoring_state(c);
    // Both the ordered key and map key own their binary center bytes.
    const i64 host_charge=add(256,multiply(static_cast<i64>(key.size()),2));
    const i64 charge=add(column->values.bytes,host_charge);
    if(add(scoring_host_bytes(c),add(add(c.grid_cache_host_bytes,c.probability_host_bytes),add(c.diagnostics.staging_bytes,host_charge)))
            >c.host_staging_limit) return;
    if(c.retain_optional(charge)) {
        bool ordered=false;
        try {
            s.order.push_back(key); ordered=true;
            const auto inserted=s.cache.emplace(key,ScoringCacheEntry{
                column,std::prev(s.order.end()),charge,host_charge}).second;
            if(!inserted) { s.order.pop_back(); c.drop_optional(charge); return; }
        } catch(...) {
            if(ordered) s.order.pop_back();
            c.drop_optional(charge); throw;
        }
        s.host_bytes=add(s.host_bytes,host_charge);
    }
}
std::vector<std::shared_ptr<ResidentColumn>> get_columns(Context& c,i64 count,const i64* handles) {
    if(count<=0 || !handles) throw std::domain_error("Expected nonempty CUDA column handles");
    check_scoring_host(c,multiply(count,32));
    auto& s=scoring_state(c);
    std::vector<std::shared_ptr<ResidentColumn>> result;
    result.reserve(static_cast<std::size_t>(count));
    for(i64 i=0;i<count;++i) result.push_back(scoring_column(s,handles[i]));
    return result;
}
__device__ double scoring_logadd(double a,double b) {
    if(a==-positive_infinity) return b;
    if(b==-positive_infinity) return a;
    const double high=a>b?a:b,low=a>b?b:a;
    return high+log1p(exp(low-high));
}
__device__ double scoring_capped_exp(double value) {
    return isnan(value)?quiet_nan:exp(value<345.?value:345.);
}
__global__ void mixture_offset_kernel(const double* const* columns,i64 n,i64 q,
        double* offsets,double* kernel,int* invalid) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
        row+=static_cast<i64>(gridDim.x)*blockDim.x) {
        double maximum=-positive_infinity;
        for(i64 k=0;k<q;++k) {
            const double value=columns[k][row];
            if(isnan(value)||value==positive_infinity) atomicExch(invalid,1);
            if(value>maximum) maximum=value;
        }
        offsets[row]=maximum;
        if(!finite_number(maximum)) atomicExch(invalid,1);
        for(i64 k=0;k<q;++k) kernel[k*n+row]=exp(columns[k][row]-maximum);
    }
}
// Both dispatches use exactly the same per-row arithmetic and component order.
__device__ double mixture_row_log_mass(const double* const* columns,const double* offsets,
        const double* kernel,const double* weights,i64 n,i64 q,i64 row) {
    const double offset=offsets[row];
    double mass=0.;
    for(i64 k=0;k<q;++k) mass+=kernel[k*n+row]*weights[k];
    double log_mass=log(mass);
    if(mass<DBL_MIN) {
        double maximum=-positive_infinity;
        for(i64 k=0;k<q;++k) {
            const double value=columns[k][row]-offset+log(weights[k]);
            if(value>maximum) maximum=value;
        }
        if(maximum==-positive_infinity) log_mass=maximum;
        else {
            double sum=0.;
            for(i64 k=0;k<q;++k) {
                sum+=exp(columns[k][row]-offset+log(weights[k])-maximum);
            }
            log_mass=maximum+log(sum);
        }
    }
    return log_mass;
}
__device__ void mixture_evaluate_row(const double* const* columns,const double* offsets,
        const double* kernel,const double* weights,i64 n,i64 q,i64 row,double* terms) {
    const double log_mass=mixture_row_log_mass(columns,offsets,kernel,weights,n,q,row);
    terms[row]=log_mass;
    terms[n+row]=offsets[row]+log_mass;
    for(i64 k=0;k<q;++k)
        terms[(k+2)*n+row]=scoring_capped_exp(columns[k][row]-offsets[row]-log_mass);
}
constexpr i64 small_mixture_rows=128,small_mixture_columns=20;
bool small_mixture(const MixtureProblem& p) {
#ifdef CLIPP2_CUDA_FORCE_GENERAL_SCORING
    // Qualification-only build: bind this flag and the binary hash separately.
    return false;
#else
    return p.n<=small_mixture_rows && p.q<=small_mixture_columns;
#endif
}
__device__ double small_mixture_sum(const double* values,i64 n) {
    // ordered_block_sum's small branch already folds into -0. The separate
    // finish kernel folds that result once more; preserve both signed-zero ops.
    // Dispatch guarantees a single <=128-value leaf. Do not instantiate the
    // unused recursive tree here: its dynamic branches inflate kernel registers.
    static_assert(small_mixture_rows<=128,"Small scoring must fit one reduction leaf");
    double partial=-0.;
    partial+=pairwise_block_sum<0>(OrderedArrayValues{values},0,n);
    double sum=-0.;
    sum+=partial;
    return sum;
}
__global__ void small_mixture_evaluate_kernel(const double* const* columns,const double* offsets,
        const double* kernel,const double* weights,i64 n,i64 q,double* sums) {
    __shared__ double terms[small_mixture_rows*(small_mixture_columns+2)];
    const i64 row=threadIdx.x;
    if(row<n) mixture_evaluate_row(columns,offsets,kernel,weights,n,q,row,terms);
    __syncthreads();  // All 128 participants publish, including inactive rows.
    if(row<q+2) sums[row]=small_mixture_sum(terms+row*n,n);
}
__global__ void mixture_finish_reduction(const double* partials,i64 chunks,i64 planes,double* sums) {
    for(i64 plane=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;plane<planes;
        plane+=static_cast<i64>(gridDim.x)*blockDim.x) {
        double sum=-0.;
        for(i64 chunk=0;chunk<chunks;++chunk) sum+=partials[plane*chunks+chunk];
        sums[plane]=sum;
    }
}
void reduce_mixture(MixtureProblem& p,i64 planes) {
    auto& c=p.context;
    mixture_combine_leaves_kernel<<<static_cast<unsigned>(std::min<i64>(65535,multiply(planes,p.chunks))),threads,0,c.stream>>>(
        p.at<double>(p.leaf_partials),p.n,planes,p.chunks,p.reduction.full_chunks,
        p.at<ScoringLeaf>(p.leaf_metadata),p.reduction.tail_count,p.reduction.leaf_count,
        p.at<double>(p.partials));
    c.launched();
    mixture_finish_reduction<<<blocks(planes),threads,0,c.stream>>>(
        p.at<double>(p.partials),p.chunks,planes,p.at<double>(p.sums));
    c.launched();
}
__global__ void mixture_log_weights_kernel(const double* weights,i64 q,double* logs) {
    for(i64 k=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;k<q;
        k+=static_cast<i64>(gridDim.x)*blockDim.x) logs[k]=log(weights[k]);
}
__device__ void mixture_pair_row(const double* const* columns,const double* offsets,
        const double* log_weights,i64 q,i64 row,i64 grow,i64 shrink,double* other,
        double* grow_log,double* shrink_log,double* numerator,double* sign) {
    const double offset=offsets[row],a=columns[grow][row]-offset,b=columns[shrink][row]-offset;
    grow_log[row]=a; shrink_log[row]=b;
    // Do not subtract two large masses from a total. The unchanged
    // complement is constructed once from the original finite log support.
    double maximum=-positive_infinity;
    for(i64 k=0;k<q;++k) if(k!=grow && k!=shrink) {
        const double value=columns[k][row]-offset+log_weights[k];
        if(value>maximum) maximum=value;
    }
    double complement=maximum;
    if(maximum!=-positive_infinity) {
        double sum=0.;
        for(i64 k=0;k<q;++k) if(k!=grow && k!=shrink)
            sum+=exp(columns[k][row]-offset+log_weights[k]-maximum);
        complement=maximum+log(sum);
    }
    other[row]=complement;
    if(a==b) { numerator[row]=-positive_infinity; sign[row]=0.; }
    else {
        const double high=a>b?a:b,delta=fabs(a-b);
        numerator[row]=log_weights[shrink]+high+log(-expm1(-delta));
        sign[row]=a>b?1.:-1.;
    }
}
__global__ void mixture_pair_kernel(const double* const* columns,const double* offsets,
        const double* log_weights,i64 n,i64 q,i64 grow,i64 shrink,double* other,
        double* grow_log,double* shrink_log,double* numerator,double* sign) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
        row+=static_cast<i64>(gridDim.x)*blockDim.x)
        mixture_pair_row(columns,offsets,log_weights,q,row,grow,shrink,
                         other,grow_log,shrink_log,numerator,sign);
}
__device__ double mixture_trial_row(i64 row,const double* other,const double* grow,
        const double* shrink,const double* numerator,const double* sign,double a,double b) {
    const double mass=scoring_logadd(other[row],scoring_logadd(grow[row]+a,shrink[row]+b));
    return sign[row]==0.?0.:sign[row]*scoring_capped_exp(numerator[row]-mass);
}
__global__ void small_mixture_trial_kernel(i64 n,const double* other,const double* grow,
        const double* shrink,const double* numerator,const double* sign,
        double grow_weight,double amount,double fraction,double* sums) {
    __shared__ double terms[small_mixture_rows];
    const double a=log(grow_weight+fraction*amount),b=log(amount*(1.-fraction));
    const i64 row=threadIdx.x;
    if(row<n) terms[row]=mixture_trial_row(row,other,grow,shrink,numerator,sign,a,b);
    __syncthreads();
    if(!row) sums[0]=small_mixture_sum(terms,n);
}
template<bool Endpoint>
__global__ void small_mixture_prepare_kernel(const double* const* columns,const double* offsets,
        const double* weights,double* log_weights,i64 n,i64 q,i64 grow,i64 shrink,
        double* other,double* grow_log,double* shrink_log,double* numerator,double* sign,
        double grow_weight,double amount,double* sums) {
    __shared__ double terms[small_mixture_rows];
    const i64 row=threadIdx.x;
    if(row<q) log_weights[row]=log(weights[row]);
    __syncthreads();
    if(row<n) {
        mixture_pair_row(columns,offsets,log_weights,q,row,grow,shrink,
                         other,grow_log,shrink_log,numerator,sign);
        if constexpr (Endpoint) {
            const double a=log(grow_weight+1.*amount),b=log(amount*(1.-1.));
            terms[row]=mixture_trial_row(row,other,grow_log,shrink_log,numerator,sign,a,b);
        }
    }
    if constexpr (Endpoint) {
        __syncthreads();
        if(!row) sums[0]=small_mixture_sum(terms,n);
    }
}
__global__ void gather_columns_kernel(const double* const* columns,const i64* labels,
        i64 first,i64 n,double* output) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
        row+=static_cast<i64>(gridDim.x)*blockDim.x) output[row]=columns[labels[row]][first+row];
}
void validate_scoring_weights(const double* weights,i64 q) {
    if(!weights) throw std::domain_error("Missing CUDA mixture weights");
    bool positive=false;
    for(i64 k=0;k<q;++k) {
        if(!std::isfinite(weights[k])||weights[k]<0.)
            throw std::domain_error("CUDA mixture weights must be nonnegative and finite");
        positive=positive||weights[k]>0.;
    }
    if(!positive) throw std::domain_error("CUDA mixture weights must be nonzero");
}
template<bool Endpoint=false>
void prepare_mixture_transfer(MixtureProblem& p,const double* weights,i64 grow,i64 shrink) {
    auto& c=p.context;
    p.pair_ready=false;
    validate_scoring_weights(weights,p.q);
    if(grow<0||grow>=p.q||shrink<0||shrink>=p.q||grow==shrink||weights[shrink]<=0.)
        throw std::domain_error("Invalid CUDA pairwise mixture transfer");
    p.grow_weight=weights[grow]; p.amount=weights[shrink];
    c.upload(p.at<double>(p.weights),weights,p.q);
    if(small_mixture(p)) {
        small_mixture_prepare_kernel<Endpoint><<<1,threads,0,c.stream>>>(
            p.at<const double*>(p.pointers),p.at<double>(p.offset),p.at<double>(p.weights),
            p.at<double>(p.log_weights),p.n,p.q,grow,shrink,p.at<double>(p.other),
            p.at<double>(p.grow_log),p.at<double>(p.shrink_log),p.at<double>(p.numerator),
            p.at<double>(p.sign),p.grow_weight,p.amount,p.at<double>(p.sums));
        c.launched();
    } else {
        mixture_log_weights_kernel<<<blocks(p.q),threads,0,c.stream>>>(
            p.at<double>(p.weights),p.q,p.at<double>(p.log_weights));
        c.launched();
        mixture_pair_kernel<<<blocks(p.n),threads,0,c.stream>>>(p.at<const double*>(p.pointers),
            p.at<double>(p.offset),p.at<double>(p.log_weights),p.n,p.q,grow,shrink,p.at<double>(p.other),
            p.at<double>(p.grow_log),p.at<double>(p.shrink_log),p.at<double>(p.numerator),p.at<double>(p.sign));
        c.launched();
    }
}
void finish_mixture_transfer(MixtureProblem& p,double* output) {
    auto& c=p.context;
    c.download(output,p.at<double>(p.sums),1); c.finish();
    *output/=p.n; ++scoring_state(c).counters[6];
}
void mixture_transfer_derivative(MixtureProblem& p,double fraction,double* output) {
    auto& c=p.context;
    if(small_mixture(p)) {
        small_mixture_trial_kernel<<<1,threads,0,c.stream>>>(p.n,p.at<double>(p.other),
            p.at<double>(p.grow_log),p.at<double>(p.shrink_log),p.at<double>(p.numerator),
            p.at<double>(p.sign),p.grow_weight,p.amount,fraction,p.at<double>(p.sums));
        c.launched();
    } else {
        mixture_leaf_trial_kernel<<<static_cast<unsigned>(std::min<i64>(65535,p.reduction.leaf_count)),threads,0,c.stream>>>(
            p.n,p.reduction.full_chunks,p.at<ScoringLeaf>(p.leaf_metadata),p.reduction.leaf_count,p.at<double>(p.other),
            p.at<double>(p.grow_log),p.at<double>(p.shrink_log),p.at<double>(p.numerator),
            p.at<double>(p.sign),p.grow_weight,p.amount,fraction,p.at<double>(p.leaf_partials));
        c.launched(); reduce_mixture(p,1);
    }
    finish_mixture_transfer(p,output);
}
#include "batch_scoring.cuh"
#include "proposal_scoring.cuh"
} // namespace

extern "C" int CliPP2GatherColumnRows(void* value,i64 q,const i64* handles,i64 count,
                                      const i64* rows,double* output) {
    return scoring_guarded([&] { gather_column_rows(context(value),q,handles,count,rows,output); });
}
extern "C" int CliPP2PositiveTransferChunk(void* value,i64 q,const i64* handles,i64 first,i64 count,
        const i64* labels,i64 capacity,i64* indices,double* gains,i64* returned) {
    return scoring_guarded([&] { positive_transfer_chunk(context(value),q,handles,first,count,labels,
                                                        capacity,indices,gains,returned); });
}
extern "C" int CliPP2CUDAProposalDiagnostics(void* value,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); require(output,"Missing CUDA proposal diagnostics");
        if(!c.scoring_state) { std::fill(output,output+4,0); return; }
        const auto& s=scoring_state(c); std::copy(s.proposal_counters,s.proposal_counters+4,output);
    });
}

extern "C" int CliPP2EvaluateWeightBatch(void* value,i64 count,const void* requests,
                                         i64 output_size,double* output) {
    return scoring_guarded([&] { evaluate_weight_batch(context(value),count,
        static_cast<const WeightBatchRequest*>(requests),output_size,output); });
}

extern "C" int CliPP2CreateJointColumns(void* value,i64 count,const double* centers,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& s=scoring_state(c);
        if(count<=0||!centers||!output) throw std::domain_error("Invalid resident CUDA column request");
        for(i64 i=0;i<multiply(count,c.data.regions);++i)
            if(!std::isfinite(centers[i])) throw std::domain_error("CUDA centers must be finite");
        // Conservative registry/key/vector staging reservation, distinct from
        // mandatory device payload and from the optional retained cache.
        const auto host_reservation=multiply(count,add(1024,multiply(c.data.regions,40)));
        check_scoring_host(c,host_reservation);
        ScoringHostScope staging(c,multiply(count,add(768,multiply(c.data.regions,40))));
        std::vector<i64> leased;
        leased.reserve(static_cast<std::size_t>(count));
        std::vector<std::shared_ptr<ResidentColumn>> requested;
        std::vector<std::string> fresh_keys;
        std::vector<double> fresh_centers;
        std::vector<double*> pointers;
        requested.reserve(static_cast<std::size_t>(count));
        fresh_keys.reserve(static_cast<std::size_t>(count));
        fresh_centers.reserve(static_cast<std::size_t>(multiply(count,c.data.regions)));
        pointers.reserve(static_cast<std::size_t>(count));
        std::map<std::string,std::shared_ptr<ResidentColumn>> fresh;
        try {
            for(i64 k=0;k<count;++k) {
                const double* center=centers+k*c.data.regions;
                const std::string key(reinterpret_cast<const char*>(center),multiply(c.data.regions,8));
                auto cached=s.cache.find(key);
                std::shared_ptr<ResidentColumn> column;
                if(cached!=s.cache.end()) {
                    column=cached->second.column;
                    s.order.splice(s.order.end(),s.order,cached->second.order); ++s.counters[1];
                } else {
                    const auto staged=fresh.find(key);
                    if(staged!=fresh.end()) { column=staged->second; ++s.counters[1]; }
                    else {
                        column=std::make_shared<ResidentColumn>(c);
                        fresh.emplace(key,column); fresh_keys.push_back(key);
                        fresh_centers.insert(fresh_centers.end(),center,center+c.data.regions);
                        pointers.push_back(column->data());
                    }
                }
                requested.push_back(std::move(column));
            }
            if(!pointers.empty()) {
                Layout layout;
                const auto dc=layout.take<double>(fresh_centers.size()),dp=layout.take<double*>(pointers.size()),
                           ds=layout.take<int>(1);
                ScoringAllocation header(c,layout.bytes);
                int status=0;
                ScoringStreamFence fence(c);
                c.upload(header.at<double>(dc),fresh_centers.data(),fresh_centers.size());
                c.upload(header.at<double*>(dp),pointers.data(),pointers.size());
                cuda_check(cudaMemsetAsync(header.at<int>(ds),0,sizeof(int),c.stream),"clear resident column status");
                const auto cells=multiply(pointers.size(),c.data.mutations);
                resident_joint_kernel<<<blocks(cells),threads,0,c.stream>>>(c.data,cells,
                    header.at<double>(dc),header.at<double*>(dp),header.at<int>(ds));
                c.launched(); c.download(&status,header.at<int>(ds),1); c.finish();
                if(status==static_cast<int>(EmitStatus::InvalidInput))
                    throw std::domain_error("Native CUDA columns received invalid CCFs");
                require(status==0,"Native CUDA columns produced invalid arithmetic");
                c.diagnostics.cell_evaluations=add(c.diagnostics.cell_evaluations,multiply(cells,c.data.regions));
                s.counters[0]=add(s.counters[0],pointers.size());
            }
            for(auto& column:requested) {
                const i64 id=s.id(); s.handles.emplace(id,column); leased.push_back(id);
            }
            s.host_bytes=add(s.host_bytes,multiply(count,256));
            for(const auto& key:fresh_keys) retain_scoring_column(c,key,fresh.at(key));
            std::copy(leased.begin(),leased.end(),output);
        } catch(...) {
            cudaStreamSynchronize(c.stream);
            for(const auto id:leased) s.handles.erase(id);
            // Registration charge is committed only after all leases exist.
            if(static_cast<i64>(leased.size())==count) s.host_bytes-=multiply(count,256);
            throw;
        }
    });
}
extern "C" int CliPP2ImportJointColumns(void* value,i64 count,const double* columns,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& s=scoring_state(c);
        if(count<=0||!columns||!output) throw std::domain_error("Invalid CUDA log-column import");
        check_scoring_host(c,multiply(count,512));
        ScoringHostScope staging(c,multiply(count,128));
        for(i64 i=0;i<multiply(count,c.data.mutations);++i)
            if(std::isnan(columns[i])||columns[i]==positive_infinity)
                throw std::domain_error("CUDA log columns must be finite or -inf");
        std::vector<i64> leased;
        leased.reserve(static_cast<std::size_t>(count));
        try {
            for(i64 k=0;k<count;++k) {
                auto column=std::make_shared<ResidentColumn>(c);
                c.upload(column->data(),columns+k*c.data.mutations,c.data.mutations);
                const i64 id=s.id(); s.handles.emplace(id,std::move(column)); leased.push_back(id);
            }
            c.finish(); s.host_bytes=add(s.host_bytes,multiply(count,256));
            std::copy(leased.begin(),leased.end(),output);
        } catch(...) { cudaStreamSynchronize(c.stream); for(const auto id:leased) s.handles.erase(id); throw; }
    });
}
extern "C" int CliPP2ReleaseJointColumns(void* value,i64 count,const i64* handles) {
    return scoring_guarded([&] {
        auto& s=scoring_state(context(value));
        if(count<0||(!handles&&count)) throw std::domain_error("Invalid CUDA column release");
        for(i64 i=0;i<count;++i) scoring_column(s,handles[i]);
        for(i64 i=0;i<count;++i) if(s.handles.erase(handles[i])) s.host_bytes-=256;
    });
}
extern "C" int CliPP2ExportJointColumns(void* value,i64 count,const i64* handles,double* output) {
    return scoring_guarded([&] {
        auto& c=context(value); const auto columns=get_columns(c,count,handles);
        ScoringHostScope staging(c,multiply(count,32));
        if(!output) throw std::domain_error("Missing CUDA column export destination");
        ScoringStreamFence fence(c);
        for(i64 k=0;k<count;++k) c.download(output+k*c.data.mutations,columns[k]->data(),c.data.mutations);
        c.finish(); ++scoring_state(c).counters[2];
    });
}
extern "C" int CliPP2GatherJointColumns(void* value,i64 count,const i64* handles,
        const i64* labels,double* output) {
    return scoring_guarded([&] {
        auto& c=context(value); const auto columns=get_columns(c,count,handles);
        ScoringHostScope staging(c,multiply(count,32));
        if(!labels||!output) throw std::domain_error("Missing CUDA assigned-column fields");
        for(i64 row=0;row<c.data.mutations;++row)
            if(labels[row]<0||labels[row]>=count) throw std::domain_error("CUDA assigned column out of range");
        ScoringRequestLease request(c,count,true);
        for(i64 k=0;k<count;++k) request.pointers()[k]=columns[k]->data();
        ScoringStreamFence fence(c);
        c.upload(request.device_pointers(),request.pointers(),count);
        const i64 tile=std::min(c.data.mutations,c.diagnostics.workspace_limit/16);
        require(tile>0,"CUDA assigned-value workspace cannot hold one row");
        Layout layout; const auto dl=layout.take<i64>(tile),dv=layout.take<double>(tile);
        c.reserve(layout.bytes);
        for(i64 first=0;first<c.data.mutations;first+=tile) {
            const i64 rows=std::min(tile,c.data.mutations-first);
            c.upload(c.pointer<i64>(dl),labels+first,rows);
            gather_columns_kernel<<<blocks(rows),threads,0,c.stream>>>(request.device_pointers(),
                c.pointer<i64>(dl),first,rows,c.pointer<double>(dv));
            c.launched(); c.download(output+first,c.pointer<double>(dv),rows); c.finish();
        }
        ++scoring_state(c).counters[3];
        request.complete=true;
    });
}
extern "C" int CliPP2CreateMixtureProblem(void* value,i64 count,const i64* handles,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); if(!output) throw std::domain_error("Missing CUDA mixture handle destination");
        *output=0;
        auto p=std::make_unique<MixtureProblem>(c,count);
        p->host_charge=add(sizeof(MixtureProblem)+256,multiply(count,64)); check_scoring_host(c,p->host_charge);
        ScoringHostScope staging(c,p->host_charge);
        p->columns=get_columns(c,count,handles);
        auto& s=scoring_state(c);
        if(s.arena_cache && s.arena_cache->bytes>=p->bytes) {
            c.drop_optional(s.arena_cache->bytes);
            p->allocation=std::move(s.arena_cache); ++s.arena_reuses;
        } else {
            p->allocation=std::make_unique<ScoringAllocation>(c,p->bytes); ++s.arena_allocations;
        }
        ScoringRequestLease request(c,count,false);
        for(i64 k=0;k<count;++k) request.pointers()[k]=p->columns[k]->data();
        int invalid=0;
        ScoringStreamFence fence(c);
        c.upload(p->at<const double*>(p->pointers),request.pointers(),count);
        if(p->reduction.tail_count)
            c.upload(p->at<ScoringLeaf>(p->leaf_metadata),p->reduction.tail,p->reduction.tail_count);
        cuda_check(cudaMemsetAsync(p->at<int>(p->invalid),0,sizeof(int),c.stream),"clear mixture status");
        mixture_offset_kernel<<<blocks(p->n),threads,0,c.stream>>>(p->at<const double*>(p->pointers),
            p->n,p->q,p->at<double>(p->offset),p->at<double>(p->kernel),p->at<int>(p->invalid));
        c.launched(); c.download(&invalid,p->at<int>(p->invalid),1); c.finish();
        if(invalid) throw std::domain_error("Every mutation needs finite likelihood in some cluster");
        const i64 id=s.id(),host_charge=p->host_charge;
        s.problems.emplace(id,std::move(p)); staging.release();
        s.host_bytes=add(s.host_bytes,host_charge); *output=id;
        request.complete=true;
    });
}
extern "C" int CliPP2ReleaseMixtureProblem(void* value,i64 handle) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        const auto charge=p.host_charge;
        auto& s=scoring_state(c);
        // Idle scratch is lower-priority than computed columns and grids. It
        // may replace an older arena, but must not evict useful emissions.
        const i64 free_optional=c.optional_cache_limit-c.optional_cache_bytes+
            (s.arena_cache?s.arena_cache->bytes:0);
        if(p.allocation->bytes<=free_optional &&
                (!s.arena_cache || s.arena_cache->bytes<p.allocation->bytes)) {
            if(s.arena_cache) { c.drop_optional(s.arena_cache->bytes); s.arena_cache.reset(); }
            if(c.retain_optional(p.allocation->bytes)) s.arena_cache=std::move(p.allocation);
        }
        s.problems.erase(handle); s.host_bytes-=charge;
    });
}
extern "C" int CliPP2CUDAMixtureRequiredBytes(i64 n,i64 q,i64* output) {
    return scoring_guarded([&] {
        if(!output) throw std::domain_error("Missing CUDA mixture memory result");
        *output=MixtureLayout(n,q).bytes;
    });
}
extern "C" int CliPP2CUDAMixtureAdmission(void* value,i64 q,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value);
        if(q<=0||!output) throw std::domain_error("Invalid CUDA mixture admission query");
        const MixtureLayout layout(c.data.mutations,q);
        const i64 packed=add(sizeof(WeightBatchItem),multiply(add(multiply(q,2),2),8));
        output[0]=add(layout.bytes,packed);
        output[1]=c.resident_limit-c.resident_bytes;
        output[2]=add(add(sizeof(MixtureProblem)+256,multiply(q,64)),
                      ScoringRequestStorage::host_bytes(add(packed,7)/8));
        if(!c.scoring_state) output[2]=add(output[2],sizeof(ScoringState)+128);
        output[3]=c.host_staging_limit-scoring_host_bytes(c)-c.grid_cache_host_bytes
                  -c.probability_host_bytes-c.diagnostics.staging_bytes;
    });
}
extern "C" int CliPP2EvaluateMixture(void* value,i64 handle,const double* weights,
        double* objective,double* gradient,double* likelihood) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        if(!objective||!gradient||!likelihood) throw std::domain_error("Missing CUDA mixture result fields");
        validate_scoring_weights(weights,p.q);
        // The result temporary precedes the fence, so a partial asynchronous
        // download cannot outlive its stack storage on an error path.
        double objective_sum=0.,small_sums[small_mixture_columns+2];
        ScoringStreamFence fence(c);
        c.upload(p.at<double>(p.weights),weights,p.q);
        if(small_mixture(p)) {
            small_mixture_evaluate_kernel<<<1,threads,0,c.stream>>>(p.at<const double*>(p.pointers),
                p.at<double>(p.offset),p.at<double>(p.kernel),p.at<double>(p.weights),p.n,p.q,p.at<double>(p.sums));
            c.launched(); c.download(small_sums,p.at<double>(p.sums),add(p.q,2)); c.finish();
            objective_sum=small_sums[0]; *likelihood=small_sums[1];
            std::copy(small_sums+2,small_sums+2+p.q,gradient);
        } else {
            mixture_leaf_evaluate_kernel<<<static_cast<unsigned>(std::min<i64>(65535,p.reduction.leaf_count)),threads,0,c.stream>>>(
                p.at<const double*>(p.pointers),p.at<double>(p.offset),p.at<double>(p.kernel),
                p.at<double>(p.weights),p.n,p.q,p.reduction.full_chunks,p.at<ScoringLeaf>(p.leaf_metadata),
                p.reduction.leaf_count,p.at<double>(p.leaf_partials));
            c.launched(); reduce_mixture(p,add(p.q,2));
            c.download(&objective_sum,p.at<double>(p.sums),1);
            c.download(likelihood,p.at<double>(p.sums)+1,1);
            c.download(gradient,p.at<double>(p.sums)+2,p.q); c.finish();
        }
        *objective=-objective_sum/p.n;
        for(i64 k=0;k<p.q;++k) gradient[k]=-gradient[k]/p.n;
        ++scoring_state(c).counters[4];
    });
}
extern "C" int CliPP2PrepareMixtureTransfer(void* value,i64 handle,const double* weights,i64 grow,i64 shrink) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        ScoringStreamFence fence(c);
        prepare_mixture_transfer(p,weights,grow,shrink);
        c.finish(); p.pair_ready=true; ++scoring_state(c).counters[5];
    });
}
extern "C" int CliPP2PrepareMixtureTransferEndpoint(void* value,i64 handle,const double* weights,
        i64 grow,i64 shrink,double* output) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        if(!output) throw std::domain_error("Missing CUDA transfer endpoint result");
        ScoringStreamFence fence(c);
        prepare_mixture_transfer<true>(p,weights,grow,shrink);
        if(small_mixture(p)) finish_mixture_transfer(p,output);
        else mixture_transfer_derivative(p,1.,output);
        p.pair_ready=true; ++scoring_state(c).counters[5];
    });
}
extern "C" int CliPP2MixtureTransferDerivative(void* value,i64 handle,double fraction,double* output) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        if(!p.pair_ready||!output||!std::isfinite(fraction)||fraction<0.||fraction>1.)
            throw std::domain_error("CUDA mixture transfer is unprepared or invalid");
        ScoringStreamFence fence(c);
        mixture_transfer_derivative(p,fraction,output);
    });
}
extern "C" int CliPP2CUDAScoringArenaDiagnostics(void* value,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); require(output,"Missing CUDA arena diagnostics");
        if(!c.scoring_state) { std::fill(output,output+3,0); return; }
        auto& s=scoring_state(c);
        output[0]=s.arena_allocations; output[1]=s.arena_reuses;
        output[2]=s.arena_cache?s.arena_cache->bytes:0;
    });
}
extern "C" int CliPP2CUDAScoringRequestDiagnostics(void* value,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); require(output,"Missing CUDA request-storage diagnostics");
        if(!c.scoring_state) { std::fill(output,output+8,0); return; }
        auto& s=scoring_state(c);
        output[0]=s.request_allocations; output[1]=s.request_reuses;
        output[2]=s.request_header_allocations; output[3]=s.request_live_objects;
        output[4]=s.request_live_host_bytes; output[5]=s.request_live_device_bytes;
        output[6]=s.request_cache?s.request_cache->host_charge:0;
        output[7]=s.request_cache&&s.request_cache->header?s.request_cache->header->bytes:0;
    });
}
extern "C" int CliPP2CUDAScoringDiagnostics(void* value,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); require(output,"Missing CUDA scoring diagnostics");
        if(!c.scoring_state) { std::fill(output,output+9,0); return; }
        auto& s=*static_cast<ScoringState*>(c.scoring_state);
        s.counters[7]=s.handles.size(); s.counters[8]=s.problems.size();
        std::copy(s.counters,s.counters+9,output);
    });
}
