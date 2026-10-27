// Context-owned immutable likelihood columns and resident mixture arithmetic.
// Included after backend.cu's kernels, Context and ordered reduction helper.
#include <list>
#include <map>
#include <cfloat>
#include <atomic>

namespace {
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
struct MixtureProblem {
    Context& context;
    std::vector<std::shared_ptr<ResidentColumn>> columns;
    std::unique_ptr<ScoringAllocation> allocation;
    i64 n=0,q=0,chunks=0,pointers=0,weights=0,offset=0,kernel=0,terms=0,partials=0,sums=0;
    i64 other=0,grow_log=0,shrink_log=0,numerator=0,sign=0,invalid=0;
    double grow_weight=0,amount=0;
    bool pair_ready=false;
    i64 host_charge=0;
    explicit MixtureProblem(Context& owner):context(owner) {}
    template<class T> T* at(i64 position) { return allocation->at<T>(position); }
};
struct ScoringCacheEntry {
    std::shared_ptr<ResidentColumn> column;
    std::list<std::string>::iterator order;
    i64 charge=0;
    i64 host_charge=0;
};
struct ScoringState {
    std::map<i64,std::shared_ptr<ResidentColumn>> handles;
    std::map<i64,std::unique_ptr<MixtureProblem>> problems;
    std::map<std::string,ScoringCacheEntry> cache;
    std::list<std::string> order;
    i64 host_bytes=0,staging_bytes=0;
    // evaluations, cache hits, exports, gathers, weight evaluations,
    // transfer preparations, O(N) trials, live handles, live problems.
    i64 counters[9]{};
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
        if(charge>c.host_staging_limit-add(c.grid_cache_host_bytes,c.diagnostics.staging_bytes))
            release_optional_grid_cache(c);
        require(charge<=c.host_staging_limit-add(c.grid_cache_host_bytes,c.diagnostics.staging_bytes),
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
void release_optional_scoring_cache(Context& c) {
    if(!c.scoring_state) return;
    auto& s=*static_cast<ScoringState*>(c.scoring_state);
    for(const auto& entry:s.cache) {
        c.drop_optional(entry.second.charge); s.host_bytes-=entry.second.host_charge;
    }
    s.cache.clear(); s.order.clear();
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
    auto used=add(scoring_host_bytes(c),add(c.grid_cache_host_bytes,c.diagnostics.staging_bytes));
    if(extra>c.host_staging_limit-used) {
        release_optional_grid_cache(c); release_optional_scoring_cache(c);
        used=add(scoring_host_bytes(c),add(c.grid_cache_host_bytes,c.diagnostics.staging_bytes));
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
    if(add(scoring_host_bytes(c),add(c.grid_cache_host_bytes,add(c.diagnostics.staging_bytes,host_charge)))
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
__global__ void mixture_evaluate_kernel(const double* const* columns,const double* offsets,const double* kernel,
        const double* weights,i64 n,i64 q,double* terms) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
        row+=static_cast<i64>(gridDim.x)*blockDim.x) {
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
                for(i64 k=0;k<q;++k)
                    sum+=exp(columns[k][row]-offset+log(weights[k])-maximum);
                log_mass=maximum+log(sum);
            }
        }
        terms[row]=log_mass;
        terms[n+row]=offset+log_mass;
        for(i64 k=0;k<q;++k)
            terms[(k+2)*n+row]=scoring_capped_exp(columns[k][row]-offset-log_mass);
    }
}
// Each block evaluates one logical 8192-value subtree. Final large chunks are
// folded sequentially, retaining common/scalar_search.h's reduction topology.
__global__ void mixture_reduce_kernel(const double* values,i64 n,i64 planes,i64 chunks,
        double* partials) {
    __shared__ double shared[128],total;
    for(i64 job=blockIdx.x;job<planes*chunks;job+=gridDim.x) {
        const i64 plane=job/chunks,first=(job%chunks)*8192;
        const i64 count=n-first<8192?n-first:8192;
        const double value=ordered_block_sum(values+plane*n+first,count,shared,&total);
        if(!threadIdx.x) partials[job]=value;
        __syncthreads();
    }
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
    mixture_reduce_kernel<<<static_cast<unsigned>(std::min<i64>(65535,multiply(planes,p.chunks))),threads,0,c.stream>>>(
        p.at<double>(p.terms),p.n,planes,p.chunks,p.at<double>(p.partials));
    c.launched();
    mixture_finish_reduction<<<blocks(planes),threads,0,c.stream>>>(
        p.at<double>(p.partials),p.chunks,planes,p.at<double>(p.sums));
    c.launched();
}
__global__ void mixture_pair_kernel(const double* const* columns,const double* offsets,
        const double* weights,i64 n,i64 q,i64 grow,i64 shrink,double* other,
        double* grow_log,double* shrink_log,double* numerator,double* sign) {
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
        row+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const double offset=offsets[row],a=columns[grow][row]-offset,b=columns[shrink][row]-offset;
        grow_log[row]=a; shrink_log[row]=b;
        // Do not subtract two large masses from a total. The unchanged
        // complement is constructed once from the original finite log support.
        double maximum=-positive_infinity;
        for(i64 k=0;k<q;++k) if(k!=grow && k!=shrink) {
            const double value=columns[k][row]-offset+log(weights[k]);
            if(value>maximum) maximum=value;
        }
        double complement=maximum;
        if(maximum!=-positive_infinity) {
            double sum=0.;
            for(i64 k=0;k<q;++k) if(k!=grow && k!=shrink)
                sum+=exp(columns[k][row]-offset+log(weights[k])-maximum);
            complement=maximum+log(sum);
        }
        other[row]=complement;
        if(a==b) { numerator[row]=-positive_infinity; sign[row]=0.; }
        else {
            const double high=a>b?a:b,delta=fabs(a-b);
            numerator[row]=log(weights[shrink])+high+log(-expm1(-delta));
            sign[row]=a>b?1.:-1.;
        }
    }
}
__global__ void mixture_trial_kernel(i64 n,const double* other,const double* grow,
        const double* shrink,const double* numerator,const double* sign,
        double grow_weight,double amount,double fraction,double* terms) {
    const double a=log(grow_weight+fraction*amount),b=log(amount*(1.-fraction));
    for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<n;
        row+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const double mass=scoring_logadd(other[row],scoring_logadd(grow[row]+a,shrink[row]+b));
        terms[row]=sign[row]==0.?0.:sign[row]*scoring_capped_exp(numerator[row]-mass);
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
} // namespace

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
        ScoringHostScope staging(c,multiply(count,64));
        if(!labels||!output) throw std::domain_error("Missing CUDA assigned-column fields");
        for(i64 row=0;row<c.data.mutations;++row)
            if(labels[row]<0||labels[row]>=count) throw std::domain_error("CUDA assigned column out of range");
        check_scoring_host(c,multiply(count,64));
        std::vector<const double*> pointers;
        for(const auto& column:columns) pointers.push_back(column->data());
        ScoringAllocation header(c,multiply(count,sizeof(const double*)));
        ScoringStreamFence fence(c);
        c.upload(header.at<const double*>(),pointers.data(),count);
        const i64 tile=std::min(c.data.mutations,c.diagnostics.workspace_limit/16);
        require(tile>0,"CUDA assigned-value workspace cannot hold one row");
        Layout layout; const auto dl=layout.take<i64>(tile),dv=layout.take<double>(tile);
        c.reserve(layout.bytes);
        for(i64 first=0;first<c.data.mutations;first+=tile) {
            const i64 rows=std::min(tile,c.data.mutations-first);
            c.upload(c.pointer<i64>(dl),labels+first,rows);
            gather_columns_kernel<<<blocks(rows),threads,0,c.stream>>>(header.at<const double*>(),
                c.pointer<i64>(dl),first,rows,c.pointer<double>(dv));
            c.launched(); c.download(output+first,c.pointer<double>(dv),rows); c.finish();
        }
        ++scoring_state(c).counters[3];
    });
}
extern "C" int CliPP2CreateMixtureProblem(void* value,i64 count,const i64* handles,i64* output) {
    return scoring_guarded([&] {
        auto& c=context(value); if(!output) throw std::domain_error("Missing CUDA mixture handle destination");
        *output=0;
        auto p=std::make_unique<MixtureProblem>(c);
        p->host_charge=add(1024,multiply(count,64)); check_scoring_host(c,p->host_charge);
        ScoringHostScope staging(c,p->host_charge);
        p->columns=get_columns(c,count,handles); p->n=c.data.mutations; p->q=count;
        p->chunks=p->n/8192+(p->n%8192!=0);
        Layout layout;
        p->pointers=layout.take<const double*>(count); p->weights=layout.take<double>(count);
        p->offset=layout.take<double>(p->n); p->kernel=layout.take<double>(multiply(p->n,count));
        p->terms=layout.take<double>(multiply(p->n,add(count,2)));
        p->partials=layout.take<double>(multiply(p->chunks,add(count,2)));
        p->sums=layout.take<double>(add(count,2)); p->other=layout.take<double>(p->n);
        p->grow_log=layout.take<double>(p->n); p->shrink_log=layout.take<double>(p->n);
        p->numerator=layout.take<double>(p->n); p->sign=layout.take<double>(p->n);
        p->invalid=layout.take<int>(1);
        p->allocation=std::make_unique<ScoringAllocation>(c,layout.bytes);
        std::vector<const double*> pointers;
        for(const auto& column:p->columns) pointers.push_back(column->data());
        int invalid=0;
        ScoringStreamFence fence(c);
        c.upload(p->at<const double*>(p->pointers),pointers.data(),count);
        cuda_check(cudaMemsetAsync(p->at<int>(p->invalid),0,sizeof(int),c.stream),"clear mixture status");
        mixture_offset_kernel<<<blocks(p->n),threads,0,c.stream>>>(p->at<const double*>(p->pointers),
            p->n,p->q,p->at<double>(p->offset),p->at<double>(p->kernel),p->at<int>(p->invalid));
        c.launched(); c.download(&invalid,p->at<int>(p->invalid),1); c.finish();
        if(invalid) throw std::domain_error("Every mutation needs finite likelihood in some cluster");
        auto& s=scoring_state(c); const i64 id=s.id(),host_charge=p->host_charge;
        s.problems.emplace(id,std::move(p)); staging.release();
        s.host_bytes=add(s.host_bytes,host_charge); *output=id;
    });
}
extern "C" int CliPP2ReleaseMixtureProblem(void* value,i64 handle) {
    return scoring_guarded([&] {
        auto& c=context(value); const auto charge=scoring_problem(c,handle).host_charge;
        auto& s=scoring_state(c); s.problems.erase(handle); s.host_bytes-=charge;
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
        double objective_sum=0.;
        ScoringStreamFence fence(c);
        c.upload(p.at<double>(p.weights),weights,p.q);
        mixture_evaluate_kernel<<<blocks(p.n),threads,0,c.stream>>>(p.at<const double*>(p.pointers),
            p.at<double>(p.offset),p.at<double>(p.kernel),p.at<double>(p.weights),p.n,p.q,p.at<double>(p.terms));
        c.launched(); reduce_mixture(p,add(p.q,2));
        c.download(&objective_sum,p.at<double>(p.sums),1);
        c.download(likelihood,p.at<double>(p.sums)+1,1);
        c.download(gradient,p.at<double>(p.sums)+2,p.q); c.finish();
        *objective=-objective_sum/p.n;
        for(i64 k=0;k<p.q;++k) gradient[k]=-gradient[k]/p.n;
        ++scoring_state(c).counters[4];
    });
}
extern "C" int CliPP2PrepareMixtureTransfer(void* value,i64 handle,const double* weights,i64 grow,i64 shrink) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        validate_scoring_weights(weights,p.q);
        if(grow<0||grow>=p.q||shrink<0||shrink>=p.q||grow==shrink||weights[shrink]<=0.)
            throw std::domain_error("Invalid CUDA pairwise mixture transfer");
        ScoringStreamFence fence(c);
        p.grow_weight=weights[grow]; p.amount=weights[shrink]; p.pair_ready=false;
        c.upload(p.at<double>(p.weights),weights,p.q);
        mixture_pair_kernel<<<blocks(p.n),threads,0,c.stream>>>(p.at<const double*>(p.pointers),
            p.at<double>(p.offset),p.at<double>(p.weights),p.n,p.q,grow,shrink,p.at<double>(p.other),
            p.at<double>(p.grow_log),p.at<double>(p.shrink_log),p.at<double>(p.numerator),p.at<double>(p.sign));
        c.launched(); c.finish(); p.pair_ready=true; ++scoring_state(c).counters[5];
    });
}
extern "C" int CliPP2MixtureTransferDerivative(void* value,i64 handle,double fraction,double* output) {
    return scoring_guarded([&] {
        auto& c=context(value); auto& p=scoring_problem(c,handle);
        if(!p.pair_ready||!output||!std::isfinite(fraction)||fraction<0.||fraction>1.)
            throw std::domain_error("CUDA mixture transfer is unprepared or invalid");
        ScoringStreamFence fence(c);
        mixture_trial_kernel<<<blocks(p.n),threads,0,c.stream>>>(p.n,p.at<double>(p.other),
            p.at<double>(p.grow_log),p.at<double>(p.shrink_log),p.at<double>(p.numerator),
            p.at<double>(p.sign),p.grow_weight,p.amount,fraction,p.at<double>(p.terms));
        c.launched(); reduce_mixture(p,1); c.download(output,p.at<double>(p.sums),1); c.finish();
        *output/=p.n; ++scoring_state(c).counters[6];
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
