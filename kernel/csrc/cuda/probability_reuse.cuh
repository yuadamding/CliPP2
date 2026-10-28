// Exact shared-center emission reuse. Bounds are deliberately not part of this
// catalog: callers keep their original assignment-box and missing-read rules.
// Host catalog begin (also exercised independently of the CUDA runtime).
struct CUDAProbabilityFamily { i64 cell, support, offset; };
struct CUDAProbabilityShape { i64 cell, offset; };
struct CUDAProbabilityRegion {
    i64 family_begin=0, families=0, shape_begin=0, shapes=0,
        component_begin=0, components=0, shape_components=0;
};
struct CUDAProbabilityCatalog {
    std::vector<CUDAProbabilityFamily> families;
    std::vector<CUDAProbabilityShape> shapes;
    std::vector<CUDAProbabilityRegion> regions;
    std::vector<i64> row_shapes, supports;
    bool valid=true;
    static std::uint64_t bits(double value) {
        std::uint64_t result; std::memcpy(&result,&value,sizeof(result)); return result;
    }
    explicit CUDAProbabilityCatalog(const ObservationSoA& data) {
        const auto cells=multiply(data.mutations,data.regions);
        row_shapes.assign(cells,-1); supports.assign(cells,0);
        families.reserve(cells); shapes.reserve(cells); regions.resize(data.regions);
        i64 components=0;
        for(i64 region=0;region<data.regions;++region) {
            auto& plan=regions[region];
            plan.family_begin=families.size(); plan.shape_begin=shapes.size();
            plan.component_begin=components;
            using FamilyKey=std::pair<std::uint64_t,std::uint8_t>;
            using ShapeKey=std::tuple<i64,i64,std::uint64_t,std::uint64_t,std::uint64_t>;
            std::map<FamilyKey,i64> family_ids;
            std::map<ShapeKey,i64> shape_ids;
            std::vector<i64> shape_families;
            for(i64 row=0;row<data.mutations;++row) {
                const auto cell=region*data.mutations+row;
                const auto o=observation(data,cell);
                if(!(o.flags&observed_flag)) continue;
                const auto family=family_ids.emplace(
                    FamilyKey{bits(o.scaling),static_cast<std::uint8_t>(o.flags&mixed_flag)},
                    families.size());
                const auto id=family.first->second;
                if(family.second) families.push_back({cell,o.support,0});
                else families[id].support=std::max(families[id].support,o.support);
                const auto shape=shape_ids.emplace(ShapeKey{id,o.support,bits(o.alt),
                    bits(o.ref),bits(o.log_constant)},shapes.size());
                if(shape.second) {
                    shapes.push_back({cell,0}); shape_families.push_back(id);
                    if(o.support>INT64_MAX-plan.shape_components) { valid=false; return; }
                    plan.shape_components=add(plan.shape_components,o.support);
                }
                row_shapes[cell]=shape.first->second; supports[cell]=o.support;
            }
            plan.families=static_cast<i64>(families.size())-plan.family_begin;
            plan.shapes=static_cast<i64>(shapes.size())-plan.shape_begin;
            for(i64 j=plan.family_begin;j<static_cast<i64>(families.size());++j) {
                families[j].offset=components;
                if(families[j].support>INT64_MAX-components) { valid=false; return; }
                components=add(components,families[j].support);
            }
            plan.components=components-plan.component_begin;
            for(i64 j=0;j<plan.shapes;++j)
                shapes[plan.shape_begin+j].offset=families[shape_families[j]].offset;
        }
    }
};
// Host catalog end.

struct CUDAProbabilityState {
    Context& owner;
    std::vector<CUDAProbabilityRegion> regions;
    std::vector<i64> supports;
    unsigned char* allocation=nullptr;
    const CUDAProbabilityFamily* families=nullptr;
    const CUDAProbabilityShape* shapes=nullptr;
    const CUDAProbabilityRegion* device_regions=nullptr;
    const i64* row_shapes=nullptr;
    i64 device_bytes=0,host_bytes=0,charge=0;
    bool live=false;
    explicit CUDAProbabilityState(Context& context):owner(context) {}
    ~CUDAProbabilityState() {
        if(allocation) owner.release_resident(allocation,device_bytes);
    }
};
struct CUDAProbabilityView {
    const CUDAProbabilityFamily* families=nullptr;
    const CUDAProbabilityShape* shapes=nullptr;
    const CUDAProbabilityRegion* regions=nullptr;
    const i64* row_shapes=nullptr;
};
struct CUDAProbabilityLease {
    Context& owner;
    CUDAProbabilityState* state;
    explicit CUDAProbabilityLease(Context& c,bool enabled=true):owner(c),
        state(enabled?static_cast<CUDAProbabilityState*>(c.probability_state):nullptr) {
        if(state) { require(!state->live,"CUDA probability metadata already leased"); state->live=true; }
    }
    CUDAProbabilityView view() const {
        return state?CUDAProbabilityView{state->families,state->shapes,state->device_regions,state->row_shapes}
                    :CUDAProbabilityView{};
    }
    ~CUDAProbabilityLease() {
        if(state) {
            if(std::uncaught_exceptions()) cudaStreamSynchronize(owner.stream);
            state->live=false;
        }
    }
};
bool evict_optional_probability_cache(Context& c) {
    auto* state=static_cast<CUDAProbabilityState*>(c.probability_state);
    if(!state || state->live) return false;
    c.drop_optional(state->charge); c.probability_host_bytes-=state->host_bytes;
    delete state; c.probability_state=nullptr; ++c.probability_counters[6];
    return true;
}
void destroy_probability_cache(Context& c) noexcept {
    evict_optional_probability_cache(c);
}
void configure_probability_reuse(Context& c,const ObservationSoA& input,bool enabled) {
    require(!c.probability_configured && c.diagnostics.calls==0,
            "CUDA probability metadata is configured once before first use");
    require(input.mutations==c.data.mutations && input.regions==c.data.regions,
            "CUDA probability metadata dimensions differ from prepared observations");
    c.probability_configured=true;
    const auto cells=multiply(input.mutations,input.regions);
    // Includes map nodes, maximum vector capacities and transfer-time overlap.
    // No optional metadata construction is attempted outside the host budget.
    const auto build_charge=add(1024,add(multiply(cells,512),
        multiply(input.regions,sizeof(CUDAProbabilityRegion))));
    if(!enabled || !c.optional_cache_limit || build_charge>c.host_staging_limit
            -c.diagnostics.staging_bytes-c.grid_cache_host_bytes-scoring_host_bytes(c)) {
        ++c.probability_counters[7]; return;
    }
    struct BuildScope {
        Context& owner; i64 bytes;
        BuildScope(Context& c,i64 charge):owner(c),bytes(charge) {
            owner.diagnostics.staging_bytes+=bytes;
        }
        ~BuildScope() { owner.diagnostics.staging_bytes-=bytes; }
    } build_scope(c,build_charge);
    CUDAProbabilityCatalog catalog(input);
    if(!catalog.valid) { ++c.probability_counters[7]; return; }
    bool useful=false;
    for(i64 region=0;region<input.regions;++region) {
        const auto& p=catalog.regions[region];
        if(p.components>c.diagnostics.workspace_limit/16 || p.shape_components>INT64_MAX/4) {
            ++c.probability_counters[7]; return;
        }
        i64 direct=0;
        for(i64 row=0;row<input.mutations;++row) {
            const auto support=catalog.supports[region*input.mutations+row];
            if(support>INT64_MAX/4-direct) { ++c.probability_counters[7]; return; }
            direct+=support;
        }
        useful|=add(multiply(p.components,2),p.shape_components)<multiply(direct,2);
    }
    if(!useful) { ++c.probability_counters[7]; return; }
    Layout layout;
    const auto df=layout.take<CUDAProbabilityFamily>(catalog.families.size()),
        ds=layout.take<CUDAProbabilityShape>(catalog.shapes.size()), dr=layout.take<i64>(cells),
        regions=layout.take<CUDAProbabilityRegion>(input.regions);
    const auto host=add(sizeof(CUDAProbabilityState)+256,
        add(multiply(catalog.regions.capacity(),sizeof(CUDAProbabilityRegion)),
            multiply(catalog.supports.capacity(),sizeof(i64))));
    const auto charge=add(host,layout.bytes);
    if(charge>c.optional_cache_limit-c.optional_cache_bytes ||
            layout.bytes>c.resident_limit-c.resident_bytes ||
            host>c.host_staging_limit-c.grid_cache_host_bytes-scoring_host_bytes(c)) {
        ++c.probability_counters[7]; return;
    }
    auto state=std::make_unique<CUDAProbabilityState>(c);
    state->regions=std::move(catalog.regions); state->supports=std::move(catalog.supports);
    state->device_bytes=layout.bytes; state->host_bytes=host; state->charge=charge;
    state->allocation=c.allocate_resident(layout.bytes);
    auto* families=reinterpret_cast<CUDAProbabilityFamily*>(state->allocation+df);
    auto* shapes=reinterpret_cast<CUDAProbabilityShape*>(state->allocation+ds);
    auto* rows=reinterpret_cast<i64*>(state->allocation+dr);
    auto* device_regions=reinterpret_cast<CUDAProbabilityRegion*>(state->allocation+regions);
    try {
        c.upload(families,catalog.families.data(),catalog.families.size());
        c.upload(shapes,catalog.shapes.data(),catalog.shapes.size());
        c.upload(rows,catalog.row_shapes.data(),cells);
        c.upload(device_regions,state->regions.data(),input.regions);
        cuda_check(cudaStreamSynchronize(c.stream),"complete CUDA probability metadata upload");
        ++c.diagnostics.synchronizations;
    } catch(...) {
        cudaStreamSynchronize(c.stream);  // Drain references before freeing host vectors.
        throw;
    }
    state->families=families; state->shapes=shapes; state->row_shapes=rows;
    state->device_regions=device_regions;
    c.optional_cache_bytes+=charge; c.probability_host_bytes+=host;
    c.probability_state=state.release(); ++c.probability_counters[0];
}

__device__ void probability_basis_component(ObservationSoA data,
        const CUDAProbabilityFamily* families,CUDAProbabilityRegion plan,
        i64 component,double center,double* log_p,double* log_q) {
    i64 lo=plan.family_begin, hi=lo+plan.families;
    const auto absolute=component+plan.component_begin;
    while(lo+1<hi) {
        const auto middle=lo+(hi-lo)/2;
        if(families[middle].offset<=absolute) lo=middle; else hi=middle;
    }
    const auto family=families[lo];
    const auto p=component_probability(observation(data,family.cell),center,absolute-family.offset+1);
    log_p[component]=::log(p); log_q[component]=::log1p(-p);
}
__global__ void probability_basis_kernel(ObservationSoA data,
        const CUDAProbabilityFamily* families,CUDAProbabilityRegion plan,
        i64 count,const double* grid,double* log_p,double* log_q) {
    for(i64 index=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
            index+=static_cast<i64>(blockDim.x)*gridDim.x) {
        const auto center=grid[index/plan.components];
        if(!finite_number(center) || center<0 || center>1) continue;
        const auto offset=(index/plan.components)*plan.components;
        probability_basis_component(data,families,plan,index%plan.components,center,
                                    log_p+offset,log_q+offset);
    }
}
__device__ double probability_mass(const Observation& o,const double* log_p,
                                   const double* log_q,i64 m) {
    return (o.alt==0?0:o.alt*log_p[m])+(o.ref==0?0:o.ref*log_q[m])+o.log_constant;
}
__device__ CellTerms probability_shape_terms(const Observation& o,double center,
        const double* p,const double* q) {
    if(!(o.flags&observed_flag)) return {0,0,0,EmitStatus::Ok};
    if(!finite_number(center) || center<0 || center>1)
        return {positive_infinity,0,1,EmitStatus::InvalidInput};
    double maximum=-positive_infinity;
    for(i64 m=0;m<o.support;++m) {
        const auto value=probability_mass(o,p,q,m); if(value>maximum) maximum=value;
    }
    if(maximum==-positive_infinity) return {positive_infinity,0,1,EmitStatus::ZeroLikelihood};
    if(!finite_number(maximum)) return {positive_infinity,0,1,EmitStatus::NumericalFailure};
    double total=0;
    for(i64 m=0;m<o.support;++m) total+=::exp(probability_mass(o,p,q,m)-maximum);
    return {-(maximum+::log(total)),0,0,EmitStatus::Ok};
}
__global__ void probability_shape_kernel(ObservationSoA data,
        const CUDAProbabilityShape* shapes,CUDAProbabilityRegion plan,i64 count,
        const double* grid,const double* log_p,const double* log_q,
        double* ll,std::uint8_t* status) {
    for(i64 index=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
            index+=static_cast<i64>(blockDim.x)*gridDim.x) {
        const auto center=grid[index/plan.shapes];
        const auto shape=shapes[plan.shape_begin+index%plan.shapes];
        const auto o=observation(data,shape.cell);
        const auto offset=(index/plan.shapes)*plan.components+shape.offset-plan.component_begin;
        const auto* p=log_p+offset; const auto* q=log_q+offset;
        const auto terms=probability_shape_terms(o,center,p,q);
        ll[index]=-terms.loss; status[index]=static_cast<std::uint8_t>(terms.status);
    }
}
__global__ void probability_gather_kernel(i64 region,i64 mutations,i64 rows,
        const i64* row_ids,const i64* row_shapes,CUDAProbabilityRegion plan,i64 count,
        const double* shape_ll,const std::uint8_t* shape_status,double* ll,std::uint8_t* status) {
    for(i64 index=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;index<count;
            index+=static_cast<i64>(blockDim.x)*gridDim.x) {
        const auto shape=row_shapes[region*mutations+row_ids[index%rows]];
        if(shape<0) {
            ll[index]=-0.0; status[index]=static_cast<std::uint8_t>(EmitStatus::Ok);
        } else {
            const auto source=(index/rows)*plan.shapes+shape-plan.shape_begin;
            ll[index]=shape_ll[source]; status[index]=shape_status[source];
        }
    }
}

bool probability_grid(Context& c,i64 region,i64 row_count,const i64* rows,
        i64 grid_count,const double* grid,double* ll,std::uint8_t* status) {
    auto* state=static_cast<CUDAProbabilityState*>(c.probability_state);
    const auto count=multiply(row_count,grid_count);
    if(!state || count<4096 || row_count<c.data.mutations/2) return false;
    const auto plan=state->regions[region];
    i64 direct=0;
    for(i64 i=0;i<row_count;++i) {
        const auto support=state->supports[region*c.data.mutations+rows[i]];
        if(support>INT64_MAX/4-direct) return false;
        direct+=support;
    }
    // Logs are shared by family; exp/support sums by exact shape. This rejects
    // all-unique families and requests too sparse to amortize extra launches.
    if(!plan.families || !plan.shapes ||
            add(multiply(plan.components,2),plan.shape_components)>=multiply(direct,2)) return false;
    const auto shapes=multiply(grid_count,plan.shapes), components=multiply(grid_count,plan.components);
    Layout layout;
    const auto dr=layout.take<i64>(row_count), dg=layout.take<double>(grid_count),
        dp=layout.take<double>(components), dq=layout.take<double>(components),
        dl=layout.take<double>(shapes), ds=layout.take<std::uint8_t>(shapes),
        out=layout.take<double>(count), os=layout.take<std::uint8_t>(count);
    if(layout.bytes>c.diagnostics.workspace_limit) return false;
    CUDAProbabilityLease lease(c);
    try {
        c.reserve(layout.bytes); c.upload(c.pointer<i64>(dr),rows,row_count);
        c.upload(c.pointer<double>(dg),grid,grid_count);
        probability_basis_kernel<<<blocks(components),threads,0,c.stream>>>(c.data,state->families,
            plan,components,c.pointer<double>(dg),c.pointer<double>(dp),c.pointer<double>(dq));
        c.launched();
        probability_shape_kernel<<<blocks(shapes),threads,0,c.stream>>>(c.data,state->shapes,plan,
            shapes,c.pointer<double>(dg),c.pointer<double>(dp),c.pointer<double>(dq),
            c.pointer<double>(dl),c.pointer<std::uint8_t>(ds));
        c.launched();
        probability_gather_kernel<<<blocks(count),threads,0,c.stream>>>(region,c.data.mutations,
            row_count,c.pointer<i64>(dr),state->row_shapes,plan,count,c.pointer<double>(dl),
            c.pointer<std::uint8_t>(ds),c.pointer<double>(out),c.pointer<std::uint8_t>(os));
        c.launched(); c.download(ll,c.pointer<double>(out),count);
        c.download(status,c.pointer<std::uint8_t>(os),count);
        c.diagnostics.cell_evaluations=add(c.diagnostics.cell_evaluations,count); c.finish();
        ++c.probability_counters[1];
        i64 valid=0;
        for(i64 i=0;i<grid_count;++i) valid+=finite_number(grid[i]) && grid[i]>=0 && grid[i]<=1;
        c.probability_counters[3]=add(c.probability_counters[3],multiply(valid,plan.components));
        c.probability_counters[4]=add(c.probability_counters[4],multiply(valid,plan.shapes));
        c.probability_counters[5]=add(c.probability_counters[5],count);
    } catch(...) {
        cudaStreamSynchronize(c.stream); throw;
    }
    return true;
}
