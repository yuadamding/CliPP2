// Device-resident joint diagnostics, pooled integration, and group refits.
// All variable-size storage is allocated by the host before kernel execution.
__global__ void joint_diagnostics_kernel(ObservationSoA data,const double* phi,const double* accepted,
        const i64* offsets,double* rows,double* maps,double* component,double* accepted_p,double* map_p,
        double* conditional,double* probability,std::uint8_t* status,double* messages) {
    for(i64 i=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;i<data.mutations;
        i+=static_cast<i64>(gridDim.x)*blockDim.x) {
        JointMass mass{data,i,phi+i,data.mutations};
        status[i]=static_cast<std::uint8_t>(joint_diagnostics_row(mass,offsets,rows+4*i,maps,
            component,accepted_p,map_p,conditional,probability,messages+i*4*(data.regions+1),accepted));
    }
}
__global__ void pooled_components_kernel(ObservationSoA data,const i64* offsets,const i64* grid_offsets,
        const double* grids,const double* weights,double* masses,double* moments) {
    const auto count=data.mutations*data.regions;
    for(i64 c=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;c<count;
        c+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const auto r=c/data.mutations;const auto o=observation(data,c);
        for(i64 j=0;j<o.support;++j)pooled_component(o,o.minimum+j,grids+grid_offsets[r],weights+grid_offsets[r],
            grid_offsets[r+1]-grid_offsets[r],masses[offsets[c]+j],moments[offsets[c]+j]);
    }
}
__global__ void pooled_joint_kernel(ObservationSoA data,const i64* offsets,const double* masses,
        const double* moments,double* pilot,std::uint8_t* status,double* cavities,double* messages) {
    for(i64 i=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;i<data.mutations;
        i+=static_cast<i64>(gridDim.x)*blockDim.x) {
        JointMass mass{data,i,nullptr,1,offsets,masses};
        status[i]=static_cast<std::uint8_t>(pooled_pilot_row(mass,moments,pilot,cavities,messages+i*4*(data.regions+1)));
    }
}
__global__ void joint_refit_kernel(ObservationSoA data,i64 groups,const i64* offsets,const i64* members,
        const double* initial,const double* lower,const double* upper,double* centers,double* ll,
        std::int32_t* status,i64* iterations,double* projected,std::uint8_t* retained,
        const i64* component_bases,const i64* component_offsets,const i64* scratch_offsets,
        const i64* branch_offsets,double* scratch,i64* branches) {
    for(i64 g=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;g<groups;
        g+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const auto count=offsets[g+1]-offsets[g],R=data.regions;
        const auto local=component_offsets+component_bases[g];
        JointGroup group{data,members+offsets[g],count,lower+g*R,upper+g*R,local};
        refit_joint_group(group,initial+g*R,centers+g*R,ll[g],status[g],iterations[g],projected[g],retained[g],
            scratch+scratch_offsets[g],branches+branch_offsets[g],local[count*R],branch_offsets[g+1]-branch_offsets[g]);
    }
}
