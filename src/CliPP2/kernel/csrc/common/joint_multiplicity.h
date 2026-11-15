#pragma once
#include "emission_math.h"

namespace clipp2 {
// Raw binomial masses: regional constants contain their independent uniform
// prior, which must be removed before applying one joint-vector normalizer.
CLIPP_HD inline double raw_component_log_mass(const Observation& o, double phi, std::int64_t m) {
    const double p=component_probability(o,phi,m);
    const double constant=o.log_constant+::log(static_cast<double>(o.support));
    return (o.alt==0?0.:o.alt*::log(p))+(o.ref==0?0.:o.ref*::log1p(-p))+constant;
}
CLIPP_HD inline double joint_add(double a, double b, bool maximum=false) {
    if(maximum) return a>b?a:b;
    if(a == -positive_infinity) return b;
    if(b == -positive_infinity) return a;
    const double high=a>b?a:b, low=a>b?b:a;
    return high+::log1p(::exp(low-high));
}
struct JointMass {
    const ObservationSoA& data;
    std::int64_t row;
    const double* phi;
    std::int64_t stride;
    const std::int64_t* offsets=nullptr;
    const double* masses=nullptr;
    bool counts=false;
    std::int64_t omit=-1;
    CLIPP_HD bool active(std::int64_t r) const {return (data.flags[r*data.mutations+row]&observed_flag)!=0;}
    CLIPP_HD std::int64_t lo(std::int64_t r) const {return data.minimum[r*data.mutations+row];}
    CLIPP_HD std::int64_t hi(std::int64_t r) const {const auto c=r*data.mutations+row;return data.minimum[c]+data.support[c]-1;}
    CLIPP_HD double operator()(std::int64_t r,std::int64_t m) const {
        if(m<lo(r)||m>hi(r)) return -positive_infinity;
        if(counts||r==omit) return 0.;
        const auto cell=r*data.mutations+row;
        return masses?masses[offsets[cell]+m-lo(r)]:raw_component_log_mass(observation(data,cell),phi[r*stride],m);
    }
};
// Canonical-minimum two-state DP. A vector with minimum h is counted only at
// anchor h. Optional fixed/prefix values support exact cavities and MAP ties.
CLIPP_HD inline double joint_reduce(const JointMass& mass,bool maximum=false,
        std::int64_t fixed=-1,std::int64_t fixed_m=0,
        const double* prefix=nullptr,std::int64_t prefix_regions=0,std::int64_t prefix_stride=1) {
    const auto& data=mass.data;
    std::int64_t L=1,U=9223372036854775807LL,active=0;
    for(std::int64_t r=0;r<data.regions;++r) if(mass.active(r)) {
        auto lo=mass.lo(r),hi=mass.hi(r);
        if(r==fixed)lo=hi=fixed_m;
        else if(prefix&&r<prefix_regions)lo=hi=static_cast<std::int64_t>(prefix[r*prefix_stride]);
        if(lo<mass.lo(r)||hi>mass.hi(r)||lo>hi)return -positive_infinity;
        if(lo>L)L=lo;if(hi<U)U=hi;++active;
    }
    if(!active)return 0.;
    if(data.max_multiplicity_difference<0) {
        double total=0.;
        for(std::int64_t r=0;r<data.regions;++r) if(mass.active(r)) {
            auto lo=mass.lo(r),hi=mass.hi(r);
            if(r==fixed)lo=hi=fixed_m;
            else if(prefix&&r<prefix_regions)lo=hi=static_cast<std::int64_t>(prefix[r*prefix_stride]);
            double value=-positive_infinity;
            for(auto m=lo;m<=hi;++m)value=joint_add(value,mass(r,m),maximum);
            total+=value;
        }
        return total;
    }
    if(L>U&&L-U>1)return -positive_infinity;
    const auto first=L>1?L-1:1;
    double total=-positive_infinity;
    for(auto h=first;h<=U;++h) {
        double f0=0.,f1=-positive_infinity;
        for(std::int64_t r=0;r<data.regions;++r) if(mass.active(r)) {
            double x=mass(r,h),y=h==9223372036854775807LL?-positive_infinity:mass(r,h+1);
            const auto selected=r==fixed?fixed_m:(prefix&&r<prefix_regions?static_cast<std::int64_t>(prefix[r*prefix_stride]):0);
            if(selected&&selected!=h)x=-positive_infinity;
            if(selected&&selected!=h+1)y=-positive_infinity;
            const double next=joint_add(joint_add(f1+x,f1+y,maximum),f0+x,maximum);
            f0+=y;f1=next;
        }
        total=joint_add(total,f1,maximum);
        if(h==9223372036854775807LL)break;
    }
    return total;
}
CLIPP_HD inline double joint_log_normalizer(const JointMass& mass) {
    auto count=mass;count.counts=true;return joint_reduce(count);
}
CLIPP_HD inline double joint_score(const JointMass& mass,bool maximum) {
    const double raw=joint_reduce(mass,maximum);
    if(maximum||raw==-positive_infinity)return raw;
    return raw-joint_log_normalizer(mass);
}
CLIPP_HD inline void joint_map(const JointMass& mass,double* output,std::int64_t stride) {
    const double best=joint_reduce(mass,true);
    for(std::int64_t r=0;r<mass.data.regions;++r) {
        output[r*stride]=quiet_nan;
        if(!mass.active(r)||!finite_number(best))continue;
        for(auto m=mass.lo(r);m<=mass.hi(r);++m) {
            const double score=joint_reduce(mass,true,r,m,output,r,stride);
            if(score==best){output[r*stride]=static_cast<double>(m);break;}
        }
    }
}
// Forward/backward cavity messages in O(HR). The caller owns ragged component
// storage and 4(R+1) doubles, shared between anchors without per-step allocation.
CLIPP_HD inline double joint_cavities(const JointMass& mass,const std::int64_t* offsets,
        double* cavities,double* scratch,bool maximum=false,std::int64_t offset_stride=1) {
    const auto R=mass.data.regions;double* f0=scratch;double* f1=f0+R+1;
    double* b0=f1+R+1;double* b1=b0+R+1;
    std::int64_t L=1,U=9223372036854775807LL,active=0;
    for(std::int64_t r=0;r<R;++r) {
        for(auto m=mass.lo(r);m<=mass.hi(r);++m)cavities[offsets[r*offset_stride]+m-mass.lo(r)]=-positive_infinity;
        if(mass.active(r)){if(mass.lo(r)>L)L=mass.lo(r);if(mass.hi(r)<U)U=mass.hi(r);++active;}
    }
    if(!active)return 0.;
    if(mass.data.max_multiplicity_difference<0) {
        f0[0]=0.;b0[R]=0.;
        for(std::int64_t r=0;r<R;++r) {
            double value=mass.active(r)?-positive_infinity:0.;
            if(mass.active(r))for(auto m=mass.lo(r);m<=mass.hi(r);++m)value=joint_add(value,mass(r,m),maximum);
            f1[r]=value;f0[r+1]=f0[r]+value;
        }
        for(std::int64_t r=R;r>0;--r)b0[r-1]=b0[r]+f1[r-1];
        for(std::int64_t r=0;r<R;++r)if(mass.active(r))for(auto m=mass.lo(r);m<=mass.hi(r);++m)
            cavities[offsets[r*offset_stride]+m-mass.lo(r)]=f0[r]+b0[r+1];
        return f0[R];
    }
    if(L>U&&L-U>1)return -positive_infinity;
    double total=-positive_infinity;
    for(auto h=L>1?L-1:1;h<=U;++h) {
        f0[0]=0.;f1[0]=-positive_infinity;
        for(std::int64_t r=0;r<R;++r) {
            if(!mass.active(r)){f0[r+1]=f0[r];f1[r+1]=f1[r];continue;}
            const double x=mass(r,h),y=h==9223372036854775807LL?-positive_infinity:mass(r,h+1);
            f0[r+1]=f0[r]+y;
            f1[r+1]=joint_add(joint_add(f1[r]+x,f1[r]+y,maximum),f0[r]+x,maximum);
        }
        total=joint_add(total,f1[R],maximum);b0[R]=-positive_infinity;b1[R]=0.;
        for(std::int64_t next=R;next>0;--next) {
            const auto r=next-1;
            if(!mass.active(r)){b0[r]=b0[next];b1[r]=b1[next];continue;}
            const double x=mass(r,h),y=h==9223372036854775807LL?-positive_infinity:mass(r,h+1);
            b0[r]=joint_add(x+b1[next],y+b0[next],maximum);
            b1[r]=joint_add(x+b1[next],y+b1[next],maximum);
        }
        for(std::int64_t r=0;r<R;++r)if(mass.active(r)) {
            if(h>=mass.lo(r)&&h<=mass.hi(r)) {
                const auto j=offsets[r*offset_stride]+h-mass.lo(r);
                cavities[j]=joint_add(cavities[j],joint_add(f0[r],f1[r],maximum)+b1[r+1],maximum);
            }
            if(h<9223372036854775807LL&&h+1>=mass.lo(r)&&h+1<=mass.hi(r)) {
                const auto j=offsets[r*offset_stride]+h+1-mass.lo(r);
                cavities[j]=joint_add(cavities[j],joint_add(f0[r]+b0[r+1],f1[r]+b1[r+1],maximum),maximum);
            }
        }
        if(h==9223372036854775807LL)break;
    }
    return total;
}
// dL is evaluated without dividing by L. Endpoint components of zero mass can
// consequently contribute a finite nonzero derivative to the mixture.
struct RawDerivative {double log_abs,sign,working;};
CLIPP_HD inline RawDerivative raw_derivative(const Observation& o,double phi,std::int64_t m) {
    const double slope=o.scaling*m,p=component_probability(o,phi,m),n=o.alt+o.ref;
    const double constant=o.log_constant+::log(static_cast<double>(o.support));
    if(slope*phi>1)return {-positive_infinity,0.,curvature_floor};
    if(p==0.) {
        if(o.alt==0)return {n*slope>0?::log(n*slope):-positive_infinity,-1.,n*slope*slope};
        if(o.alt==1)return {constant+::log(slope),1.,1.};
        return {-positive_infinity,0.,1.};
    }
    if(p==1.) {
        if(o.ref==0)return {n*slope>0?::log(n*slope):-positive_infinity,1.,n*slope*slope};
        if(o.ref==1)return {constant+::log(slope),-1.,1.};
        return {-positive_infinity,0.,1.};
    }
    const double score=slope*(o.alt/p-o.ref/(1-p));
    const double curvature=slope*slope*(o.alt/(p*p)+o.ref/((1-p)*(1-p)));
    return {score==0?-positive_infinity:raw_component_log_mass(o,phi,m)+::log(::fabs(score)),
            score>0?1.:score<0?-1.:0.,curvature};
}
CLIPP_HD inline EmitStatus joint_terms(const JointMass& mass,double& loss,double* gradient,double* curvature,
        std::int64_t out_stride=1,double* right=nullptr,double* left=nullptr,
        const std::int64_t* component_offsets=nullptr,double* cavities=nullptr,double* messages=nullptr) {
    const auto& data=mass.data;
    for(std::int64_t r=0;r<data.regions;++r) if(!finite_number(mass.phi[r*mass.stride])||mass.phi[r*mass.stride]<0||mass.phi[r*mass.stride]>1)
        {loss=positive_infinity;return EmitStatus::InvalidInput;}
    const bool profile=data.aggregation==LikelihoodAggregation::IntegerProfile;
    const double raw=component_offsets&&cavities&&messages?
        joint_cavities(mass,component_offsets,cavities,messages,profile):joint_reduce(mass,profile);
    if(raw==-positive_infinity){loss=positive_infinity;return EmitStatus::ZeroLikelihood;}
    loss=profile?-raw:joint_log_normalizer(mass)-raw;
    if(!finite_number(loss))return EmitStatus::NumericalFailure;
    if(!gradient)return EmitStatus::Ok;
    for(std::int64_t r=0;r<data.regions;++r) {
        gradient[r*out_stride]=0.;curvature[r*out_stride]=curvature_floor;
        if(right)right[r*out_stride]=0.;if(left)left[r*out_stride]=0.;
        if(!mass.active(r))continue;
        const auto o=observation(data,r*data.mutations+mass.row);
        double derivative=0.,working=0.,low=positive_infinity,high=-positive_infinity;
        auto cavity=mass;cavity.omit=r;
        for(auto m=mass.lo(r);m<=mass.hi(r);++m) {
            const auto term=raw_derivative(o,mass.phi[r*mass.stride],m);
            if(profile) {
                if(joint_reduce(mass,true,r,m)!=raw)continue;
                const double slope=o.scaling*m,phi=mass.phi[r*mass.stride],p=component_probability(o,phi,m);
                const double d=(o.alt==0?0.:-o.alt/phi)+(o.ref==0?0.:o.ref*slope/(1-p));
                const double directional_right=o.ref==0&&slope*phi==1?0.:d;
                if(directional_right<low)low=directional_right;if(d>high)high=d;
                if(working==0){derivative=d;working=term.working;}
            } else {
                const double c=cavities?cavities[component_offsets[r]+m-mass.lo(r)]:joint_reduce(cavity,false,r,m);
                if(c==-positive_infinity)continue;
                if(term.sign)derivative-=term.sign*::exp(c+term.log_abs-raw);
                const double component=mass(r,m);
                if(finite_number(component))working+=::exp(c+component-raw)*term.working;
            }
        }
        if(!finite_number(derivative))return EmitStatus::NumericalFailure;
        gradient[r*out_stride]=derivative;
        curvature[r*out_stride]=!finite_number(working)?1e12:(working<curvature_floor?curvature_floor:(working>1e12?1e12:working));
        if(right)right[r*out_stride]=profile?low:derivative;
        if(left)left[r*out_stride]=profile?high:derivative;
    }
    return EmitStatus::Ok;
}
}

namespace clipp2 {
// Ragged diagnostic buffers follow region-major cell offsets. Row fields are
// [marginal, profile, log-normalizer, independent-profile minus joint-profile].
CLIPP_HD inline EmitStatus joint_diagnostics_row(const JointMass& mass,const std::int64_t* offsets,
        double* row_values,double* maps,double* component_at_map,double* accepted_logp,double* map_logp,
        double* conditional,double* probabilities,double* messages,const double* accepted=nullptr) {
    const auto n=mass.data.mutations,rn=mass.data.regions,i=mass.row;
    const double z=joint_cavities(mass,offsets+i,probabilities,messages,false,n);
    const double best=joint_reduce(mass,true),normalizer=joint_log_normalizer(mass);
    row_values[0]=z-normalizer;row_values[1]=best;row_values[2]=normalizer;
    double independent=0.;
    for(std::int64_t r=0;r<rn;++r)if(mass.active(r)) {
        double local=-positive_infinity;
        for(auto m=mass.lo(r);m<=mass.hi(r);++m)if(mass(r,m)>local)local=mass(r,m);
        independent+=local;
    }
    row_values[3]=finite_number(best)?(independent>best?independent-best:0.):quiet_nan;
    joint_map(mass,maps+i,n);
    for(std::int64_t r=0;r<rn;++r) {
        const auto cell=r*n+i;
        component_at_map[cell]=accepted_logp[cell]=map_logp[cell]=quiet_nan;
        for(auto m=mass.lo(r);m<=mass.hi(r);++m) {
            const auto index=offsets[cell]+m-mass.lo(r);
            conditional[index]=mass.active(r)?joint_reduce(mass,true,r,m):quiet_nan;
            const double numerator=mass.active(r)?probabilities[index]+mass(r,m):quiet_nan;
            probabilities[index]=finite_number(z)&&mass.active(r)?::exp(numerator-z):quiet_nan;
            if(mass.active(r)&&finite_number(z)) {
                if(m==maps[cell]){component_at_map[cell]=mass(r,m);map_logp[cell]=numerator-z;}
                const double call=accepted?accepted[cell]:maps[cell];
                if(m==call)accepted_logp[cell]=numerator-z;
            }
        }
        if(mass.active(r)&&accepted&&finite_number(accepted[cell])&&!finite_number(accepted_logp[cell]))
            accepted_logp[cell]=-positive_infinity;
    }
    return z==-positive_infinity?EmitStatus::ZeroLikelihood:
        finite_number(z)&&finite_number(normalizer)?EmitStatus::Ok:EmitStatus::NumericalFailure;
}
CLIPP_HD inline void pooled_component(const Observation& o,std::int64_t m,const double* grid,
        const double* weights,std::int64_t count,double& logmass,double& moment) {
    double maximum=-positive_infinity;
    for(std::int64_t j=0;j<count;++j)if(weights[j]>0) {
        const double x=::log(weights[j])+raw_component_log_mass(o,grid[j],m);
        if(x>maximum)maximum=x;
    }
    if(!finite_number(maximum)){logmass=-positive_infinity;moment=quiet_nan;return;}
    double z=0.,v=0.;
    for(std::int64_t j=0;j<count;++j)if(weights[j]>0) {
        const double x=::exp(::log(weights[j])+raw_component_log_mass(o,grid[j],m)-maximum);
        z+=x;v+=x*grid[j];
    }
    logmass=maximum+::log(z);moment=v/z;
}
CLIPP_HD inline EmitStatus pooled_pilot_row(const JointMass& mass,const double* moments,
        double* pilot,double* cavities=nullptr,double* messages=nullptr) {
    const double z=cavities?joint_cavities(mass,mass.offsets+mass.row,cavities,messages,false,mass.data.mutations):joint_reduce(mass);
    for(std::int64_t r=0;r<mass.data.regions;++r) {
        const auto cell=r*mass.data.mutations+mass.row;
        pilot[cell]=quiet_nan;
        if(!mass.active(r)||!finite_number(z))continue;
        double mean=0.;
        for(auto m=mass.lo(r);m<=mass.hi(r);++m) {
            const double score=cavities?cavities[mass.offsets[cell]+m-mass.lo(r)]+mass(r,m):joint_reduce(mass,false,r,m);
            if(!finite_number(score))continue;
            mean+=::exp(score-z)*moments[mass.offsets[cell]+m-mass.lo(r)];
        }
        pilot[cell]=mean;
    }
    return z==-positive_infinity?EmitStatus::ZeroLikelihood:finite_number(z)?EmitStatus::Ok:EmitStatus::NumericalFailure;
}
}

namespace clipp2 {
CLIPP_HD inline EmitStatus joint_column_score(const JointMass& mass,double& score) {
    for(std::int64_t r=0;r<mass.data.regions;++r) {
        const double phi=mass.phi[r*mass.stride];const auto cell=r*mass.data.mutations+mass.row;
        if(!finite_number(phi)){score=-positive_infinity;return EmitStatus::InvalidInput;}
        if(phi<mass.data.lower[cell]||phi>mass.data.upper[cell]){score=-positive_infinity;return EmitStatus::ZeroLikelihood;}
    }
    double loss;const auto status=joint_terms(mass,loss,nullptr,nullptr);score=-loss;return status;
}
}
