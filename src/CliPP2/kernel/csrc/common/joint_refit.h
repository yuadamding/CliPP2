#pragma once
#include "joint_multiplicity.h"

namespace clipp2 {
struct JointGroup {
    const ObservationSoA& data;
    const std::int64_t* rows;
    std::int64_t count;
    const double *lower,*upper;
    const std::int64_t* offsets; // group-major [member, region] ragged components
    CLIPP_HD double objective(const double* center)const {
        double value=0.;
        const bool profile=data.aggregation==LikelihoodAggregation::IntegerProfile;
        for(std::int64_t i=0;i<count;++i)value+=joint_score(JointMass{data,rows[i],center,1},profile);
        return value;
    }
    CLIPP_HD void posterior(const double* center,double* weights,double* messages,bool maximum=false)const {
        for(std::int64_t i=0;i<count;++i) {
            JointMass mass{data,rows[i],center,1};
            const double z=joint_cavities(mass,offsets+i*data.regions,weights,messages,maximum);
            for(std::int64_t r=0;r<data.regions;++r)for(auto m=mass.lo(r);m<=mass.hi(r);++m) {
                const auto target=offsets[i*data.regions+r]+m-mass.lo(r);
                if(!mass.active(r))weights[target]=0.;
                else if(!maximum)weights[target]=::exp(weights[target]+mass(r,m)-z);
            }
        }
    }
    CLIPP_HD double expected_derivative(std::int64_t r,double x,const double* weights)const {
        double value=0.;
        for(std::int64_t i=0;i<count;++i) {
            const auto o=observation(data,r*data.mutations+rows[i]);
            if(!(o.flags&observed_flag))continue;
            for(std::int64_t j=0;j<o.support;++j) {
                const double w=weights[offsets[i*data.regions+r]+j];if(w==0)continue;
                const double slope=o.scaling*(o.minimum+j),p=slope*x;
                const double positive=o.alt==0?0.:(x==0?positive_infinity:o.alt/x);
                const double negative=o.ref==0?0.:(p>=1?positive_infinity:o.ref*slope/(1-p));
                value+=w*(positive-negative);
            }
        }
        return value;
    }
    CLIPP_HD void gradient(const double* center,double* output,double* working,double* temp,double* cavities,double* messages)const {
        for(std::int64_t r=0;r<data.regions;++r)output[r]=working[r]=0.;
        for(std::int64_t i=0;i<count;++i) {
            JointMass mass{data,rows[i],center,1};double loss;
            joint_terms(mass,loss,temp,temp+data.regions,1,nullptr,nullptr,offsets+i*data.regions,cavities,messages);
            for(std::int64_t r=0;r<data.regions;++r){output[r]+=temp[r];working[r]+=temp[data.regions+r];}
        }
    }
    CLIPP_HD double profile_projected(const double* center,double* right,double* left,double* temp)const {
        for(std::int64_t r=0;r<data.regions;++r)right[r]=left[r]=0.;
        for(std::int64_t i=0;i<count;++i) {
            JointMass mass{data,rows[i],center,1};double loss;
            joint_terms(mass,loss,temp,temp+data.regions,1,temp,temp+data.regions);
            for(std::int64_t r=0;r<data.regions;++r){right[r]+=temp[r];left[r]+=temp[data.regions+r];}
        }
        double residual=0.;
        for(std::int64_t r=0;r<data.regions;++r) {
            if(center[r]<upper[r]&&-right[r]>residual)residual=-right[r];
            if(center[r]>lower[r]&&left[r]>residual)residual=left[r];
        }
        return residual/(1.+count);
    }
    CLIPP_HD double projected(const double* center,const double* gradient)const {
        double result=0.;
        for(std::int64_t r=0;r<data.regions;++r) {
            double g=gradient[r];if(center[r]<=lower[r]&&g>=0)g=0.;if(center[r]>=upper[r]&&g<=0)g=0.;
            const double residual=::fabs(g)/(1.+count);if(residual>result)result=residual;
        }
        return result;
    }
};
CLIPP_HD inline double joint_clip(double x,double lo,double hi){return x<lo?lo:x>hi?hi:x;}
// Exact conditional profile envelope. Each observation's winning multiplicity
// decreases as phi increases; only consecutive envelope branches intersect.
CLIPP_HD inline double profile_cross(const Observation& o,std::int64_t q,std::int64_t m,double gq,double gm) {
    const double numerator=o.alt*::log(static_cast<double>(q)/m)+gq-gm;
    if(o.ref==0)return numerator>=0?positive_infinity:0.;
    const double z=numerator/o.ref;
    if(z<=0)return 0.;
    const double t=-::expm1(-z);
    return t/(o.scaling*((q-m)+m*t));
}
CLIPP_HD inline double profile_coordinate_value(const JointGroup& group,std::int64_t r,double x,const double* offsets) {
    double sum=0.;
    for(std::int64_t i=0;i<group.count;++i) {
        const auto o=observation(group.data,r*group.data.mutations+group.rows[i]);
        if(!(o.flags&observed_flag))continue;
        double best=-positive_infinity;
        for(std::int64_t j=0;j<o.support;++j) {
            const double v=raw_component_log_mass(o,x,o.minimum+j)+offsets[group.offsets[i*group.data.regions+r]+j];
            if(v>best)best=v;
        }
        sum+=best;
    }
    return sum;
}
CLIPP_HD inline double profile_coordinate_derivative(const JointGroup& group,std::int64_t r,double x,
        const double* offsets,double branch_x) {
    double derivative=0.;
    for(std::int64_t i=0;i<group.count;++i) {
        const auto o=observation(group.data,r*group.data.mutations+group.rows[i]);
        if(!(o.flags&observed_flag))continue;
        double best=-positive_infinity;std::int64_t chosen=o.minimum;
        for(std::int64_t j=0;j<o.support;++j) {
            const double v=raw_component_log_mass(o,branch_x,o.minimum+j)+offsets[group.offsets[i*group.data.regions+r]+j];
            if(v>best){best=v;chosen=o.minimum+j;}
        }
        const double slope=o.scaling*chosen;
        derivative+=(o.alt==0?0.:(x==0?positive_infinity:o.alt/x))-
            (o.ref==0?0.:(slope*x>=1?positive_infinity:o.ref*slope/(1-slope*x)));
    }
    return derivative;
}
CLIPP_HD inline double optimize_profile_coordinate(const JointGroup& group,std::int64_t r,const double* offsets,
        double initial,double* breaks,double* starts,std::int64_t* branches) {
    const double lo=group.lower[r],hi=group.upper[r];std::int64_t count=2;breaks[0]=lo;breaks[1]=hi;
    for(std::int64_t i=0;i<group.count;++i) {
        const auto o=observation(group.data,r*group.data.mutations+group.rows[i]);
        if(!(o.flags&observed_flag))continue;
        const auto first=group.offsets[i*group.data.regions+r];std::int64_t size=0;
        for(std::int64_t j=o.support;j>0;--j) {
            const auto m=o.minimum+j-1;const double gm=offsets[first+j-1];if(!finite_number(gm))continue;
            double crossing=lo;
            while(size) {
                const auto q=branches[size-1];crossing=profile_cross(o,q,m,offsets[first+q-o.minimum],gm);
                if(crossing>starts[size-1])break;
                --size;
            }
            if(size&&crossing>=hi)continue;
            branches[size]=m;starts[size]=size?crossing:lo;++size;
        }
        for(std::int64_t j=1;j<size;++j)if(starts[j]>lo&&starts[j]<hi)breaks[count++]=starts[j];
    }
    // Insertion sorting is bounded by the sum of regional support widths.
    for(std::int64_t i=1;i<count;++i){const double v=breaks[i];auto j=i;while(j&&breaks[j-1]>v){breaks[j]=breaks[j-1];--j;}breaks[j]=v;}
    double best=profile_coordinate_value(group,r,initial,offsets),selected=initial;
    for(std::int64_t j=0;j<count;++j) {
        const double v=profile_coordinate_value(group,r,breaks[j],offsets);
        if(v>best||(v==best&&breaks[j]<selected)){best=v;selected=breaks[j];}
        if(j+1==count||breaks[j+1]<=breaks[j])continue;
        double a=breaks[j],b=breaks[j+1];const double midpoint=(a+b)*.5;
        for(int step=0;step<60;++step) {
            const double x=(a+b)*.5;if(x==a||x==b)break;
            const double derivative=profile_coordinate_derivative(group,r,x,offsets,midpoint);
            if(derivative>0)a=x;else b=x;
        }
        const double x=(a+b)*.5,value=profile_coordinate_value(group,r,x,offsets);
        if(value>best||(value==best&&x<selected)){best=value;selected=x;}
    }
    return selected;
}
// Scratch: posterior/conditional arrays (components), breaks (components+2),
// envelope starts (max_support), plus 12R+4 doubles; integer envelope max_support.
CLIPP_HD inline void refit_joint_group(const JointGroup& group,const double* initial,
        double* output,double& result,std::int32_t& status,std::int64_t& iterations,
        double& projected,std::uint8_t& retained,double* scratch,std::int64_t* branches,
        std::int64_t components,std::int64_t max_support) {
    const auto R=group.data.regions;
    double* weights=scratch;double* breaks=weights+components;double* starts=breaks+components+2;
    double* current=starts+max_support;double* proposal=current+R;double* gradient=proposal+R;
    double* scale=gradient+R;double* temp=scale+R;double* incumbent=temp+2*R;double* messages=incumbent+R;
    const bool profile=group.data.aggregation==LikelihoodAggregation::IntegerProfile;
    for(std::int64_t r=0;r<R;++r)incumbent[r]=finite_number(initial[r])?joint_clip(initial[r],group.lower[r],group.upper[r]):(group.lower[r]+group.upper[r])*.5;
    const double original=group.objective(incumbent);result=original;retained=1;iterations=0;status=2;projected=positive_infinity;
    for(std::int64_t r=0;r<R;++r)output[r]=incumbent[r];
    // A supplied stationary point remains a valid incumbent even when a root
    // rounded to the adjacent float has a slightly smaller computed objective.
    if(finite_number(original)) {
        if(profile)projected=group.profile_projected(incumbent,gradient,scale,temp);
        else {group.gradient(incumbent,gradient,scale,temp,weights,messages);projected=group.projected(incumbent,gradient);}
        if(projected<=1e-5)status=0;
    }
    // Endpoint starts can be false EM fixed points. A second genuinely interior
    // start is always tried, preserving the supplied finite center as incumbent.
    for(int start=0;start<2;++start) {
        for(std::int64_t r=0;r<R;++r)current[r]=start?(incumbent[r]+(group.lower[r]+group.upper[r])*.5)*.5:incumbent[r];
        double value=group.objective(current);if(!finite_number(value))continue;
        double residual=positive_infinity;
        for(int iteration=0;iteration<2000;++iteration) {
            ++iterations;
            if(profile) {
                for(std::int64_t r=0;r<R;++r) {
                    group.posterior(current,weights,messages,true);
                    current[r]=optimize_profile_coordinate(group,r,weights,current[r],breaks,starts,branches);
                }
                const double next=group.objective(current);
                if(next<value-1e-9*(1+::fabs(value))){status=3;return;}
                const double gain=next-value;value=next;
                if(gain<=1e-10*(1+::fabs(value))) {
                    residual=group.profile_projected(current,gradient,scale,temp);
                    if(residual<=1e-7)break;
                }
            } else {
                group.posterior(current,weights,messages,false);
                for(std::int64_t r=0;r<R;++r) {
                    double a=group.lower[r],b=group.upper[r];
                    if(group.expected_derivative(r,a,weights)<=0)proposal[r]=a;
                    else if(group.expected_derivative(r,b,weights)>=0)proposal[r]=b;
                    else {for(int step=0;step<60;++step){const double x=(a+b)*.5;if(x==a||x==b)break;
                        if(group.expected_derivative(r,x,weights)>0)a=x;else b=x;}proposal[r]=(a+b)*.5;}
                }
                const double next=group.objective(proposal);
                if(next>=value-1e-10*(1+::fabs(value))) {
                    for(std::int64_t r=0;r<R;++r)current[r]=proposal[r];value=next;
                }
                group.gradient(current,gradient,scale,temp,weights,messages);residual=group.projected(current,gradient);
                if(residual<=1e-7)break;
                // Exact feasible gradient rescue, including finite derivatives
                // from components with zero posterior at an endpoint.
                double step=1.;bool improved=false;
                for(int attempt=0;attempt<45;++attempt) {
                    for(std::int64_t r=0;r<R;++r)proposal[r]=joint_clip(current[r]-step*gradient[r]/(scale[r]>1?scale[r]:1),group.lower[r],group.upper[r]);
                    const double trial=group.objective(proposal);
                    if(trial>value){for(std::int64_t r=0;r<R;++r)current[r]=proposal[r];value=trial;improved=true;break;}
                    step*=.5;
                }
                if(!improved&&residual<1e-5)break;
            }
        }
        const bool converged=residual<=1e-5;
        if(converged&&(!finite_number(original)||value>=original)&&(value>result||!finite_number(result)||status!=0)) {
            result=value;projected=residual;status=0;
            for(std::int64_t r=0;r<R;++r)output[r]=current[r];
        }
    }
    retained=result<=original; // the incumbent is never discarded for a worse fit
    if(finite_number(original)&&result<original){result=original;status=2;projected=positive_infinity;for(std::int64_t r=0;r<R;++r)output[r]=incumbent[r];}
}
}
