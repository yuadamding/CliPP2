// CliPP2-owned chain kernel, derived from CliPP1.5 9e51e30ac8b0c8c9ab1f46821b496909843bda0c.
// AGPL-3.0; original attribution and local changes are recorded in the project NOTICE.
#include "chain.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace clipp2kernel {

ChainData prepare_chain_inputs(int count, const int* alt, const int* depth,
    const int* major, const int* total, double purity, const double* pilot_cp,
    const int* requested_k, int k_count)
{
    if(count <= 0 || !alt || !depth || !major || !total || !pilot_cp || !requested_k)
        throw std::invalid_argument("Nonempty chain inputs are required.");
    if(!std::isfinite(purity) || purity <= 0.0 || purity > 1.0)
        throw std::invalid_argument("Purity must be finite and in (0,1].");
    if(k_count < 1 || k_count > std::min(kMaxClusters,count))
        throw std::invalid_argument("Require 1..min(10,N) requested K values.");
    std::vector<int> seen;
    for(int j=0;j<k_count;++j){
        if(requested_k[j]<1 || requested_k[j]>std::min(kMaxClusters,count) ||
            std::find(seen.begin(),seen.end(),requested_k[j])!=seen.end())
            throw std::invalid_argument("Requested K must be distinct and in 1..min(10,N).");
        seen.push_back(requested_k[j]);
    }
    ChainData data;
    data.count=count; data.purity=purity;
    data.alt.assign(alt,alt+count); data.depth.assign(depth,depth+count);
    data.major.assign(major,major+count);
    data.scale.resize(count); data.log_choose.resize(count); data.pilot_x.resize(count);
    for(int i=0;i<count;++i){
        if(alt[i]<0 || depth[i]<=0 || alt[i]>depth[i] || major[i]<1 || total[i]<major[i])
            throw std::invalid_argument("Invalid binomial counts or full-major multiplicity support.");
        if(!std::isfinite(pilot_cp[i]) || pilot_cp[i]<0.0 || pilot_cp[i]>purity*(1.0+1e-12))
            throw std::invalid_argument("Pilot CP must be finite and in [0,purity].");
        data.scale[i]=purity/(2.0*(1.0-purity)+purity*total[i]);
        if(!(data.scale[i]>0.0)) throw std::invalid_argument("Purity is below representable likelihood scaling.");
        data.log_choose[i]=std::lgamma(double(depth[i])+1.0)-std::lgamma(double(alt[i])+1.0)-std::lgamma(double(depth[i]-alt[i])+1.0);
        data.pilot_x[i]=std::max(kChainLowerCCF,std::min(kChainUpperCCF,pilot_cp[i]/purity));
    }
    return data;
}

std::vector<double> project_chain_jumps(const std::vector<double>& x,int k)
{
    std::vector<double> projected(x.size()>0?x.size()-1:0,0.0);
    std::vector<std::pair<double,int>> best;
    for(int i=0;i+1<int(x.size());++i){
        const double magnitude=std::fabs(x[i+1]-x[i]);
        if(magnitude==0.0 || k<=1) continue;
        auto item=std::make_pair(magnitude,i);
        auto position=std::find_if(best.begin(),best.end(),[&](const auto& other){
            return magnitude>other.first || (magnitude==other.first && i<other.second);
        });
        best.insert(position,item);
        if(int(best.size())>k-1) best.pop_back();
    }
    for(const auto& item:best) projected[item.second]=x[item.second+1]-x[item.second];
    return projected;
}

std::vector<int> chain_labels(const std::vector<double>& projected)
{
    std::vector<int> labels(projected.size()+1,0);
    for(size_t i=0;i<projected.size();++i) labels[i+1]=labels[i]+(projected[i]!=0.0);
    return labels;
}

double chain_support_penalty(const std::vector<double>& x,
    const std::vector<double>& projected,double rho)
{
    double value=0.0;
    for(size_t i=0;i<projected.size();++i){
        if(projected[i]!=0.0) continue;
        const double difference=x[i+1]-x[i];
        value+=difference*difference;
    }
    return 0.5*rho*value;
}

std::vector<double> boxed_chain_laplacian_quadratic(const std::vector<double>& curvature,
    const std::vector<double>& rhs,const std::vector<double>& projected,
    double rho,bool* solved,double lower,double upper)
{
    const int n=int(curvature.size());
    std::vector<double> x(n,lower);
    std::vector<int> active(n,0);
    std::vector<long double> effective(n),forward(n);
    if(solved) *solved=false;
    const auto edge=[&](int i){return i>=0 && i<n-1 && projected[i]==0.0;};
    // A usual Thomas pivot subtracts two O(rho) values to recover an O(h)
    // likelihood curvature. Instead eliminate a spring in series with the
    // previous positive curvature: rho*h/(rho+h). No large subtraction is
    // needed, including at the last node of a long, nearly fused block.
    for(long long sweep=0;sweep<2LL*n+64;++sweep){
        for(int begin=0;begin<n;){
            if(active[begin]){++begin;continue;}
            int end=begin+1;
            while(end<n && !active[end] && edge(end-1)) ++end;
            for(int i=begin;i<end;++i){
                long double h=curvature[i],b=rhs[i];
                if(i==begin && edge(i-1)){h+=rho;b+=static_cast<long double>(rho)*x[i-1];}
                if(i==end-1 && edge(i)){h+=rho;b+=static_cast<long double>(rho)*x[i+1];}
                if(i>begin){
                    const long double ratio=static_cast<long double>(rho)/(rho+effective[i-1]);
                    h+=ratio*effective[i-1];b+=ratio*forward[i-1];
                }
                if(!(h>0.0L) || !std::isfinite(h))
                    throw std::runtime_error("Nonpositive effective chain curvature.");
                effective[i]=h;forward[i]=b;
            }
            x[end-1]=double(forward[end-1]/effective[end-1]);
            for(int i=end-2;i>=begin;--i)
                x[i]=double(static_cast<long double>(x[i+1])+
                    (forward[i]-effective[i]*x[i+1])/(rho+effective[i]));
            begin=end;
        }
        bool bounded=false;
        for(int i=0;i<n;++i){
            if(!active[i] && x[i]<lower){x[i]=lower;active[i]=-1;bounded=true;}
            else if(!active[i] && x[i]>upper){x[i]=upper;active[i]=1;bounded=true;}
        }
        if(bounded) continue;
        int release=-1;long double violation=1e-9L;
        for(int i=0;i<n;++i){
            long double g=static_cast<long double>(curvature[i])*x[i]-rhs[i];
            if(edge(i-1)) g+=static_cast<long double>(rho)*(x[i]-x[i-1]);
            if(edge(i)) g+=static_cast<long double>(rho)*(x[i]-x[i+1]);
            const long double wrong=active[i]<0?-g:(active[i]>0?g:0.0L);
            if(wrong>violation){violation=wrong;release=i;}
        }
        if(release<0){if(solved)*solved=true;return x;}
        active[release]=0;
    }
    for(double& value:x) value=std::max(lower,std::min(upper,value));
    return x;
}

namespace {
double sum(const std::vector<double>& values){return std::accumulate(values.begin(),values.end(),0.0);}
std::vector<double> augmented_gradient(const std::vector<double>& x,const std::vector<double>& z,
    const std::vector<double>& gradient,double rho){
    std::vector<double> g=gradient;for(size_t i=0;i<z.size();++i){double v=rho*(x[i+1]-x[i]-z[i]);g[i]-=v;g[i+1]+=v;}return g;
}
double stationarity(const std::vector<double>& x,const std::vector<double>& g){
    double residual=0.0;for(size_t i=0;i<x.size();++i)residual=std::max(residual,std::fabs(x[i]-std::max(kChainLowerCCF,std::min(kChainUpperCCF,x[i]-g[i]))));return residual;
}

struct BlockScale {int longest;double range;double likelihood_curvature;};
BlockScale block_scale(const std::vector<double>& x,const std::vector<double>& z,
    const std::vector<double>& curvature){
    BlockScale result{1,0.0,0.0};
    for(int begin=0;begin<int(x.size());){
        int end=begin+1;
        while(end<int(x.size()) && z[end-1]==0.0) ++end;
        const auto range=std::minmax_element(x.begin()+begin,x.begin()+end);
        result.range=std::max(result.range,*range.second-*range.first);
        if(end-begin>=result.longest){
            result.longest=end-begin;
            result.likelihood_curvature=std::accumulate(curvature.begin()+begin,curvature.begin()+end,0.0)/(end-begin);
        }
        begin=end;
    }
    return result;
}

// This is only a warm start on the same chain support, not a replacement
// likelihood, a new partition, or a constrained/global-optimality certificate.
// The caller accepts it only if it decreases the actual next-rho objective.
void feasible_support_warm_start(const std::vector<double>& raw,const std::vector<int>& labels,
    const ChainEvaluator& evaluate,std::vector<double>& x,std::vector<double>& values,
    std::vector<double>& gradient,std::vector<double>& curvature){
    const int n=int(raw.size()),q=labels.back()+1;
    std::vector<double> centers(q,0.0),sizes(q,0.0);
    for(int i=0;i<n;++i){centers[labels[i]]+=raw[i];sizes[labels[i]]+=1.0;}
    for(int j=0;j<q;++j) centers[j]=std::max(kChainLowerCCF,std::min(kChainUpperCCF,centers[j]/sizes[j]));
    x.resize(n);
    for(int i=0;i<n;++i) x[i]=centers[labels[i]];
    evaluate(x,values,gradient,curvature);
    for(int iteration=0;iteration<50;++iteration){
        std::vector<double> g(q,0.0),h(q,0.0),direction(q);
        for(int i=0;i<n;++i){g[labels[i]]+=gradient[i];h[labels[i]]+=curvature[i];}
        double residual=0.0,slope=0.0;
        for(int j=0;j<q;++j){
            residual=std::max(residual,std::fabs(centers[j]-std::max(kChainLowerCCF,std::min(kChainUpperCCF,centers[j]-g[j]/sizes[j]))));
            direction[j]=std::max(kChainLowerCCF,std::min(kChainUpperCCF,centers[j]-g[j]/h[j]))-centers[j];
            slope+=g[j]*direction[j];
        }
        if(residual<=1e-8 || !(slope<0.0)) break;
        std::vector<double> trial(n),v(n),grad(n),hessian(n);
        const double previous=sum(values);
        bool accepted=false;double step=1.0;
        for(int backtrack=0;backtrack<40;++backtrack){
            for(int i=0;i<n;++i) trial[i]=centers[labels[i]]+step*direction[labels[i]];
            evaluate(trial,v,grad,hessian);
            if(std::isfinite(sum(v)) && sum(v)<=previous+1e-4*step*slope){
                for(int j=0;j<q;++j) centers[j]+=step*direction[j];
                x.swap(trial);values.swap(v);gradient.swap(grad);curvature.swap(hessian);
                accepted=true;break;
            }
            step*=0.5;
        }
        if(!accepted) break;
    }
}
}

void solve_chain(const ChainData& data,const int* requested_k,int k_count,
    const ChainEvaluator& evaluate,int* labels_out,double* cp_out,
    double* diagnostics_out,int* statuses_out)
{
    constexpr int levels=kChainLevels, iterations_per_level=kChainIterationsPerLevel;
    constexpr double stationarity_tolerance=kChainStationarityTolerance, constraint_tolerance=kChainConstraintTolerance;
    // An engineering guard, not an assertion that this finite penalty attains
    // the constrained solution. Higher rho makes rho*diff(x) unreliable in
    // double precision even with a stable tridiagonal factorization.
    constexpr double maximum_rho=kChainMaximumRho;
    const int n=data.count;
    for(int request=0;request<k_count;++request){
        const int k=requested_k[request];
        std::vector<double> x=data.pilot_x,values(n),gradient(n),curvature(n);
        evaluate(x,values,gradient,curvature);
        if(!std::isfinite(sum(values)))throw std::runtime_error("Initial chain likelihood is nonfinite.");
        double rho=1.0,stat=INFINITY,violation=INFINITY;
        int total_iterations=0,level_used=0,qp_budget_hits=0,warm_starts=0;
        bool stalled=false,numerical_limit=false;
        std::vector<int> warm_labels;
        std::vector<double> warm_x,warm_values,warm_gradient,warm_curvature;
        for(int level=0;level<levels;++level){
            level_used=level+1;stalled=false;
            for(int iteration=0;iteration<iterations_per_level;++iteration){
                ++total_iterations;
                const auto z=project_chain_jumps(x,k);
                const auto g=augmented_gradient(x,z,gradient,rho);
                stat=stationarity(x,g);
                if(stat<=stationarity_tolerance) break;
                std::vector<double> diagonal(n),rhs(n);
                for(int i=0;i<n;++i){
                    const int degree=(i>0 && z[i-1]==0.0)+(i+1<n && z[i]==0.0);
                    diagonal[i]=curvature[i]+rho*degree;
                    rhs[i]=curvature[i]*x[i]-gradient[i];
                }
                bool qp_solved=false;
                // Holding sparse values fixed adds artificial rho stiffness at
                // retained jumps. Holding only their support instead minimizes
                // those values analytically and gives a tighter MM upper bound.
                const auto qp=boxed_chain_laplacian_quadratic(curvature,rhs,z,rho,
                    &qp_solved,kChainLowerCCF,kChainUpperCCF);
                if(!qp_solved) ++qp_budget_hits;
                const double old_objective=sum(values)+chain_support_penalty(x,z,rho);
                bool accepted=false;
                std::vector<double> trial(n),trial_values(n),trial_gradient(n),trial_curvature(n);
                for(int fallback=0;fallback<2 && !accepted;++fallback){
                    std::vector<double> direction(n);double slope=0.0;
                    for(int i=0;i<n;++i){
                        const double candidate=fallback?std::max(kChainLowerCCF,std::min(kChainUpperCCF,x[i]-g[i]/diagonal[i])):qp[i];
                        direction[i]=candidate-x[i];slope+=g[i]*direction[i];
                    }
                    if(slope>=0.0) continue;
                    double step=1.0;
                    for(int backtrack=0;backtrack<40;++backtrack){
                        for(int i=0;i<n;++i) trial[i]=std::max(kChainLowerCCF,std::min(kChainUpperCCF,x[i]+step*direction[i]));
                        evaluate(trial,trial_values,trial_gradient,trial_curvature);
                        // This support-conditional objective bounds the exact
                        // distance-to-set objective above, with equality at x.
                        // Armijo descent therefore also decreases that objective,
                        // including when projection support changes or ties.
                        const double value=sum(trial_values)+chain_support_penalty(trial,z,rho);
                        if(std::isfinite(value) && value<=old_objective+1e-4*step*slope+1e-12*std::max(1.0,std::fabs(old_objective))){
                            double displacement=0.0;
                            for(int i=0;i<n;++i) displacement=std::max(displacement,std::fabs(trial[i]-x[i]));
                            x.swap(trial);values.swap(trial_values);gradient.swap(trial_gradient);curvature.swap(trial_curvature);accepted=true;
                            if(displacement<=8.0*std::numeric_limits<double>::epsilon()) stalled=true;
                            break;
                        }
                        step*=0.5;
                    }
                }
                if(!accepted){stalled=true;break;}
                if(stalled) break;
            }
            const auto z=project_chain_jumps(x,k);
            stat=stationarity(x,augmented_gradient(x,z,gradient,rho));
            violation=std::sqrt(2.0*chain_support_penalty(x,z,1.0));
            const auto scale=block_scale(x,z,curvature);
            if(stat<=stationarity_tolerance && violation<=constraint_tolerance && scale.range<=constraint_tolerance) break;
            const double roundoff_scale=8.0*rho*std::numeric_limits<double>::epsilon();
            if(rho>=maximum_rho || (stalled && violation<=constraint_tolerance &&
                scale.range<=constraint_tolerance && roundoff_scale>=stationarity_tolerance)){
                numerical_limit=true;break;
            }
            if(level+1<levels){
                // The slowest nonconstant mode of an L-node path has
                // eigenvalue 4*sin(pi/(2L))^2. Account for that mode as well as
                // achieved feasibility; a fixed rho cap is length dependent.
                const double gap=4.0*std::pow(std::sin(std::acos(-1.0)/(2.0*scale.longest)),2);
                const double spectral_rho=scale.likelihood_curvature/gap;
                const double residual_ratio=std::max(violation,scale.range)/constraint_tolerance;
                const double factor=std::max(2.0,std::min(10.0,std::max(spectral_rho/rho,residual_ratio)));
                const double next_rho=std::min(maximum_rho,rho*factor);
                const auto labels=chain_labels(z);
                if(labels!=warm_labels){
                    feasible_support_warm_start(x,labels,evaluate,warm_x,warm_values,warm_gradient,warm_curvature);
                    warm_labels=labels;
                }
                const double old_objective=sum(values)+chain_support_penalty(x,z,next_rho);
                if(std::isfinite(sum(warm_values)) && sum(warm_values)<old_objective){
                    x=warm_x;values=warm_values;gradient=warm_gradient;curvature=warm_curvature;++warm_starts;
                }
                rho=next_rho;
            }
        }
        // CP is the public native parameterization. At large rho, even the
        // one-ulp CP/CCF round trip can change the raw gradient appreciably;
        // compute diagnostics from exactly the values a reader reconstructs.
        std::vector<double> published_cp(n);
        for(int i=0;i<n;++i){published_cp[i]=data.purity*x[i];x[i]=published_cp[i]/data.purity;}
        evaluate(x,values,gradient,curvature);
        const auto z=project_chain_jumps(x,k);
        stat=stationarity(x,augmented_gradient(x,z,gradient,rho));
        violation=std::sqrt(2.0*chain_support_penalty(x,z,1.0));
        const auto labels=chain_labels(z);
        const int actual=labels.back()+1;
        const double objective=sum(values)+chain_support_penalty(x,z,rho);
        const auto scale=block_scale(x,z,curvature);
        const bool tolerance_met=stat<=stationarity_tolerance && violation<=constraint_tolerance && scale.range<=constraint_tolerance;
        if(!tolerance_met && violation<=constraint_tolerance && scale.range<=constraint_tolerance &&
            8.0*rho*std::numeric_limits<double>::epsilon()>=stationarity_tolerance) numerical_limit=true;
        statuses_out[request]=tolerance_met?0:(numerical_limit?1:(stalled?2:3));
        const auto offset=static_cast<std::size_t>(request)*n;
        std::copy(labels.begin(),labels.end(),labels_out+offset);
        std::copy(published_cp.begin(),published_cp.end(),cp_out+offset);
        const double diagnostics[kDiagnosticCount]={
            double(k),double(actual),rho,rho/data.purity/data.purity,
            std::log(rho)-2.0*std::log(data.purity),objective,sum(values),
            violation,data.purity*violation,stat,double(total_iterations),
            double(level_used),double(qp_budget_hits),kChainLowerCCF,kChainUpperCCF,
            scale.range,double(scale.longest),double(warm_starts),
            8.0*rho*std::numeric_limits<double>::epsilon()
        };
        std::copy(diagnostics,diagnostics+kDiagnosticCount,
                  diagnostics_out+static_cast<std::size_t>(request)*kDiagnosticCount);
    }
}

} // namespace clipp2kernel
