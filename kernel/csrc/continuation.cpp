// Ordered CPU continuation. This is the same finite-penalty search policy as
// solver.py, not a new objective, convergence gate or candidate-search policy.
#include "continuation.h"
#include "forest.h"
#include "prepared_tree.h"
#include "executor.h"
#include "common/emission_math.h"
#include "common/scalar_search.h"
#include "common/validation.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Vector = std::vector<double>;
thread_local char error_message[1024] = "";
thread_local std::int64_t evaluations = 0;
constexpr double epsilon = std::numeric_limits<double>::epsilon();
double clip(double x, double lo, double hi) { return std::min(std::max(x, lo), hi); }
double sum(const Vector& x) {
    return clipp2::pairwise_sum([&](std::int64_t i) { return x[i]; }, 0, x.size());
}
struct Terms { Vector value, gradient, curvature; };
struct Point { Vector x; Terms terms; };
void resize(Point& point,std::size_t cells) {
    point.x.resize(cells);point.terms.value.resize(cells);
    point.terms.gradient.resize(cells);point.terms.curvature.resize(cells);
}

// Every capacity reserves its maximum shape before entering continuation.
// Main-step and warm-start work are disjoint lifetimes and reuse these arrays.
struct Arena {
    unsigned long coordinate_epoch=0,projection_epoch=~0UL;
    Point current,trial,warm;
    Vector gradient,delta,spring_delta,trial_delta,h,rhs,qp,diagonal,direction,products;
    Vector minima,maxima,means,lo,hi,center,next,group_gradient,group_h,group_direction,group_products;
    std::vector<long double> scores,cut_delta,audit_gradient,audit_scale,audit_delta;
    std::vector<int> selected,graph,parent,labels,mapping,sizes,degree,warm_labels,qp_diagnostics;
    std::vector<unsigned char> cuts;
    explicit Arena(int n,int r):
        scores(n-1),cut_delta(static_cast<std::size_t>(n-1)*r),
        audit_gradient(static_cast<std::size_t>(n)*r),audit_scale(audit_gradient.size()),
        audit_delta(cut_delta.size()),parent(n),labels(n),mapping(n),sizes(n),degree(n),
        qp_diagnostics(2*r),cuts(n-1) {
        const auto cells=static_cast<std::size_t>(n)*r;
        resize(current,cells);resize(trial,cells);resize(warm,cells);
        for(auto* value:{&gradient,&delta,&spring_delta,&trial_delta,&h,&rhs,&qp,&diagonal,&direction,&products,
                        &minima,&maxima,&means,&lo,&hi,&center,&next,&group_gradient,&group_h,&group_direction,&group_products})
            value->resize(cells);
        selected.reserve(n-1);graph.reserve(2*static_cast<std::size_t>(n-1));warm_labels.reserve(n);
    }
};

struct Engine {
    const clipp2::ObservationSoA& data;
    int n, r, e;
    std::size_t cells;
    const int *edges, *canonical;
    const double *lower, *upper, *gaps;
    const CliPP2ContinuationPolicy& policy;
    CliPP2Norm norm;
    CliPP2CutScores cut_scores;
    const clipp2kernel::PreparedTree& tree;
    clipp2kernel::Executor* team;
    Arena& arena;

    void evaluate(const Vector& x,Terms& out) const {
        ++evaluations;
        auto cell_terms=[&](int i) {for(int region=0;region<r;++region) {
            const auto cell=static_cast<std::size_t>(i)*r+region;
            const auto observation=clipp2::observation(data, static_cast<std::int64_t>(region)*n+i);
            const auto result=clipp2::evaluate_observation<clipp2::EmitMode::LossGradientCurvature>(observation,x[cell]);
            if (result.status==clipp2::EmitStatus::InvalidInput)
                throw std::invalid_argument("Native continuation received invalid CCFs");
            if (result.status==clipp2::EmitStatus::NumericalFailure)
                throw std::runtime_error("Native continuation likelihood produced invalid arithmetic");
            out.value[cell]=result.loss; out.gradient[cell]=result.gradient;
            out.curvature[cell]=result.curvature;
        }};
        if(team)team->run(n,cell_terms);else for(int i=0;i<n;++i)cell_terms(i);
    }
    const std::vector<unsigned char>& project(const Vector& x, int k) const {
        if(arena.coordinate_epoch==arena.projection_epoch)return arena.cuts;
        arena.projection_epoch=arena.coordinate_epoch;
        auto& scores=arena.scores;auto& delta=arena.cut_delta;auto& selected=arena.selected;
        auto& mask=arena.cuts;std::fill(mask.begin(),mask.end(),0);selected.clear();
        if(k==1)return mask;
        for (int edge=0;edge<e;++edge) {
            for (int region=0;region<r;++region) {
                delta[static_cast<std::size_t>(edge)*r+region]=static_cast<long double>(x[static_cast<std::size_t>(edges[2*edge])*r+region])
                    -x[static_cast<std::size_t>(edges[2*edge+1])*r+region];
            }
        }
        if (e && cut_scores(delta.data(),e,r,scores.data()))
            throw std::runtime_error("Continuation cut-score reduction failed");
        for(int edge=0;edge<e;++edge)if(scores[edge]>0)selected.push_back(edge);
        const int keep=std::min(k-1,int(selected.size()));
        if(keep<int(selected.size()))std::nth_element(selected.begin(),selected.begin()+keep,selected.end(),
            [&](int a,int b){return scores[a]==scores[b]?a<b:scores[a]>scores[b];});
        for (int i=0;i<keep;++i) mask[selected[i]]=1;
        return mask;
    }
    const std::vector<int>& forest(const std::vector<unsigned char>& cuts) const {
        auto& result=arena.graph;result.clear();
        for (int edge=0;edge<e;++edge) if (!cuts[edge]) {
            result.push_back(edges[2*edge]);result.push_back(edges[2*edge+1]);
        }
        return result;
    }
    const Vector& jumps(const Vector& x,const std::vector<int>& graph,Vector& result) const {
        result.resize(graph.size()/2*r);
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region)
            result[edge*r+region]=x[static_cast<std::size_t>(graph[2*edge])*r+region]
                -x[static_cast<std::size_t>(graph[2*edge+1])*r+region];
        return result;
    }
    double square_sum(const Vector& values) const {
        return clipp2::pairwise_sum([&](std::int64_t i){return values[i]*values[i];},0,values.size());
    }
    const Vector& penalized_gradient(const Point& point,const std::vector<int>& graph,double rho) const {
        auto& spring=arena.gradient;std::fill(spring.begin(),spring.end(),0);
        jumps(point.x,graph,arena.spring_delta);auto& delta=arena.spring_delta;
        for (double& value:delta) value*=rho;
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region)
            spring[static_cast<std::size_t>(graph[2*edge])*r+region]+=delta[edge*r+region];
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region)
            spring[static_cast<std::size_t>(graph[2*edge+1])*r+region]-=delta[edge*r+region];
        for (std::size_t i=0;i<cells;++i) spring[i]+=point.terms.gradient[i];
        return spring;
    }
    double stationarity(const Vector& x,const Vector& gradient) const {
        double result=0;
        for (std::size_t i=0;i<cells;++i)
            result=std::max(result,std::abs(x[i]-clip(x[i]-gradient[i],lower[i],upper[i])));
        return result;
    }
    bool quadratic(const Vector& h,const Vector& rhs,const std::vector<int>& graph,
                   double rho,Vector& result) const {
        int diagnostics[2]={0,0};
        if(team) {
            team->run(r,[&](int region){clipp2kernel::solve_prepared_forest(tree,arena.cuts.data(),r,
                h.data(),rhs.data(),rho,lower,upper,result.data(),arena.qp_diagnostics.data()+2*region,region,region+1);});
            for(int region=0;region<r;++region){diagnostics[0]+=arena.qp_diagnostics[2*region];diagnostics[1]+=arena.qp_diagnostics[2*region+1];}
        } else clipp2kernel::solve_prepared_forest(tree,arena.cuts.data(),r,h.data(),rhs.data(),rho,
                                 lower,upper,result.data(),diagnostics);
        auto& gradient=arena.audit_gradient;auto& scale=arena.audit_scale;auto& delta=arena.audit_delta;
        for (std::size_t i=0;i<cells;++i) {
            const long double product=static_cast<long double>(h[i])*result[i];
            gradient[i]=product-rhs[i]; scale[i]=1+std::abs(product)+std::abs(rhs[i]);
        }
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region)
            delta[edge*r+region]=static_cast<long double>(rho)*
                (static_cast<long double>(result[static_cast<std::size_t>(graph[2*edge])*r+region])
                 -result[static_cast<std::size_t>(graph[2*edge+1])*r+region]);
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region) {
            const auto i=static_cast<std::size_t>(graph[2*edge])*r+region;
            gradient[i]+=delta[edge*r+region]; scale[i]+=std::abs(delta[edge*r+region]);
        }
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region) {
            const auto i=static_cast<std::size_t>(graph[2*edge+1])*r+region;
            gradient[i]-=delta[edge*r+region]; scale[i]+=std::abs(delta[edge*r+region]);
        }
        long double error=0;
        double violation=0;
        for (std::size_t i=0;i<cells;++i) {
            auto residual=gradient[i];
            if (result[i]==lower[i]) residual=std::min(residual,0.0L);
            if (result[i]==upper[i]) residual=std::max(residual,0.0L);
            if (lower[i]==upper[i]) residual=0;
            error=std::max(error,std::abs(residual)/scale[i]);
            violation=std::max({violation,lower[i]-result[i],result[i]-upper[i]});
        }
        return !diagnostics[1] && double(error)<=1e-7 && violation==0;
    }
    const std::vector<int>& components(const std::vector<int>& graph) const {
        auto& parent=arena.parent;auto& labels=arena.labels;auto& mapping=arena.mapping;
        std::fill(mapping.begin(),mapping.end(),-1);
        std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
        for (std::size_t edge=0;edge<graph.size()/2;++edge) {
            const int a=root(graph[2*edge]),b=root(graph[2*edge+1]);parent[b]=a;
        }
        for (int i=0;i<n;++i) labels[i]=root(i);
        int next=0;
        for (int i=0;i<n;++i) if(mapping[labels[canonical[i]]]<0) mapping[labels[canonical[i]]]=next++;
        for (int& label:labels) label=mapping[label];
        return labels;
    }
    void component_scale(const Point& point,const std::vector<int>& labels,
                         double& range,int& longest,double& mean_h) const {
        const int q=*std::max_element(labels.begin(),labels.end())+1;
        auto& minima=arena.minima;auto& maxima=arena.maxima;auto& means=arena.means;auto& sizes=arena.sizes;
        std::fill(minima.begin(),minima.end(),std::numeric_limits<double>::infinity());
        std::fill(maxima.begin(),maxima.end(),-std::numeric_limits<double>::infinity());
        std::fill(means.begin(),means.end(),0);std::fill(sizes.begin(),sizes.end(),0);
        for (int i=0;i<n;++i) {
            ++sizes[labels[i]];
            for (int region=0;region<r;++region) {
                const auto cell=static_cast<std::size_t>(i)*r+region,target=static_cast<std::size_t>(labels[i])*r+region;
                minima[target]=std::min(minima[target],point.x[cell]);
                maxima[target]=std::max(maxima[target],point.x[cell]);
                means[target]+=point.terms.curvature[cell];
            }
        }
        longest=*std::max_element(sizes.begin(),sizes.end());range=0;mean_h=-std::numeric_limits<double>::infinity();
        for (int group=0;group<q;++group) for (int region=0;region<r;++region) {
            const auto cell=static_cast<std::size_t>(group)*r+region;
            range=std::max(range,maxima[cell]-minima[cell]);
            if(sizes[group]==longest) mean_h=std::max(mean_h,means[cell]/sizes[group]);
        }
    }
    bool warm_start(const Point& point,const std::vector<int>& labels,Point& warm) const {
        const int q=*std::max_element(labels.begin(),labels.end())+1;
        auto& lo=arena.lo;auto& hi=arena.hi;auto& center=arena.center;auto& sizes=arena.sizes;
        std::fill(lo.begin(),lo.end(),-std::numeric_limits<double>::infinity());
        std::fill(hi.begin(),hi.end(),std::numeric_limits<double>::infinity());
        std::fill(center.begin(),center.end(),0);std::fill(sizes.begin(),sizes.end(),0);
        for (int i=0;i<n;++i) {
            ++sizes[labels[i]];
            for(int region=0;region<r;++region) {
                const auto cell=static_cast<std::size_t>(i)*r+region,target=static_cast<std::size_t>(labels[i])*r+region;
                lo[target]=std::max(lo[target],lower[cell]);hi[target]=std::min(hi[target],upper[cell]);
                center[target]+=point.x[cell];
            }
        }
        for(int group=0;group<q;++group) for(int region=0;region<r;++region) {
            const auto cell=static_cast<std::size_t>(group)*r+region;
            if(lo[cell]>hi[cell]) return false;
            center[cell]=clip(center[cell]/sizes[group],lo[cell],hi[cell]);
        }
        auto expand=[&](const Vector& values,Vector& result) {
            for(int i=0;i<n;++i) for(int region=0;region<r;++region)
                result[static_cast<std::size_t>(i)*r+region]=values[static_cast<std::size_t>(labels[i])*r+region];
        };
        expand(center,warm.x);evaluate(warm.x,warm.terms);
        for(int iteration=0;iteration<50;++iteration) {
            auto& gradient=arena.group_gradient;auto& curvature=arena.group_h;
            auto& direction=arena.group_direction;auto& products=arena.group_products;
            products.resize(static_cast<std::size_t>(q)*r);
            std::fill(gradient.begin(),gradient.end(),0);std::fill(curvature.begin(),curvature.end(),0);
            for(int i=0;i<n;++i) for(int region=0;region<r;++region) {
                const auto cell=static_cast<std::size_t>(i)*r+region,target=static_cast<std::size_t>(labels[i])*r+region;
                gradient[target]+=warm.terms.gradient[cell];curvature[target]+=warm.terms.curvature[cell];
            }
            double residual=0;
            for(int group=0;group<q;++group) for(int region=0;region<r;++region) {
                const auto cell=static_cast<std::size_t>(group)*r+region;
                direction[cell]=clip(center[cell]-gradient[cell]/std::max(curvature[cell],1e-8),lo[cell],hi[cell])-center[cell];
                residual=std::max(residual,std::abs(center[cell]-clip(center[cell]-gradient[cell]/sizes[group],lo[cell],hi[cell])));
                products[cell]=gradient[cell]*direction[cell];
            }
            const double slope=sum(products);
            if(residual<=1e-8 || !(slope<0)) break;
            const double previous=sum(warm.terms.value);
            bool accepted=false;
            for(int backtrack=0;backtrack<40;++backtrack) {
                const double step=std::ldexp(1.0,-backtrack);
                auto& next=arena.next;auto& trial=arena.trial;
                for(std::size_t i=0;i<static_cast<std::size_t>(q)*r;++i) next[i]=center[i]+step*direction[i];
                expand(next,trial.x);evaluate(trial.x,trial.terms);
                const double objective=sum(trial.terms.value);
                if(std::isfinite(objective) && objective<=previous+1e-4*step*slope) {
                    center.swap(next);std::swap(warm,trial);accepted=true;break;
                }
            }
            if(!accepted) break;
        }
        return true;
    }
    void run(const Point& initial,int capacity,double* output,int* label_output,
             unsigned char* cut_output,CliPP2ContinuationReport& report) const {
        auto& current=arena.current;current=initial;auto& warm=arena.warm;
        auto& warm_labels=arena.warm_labels;auto& labels=arena.labels;auto& cuts=arena.cuts;
        bool have_warm=false,numerical_limit=false,stalled=false;
        double rho=1,stationarity_value=0,violation=0,block_range=0;
        std::int64_t iterations=0,qp_calls=0,qp_unqualified=0,warm_starts=0,accepted_steps=0;
        evaluations=0;
        int longest=1,level=0;
        for(;level<policy.levels;++level) {
            stalled=false;
            for(int iteration=0;iteration<policy.iterations;++iteration) {
                ++iterations;project(current.x,capacity);
                const auto& graph=forest(cuts);
                const auto& delta=jumps(current.x,graph,arena.delta);
                const auto& gradient=penalized_gradient(current,graph,rho);
                stationarity_value=stationarity(current.x,gradient);
                if(stationarity_value<=policy.stationarity) break;
                auto& h=arena.h;auto& rhs=arena.rhs;auto& qp=arena.qp;auto& diagonal=arena.diagonal;
                for(std::size_t i=0;i<cells;++i) {h[i]=std::max(current.terms.curvature[i],1e-8);rhs[i]=h[i]*current.x[i]-current.terms.gradient[i];}
                const bool solved=quadratic(h,rhs,graph,rho,qp);qp_unqualified+=!solved;++qp_calls;
                const double previous=sum(current.terms.value)+0.5*rho*square_sum(delta);
                bool accepted=false;
                for(int proposal=solved?0:1;proposal<2;++proposal) {
                    if(proposal) {
                        auto& degree=arena.degree;std::fill(degree.begin(),degree.end(),0);
                        for(int endpoint:graph)++degree[endpoint];
                        for(int i=0;i<n;++i)for(int region=0;region<r;++region) {
                            const auto cell=static_cast<std::size_t>(i)*r+region;
                            diagonal[cell]=clip(current.x[cell]-gradient[cell]/(h[cell]+rho*degree[i]),lower[cell],upper[cell]);
                        }
                    }
                    const auto& target=proposal?diagonal:qp;
                    auto& direction=arena.direction;auto& products=arena.products;
                    for(std::size_t i=0;i<cells;++i){direction[i]=target[i]-current.x[i];products[i]=gradient[i]*direction[i];}
                    const double slope=sum(products);if(!(slope<0))continue;
                    for(int backtrack=0;backtrack<40;++backtrack) {
                        const double step=std::ldexp(1.0,-backtrack);
                        auto& trial=arena.trial;
                        for(std::size_t i=0;i<cells;++i)trial.x[i]=clip(current.x[i]+step*direction[i],lower[i],upper[i]);
                        evaluate(trial.x,trial.terms);
                        const double objective=sum(trial.terms.value)+0.5*rho*square_sum(jumps(trial.x,graph,arena.trial_delta));
                        const double slack=1e-12*std::max(1.0,std::abs(previous));
                        if(std::isfinite(objective) && objective<=previous+1e-4*step*slope+slack) {
                            double maximum_step=0;
                            for(std::size_t i=0;i<cells;++i)maximum_step=std::max(maximum_step,std::abs(trial.x[i]-current.x[i]));
                            stalled=maximum_step<=8*epsilon;std::swap(current,trial);++arena.coordinate_epoch;
                            accepted=true;++accepted_steps;break;
                        }
                    }
                    if(accepted)break;
                }
                if(!accepted)stalled=true;
                if(stalled)break;
            }
            project(current.x,capacity);const auto& graph=forest(cuts);
            stationarity_value=stationarity(current.x,penalized_gradient(current,graph,rho));
            const auto& delta=jumps(current.x,graph,arena.delta);
            violation=norm(delta.data(),delta.size());
            if(!std::isfinite(violation))throw std::runtime_error("Continuation norm reduction failed");
            components(graph);double mean_h;
            component_scale(current,labels,block_range,longest,mean_h);
            const bool feasible=std::max(violation,block_range)<=policy.constraint;
            if(stationarity_value<=policy.stationarity && feasible)break;
            if(rho>=policy.maximum_penalty || (stalled && feasible && 8*rho*epsilon>=policy.stationarity)) {numerical_limit=true;break;}
            if(level+1<policy.levels) {
                const double factor=std::max(2.0,std::min(10.0,std::max(mean_h/gaps[longest]/rho,std::max(violation,block_range)/policy.constraint)));
                const double next_rho=std::min(policy.maximum_penalty,rho*factor);
                if(warm_labels!=labels) {have_warm=warm_start(current,labels,warm);warm_labels=labels;}
                if(have_warm) {
                    const double previous=sum(current.terms.value)+0.5*next_rho*(violation*violation);
                    if(std::all_of(warm.terms.value.begin(),warm.terms.value.end(),[](double x){return std::isfinite(x);}) && sum(warm.terms.value)<previous) {current=warm;++arena.coordinate_epoch;++warm_starts;}
                }
                rho=next_rho;
            }
        }
        const bool tolerance=stationarity_value<=policy.stationarity && std::max(violation,block_range)<=policy.constraint;
        if(output)std::copy(current.x.begin(),current.x.end(),output);
        std::copy(labels.begin(),labels.end(),label_output);
        std::copy(cuts.begin(),cuts.end(),cut_output);
        const double nll=sum(current.terms.value);
        report={rho,stationarity_value,violation,block_range,nll,nll+0.5*rho*(violation*violation),
                iterations,evaluations,qp_calls,qp_unqualified,warm_starts,accepted_steps,std::min(level+1,policy.levels),longest,
                tolerance?0:(numerical_limit?1:(stalled?2:3))};
    }
};
}

extern "C" const char* CliPP2ContinuationLastError() {return error_message;}
extern "C" int CliPP2ContinuationMemory(int n,int r,int workers,int tasks,
    long long* shared,long long* state,long long* per_thread) {
    if(n<1||r<1||workers<1||tasks<1||!shared||!state||!per_thread)return 1;
    const long double cells=static_cast<long double>(n)*r;
    const long double a=4*sizeof(double)*cells+12.0L*sizeof(int)*n+4096;
    const long double b=(33*sizeof(double)+4*sizeof(long double))*cells+
        (sizeof(long double)+10*sizeof(int)+1)*static_cast<long double>(n)+8.0L*r+4096;
    const long double c=(5*sizeof(double)+4*sizeof(long double)+5*sizeof(int)+1)*static_cast<long double>(n)+4096;
    if(std::max({a,b,c})>std::numeric_limits<long long>::max())return 1;
    *shared=static_cast<long long>(a);*state=static_cast<long long>(b);*per_thread=static_cast<long long>(c);return 0;
}
extern "C" int CliPP2ContinueCPU(const clipp2::ObservationSoA* data,const int* edges,
    const int* canonical,const double* initial,const double* lower,const double* upper,
    int tasks,const int* capacities,const CliPP2ContinuationPolicy* policy,const double* gaps,
    CliPP2Norm norm,CliPP2CutScores cut_scores,int workers,long long workspace_bytes,long long host_budget_bytes,
    int output_flags,double* output,int* labels,
    unsigned char* cuts,CliPP2ContinuationReport* reports,int* actual_workers) {
    error_message[0]='\0';
    try {
        if(!data || data->mutations<1 || data->regions<1 || data->mutations>std::numeric_limits<int>::max()
            || data->regions>std::numeric_limits<int>::max() || !canonical || !initial || !lower || !upper
            || !policy || !gaps || !norm || !cut_scores || (output_flags && !output) || !labels || !cuts || !reports || !actual_workers
            || !capacities || tasks<1 || workers<1 || workspace_bytes<0 || host_budget_bytes<0
            || (output_flags!=0 && output_flags!=1) || (data->mutations>1 && !edges))
            throw std::invalid_argument("Invalid native continuation dimensions or buffers");
        const auto n=static_cast<int>(data->mutations),r=static_cast<int>(data->regions),e=n-1;
        if(static_cast<std::size_t>(n)>std::numeric_limits<std::size_t>::max()/r)
            throw std::invalid_argument("Native continuation dimensions overflow");
        const auto cells=static_cast<std::size_t>(n)*r;
        long long shared=0,state_bytes=0,thread_bytes=0;
        if(CliPP2ContinuationMemory(n,r,workers,tasks,&shared,&state_bytes,&thread_bytes))
            throw std::invalid_argument("Native continuation allocation sizes overflow");
        const auto required=[&](int team){return static_cast<long double>(shared)+
            state_bytes*(tasks>=team?team:1.0L)+static_cast<long double>(thread_bytes)*team;};
        const auto minimum=required(1);
        if(minimum>host_budget_bytes) {
            char reason[256];std::snprintf(reason,sizeof(reason),
                "CPU continuation requires %.0Lf tracked native host bytes; available %lld bytes",minimum,host_budget_bytes);
            throw std::runtime_error(reason);
        }
        while(workers>1 && required(workers)>host_budget_bytes)--workers;
        clipp2::validate_observations(*data,workspace_bytes);
        if(policy->levels<1 || policy->iterations<1 || !std::isfinite(policy->stationarity)
            || policy->stationarity<=0 || !std::isfinite(policy->constraint) || policy->constraint<=0
            || !std::isfinite(policy->maximum_penalty) || policy->maximum_penalty<=0)
            throw std::invalid_argument("Invalid native continuation policy");
        std::vector<int> seen(n,0);
        for(int i=0;i<n;++i) {
            if(canonical[i]<0 || canonical[i]>=n || seen[canonical[i]]++)
                throw std::invalid_argument("Invalid native canonical order");
            if(!std::isfinite(gaps[i+1]) || gaps[i+1]<=0)
                throw std::invalid_argument("Invalid native continuation gap");
        }
        for(int i=0;i<tasks;++i)if(capacities[i]<1 || capacities[i]>n)
            throw std::invalid_argument("Native continuation capacity outside 1..N");
        for(std::size_t i=0;i<cells;++i)if(!std::isfinite(initial[i]) || !std::isfinite(lower[i])
            || !std::isfinite(upper[i]) || lower[i]<0 || upper[i]>1 || lower[i]>upper[i]
            || initial[i]<lower[i] || initial[i]>upper[i])
            throw std::invalid_argument("Native continuation has invalid coordinates or box");
        // A caller may previously have solved a much larger QP on this thread.
        // Release that unrelated cache before admitting this batch's scratch.
        clipp2kernel::reset_forest_workspace();
        struct ReleaseForest {~ReleaseForest(){clipp2kernel::reset_forest_workspace();}} release_forest;
        const clipp2kernel::PreparedTree tree(n,e,edges);
        clipp2kernel::Executor pool(workers);
        pool.each([&](int){clipp2kernel::reset_forest_workspace();clipp2kernel::prepare_forest_workspace(n,e);});
        const bool capacity_parallel=tasks>=workers;
        Point start;resize(start,cells);start.x.assign(initial,initial+cells);
        // Initial emissions reuse the same pool but need no capacity arena.
        pool.run(n,[&](int i){for(int region=0;region<r;++region) {
            const auto cell=static_cast<std::size_t>(i)*r+region;
            const auto result=clipp2::evaluate_observation<clipp2::EmitMode::LossGradientCurvature>(
                clipp2::observation(*data,static_cast<std::int64_t>(region)*n+i),start.x[cell]);
            if(result.status==clipp2::EmitStatus::InvalidInput || result.status==clipp2::EmitStatus::NumericalFailure)
                throw std::runtime_error("Invalid initial continuation likelihood");
            start.terms.value[cell]=result.loss;start.terms.gradient[cell]=result.gradient;start.terms.curvature[cell]=result.curvature;
        }});
        if(!std::all_of(start.terms.value.begin(),start.terms.value.end(),[](double x){return std::isfinite(x);}))
            throw std::runtime_error("Initial tree likelihood is nonfinite");
        auto work=[&](int task){
            Arena arena(n,r);
            Engine engine{*data,n,r,e,cells,edges,canonical,lower,upper,gaps,*policy,norm,cut_scores,
                          tree,capacity_parallel?nullptr:&pool,arena};
            engine.run(start,capacities[task],output_flags?output+static_cast<std::size_t>(task)*cells:nullptr,
                labels+static_cast<std::size_t>(task)*n,cuts+static_cast<std::size_t>(task)*e,reports[task]);
        };
        if(capacity_parallel)pool.run(tasks,work);else for(int task=0;task<tasks;++task)work(task);
        *actual_workers=workers;return 0;
    } catch(const std::exception& error) {std::snprintf(error_message,sizeof(error_message),"%s",error.what());}
      catch(...) {std::snprintf(error_message,sizeof(error_message),"%s","Unknown native continuation failure");}
    return 1;
}
