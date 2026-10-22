// Ordered CPU continuation. This is the same finite-penalty search policy as
// solver.py, not a new objective, convergence gate or candidate-search policy.
#include "continuation.h"
#include "forest.h"
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

struct Engine {
    const clipp2::ObservationSoA& data;
    int n, r, e;
    std::size_t cells;
    const int *edges, *canonical;
    const double *lower, *upper, *gaps;
    const CliPP2ContinuationPolicy& policy;
    CliPP2Norm norm;
    CliPP2CutScores cut_scores;

    Terms evaluate(const Vector& x) const {
        ++evaluations;
        Terms out{Vector(cells), Vector(cells), Vector(cells)};
        for (int i=0; i<n; ++i) for (int region=0; region<r; ++region) {
            const auto cell=static_cast<std::size_t>(i)*r+region;
            const auto observation=clipp2::observation(data, static_cast<std::int64_t>(region)*n+i);
            const auto result=clipp2::evaluate_observation<clipp2::EmitMode::LossGradientCurvature>(observation,x[cell]);
            if (result.status==clipp2::EmitStatus::InvalidInput)
                throw std::invalid_argument("Native continuation received invalid CCFs");
            if (result.status==clipp2::EmitStatus::NumericalFailure)
                throw std::runtime_error("Native continuation likelihood produced invalid arithmetic");
            out.value[cell]=result.loss; out.gradient[cell]=result.gradient;
            out.curvature[cell]=result.curvature;
        }
        return out;
    }
    std::vector<unsigned char> project(const Vector& x, int k) const {
        std::vector<long double> scores(e);
        std::vector<long double> delta(static_cast<std::size_t>(e)*r);
        std::vector<int> selected;
        selected.reserve(e);
        for (int edge=0;edge<e;++edge) {
            for (int region=0;region<r;++region) {
                delta[static_cast<std::size_t>(edge)*r+region]=static_cast<long double>(x[static_cast<std::size_t>(edges[2*edge])*r+region])
                    -x[static_cast<std::size_t>(edges[2*edge+1])*r+region];
            }
        }
        if (e && cut_scores(delta.data(),e,r,scores.data()))
            throw std::runtime_error("Continuation cut-score reduction failed");
        for(int edge=0;edge<e;++edge)if(scores[edge]>0)selected.push_back(edge);
        std::stable_sort(selected.begin(),selected.end(),[&](int a,int b){return scores[a]>scores[b];});
        std::vector<unsigned char> mask(e,0);
        for (int i=0;i<std::min(k-1,int(selected.size()));++i) mask[selected[i]]=1;
        return mask;
    }
    std::vector<int> forest(const std::vector<unsigned char>& cuts) const {
        std::vector<int> result; result.reserve(2*e);
        for (int edge=0;edge<e;++edge) if (!cuts[edge]) {
            result.push_back(edges[2*edge]);result.push_back(edges[2*edge+1]);
        }
        return result;
    }
    Vector jumps(const Vector& x,const std::vector<int>& graph) const {
        Vector result(graph.size()/2*r);
        for (std::size_t edge=0;edge<graph.size()/2;++edge) for (int region=0;region<r;++region)
            result[edge*r+region]=x[static_cast<std::size_t>(graph[2*edge])*r+region]
                -x[static_cast<std::size_t>(graph[2*edge+1])*r+region];
        return result;
    }
    double square_sum(const Vector& values) const {
        return clipp2::pairwise_sum([&](std::int64_t i){return values[i]*values[i];},0,values.size());
    }
    Vector penalized_gradient(const Point& point,const std::vector<int>& graph,double rho) const {
        Vector spring(cells,0);
        Vector delta=jumps(point.x,graph);
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
        int diagnostics[2];
        clipp2kernel::solve_forest(n,r,graph.size()/2,graph.data(),h.data(),rhs.data(),rho,
                                 lower,upper,result.data(),diagnostics);
        std::vector<long double> gradient(cells),scale(cells),delta(graph.size()/2*r);
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
    std::vector<int> components(const std::vector<int>& graph) const {
        std::vector<int> parent(n),labels(n),mapping(n,-1);
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
        Vector minima(q*r,std::numeric_limits<double>::infinity());
        Vector maxima(q*r,-std::numeric_limits<double>::infinity()),means(q*r,0);
        std::vector<int> sizes(q,0);
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
        Vector lo(q*r,-std::numeric_limits<double>::infinity()),hi(q*r,std::numeric_limits<double>::infinity());
        Vector center(q*r,0);std::vector<int> sizes(q,0);
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
        auto expand=[&](const Vector& values) {
            Vector result(cells);
            for(int i=0;i<n;++i) for(int region=0;region<r;++region)
                result[static_cast<std::size_t>(i)*r+region]=values[static_cast<std::size_t>(labels[i])*r+region];
            return result;
        };
        warm.x=expand(center);warm.terms=evaluate(warm.x);
        for(int iteration=0;iteration<50;++iteration) {
            Vector gradient(q*r,0),curvature(q*r,0),direction(q*r),products(q*r);
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
                Vector next(center.size());
                for(std::size_t i=0;i<next.size();++i) next[i]=center[i]+step*direction[i];
                Point trial;trial.x=expand(next);trial.terms=evaluate(trial.x);
                const double objective=sum(trial.terms.value);
                if(std::isfinite(objective) && objective<=previous+1e-4*step*slope) {
                    center=std::move(next);warm=std::move(trial);accepted=true;break;
                }
            }
            if(!accepted) break;
        }
        return true;
    }
    void run(const Point& initial,int capacity,double* output,int* label_output,
             unsigned char* cut_output,CliPP2ContinuationReport& report) const {
        Point current=initial,warm;
        std::vector<int> warm_labels,labels;
        std::vector<unsigned char> cuts;
        bool have_warm=false,numerical_limit=false,stalled=false;
        double rho=1,stationarity_value=0,violation=0,block_range=0;
        std::int64_t iterations=0,qp_calls=0,qp_unqualified=0,warm_starts=0,accepted_steps=0;
        evaluations=0;
        int longest=1,level=0;
        for(;level<policy.levels;++level) {
            stalled=false;
            for(int iteration=0;iteration<policy.iterations;++iteration) {
                ++iterations;cuts=project(current.x,capacity);
                const auto graph=forest(cuts);
                const auto delta=jumps(current.x,graph);
                const auto gradient=penalized_gradient(current,graph,rho);
                stationarity_value=stationarity(current.x,gradient);
                if(stationarity_value<=policy.stationarity) break;
                Vector h(cells),rhs(cells),qp(cells),diagonal(cells);
                for(std::size_t i=0;i<cells;++i) {h[i]=std::max(current.terms.curvature[i],1e-8);rhs[i]=h[i]*current.x[i]-current.terms.gradient[i];}
                const bool solved=quadratic(h,rhs,graph,rho,qp);qp_unqualified+=!solved;++qp_calls;
                std::vector<int> degree(n,0);for(int endpoint:graph)++degree[endpoint];
                for(int i=0;i<n;++i)for(int region=0;region<r;++region) {
                    const auto cell=static_cast<std::size_t>(i)*r+region;
                    diagonal[cell]=clip(current.x[cell]-gradient[cell]/(h[cell]+rho*degree[i]),lower[cell],upper[cell]);
                }
                const double previous=sum(current.terms.value)+0.5*rho*square_sum(delta);
                bool accepted=false;
                for(int proposal=solved?0:1;proposal<2;++proposal) {
                    const auto& target=proposal?diagonal:qp;
                    Vector direction(cells),products(cells);
                    for(std::size_t i=0;i<cells;++i){direction[i]=target[i]-current.x[i];products[i]=gradient[i]*direction[i];}
                    const double slope=sum(products);if(!(slope<0))continue;
                    for(int backtrack=0;backtrack<40;++backtrack) {
                        const double step=std::ldexp(1.0,-backtrack);
                        Point trial;trial.x.resize(cells);
                        for(std::size_t i=0;i<cells;++i)trial.x[i]=clip(current.x[i]+step*direction[i],lower[i],upper[i]);
                        trial.terms=evaluate(trial.x);
                        const double objective=sum(trial.terms.value)+0.5*rho*square_sum(jumps(trial.x,graph));
                        const double slack=1e-12*std::max(1.0,std::abs(previous));
                        if(std::isfinite(objective) && objective<=previous+1e-4*step*slope+slack) {
                            double maximum_step=0;
                            for(std::size_t i=0;i<cells;++i)maximum_step=std::max(maximum_step,std::abs(trial.x[i]-current.x[i]));
                            stalled=maximum_step<=8*epsilon;current=std::move(trial);accepted=true;++accepted_steps;break;
                        }
                    }
                    if(accepted)break;
                }
                if(!accepted)stalled=true;
                if(stalled)break;
            }
            cuts=project(current.x,capacity);const auto graph=forest(cuts);
            stationarity_value=stationarity(current.x,penalized_gradient(current,graph,rho));
            const auto delta=jumps(current.x,graph);
            violation=norm(delta.data(),delta.size());
            if(!std::isfinite(violation))throw std::runtime_error("Continuation norm reduction failed");
            labels=components(graph);double mean_h;
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
                    if(std::all_of(warm.terms.value.begin(),warm.terms.value.end(),[](double x){return std::isfinite(x);}) && sum(warm.terms.value)<previous) {current=warm;++warm_starts;}
                }
                rho=next_rho;
            }
        }
        const bool tolerance=stationarity_value<=policy.stationarity && std::max(violation,block_range)<=policy.constraint;
        std::copy(current.x.begin(),current.x.end(),output);std::copy(labels.begin(),labels.end(),label_output);
        std::copy(cuts.begin(),cuts.end(),cut_output);
        const double nll=sum(current.terms.value);
        report={rho,stationarity_value,violation,block_range,nll,nll+0.5*rho*(violation*violation),
                iterations,evaluations,qp_calls,qp_unqualified,warm_starts,accepted_steps,std::min(level+1,policy.levels),longest,
                tolerance?0:(numerical_limit?1:(stalled?2:3))};
    }
};
}

extern "C" const char* CliPP2ContinuationLastError() {return error_message;}
extern "C" int CliPP2ContinueCPU(const clipp2::ObservationSoA* data,const int* edges,
    const int* canonical,const double* initial,const double* lower,const double* upper,
    int tasks,const int* capacities,const CliPP2ContinuationPolicy* policy,const double* gaps,
    CliPP2Norm norm,CliPP2CutScores cut_scores,int workers,long long workspace_bytes,double* output,int* labels,
    unsigned char* cuts,CliPP2ContinuationReport* reports,int* actual_workers) {
    error_message[0]='\0';
    try {
        if(!data || data->mutations<1 || data->regions<1 || data->mutations>std::numeric_limits<int>::max()
            || data->regions>std::numeric_limits<int>::max() || !canonical || !initial || !lower || !upper
            || !policy || !gaps || !norm || !cut_scores || !output || !labels || !cuts || !reports || !actual_workers
            || !capacities || tasks<1 || workers<1 || workspace_bytes<0 || (data->mutations>1 && !edges))
            throw std::invalid_argument("Invalid native continuation dimensions or buffers");
        const auto n=static_cast<int>(data->mutations),r=static_cast<int>(data->regions),e=n-1;
        if(static_cast<std::size_t>(n)>std::numeric_limits<std::size_t>::max()/r)
            throw std::invalid_argument("Native continuation dimensions overflow");
        const auto cells=static_cast<std::size_t>(n)*r;
        clipp2::validate_observations(*data,workspace_bytes);
        if(policy->levels<1 || policy->iterations<1 || !std::isfinite(policy->stationarity)
            || policy->stationarity<=0 || !std::isfinite(policy->constraint) || policy->constraint<=0
            || !std::isfinite(policy->maximum_penalty) || policy->maximum_penalty<=0)
            throw std::invalid_argument("Invalid native continuation policy");
        std::vector<int> roots(n),seen(n,0);std::iota(roots.begin(),roots.end(),0);
        auto root=[&](int i){while(roots[i]!=i){roots[i]=roots[roots[i]];i=roots[i];}return i;};
        for(int i=0;i<n;++i) {
            if(canonical[i]<0 || canonical[i]>=n || seen[canonical[i]]++)
                throw std::invalid_argument("Invalid native canonical order");
            if(!std::isfinite(gaps[i+1]) || gaps[i+1]<=0)
                throw std::invalid_argument("Invalid native continuation gap");
        }
        for(int edge=0;edge<e;++edge) {
            if(edges[2*edge]<0 || edges[2*edge]>=n || edges[2*edge+1]<0 || edges[2*edge+1]>=n)
                throw std::invalid_argument("Native continuation tree endpoint outside node range");
            const int a=root(edges[2*edge]),b=root(edges[2*edge+1]);
            if(a==b)throw std::invalid_argument("Native continuation graph is not a tree");
            roots[b]=a;
        }
        for(int i=0;i<tasks;++i)if(capacities[i]<1 || capacities[i]>n)
            throw std::invalid_argument("Native continuation capacity outside 1..N");
        for(std::size_t i=0;i<cells;++i)if(!std::isfinite(initial[i]) || !std::isfinite(lower[i])
            || !std::isfinite(upper[i]) || lower[i]<0 || upper[i]>1 || lower[i]>upper[i]
            || initial[i]<lower[i] || initial[i]>upper[i])
            throw std::invalid_argument("Native continuation has invalid coordinates or box");
        // Conservative upper bound includes controller vectors and per-thread
        // forest storage. Caller-owned outputs and immutable prepared inputs
        // are separate. No extra worker is admitted without its whole scratch.
        const long double per_worker=416.0L*cells+512.0L*n+64.0L*e;
        const long double shared=32.0L*cells;
        if(per_worker+shared>workspace_bytes)
            throw std::runtime_error("CPU continuation workspace cannot hold one capacity state");
        workers=std::min({workers,tasks,static_cast<int>(std::min<long double>(
            std::numeric_limits<int>::max(),(workspace_bytes-shared)/per_worker))});
        // A caller may previously have solved a much larger QP on this thread.
        // Release that unrelated cache before admitting this batch's scratch.
        clipp2kernel::reset_forest_workspace();
        Engine engine{*data,n,r,e,cells,edges,canonical,lower,upper,gaps,*policy,norm,cut_scores};
        Point start;start.x.assign(initial,initial+cells);start.terms=engine.evaluate(start.x);
        if(!std::all_of(start.terms.value.begin(),start.terms.value.end(),[](double x){return std::isfinite(x);}))
            throw std::runtime_error("Initial tree likelihood is nonfinite");
        std::atomic<int> next{0};std::atomic<bool> failed{false};std::exception_ptr failure;std::mutex failure_mutex;
        auto work=[&](){try {while(!failed.load()) {const int task=next.fetch_add(1);if(task>=tasks)break;
            engine.run(start,capacities[task],output+static_cast<std::size_t>(task)*cells,
                labels+static_cast<std::size_t>(task)*n,cuts+static_cast<std::size_t>(task)*e,reports[task]);}}
            catch(...){failed.store(true);std::lock_guard<std::mutex> lock(failure_mutex);if(!failure)failure=std::current_exception();}};
        std::vector<std::thread> pool;
        try {for(int i=1;i<workers;++i)pool.emplace_back(work);work();}
        catch(...){failed.store(true);for(auto& thread:pool)thread.join();throw;}
        for(auto& thread:pool)thread.join();
        if(failure)std::rethrow_exception(failure);
        *actual_workers=workers;return 0;
    } catch(const std::exception& error) {std::snprintf(error_message,sizeof(error_message),"%s",error.what());}
      catch(...) {std::snprintf(error_message,sizeof(error_message),"%s","Unknown native continuation failure");}
    return 1;
}
