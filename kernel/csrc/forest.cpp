// CliPP2-owned shared-forest, multi-region quadratic solver; AGPL-3.0.
// Positive-message elimination generalizes the inherited scalar chain method.
// See the project NOTICE. Numerical qualification is separate from sweep completion.
#include "chain.h"
#include <limits>
#include <numeric>
#include <stdexcept>

namespace clipp2kernel {
namespace {

void free_forest(const std::vector<double>& curvature,const std::vector<double>& rhs,
    const std::vector<std::vector<int>>& adjacency,const std::vector<int>& active,
    std::vector<double>& x,double penalty) {
    const int n=int(x.size());
    std::vector<int> parent(n,-2),order,stack;
    order.reserve(n);stack.reserve(n);
    // Match the reference's ascending roots, input-edge adjacency ordering,
    // assignment on push, and LIFO traversal. This fixes rounding and tie order.
    for(int root=0;root<n;++root) {
        if(active[root] || parent[root]!=-2) continue;
        parent[root]=-1;stack.push_back(root);
        while(!stack.empty()) {
            const int i=stack.back();stack.pop_back();order.push_back(i);
            for(int j:adjacency[i]) {
                if(!active[j] && parent[j]==-2) {
                    parent[j]=i;stack.push_back(j);
                }
            }
        }
    }
    std::vector<long double> h(curvature.begin(),curvature.end()),b(rhs.begin(),rhs.end());
    const long double rho=penalty;
    for(auto cursor=order.rbegin();cursor!=order.rend();++cursor) {
        const int i=*cursor;
        for(int j:adjacency[i]) {
            if(active[j]) {
                h[i]+=rho;b[i]+=rho*static_cast<long double>(x[j]);
            }
        }
        if(!std::isfinite(h[i]) || h[i]<=0.0L)
            throw std::runtime_error("Nonpositive or nonfinite forest elimination curvature");
        if(parent[i]>=0) {
            const int j=parent[i];
            const long double ratio=rho/(rho+h[i]);
            h[j]+=ratio*h[i];b[j]+=ratio*b[i];
        }
    }
    for(int i:order) {
        if(parent[i]<0) x[i]=double(b[i]/h[i]);
        else {
            const long double p=x[parent[i]];
            x[i]=double(p+(b[i]-h[i]*p)/(rho+h[i]));
        }
    }
}

} // namespace

void solve_forest(int n,int regions,int edge_count,const int* edges,
    const double* curvature,const double* rhs,double penalty,
    const double* lower,const double* upper,double* output,int* diagnostics) {
    if(n<1 || regions<1 || edge_count<0 || edge_count>n-1)
        throw std::invalid_argument("Forest requires positive N,R and 0<=E<N");
    if(!curvature || !rhs || !lower || !upper || !output || !diagnostics || (edge_count && !edges))
        throw std::invalid_argument("Non-null forest array buffers are required");
    if(!std::isfinite(penalty) || penalty<0.0)
        throw std::invalid_argument("Quadratic penalty must be finite and nonnegative");
    if(static_cast<std::size_t>(n)>std::numeric_limits<std::size_t>::max()/regions)
        throw std::invalid_argument("Forest matrix dimensions overflow addressable memory");
    const auto cells=static_cast<std::size_t>(n)*regions;
    for(std::size_t i=0;i<cells;++i) {
        if(!std::isfinite(curvature[i]) || curvature[i]<=0.0 || !std::isfinite(rhs[i])
            || !std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>upper[i])
            throw std::invalid_argument("Quadratic requires finite positive curvature and feasible finite bounds");
    }
    std::vector<std::vector<int>> adjacency(n);
    std::vector<int> roots(n);
    std::iota(roots.begin(),roots.end(),0);
    for(int e=0;e<edge_count;++e) {
        const auto offset=static_cast<std::size_t>(e)*2;
        const int i=edges[offset],j=edges[offset+1];
        if(i<0 || i>=n || j<0 || j>=n)
            throw std::invalid_argument("Forest endpoint outside node range");
        int a=i,c=j;
        while(roots[a]!=a) {roots[a]=roots[roots[a]];a=roots[a];}
        while(roots[c]!=c) {roots[c]=roots[roots[c]];c=roots[c];}
        if(a==c) throw std::invalid_argument("Quadratic graph must be a forest");
        roots[c]=a;
        adjacency[i].push_back(j);adjacency[j].push_back(i);
    }
    long long sweeps=0;
    int budget_hits=0;
    std::vector<double> h(n),b(n),lo(n),hi(n),x(n);
    std::vector<int> active(n);
    std::vector<bool> fixed(n);
    std::vector<long double> gradient(n),delta(edge_count);
    const long double rho=penalty;
    for(int region=0;region<regions;++region) {
        for(int i=0;i<n;++i) {
            const auto cell=static_cast<std::size_t>(i)*regions+region;
            h[i]=curvature[cell];b[i]=rhs[cell];lo[i]=lower[cell];hi[i]=upper[cell];
            x[i]=lo[i];fixed[i]=lo[i]==hi[i];active[i]=fixed[i]?-1:0;
        }
        bool finished=false;
        for(long long sweep=0;sweep<2LL*n+64;++sweep) {
            ++sweeps;
            free_forest(h,b,adjacency,active,x,penalty);
            bool bounded=false;
            for(int i=0;i<n;++i) {
                if(!active[i] && x[i]<lo[i]) {x[i]=lo[i];active[i]=-1;bounded=true;}
                else if(!active[i] && x[i]>hi[i]) {x[i]=hi[i];active[i]=1;bounded=true;}
            }
            if(bounded) continue;
            for(int i=0;i<n;++i) gradient[i]=static_cast<long double>(h[i])*x[i]-b[i];
            for(int e=0;e<edge_count;++e) {
                const auto offset=static_cast<std::size_t>(e)*2;
                delta[e]=rho*(static_cast<long double>(x[edges[offset]])-x[edges[offset+1]]);
            }
            // NumPy uses two add.at passes: all positive endpoints first, then
            // all negative endpoints. Do not interleave their accumulation.
            for(int e=0;e<edge_count;++e) gradient[edges[static_cast<std::size_t>(e)*2]]+=delta[e];
            for(int e=0;e<edge_count;++e) gradient[edges[static_cast<std::size_t>(e)*2+1]]-=delta[e];
            int release=0;
            long double largest=-std::numeric_limits<long double>::infinity();
            for(int i=0;i<n;++i) {
                const long double wrong=fixed[i]?0.0L:(active[i]<0?-gradient[i]:(active[i]>0?gradient[i]:0.0L));
                if(wrong>largest) {largest=wrong;release=i;}
            }
            if(largest<=1e-9) {finished=true;break;}
            active[release]=0;
        }
        budget_hits+=!finished;
        for(int i=0;i<n;++i) {
            if(!std::isfinite(x[i])) throw std::runtime_error("Nonfinite forest quadratic iterate");
            if(x[i]<lo[i]) x[i]=lo[i];
            if(x[i]>hi[i]) x[i]=hi[i];
            output[static_cast<std::size_t>(i)*regions+region]=x[i];
        }
    }
    if(sweeps>std::numeric_limits<int>::max())
        throw std::runtime_error("Forest sweep count exceeds the ABI diagnostic integer range");
    diagnostics[0]=int(sweeps);diagnostics[1]=budget_hits;
}

} // namespace clipp2kernel
