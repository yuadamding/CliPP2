// CliPP2-owned shared-forest, multi-region quadratic solver; AGPL-3.0.
// Positive-message elimination generalizes the inherited scalar chain method.
// See the project NOTICE. Numerical qualification is separate from sweep completion.
#include "forest.h"
#include "prepared_tree.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace clipp2kernel {
namespace {

// One most-recent topology per calling thread, keyed by ordered edge contents.
// Never retain caller buffers or a history of forests. Active-set traversal is
// recomputed each sweep; only storage and immutable adjacency are reused.
struct ForestWorkspace {
    std::vector<int> parent, order, stack, active;
    std::vector<double> h, b, lo, hi, x;
    std::vector<bool> fixed;
    std::vector<long double> elimination_h, elimination_b, gradient, delta;

    void prepare(int n, int edge_count) {
        parent.resize(n);order.reserve(n);stack.reserve(n);active.resize(n);fixed.resize(n);
        h.resize(n);b.resize(n);lo.resize(n);hi.resize(n);x.resize(n);
        elimination_h.resize(n);elimination_b.resize(n);gradient.resize(n);delta.resize(edge_count);
    }
};
thread_local ForestWorkspace forest_workspace;

void free_forest(const std::vector<double>& curvature,const std::vector<double>& rhs,
    const PreparedTree& tree,const unsigned char* cuts,const std::vector<int>& active,
    std::vector<double>& x,double penalty,ForestWorkspace& workspace) {
    const int n=int(x.size());
    auto& parent=workspace.parent;auto& order=workspace.order;auto& stack=workspace.stack;
    std::fill(parent.begin(),parent.end(),-2);order.clear();stack.clear();
    // Match the reference's ascending roots, input-edge adjacency ordering,
    // assignment on push, and LIFO traversal. This fixes rounding and tie order.
    for(int root=0;root<n;++root) {
        if(active[root] || parent[root]!=-2) continue;
        parent[root]=-1;stack.push_back(root);
        while(!stack.empty()) {
            const int i=stack.back();stack.pop_back();order.push_back(i);
            for(int slot=tree.offsets[i];slot<tree.offsets[i+1];++slot) {
                if(cuts && cuts[tree.edge_ids[slot]])continue;
                const int j=tree.adjacent[slot];
                if(!active[j] && parent[j]==-2) {
                    parent[j]=i;stack.push_back(j);
                }
            }
        }
    }
    auto& h=workspace.elimination_h;auto& b=workspace.elimination_b;
    std::copy(curvature.begin(),curvature.end(),h.begin());
    std::copy(rhs.begin(),rhs.end(),b.begin());
    const long double rho=penalty;
    for(auto cursor=order.rbegin();cursor!=order.rend();++cursor) {
        const int i=*cursor;
        for(int slot=tree.offsets[i];slot<tree.offsets[i+1];++slot) {
            if(cuts && cuts[tree.edge_ids[slot]])continue;
            const int j=tree.adjacent[slot];
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

void reset_forest_workspace() { forest_workspace=ForestWorkspace{}; }
void prepare_forest_workspace(int nodes,int edges) {forest_workspace.prepare(nodes,edges);}

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
    const PreparedTree tree(n,edge_count,edges);
    solve_prepared_forest(tree,nullptr,regions,curvature,rhs,penalty,lower,upper,output,diagnostics);
}

void solve_prepared_forest(const PreparedTree& tree,const unsigned char* cuts,int regions,
    const double* curvature,const double* rhs,double penalty,const double* lower,
    const double* upper,double* output,int* diagnostics,int region_begin,int region_end) {
    const int n=tree.n,edge_count=tree.edges.size()/2;
    const auto* edges=tree.edges.data();
    if(region_end<0)region_end=regions;
    const auto cells=static_cast<std::size_t>(n)*regions;
    for(std::size_t i=0;i<cells;++i) {
        if(!std::isfinite(curvature[i]) || curvature[i]<=0.0 || !std::isfinite(rhs[i])
            || !std::isfinite(lower[i]) || !std::isfinite(upper[i]) || lower[i]>upper[i])
            throw std::invalid_argument("Quadratic requires finite positive curvature and feasible finite bounds");
    }
    auto& workspace=forest_workspace;
    workspace.prepare(n,edge_count);
    long long sweeps=0;
    int budget_hits=0;
    auto& h=workspace.h;auto& b=workspace.b;auto& lo=workspace.lo;
    auto& hi=workspace.hi;auto& x=workspace.x;auto& active=workspace.active;
    auto& fixed=workspace.fixed;auto& gradient=workspace.gradient;auto& delta=workspace.delta;
    const long double rho=penalty;
    for(int region=region_begin;region<region_end;++region) {
        for(int i=0;i<n;++i) {
            const auto cell=static_cast<std::size_t>(i)*regions+region;
            h[i]=curvature[cell];b[i]=rhs[cell];lo[i]=lower[cell];hi[i]=upper[cell];
            x[i]=lo[i];fixed[i]=lo[i]==hi[i];active[i]=fixed[i]?-1:0;
        }
        bool finished=false;
        for(long long sweep=0;sweep<2LL*n+64;++sweep) {
            ++sweeps;
            free_forest(h,b,tree,cuts,active,x,penalty,workspace);
            bool bounded=false;
            for(int i=0;i<n;++i) {
                if(!active[i] && x[i]<lo[i]) {x[i]=lo[i];active[i]=-1;bounded=true;}
                else if(!active[i] && x[i]>hi[i]) {x[i]=hi[i];active[i]=1;bounded=true;}
            }
            if(bounded) continue;
            for(int i=0;i<n;++i) gradient[i]=static_cast<long double>(h[i])*x[i]-b[i];
            for(int e=0;e<edge_count;++e) {
                if(cuts && cuts[e])continue;
                const auto offset=static_cast<std::size_t>(e)*2;
                delta[e]=rho*(static_cast<long double>(x[edges[offset]])-x[edges[offset+1]]);
            }
            // NumPy uses two add.at passes: all positive endpoints first, then
            // all negative endpoints. Do not interleave their accumulation.
            for(int e=0;e<edge_count;++e)if(!cuts || !cuts[e])gradient[edges[static_cast<std::size_t>(e)*2]]+=delta[e];
            for(int e=0;e<edge_count;++e)if(!cuts || !cuts[e])gradient[edges[static_cast<std::size_t>(e)*2+1]]-=delta[e];
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
