// Ordered immutable tree adjacency, shared across continuation capacities.
#pragma once
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace clipp2kernel {
struct PreparedTree {
    int n;
    std::vector<int> edges, offsets, adjacent, edge_ids;
    PreparedTree(int nodes,int count,const int* input):n(nodes),offsets(nodes+1,0) {
        if(count)edges.assign(input,input+2*static_cast<std::size_t>(count));
        std::vector<int> roots(n);std::iota(roots.begin(),roots.end(),0);
        auto root=[&](int i){while(roots[i]!=i){roots[i]=roots[roots[i]];i=roots[i];}return i;};
        for(int e=0;e<count;++e) {
            const int a=edges[2*e],b=edges[2*e+1];
            if(a<0||a>=n||b<0||b>=n)throw std::invalid_argument("Forest endpoint outside node range");
            const int x=root(a),y=root(b);
            if(x==y)throw std::invalid_argument("Quadratic graph must be a forest");
            roots[y]=x;++offsets[a+1];++offsets[b+1];
        }
        std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
        adjacent.resize(2*static_cast<std::size_t>(count));edge_ids.resize(adjacent.size());
        auto cursor=offsets;
        for(int e=0;e<count;++e)for(int side=0;side<2;++side) {
            const int slot=cursor[edges[2*e+side]]++;
            adjacent[slot]=edges[2*e+1-side];edge_ids[slot]=e;
        }
    }
};
}
