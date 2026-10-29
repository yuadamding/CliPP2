// Terminal leaves of the existing eight-aligned/8192-value reduction tree.
// Scheduling is parallel; floating-point parentheses remain unchanged.
struct ScoringLeaf {
    i64 first, count, slot;
};
struct ScoringReductionPlan {
    i64 full_chunks, tail_count=0, leaf_count;
    ScoringLeaf tail[128]{};
    explicit ScoringReductionPlan(i64 n):full_chunks(n/8192) {
        if(n<=0) throw std::domain_error("CUDA scoring rows must be positive");
        if(n%8192) append(full_chunks*8192,n%8192,0,128);
        leaf_count=add(multiply(full_chunks,64),tail_count);
    }
    void append(i64 first,i64 count,i64 slot,i64 width) {
        if(count<=128) { tail[tail_count++]={first,count,slot}; return; }
        i64 left=count/2; left-=left%8;
        append(first,left,slot,width/2);
        append(first+left,count-left,slot+width/2,width/2);
    }
};

// Defined in scoring.cuh; both small and scalable dispatches share these exact
// row operations. No exp/log or accumulation order is changed by the tiling.
__device__ double mixture_row_log_mass(const double* const* columns,const double* offsets,
        const double* kernel,const double* weights,i64 n,i64 q,i64 row);
__device__ double scoring_capped_exp(double value);
__device__ double mixture_trial_row(i64 row,const double* other,const double* grow,
        const double* shrink,const double* numerator,const double* sign,double a,double b);

constexpr int scoring_plane_tile=16;
__device__ ScoringLeaf scoring_leaf(i64 leaf,i64 full_chunks,const ScoringLeaf* tail) {
    if(leaf<full_chunks*64) return {leaf*128,128,2*(leaf%64)};
    return tail[leaf-full_chunks*64];
}

// Eight lanes own the eight independent accumulation chains of each plane.
// The final four-pair fold and remainder are still performed in original order.
// Every block participant must enter, including inactive planes and rows.
__device__ void scoring_leaf_reduce(const double* values,i64 count,i64 first_plane,
        i64 planes,i64 leaf,i64 leaves,double* chains,double* partials) {
    const int plane=threadIdx.x/8,lane=threadIdx.x%8;
    if(plane<planes && count>=8) {
        double sum=values[plane*128+lane];
        for(i64 row=8;row<count-count%8;row+=8) sum+=values[plane*128+row+lane];
        chains[threadIdx.x]=sum;
    }
    __syncthreads();
    if(plane<planes && !lane) {
        double sum=-0.;
        if(count<8) {
            for(i64 row=0;row<count;++row) sum+=values[plane*128+row];
        } else {
            const double* a=chains+plane*8;
            sum=((a[0]+a[1])+(a[2]+a[3]))+((a[4]+a[5])+(a[6]+a[7]));
            for(i64 row=count-count%8;row<count;++row) sum+=values[plane*128+row];
        }
        partials[(first_plane+plane)*leaves+leaf]=sum;
    }
    __syncthreads(); // All readers finish before the next tile overwrites input.
}

// Scalar, host-batched and resident adapters enter these collectives uniformly.
// Only scheduling differs: the row arithmetic and reduction parentheses live here.
__device__ void scoring_evaluate_leaf(const double* const* columns,const double* offsets,
        const double* kernel,const double* weights,i64 n,i64 q,ScoringLeaf range,
        i64 leaf,i64 leaves,double* partials,double* values,double* chains,double* masses) {
    const i64 row=threadIdx.x;
    if(row<range.count)
        masses[row]=mixture_row_log_mass(columns,offsets,kernel,weights,n,q,range.first+row);
    __syncthreads();
    for(i64 first=0;first<q+2;first+=scoring_plane_tile) {
        const i64 count=q+2-first<scoring_plane_tile?q+2-first:scoring_plane_tile;
        if(row<range.count) {
            const i64 source=range.first+row;
            const double offset=offsets[source],mass=masses[row];
            for(i64 plane=0;plane<count;++plane) {
                const i64 index=first+plane;
                values[plane*128+row]=(index==0?mass:(index==1?offset+mass:
                    scoring_capped_exp(columns[index-2][source]-offset-mass)));
            }
        }
        __syncthreads();
        scoring_leaf_reduce(values,range.count,first,count,leaf,leaves,chains,partials);
    }
}
__device__ void scoring_trial_leaf(ScoringLeaf range,i64 leaf,i64 leaves,
        const double* other,const double* grow,const double* shrink,const double* numerator,
        const double* sign,double a,double b,double* partials,double* values,double* chains) {
    const i64 row=threadIdx.x;
    if(row<range.count)
        values[row]=mixture_trial_row(range.first+row,other,grow,shrink,numerator,sign,a,b);
    __syncthreads();
    scoring_leaf_reduce(values,range.count,0,1,leaf,leaves,chains,partials);
}
// Reconstruct internal nodes from compact leaf results. Unoccupied slots are
// never read: each input is a terminal leaf or an already-combined child.
__device__ void scoring_combine_chunk(const double* leaves,i64 n,i64 chunks,
        i64 full_chunks,const ScoringLeaf* tail,i64 tail_count,i64 leaf_count,
        i64 job,double* partials,double* values) {
    const i64 plane=job/chunks,chunk=job%chunks;
    const bool full=chunk<full_chunks;
    const i64 count=full?8192:n%8192,first=chunk*64;
    if(threadIdx.x<(full?64:tail_count)) {
        const i64 slot=full?2*threadIdx.x:tail[threadIdx.x].slot;
        values[slot]=leaves[plane*leaf_count+first+threadIdx.x];
    }
    __syncthreads();
    int deepest=0;
    for(i64 maximum=count;maximum>128;++deepest) {
        i64 split=maximum/2; split-=split%8;
        maximum-=split;
    }
    for(int depth=deepest-1;depth>=0;--depth) {
        const int nodes=1<<depth;
        for(int node=threadIdx.x;node<nodes;node+=blockDim.x) {
            i64 offset,size;
            if(ordered_tree_node(count,depth,node,offset,size) && size>128) {
                const int left=node<<(7-depth),right=left+(1<<(6-depth));
                values[left]=values[left]+values[right];
            }
        }
        __syncthreads();
    }
    if(!threadIdx.x) {
        double sum=-0.; sum+=values[0]; // Preserve ordered_block_sum's outer fold.
        partials[job]=sum;
    }
    __syncthreads();
}
__device__ double scoring_finish_plane(const double* partials,i64 chunks,i64 plane) {
    double sum=-0.;
    for(i64 chunk=0;chunk<chunks;++chunk) sum+=partials[plane*chunks+chunk];
    return sum;
}

__global__ void mixture_leaf_evaluate_kernel(const double* const* columns,const double* offsets,
        const double* kernel,const double* weights,i64 n,i64 q,i64 full_chunks,
        const ScoringLeaf* tail,i64 leaves,double* partials) {
    static_assert(threads==128,"Scoring leaves require 128 physical participants");
    __shared__ double values[scoring_plane_tile*128],chains[128],masses[128];
    for(i64 leaf=blockIdx.x;leaf<leaves;leaf+=gridDim.x)
        scoring_evaluate_leaf(columns,offsets,kernel,weights,n,q,scoring_leaf(leaf,full_chunks,tail),
                              leaf,leaves,partials,values,chains,masses);
}
__global__ void mixture_leaf_trial_kernel(i64 n,i64 full_chunks,const ScoringLeaf* tail,i64 leaves,
        const double* other,const double* grow,const double* shrink,const double* numerator,
        const double* sign,double grow_weight,double amount,double fraction,double* partials) {
    __shared__ double values[128],chains[128];
    const double a=log(grow_weight+fraction*amount),b=log(amount*(1.-fraction));
    for(i64 leaf=blockIdx.x;leaf<leaves;leaf+=gridDim.x)
        scoring_trial_leaf(scoring_leaf(leaf,full_chunks,tail),leaf,leaves,
            other,grow,shrink,numerator,sign,a,b,partials,values,chains);
}
__global__ void mixture_combine_leaves_kernel(const double* leaves,i64 n,i64 planes,i64 chunks,
        i64 full_chunks,const ScoringLeaf* tail,i64 tail_count,i64 leaf_count,double* partials) {
    __shared__ double values[128];
    for(i64 job=blockIdx.x;job<planes*chunks;job+=gridDim.x)
        scoring_combine_chunk(leaves,n,chunks,full_chunks,tail,tail_count,leaf_count,job,partials,values);
}
