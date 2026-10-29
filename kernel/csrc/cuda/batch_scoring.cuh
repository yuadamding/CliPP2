// One submission owner; independent mixture states keep their request order.
// Included inside scoring.cuh's anonymous namespace, after scalar helpers.
struct WeightBatchRequest {
    i64 handle, operation, grow, shrink;
    double fraction;
    const double* weights;
};
struct WeightBatchItem {
    const double* const* columns;
    const double *offsets,*kernel,*weights;
    double *logs,*other,*grow_log,*shrink_log,*numerator,*sign,*sums;
    const ScoringLeaf* tail;
    double *leaf_partials,*chunk_partials;
    i64 n,q,operation,grow,shrink,output;
    i64 full_chunks,tail_count,leaf_count,chunks;
    double grow_weight,amount,fraction;
    int small;
};

__global__ void weight_batch_small_kernel(const WeightBatchItem* items,i64 count,double* output) {
    __shared__ double terms[small_mixture_rows*(small_mixture_columns+2)];
    const i64 row=threadIdx.x;
    for(i64 index=blockIdx.x;index<count;index+=gridDim.x) {
        const auto& p=items[index];
        if(!p.small) continue; // Uniform for every participant in this block.
        if(p.operation==0) {
            if(row<p.n) mixture_evaluate_row(p.columns,p.offsets,p.kernel,p.weights,
                                            p.n,p.q,row,terms);
            __syncthreads();
            if(row<p.q+2) output[p.output+row]=small_mixture_sum(terms+row*p.n,p.n);
        } else {
            if(p.operation==1) {
                if(row<p.q) p.logs[row]=log(p.weights[row]);
                __syncthreads();
                if(row<p.n) mixture_pair_row(p.columns,p.offsets,p.logs,p.q,row,
                    p.grow,p.shrink,p.other,p.grow_log,p.shrink_log,p.numerator,p.sign);
                // Each row subsequently reads only its own pair fields.
            }
            const double a=log(p.grow_weight+p.fraction*p.amount),b=log(p.amount*(1.-p.fraction));
            if(row<p.n) terms[row]=mixture_trial_row(row,p.other,p.grow_log,
                p.shrink_log,p.numerator,p.sign,a,b);
            __syncthreads();
            if(!row) output[p.output]=small_mixture_sum(terms,p.n);
        }
        __syncthreads(); // Do not overwrite shared rows while another lane reads.
    }
}
// General requests share every numerical stage. Grid.y owns independent
// problems; grid.x owns rows, existing reduction leaves or chunk/plane jobs.
// Ready requests execute immediately, including a partial batch of one.
__global__ void weight_batch_logs_kernel(const WeightBatchItem* items,i64 count) {
    for(i64 index=blockIdx.y;index<count;index+=gridDim.y) {
        const auto& p=items[index];
        if(p.small||p.operation!=1) continue;
        for(i64 k=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;k<p.q;
            k+=static_cast<i64>(gridDim.x)*blockDim.x) p.logs[k]=log(p.weights[k]);
    }
}
__global__ void weight_batch_pair_kernel(const WeightBatchItem* items,i64 count) {
    for(i64 index=blockIdx.y;index<count;index+=gridDim.y) {
        const auto& p=items[index];
        if(p.small||p.operation!=1) continue;
        for(i64 row=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;row<p.n;
            row+=static_cast<i64>(gridDim.x)*blockDim.x)
            mixture_pair_row(p.columns,p.offsets,p.logs,p.q,row,p.grow,p.shrink,
                             p.other,p.grow_log,p.shrink_log,p.numerator,p.sign);
    }
}
__global__ void weight_batch_leaves_kernel(const WeightBatchItem* items,i64 count) {
    static_assert(threads==128,"Scoring leaves require 128 physical participants");
    __shared__ double values[scoring_plane_tile*128],chains[128],masses[128];
    for(i64 index=blockIdx.y;index<count;index+=gridDim.y) {
        const auto& p=items[index];
        if(p.small) continue;
        if(p.operation==0) {
            for(i64 leaf=blockIdx.x;leaf<p.leaf_count;leaf+=gridDim.x)
                scoring_evaluate_leaf(p.columns,p.offsets,p.kernel,p.weights,p.n,p.q,
                    scoring_leaf(leaf,p.full_chunks,p.tail),leaf,p.leaf_count,
                    p.leaf_partials,values,chains,masses);
        } else {
            const double a=log(p.grow_weight+p.fraction*p.amount),b=log(p.amount*(1.-p.fraction));
            for(i64 leaf=blockIdx.x;leaf<p.leaf_count;leaf+=gridDim.x)
                scoring_trial_leaf(scoring_leaf(leaf,p.full_chunks,p.tail),leaf,p.leaf_count,
                    p.other,p.grow_log,p.shrink_log,p.numerator,p.sign,a,b,p.leaf_partials,values,chains);
        }
        __syncthreads();
    }
}
__global__ void weight_batch_combine_kernel(const WeightBatchItem* items,i64 count) {
    __shared__ double values[128];
    for(i64 index=blockIdx.y;index<count;index+=gridDim.y) {
        const auto& p=items[index];
        if(p.small) continue;
        const i64 planes=p.operation==0?p.q+2:1;
        for(i64 job=blockIdx.x;job<planes*p.chunks;job+=gridDim.x)
            scoring_combine_chunk(p.leaf_partials,p.n,p.chunks,p.full_chunks,p.tail,p.tail_count,
                                  p.leaf_count,job,p.chunk_partials,values);
        __syncthreads();
    }
}
__global__ void weight_batch_finish_kernel(const WeightBatchItem* items,i64 count,double* output) {
    for(i64 index=blockIdx.y;index<count;index+=gridDim.y) {
        const auto& p=items[index];
        if(p.small) continue;
        const i64 size=p.operation==0?p.q+2:1;
        for(i64 k=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;k<size;
            k+=static_cast<i64>(gridDim.x)*blockDim.x) {
            const double sum=scoring_finish_plane(p.chunk_partials,p.chunks,k);
            p.sums[k]=sum;
            output[p.output+k]=sum;
        }
    }
}

void evaluate_weight_batch(Context& c,i64 count,const WeightBatchRequest* requests,
                           i64 output_size,double* output) {
    if(count<=0||!requests||!output) throw std::domain_error("Invalid CUDA weight batch");
    // Validate the entire batch before changing any transfer state or submitting.
    i64 weights_count=0,results=0;
    for(i64 index=0;index<count;++index) {
        const auto& request=requests[index];
        auto& p=scoring_problem(c,request.handle);
        for(i64 prior=0;prior<index;++prior)
            if(requests[prior].handle==request.handle)
                throw std::domain_error("CUDA weight batch requires independent mixture states");
        if(request.operation<0||request.operation>2)
            throw std::domain_error("Unknown CUDA weight batch operation");
        if(request.operation!=2) {
            validate_scoring_weights(request.weights,p.q);
            weights_count=add(weights_count,p.q);
        }
        if(request.operation==1 && (request.grow<0||request.grow>=p.q||
                request.shrink<0||request.shrink>=p.q||request.grow==request.shrink||
                request.weights[request.shrink]<=0.))
            throw std::domain_error("Invalid CUDA batched transfer");
        if(request.operation==2 && (!p.pair_ready||!std::isfinite(request.fraction)||
                request.fraction<0.||request.fraction>1.))
            throw std::domain_error("Unprepared or invalid CUDA batched trial");
        results=add(results,request.operation==0?add(p.q,2):1);
    }
    if(results!=output_size) throw std::domain_error("CUDA batch output size mismatch");
    Layout layout;
    const i64 descriptors=layout.take<WeightBatchItem>(count);
    const i64 weights_offset=layout.take<double>(weights_count);
    const i64 output_offset=layout.take<double>(results),input_bytes=output_offset;
    // Lease the already-budgeted request pool. Byte copies never read invalid
    // pointer representations from its host array; pointer users overwrite it.
    ScoringRequestLease lease(c,add(layout.bytes,7)/8,true);
    auto* host=reinterpret_cast<unsigned char*>(lease.pointers());
    auto* device=reinterpret_cast<unsigned char*>(lease.device_pointers());
    i64 cursor=0,result_offset=0;
    bool has_small=false,has_general=false,has_general_prepare=false;
    i64 maximum_q=0,maximum_leaves=0,maximum_jobs=0;
    for(i64 index=0;index<count;++index) {
        const auto& request=requests[index];
        auto& p=scoring_problem(c,request.handle);
        WeightBatchItem item{};
        item.columns=p.at<const double*>(p.pointers); item.offsets=p.at<double>(p.offset);
        item.kernel=p.at<double>(p.kernel); item.weights=reinterpret_cast<double*>(device+weights_offset)+cursor;
        item.logs=p.at<double>(p.log_weights); item.other=p.at<double>(p.other);
        item.grow_log=p.at<double>(p.grow_log); item.shrink_log=p.at<double>(p.shrink_log);
        item.numerator=p.at<double>(p.numerator); item.sign=p.at<double>(p.sign); item.sums=p.at<double>(p.sums);
        item.tail=p.at<ScoringLeaf>(p.leaf_metadata);
        item.leaf_partials=p.at<double>(p.leaf_partials); item.chunk_partials=p.at<double>(p.partials);
        item.full_chunks=p.reduction.full_chunks; item.tail_count=p.reduction.tail_count;
        item.leaf_count=p.reduction.leaf_count; item.chunks=p.chunks;
        item.n=p.n; item.q=p.q; item.operation=request.operation;
        item.grow=request.grow; item.shrink=request.shrink; item.output=result_offset;
        item.small=small_mixture(p); has_small=has_small||item.small; has_general=has_general||!item.small;
        if(!item.small) {
            has_general_prepare=has_general_prepare||request.operation==1;
            maximum_q=std::max(maximum_q,p.q);
            maximum_leaves=std::max(maximum_leaves,p.reduction.leaf_count);
            maximum_jobs=std::max(maximum_jobs,multiply(request.operation==0?add(p.q,2):1,p.chunks));
        }
        if(request.operation!=2) {
            std::memcpy(host+weights_offset+cursor*8,request.weights,static_cast<std::size_t>(p.q*8));
            cursor+=p.q;
        }
        if(request.operation==1) {
            p.pair_ready=false;
            p.grow_weight=request.weights[request.grow]; p.amount=request.weights[request.shrink];
        }
        item.grow_weight=p.grow_weight; item.amount=p.amount;
        item.fraction=request.operation==1?1.:request.fraction;
        std::memcpy(host+descriptors+index*sizeof(item),&item,sizeof(item));
        result_offset+=request.operation==0?p.q+2:1;
    }
    c.upload(device,host,input_bytes);
    auto* items=reinterpret_cast<WeightBatchItem*>(device+descriptors);
    auto* device_output=reinterpret_cast<double*>(device+output_offset);
    if(has_small) {
        weight_batch_small_kernel<<<static_cast<unsigned>(std::min<i64>(65535,count)),threads,0,c.stream>>>(
            items,count,device_output);
        c.launched();
    }
    if(has_general) {
        const unsigned problems=static_cast<unsigned>(std::min<i64>(65535,count));
        if(has_general_prepare) {
            weight_batch_logs_kernel<<<dim3(blocks(maximum_q),problems),threads,0,c.stream>>>(items,count);
            c.launched();
            weight_batch_pair_kernel<<<dim3(blocks(c.data.mutations),problems),threads,0,c.stream>>>(items,count);
            c.launched();
        }
        weight_batch_leaves_kernel<<<dim3(static_cast<unsigned>(std::min<i64>(65535,maximum_leaves)),problems),threads,0,c.stream>>>(
            items,count);
        c.launched();
        weight_batch_combine_kernel<<<dim3(static_cast<unsigned>(std::min<i64>(65535,maximum_jobs)),problems),threads,0,c.stream>>>(
            items,count);
        c.launched();
        weight_batch_finish_kernel<<<dim3(blocks(add(maximum_q,2)),problems),threads,0,c.stream>>>(
            items,count,device_output);
        c.launched();
    }
    c.download(host+output_offset,device+output_offset,multiply(results,8)); c.finish();
    std::memcpy(output,host+output_offset,static_cast<std::size_t>(results*8));
    result_offset=0;
    for(i64 index=0;index<count;++index) {
        const auto& request=requests[index]; auto& p=scoring_problem(c,request.handle);
        if(request.operation==0) {
            output[result_offset]=-output[result_offset]/p.n;
            for(i64 k=0;k<p.q;++k) output[result_offset+k+2]=-output[result_offset+k+2]/p.n;
            ++scoring_state(c).counters[4]; result_offset+=p.q+2;
        } else {
            output[result_offset++]/=p.n;
            if(request.operation==1) { p.pair_ready=true; ++scoring_state(c).counters[5]; }
            ++scoring_state(c).counters[6];
        }
    }
    lease.complete=true;
}
