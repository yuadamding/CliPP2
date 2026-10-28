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
    i64 n,q,operation,grow,shrink,output;
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
__global__ void weight_batch_gather_kernel(const WeightBatchItem* items,i64 count,double* output) {
    for(i64 index=blockIdx.x;index<count;index+=gridDim.x) {
        const auto& p=items[index];
        if(p.small) continue;
        const i64 size=p.operation==0?p.q+2:1;
        for(i64 k=threadIdx.x;k<size;k+=blockDim.x) output[p.output+k]=p.sums[k];
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
    bool has_small=false,has_general=false;
    for(i64 index=0;index<count;++index) {
        const auto& request=requests[index];
        auto& p=scoring_problem(c,request.handle);
        WeightBatchItem item{};
        item.columns=p.at<const double*>(p.pointers); item.offsets=p.at<double>(p.offset);
        item.kernel=p.at<double>(p.kernel); item.weights=reinterpret_cast<double*>(device+weights_offset)+cursor;
        item.logs=p.at<double>(p.log_weights); item.other=p.at<double>(p.other);
        item.grow_log=p.at<double>(p.grow_log); item.shrink_log=p.at<double>(p.shrink_log);
        item.numerator=p.at<double>(p.numerator); item.sign=p.at<double>(p.sign); item.sums=p.at<double>(p.sums);
        item.n=p.n; item.q=p.q; item.operation=request.operation;
        item.grow=request.grow; item.shrink=request.shrink; item.output=result_offset;
        item.small=small_mixture(p); has_small=has_small||item.small; has_general=has_general||!item.small;
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
    // General problems initially share the packed transfer/final synchronization;
    // each still uses its exact multi-block reduction plan, not a small-path fallback.
    for(i64 index=0;index<count;++index) {
        const auto& request=requests[index]; auto& p=scoring_problem(c,request.handle);
        if(small_mixture(p)) continue;
        WeightBatchItem item;
        std::memcpy(&item,host+descriptors+index*sizeof(item),sizeof(item));
        if(request.operation==0) {
            mixture_leaf_evaluate_kernel<<<static_cast<unsigned>(std::min<i64>(65535,p.reduction.leaf_count)),threads,0,c.stream>>>(
                item.columns,item.offsets,item.kernel,item.weights,p.n,p.q,p.reduction.full_chunks,
                p.at<ScoringLeaf>(p.leaf_metadata),p.reduction.leaf_count,p.at<double>(p.leaf_partials));
            c.launched(); reduce_mixture(p,add(p.q,2));
        } else {
            if(request.operation==1) {
                mixture_log_weights_kernel<<<blocks(p.q),threads,0,c.stream>>>(item.weights,p.q,item.logs);
                c.launched();
                mixture_pair_kernel<<<blocks(p.n),threads,0,c.stream>>>(item.columns,item.offsets,item.logs,
                    p.n,p.q,item.grow,item.shrink,item.other,item.grow_log,item.shrink_log,item.numerator,item.sign);
                c.launched();
            }
            mixture_leaf_trial_kernel<<<static_cast<unsigned>(std::min<i64>(65535,p.reduction.leaf_count)),threads,0,c.stream>>>(
                p.n,p.reduction.full_chunks,p.at<ScoringLeaf>(p.leaf_metadata),p.reduction.leaf_count,
                item.other,item.grow_log,item.shrink_log,item.numerator,item.sign,
                item.grow_weight,item.amount,item.fraction,p.at<double>(p.leaf_partials));
            c.launched(); reduce_mixture(p,1);
        }
    }
    if(has_general) {
        weight_batch_gather_kernel<<<static_cast<unsigned>(std::min<i64>(65535,count)),threads,0,c.stream>>>(
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
