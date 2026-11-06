// Compact proposal summaries from live immutable columns. Floating-point work
// is subtraction only; ordering, support legality and scientific ties stay host-side.
__global__ void subset_columns_kernel(const double* const* columns,const i64* rows,
                                     i64 count,i64 q,double* output) {
    for(i64 item=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;item<count*q;
        item+=static_cast<i64>(gridDim.x)*blockDim.x)
        output[item]=columns[item%q][rows[item/q]];
}
__global__ void proposal_counts_kernel(const double* const* columns,const i64* labels,
        i64 first,i64 count,i64 q,i64* counts,unsigned long long* header) {
    for(i64 local=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;local<count;
        local+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const i64 row=first+local;
        const double assigned=columns[labels[local]][row];
        if(!isfinite(assigned)) atomicOr(header+1,1ULL);
        i64 positive=0;
        for(i64 k=0;k<q;++k) {
            const double value=columns[k][row],gain=value-assigned;
            if(isnan(value)||value==positive_infinity) atomicOr(header+1,2ULL);
            positive+=isfinite(gain)&&gain>0.;
        }
        counts[local]=positive;
    }
}
__global__ void proposal_prefix_kernel(i64 count,i64* offsets,unsigned long long* header) {
    // Integer-only stable compaction; rows and destinations keep NumPy nonzero
    // order, including exact gain ties with identical scientific group keys.
    if(!threadIdx.x) {
        i64 total=0;
        for(i64 row=0;row<count;++row) { const i64 size=offsets[row]; offsets[row]=total; total+=size; }
        header[0]=static_cast<unsigned long long>(total);
    }
}
__global__ void proposal_scatter_kernel(const double* const* columns,const i64* labels,
        i64 first,i64 count,i64 q,const i64* offsets,i64* indices,double* gains) {
    for(i64 local=static_cast<i64>(blockIdx.x)*blockDim.x+threadIdx.x;local<count;
        local+=static_cast<i64>(gridDim.x)*blockDim.x) {
        const i64 row=first+local;
        const double assigned=columns[labels[local]][row];
        i64 offset=offsets[local];
        for(i64 k=0;k<q;++k) {
            const double gain=columns[k][row]-assigned;
            if(isfinite(gain)&&gain>0.) { indices[offset]=row*q+k; gains[offset++]=gain; }
        }
    }
}

template<class Function> void proposal_inputs(Context& c,i64 q,const i64* handles,
        i64 count,const i64* rows,i64 row_limit,Function function) {
    if(q<=0||count<0||!handles||(count&&!rows)) throw std::domain_error("Invalid CUDA proposal input");
    for(i64 i=0;i<count;++i) if(rows[i]<0||rows[i]>=row_limit)
        throw std::domain_error("CUDA proposal row/label outside range");
    ScoringHostScope metadata(c,add(512,multiply(q,64)));
    const auto columns=get_columns(c,q,handles);
    ScoringRequestLease request(c,add(q,count),true);
    auto* host=reinterpret_cast<unsigned char*>(request.pointers());
    for(i64 k=0;k<q;++k) request.pointers()[k]=columns[k]->data();
    std::memcpy(host+q*8,rows,static_cast<std::size_t>(count*8));
    auto* device=reinterpret_cast<unsigned char*>(request.device_pointers());
    c.upload(device,host,multiply(add(q,count),8));
    function(reinterpret_cast<const double* const*>(device),reinterpret_cast<i64*>(device+q*8));
    request.complete=true;
}

void gather_column_rows(Context& c,i64 q,const i64* handles,i64 count,const i64* rows,double* output) {
    if(count<0||(count&&!output)) throw std::domain_error("Invalid CUDA subset destination");
    if(!count) return;
    c.reserve(multiply(multiply(count,q),8));
    proposal_inputs(c,q,handles,count,rows,c.data.mutations,[&](const double* const* columns,const i64* selected) {
        subset_columns_kernel<<<blocks(multiply(count,q)),threads,0,c.stream>>>(
            columns,selected,count,q,c.pointer<double>(0));
        c.launched(); c.download(output,c.pointer<double>(0),multiply(count,q)); c.finish();
    });
    ++scoring_state(c).proposal_counters[2];
    scoring_state(c).proposal_counters[3]=add(scoring_state(c).proposal_counters[3],multiply(count,q));
}

void positive_transfer_chunk(Context& c,i64 q,const i64* handles,i64 first,i64 count,
        const i64* labels,i64 capacity,i64* indices,double* gains,i64* returned) {
    if(first<0||count<=0||count>c.data.mutations||first>c.data.mutations-count||q<=0||
            !returned||!indices||!gains||capacity<multiply(count,q))
        throw std::domain_error("Invalid CUDA positive-transfer chunk");
    *returned=0;
    Layout layout;
    const i64 header=layout.take<unsigned long long>(2),offsets=layout.take<i64>(count);
    const i64 index_offset=layout.take<i64>(multiply(count,q));
    const i64 gain_offset=layout.take<double>(multiply(count,q));
    c.reserve(layout.bytes);
    unsigned long long result[2]{};
    ScoringStreamFence fence(c);
    proposal_inputs(c,q,handles,count,labels,q,[&](const double* const* columns,const i64* assignments) {
        cuda_check(cudaMemsetAsync(c.pointer<unsigned long long>(header),0,16,c.stream),"clear proposal status");
        proposal_counts_kernel<<<blocks(count),threads,0,c.stream>>>(columns,assignments,first,count,q,
            c.pointer<i64>(offsets),c.pointer<unsigned long long>(header)); c.launched();
        proposal_prefix_kernel<<<1,threads,0,c.stream>>>(count,c.pointer<i64>(offsets),
            c.pointer<unsigned long long>(header)); c.launched();
        c.download(result,c.pointer<unsigned long long>(header),2); c.finish();
        if(result[1]&2) throw std::domain_error("Proposal likelihood columns must be finite or -inf");
        if(result[1]&1) throw std::domain_error("A reassignment parent has inadmissible assigned centers");
        require(result[0]<=static_cast<unsigned long long>(capacity),"CUDA proposal compaction exceeds capacity");
        if(result[0]) {
            proposal_scatter_kernel<<<blocks(count),threads,0,c.stream>>>(columns,assignments,first,count,q,
                c.pointer<i64>(offsets),c.pointer<i64>(index_offset),c.pointer<double>(gain_offset)); c.launched();
            c.download(indices,c.pointer<i64>(index_offset),result[0]);
            c.download(gains,c.pointer<double>(gain_offset),result[0]); c.finish();
        }
        *returned=static_cast<i64>(result[0]);
    });
    ++scoring_state(c).proposal_counters[0];
    scoring_state(c).proposal_counters[1]=add(scoring_state(c).proposal_counters[1],*returned);
}
