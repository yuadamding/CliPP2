#pragma once
#include "views.h"
#include <limits>
#include <stdexcept>
#include <vector>
namespace clipp2 {
// Host-owned ragged workspace. Prepared once per context/continuation arena;
// numerical iterations only reuse its buffers.
struct JointWorkspace {
    std::vector<std::int64_t> offsets,rows;
    std::vector<double> values;
    std::int64_t regions=0;
    static std::int64_t add(std::int64_t a,std::int64_t b) {
        if(a<0||b<0||a>std::numeric_limits<std::int64_t>::max()-b)
            throw std::overflow_error("Joint workspace offset overflow");
        return a+b;
    }
    static std::int64_t size(std::int64_t a,std::int64_t b=1) {
        const auto limit=std::numeric_limits<std::int64_t>::max()/sizeof(double);
        if(a<0||b<0||(b&&static_cast<std::uint64_t>(a)>limit/static_cast<std::uint64_t>(b)))
            throw std::overflow_error("Joint workspace byte-size overflow");
        return a*b;
    }
    void prepare(const ObservationSoA& data,std::int64_t batches=1) {
        regions=data.regions;
        if(offsets.empty()) {
            offsets.resize(size(data.mutations,add(regions,1)));rows.resize(size(add(data.mutations,1)),0);
            for(std::int64_t i=0;i<data.mutations;++i) {
                auto out=offsets.data()+i*(regions+1);out[0]=0;
                for(std::int64_t r=0;r<regions;++r)out[r+1]=add(out[r],data.support[r*data.mutations+i]);
                rows[i+1]=add(rows[i],add(out[regions],size(4,add(regions,1))));
            }
        }
        const auto count=size(rows.back(),batches);
        if(values.size()<static_cast<std::size_t>(count))values.resize(count);
    }
    const std::int64_t* component_offsets(std::int64_t row)const{return offsets.data()+row*(regions+1);}
    double* cavity(std::int64_t row,std::int64_t batch=0){return values.data()+batch*rows.back()+rows[row];}
    double* messages(std::int64_t row,std::int64_t batch=0){return cavity(row,batch)+component_offsets(row)[regions];}
};
}
