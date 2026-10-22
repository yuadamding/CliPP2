#pragma once
#include "backend.h"
#include "validation.h"
#include "refit_validation.h"
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <vector>

// Resident API v1 is an ordered synchronous scheduler on BOTH backends. It
// establishes ownership, not asynchronous CUDA execution: submit completes
// device work/transfers; collect copies a retained host result. The backend
// context must outlive its engine. No CUDA headers enter this public interface.
namespace clipp2_resident {
using Handle = std::uint64_t;
inline std::int64_t sum(std::initializer_list<std::int64_t> values) {
    std::int64_t total=0;
    for (auto value:values) {
        clipp2::require(value>=0 && value<=INT64_MAX-total,"Resident byte count overflow"); total+=value;
    }
    return total;
}
inline std::int64_t bytes(std::int64_t count, std::int64_t width) {
    return clipp2::checked_product(count,width,1);
}
struct Request { Handle plan; const double* incumbents; };
struct Budget {
    std::int64_t used = 0, limit;
    explicit Budget(std::int64_t value) : limit(value) {}
    void acquire(std::int64_t bytes) {
        clipp2::require(bytes >= 0 && bytes <= limit - used, "Resident host memory cap exceeded");
        used += bytes;
    }
};
struct Object {
    std::shared_ptr<Budget> budget;
    std::int64_t bytes;
    Object(std::shared_ptr<Budget> owner, std::int64_t count) : budget(std::move(owner)), bytes(count) {
        budget->acquire(bytes);
    }
    virtual ~Object() { budget->used -= bytes; }
};
struct Plan final : Object {
    std::vector<std::int64_t> regions, offsets, rows, grid_offsets;
    std::vector<double> grid, lower, upper, tolerance;
    Plan(std::shared_ptr<Budget> budget, const clipp2::RefitBatch& b)
        : Object(std::move(budget), sum({clipp2_resident::bytes(b.tasks,48),16,
                 clipp2_resident::bytes(b.members,8),clipp2_resident::bytes(b.grid_points,8),512})),
          regions(b.regions, b.regions+b.tasks), offsets(b.member_offsets, b.member_offsets+b.tasks+1),
          rows(b.member_rows, b.member_rows+b.members), grid_offsets(b.grid_offsets, b.grid_offsets+b.tasks+1),
          grid(b.grid, b.grid+b.grid_points), lower(b.lower, b.lower+b.tasks),
          upper(b.upper, b.upper+b.tasks), tolerance(b.xatol, b.xatol+b.tasks) {}
    clipp2::RefitBatch batch(const double* incumbents) const {
        return {static_cast<std::int64_t>(regions.size()), static_cast<std::int64_t>(rows.size()),
                static_cast<std::int64_t>(grid.size()), regions.data(), offsets.data(), rows.data(),
                grid_offsets.data(), grid.data(), lower.data(), upper.data(), tolerance.data(), incumbents};
    }
};
struct Job final : Object {
    std::vector<std::shared_ptr<Plan>> plans;
    std::vector<clipp2::RefitResult> results;
    Job(std::shared_ptr<Budget> budget, std::int64_t count, std::int64_t requests)
        : Object(std::move(budget), sum({clipp2_resident::bytes(count,sizeof(clipp2::RefitResult)),
                 clipp2_resident::bytes(requests,sizeof(std::shared_ptr<Plan>)),256})), results(count) {
        plans.reserve(requests);
    }
};
struct Columns final : Object {
    std::vector<double> values;
    std::vector<std::uint8_t> status;
    Columns(std::shared_ptr<Budget> budget, std::int64_t count)
        : Object(std::move(budget), sum({clipp2_resident::bytes(count,9),256})), values(count), status(count) {}
};
struct Slot { std::uint32_t generation = 1; std::shared_ptr<Object> value; };
struct Engine {
    void* backend;
    std::int64_t n, r;
    std::shared_ptr<Budget> budget;
    std::vector<Slot> slots;
    std::mutex mutex;
    Engine(void* context, std::int64_t rows, std::int64_t regions, std::int64_t limit)
        : backend(context), n(rows), r(regions), budget(std::make_shared<Budget>(limit)) {}
    Handle put(std::shared_ptr<Object> object) {
        for (std::size_t i=0; i<slots.size(); ++i)
            if (!slots[i].value && slots[i].generation != 0) {
                slots[i].value = std::move(object);
                return (Handle(slots[i].generation)<<32) | Handle(i+1);
            }
        clipp2::require(slots.size() < UINT32_MAX, "Resident handle space exhausted");
        // Reserve amortized vector growth and control metadata conservatively.
        budget->acquire(2*sizeof(Slot));
        try { slots.push_back({1, std::move(object)}); }
        catch (...) { budget->used -= 2*sizeof(Slot); throw; }
        return (Handle(1)<<32) | Handle(slots.size());
    }
    Slot& slot(Handle handle) {
        const auto index = static_cast<std::uint32_t>(handle);
        clipp2::require(index && index <= slots.size(), "Unknown resident handle");
        auto& value = slots[index-1];
        clipp2::require(value.value && value.generation == handle>>32, "Stale resident handle");
        return value;
    }
    template<class T> std::shared_ptr<T> get(Handle handle) {
        auto result = std::dynamic_pointer_cast<T>(slot(handle).value);
        clipp2::require(bool(result), "Wrong resident handle type"); return result;
    }
};
inline thread_local char error[1024] = "";
template<class F> int checked(Engine* e, F operation) {
    error[0] = 0;
    try {
        clipp2::require(e != nullptr, "Null resident engine");
        std::lock_guard<std::mutex> lock(e->mutex); operation(); return 0;
    } catch (const std::exception& failure) { std::snprintf(error, sizeof(error), "%s", failure.what()); }
    catch (...) { std::snprintf(error, sizeof(error), "Unknown resident execution failure"); }
    return 1;
}
inline void backend_check(int code) {
    if (code) throw std::runtime_error(CliPP2BackendLastError());
}
}

extern "C" {
int CliPP2ResidentABI() { return 1; }
const char* CliPP2ResidentLastError() { return clipp2_resident::error; }
int CliPP2ResidentCreate(void* backend, std::int64_t n, std::int64_t r,
                          std::int64_t host_bytes, clipp2_resident::Engine** output) {
    if (output) *output = nullptr;
    try {
        clipp2::require(backend && output && n>0 && r>0 && host_bytes>=0, "Invalid resident engine configuration");
        std::int64_t actual_n=0,actual_r=0;
        clipp2_resident::backend_check(CliPP2BackendDimensions(backend,&actual_n,&actual_r));
        clipp2::require(n==actual_n && r==actual_r,"Resident dimensions do not match backend context");
        *output = new clipp2_resident::Engine(backend,n,r,host_bytes); return 0;
    } catch (const std::exception& e) { std::snprintf(clipp2_resident::error,sizeof(clipp2_resident::error),"%s",e.what()); }
    catch (...) { std::snprintf(clipp2_resident::error,sizeof(clipp2_resident::error),"Resident allocation failure"); }
    return 1;
}
void CliPP2ResidentDestroy(clipp2_resident::Engine* e) { delete e; }
int CliPP2ResidentBytes(clipp2_resident::Engine* e, std::int64_t* output) {
    return clipp2_resident::checked(e,[&] { clipp2::require(output,"Null resident bytes output"); *output=e->budget->used; });
}
int CliPP2ReleaseHandle(clipp2_resident::Engine* e, clipp2_resident::Handle handle) {
    return clipp2_resident::checked(e,[&] { auto& slot=e->slot(handle); slot.value.reset(); ++slot.generation; });
}
int CliPP2PrepareCoordinates(clipp2_resident::Engine* e, const clipp2::RefitBatch* batch,
                               clipp2_resident::Handle* output) {
    return clipp2_resident::checked(e,[&] {
        clipp2::require(batch && output && batch->tasks>0, "Invalid resident coordinate batch");
        // Existing validation checks CSR, grids and identities before copying.
        // Bounds remain in the plan; backend execution independently checks the
        // all-member box against authoritative prepared observations.
        clipp2::ObservationSoA shape{}; shape.mutations=e->n; shape.regions=e->r;
        clipp2::RefitResult sentinel{};
        clipp2::validate_refit_batch(shape,*batch,&sentinel,false);
        *output=e->put(std::make_shared<clipp2_resident::Plan>(e->budget,*batch));
    });
}
int CliPP2SubmitRefits(clipp2_resident::Engine* e, const clipp2_resident::Request* requests,
                         std::int64_t count, clipp2_resident::Handle* output) {
    return clipp2_resident::checked(e,[&] {
        clipp2::require(count>0 && requests && output,"Invalid resident refit requests");
        std::int64_t total=0;
        for (std::int64_t i=0;i<count;++i) {
            const auto size=e->get<clipp2_resident::Plan>(requests[i].plan)->regions.size();
            clipp2::require(size <= static_cast<std::size_t>(INT64_MAX-total),"Resident result overflow"); total+=size;
        }
        auto job=std::make_shared<clipp2_resident::Job>(e->budget,total,count);
        std::int64_t offset=0;
        for (std::int64_t i=0;i<count;++i) {
            auto plan=e->get<clipp2_resident::Plan>(requests[i].plan);
            auto batch=plan->batch(requests[i].incumbents);
            clipp2_resident::backend_check(CliPP2RefitCentersBatch(e->backend,&batch,job->results.data()+offset));
            offset+=batch.tasks; job->plans.push_back(std::move(plan));
        }
        *output=e->put(std::move(job));
    });
}
int CliPP2CollectRefits(clipp2_resident::Engine* e, clipp2_resident::Handle handle,
                          std::int64_t count, clipp2::RefitResult* output) {
    return clipp2_resident::checked(e,[&] {
        auto job=e->get<clipp2_resident::Job>(handle);
        clipp2::require(output && count==static_cast<std::int64_t>(job->results.size()),"Resident result size mismatch");
        std::copy(job->results.begin(),job->results.end(),output);
    });
}
int CliPP2CreateJointColumns(clipp2_resident::Engine* e, std::int64_t count,
                               const double* centers, clipp2_resident::Handle* output) {
    return clipp2_resident::checked(e,[&] {
        clipp2::require(count>0 && centers && output,"Invalid resident center batch");
        auto columns=std::make_shared<clipp2_resident::Columns>(e->budget,clipp2::checked_product(count,e->n));
        clipp2_resident::backend_check(CliPP2EvaluateJointColumns(e->backend,count,centers,
                                      columns->values.data(),columns->status.data()));
        *output=e->put(std::move(columns));
    });
}
int CliPP2CollectJointColumns(clipp2_resident::Engine* e, clipp2_resident::Handle handle,
                                std::int64_t count, double* values, std::uint8_t* status) {
    return clipp2_resident::checked(e,[&] {
        auto columns=e->get<clipp2_resident::Columns>(handle);
        clipp2::require(values && status && count==static_cast<std::int64_t>(columns->values.size()),"Resident column size mismatch");
        std::copy(columns->values.begin(),columns->values.end(),values);
        std::copy(columns->status.begin(),columns->status.end(),status);
    });
}
}
