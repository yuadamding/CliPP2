#pragma once
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define CLIPP_HD __host__ __device__
#else
#define CLIPP_HD
#endif

#if defined(_OPENMP)
#define CLIPP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#else
#define CLIPP_PARALLEL_FOR
#endif

namespace clipp2 {
// Value-only constexpr constants are valid in CUDA device expressions; direct
// std::numeric_limits calls are host-only with CUDA 12.2's GCC 8 headers.
constexpr double maximum_finite = std::numeric_limits<double>::max();
constexpr double positive_infinity = std::numeric_limits<double>::infinity();
constexpr double quiet_nan = std::numeric_limits<double>::quiet_NaN();
constexpr int backend_abi = 5;
constexpr const char* numerical_policy = "compact_fp64_ordered_emission_v1";
constexpr double mixed_epsilon = 1e-6;
constexpr double curvature_floor = 1e-8;
constexpr std::uint8_t observed_flag = 1;
constexpr std::uint8_t mixed_flag = 2;
enum class EmitMode : std::int32_t { LossOnly = 0, LossGradientCurvature = 1 };
enum class EmitStatus : std::uint8_t {
    Ok = 0, ZeroLikelihood = 1, InvalidInput = 2, NumericalFailure = 3
};
}
