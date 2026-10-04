// CliPP2-owned chain kernel, derived from CliPP1.5 9e51e30ac8b0c8c9ab1f46821b496909843bda0c.
// AGPL-3.0; original attribution and local changes are recorded in the project NOTICE.
#pragma once

#include <cmath>
#include <functional>
#include <vector>

namespace clipp2kernel {

constexpr int kMaxClusters = 10;
constexpr int kDiagnosticCount = 19;
constexpr double kChainLowerCCF = 1e-8;
constexpr double kChainUpperCCF = 1.0-1e-8;
constexpr int kChainLevels = 20;
constexpr int kChainIterationsPerLevel = 300;
constexpr double kChainStationarityTolerance = 1e-6;
constexpr double kChainConstraintTolerance = 1e-6;
constexpr double kChainMaximumRho = 1e12;

struct MultiplicityLikelihood {
    double nll;
    double gradient;
    double curvature;
};

// x is CCF, so p_m = x * purity * m / (2*(1-purity)+purity*total_cn).
// Curvature is the positive posterior mean complete-data observed curvature;
// the actual marginalized objective is checked by line search.
inline MultiplicityLikelihood multiplicity_likelihood_x(
    double x, double r, double n, int major, double scale, double log_choose)
{
    double maximum = -INFINITY;
    for(int m = 1; m <= major; ++m){
        const double a = std::fmin(1.0, scale * m);
        const double p = std::fmin(1.0, x * a);
        const double log_p = x > 0.0 ? std::fmin(0.0,std::log(x) + std::log(scale) + std::log(double(m))) : -INFINITY;
        const double ell = (r > 0.0 ? r * log_p : 0.0) + (n > r ? (n-r) * std::log1p(-p) : 0.0);
        maximum = std::fmax(maximum, ell);
    }
    if(!std::isfinite(maximum)) return {INFINITY, 0.0, 1.0};
    double mass = 0.0, gradient = 0.0, curvature = 0.0, boundary_gradient = 0.0;
    bool infinite_boundary_curvature=false;
    for(int m = 1; m <= major; ++m){
        const double a = std::fmin(1.0, scale * m);
        const double p = std::fmin(1.0, x * a);
        const double log_p = x > 0.0 ? std::fmin(0.0,std::log(x) + std::log(scale) + std::log(double(m))) : -INFINITY;
        const double ell = (r > 0.0 ? r * log_p : 0.0) + (n > r ? (n-r) * std::log1p(-p) : 0.0);
        if(!std::isfinite(ell)){
            // A vanishing binomial state can still have a nonzero one-sided
            // derivative. Optimizer iterates stay in the declared interior.
            if(p==1.0 && n-r==1.0){boundary_gradient+=a*std::exp(-maximum);infinite_boundary_curvature=true;}
            if(p==1.0 && n-r==2.0) curvature+=2.0*a*a*std::exp(-maximum);
            continue;
        }
        const double weight = std::exp(ell-maximum);
        const double g = (r > 0.0 ? -r/x : 0.0) + (n > r ? (n-r)*a/(1.0-p) : 0.0);
        const double h = (r > 0.0 ? r/(x*x) : 0.0) + (n > r ? (n-r)*a*a/((1.0-p)*(1.0-p)) : 0.0);
        mass += weight;
        gradient += weight*g;
        curvature += weight*h;
    }
    return {-maximum-std::log(mass)+std::log(double(major))-log_choose,
            (gradient+boundary_gradient)/mass,
            infinite_boundary_curvature?INFINITY:std::fmax(1e-8, curvature/mass)};
}

struct ChainData {
    int count;
    double purity;
    std::vector<int> alt, depth, major;
    std::vector<double> scale, log_choose, pilot_x;
};

using ChainEvaluator = std::function<void(const std::vector<double>&,
    std::vector<double>&, std::vector<double>&, std::vector<double>&)>;

ChainData prepare_chain_inputs(int count, const int* alt, const int* depth,
    const int* major, const int* total, double purity, const double* pilot_cp,
    const int* requested_k, int k_count);

void solve_chain(const ChainData& data, const int* requested_k, int k_count,
    const ChainEvaluator& evaluate, int* labels_out, double* cp_out,
    double* diagnostics_out, int* statuses_out);

void solve_forest(int count, int regions, int edge_count, const int* edges,
    const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics);

} // namespace clipp2kernel

// All output buffers are caller-owned and valid only when the return code is zero.
// Row-major shapes: labels/CP [k_count,count], diagnostics [k_count,19], status [k_count].
// Status: 0 tolerance met, 1 numerical limit, 2 line-search stall, 3 finite budget.
// The callback receives CCF and writes loss, gradient and positive curvature.
using CliPP2Callback = int (*)(int, const double*, double*, double*, double*);
extern "C" {
int CliPP2KernelABI();
const char* CliPP2KernelBuildId();
const char* CliPP2KernelLastError();
int CliPP2SolveChain(int count, const int* alt, const int* depth, const int* major,
    const int* total, double purity, const double* pilot_cp, const int* capacities,
    int k_count, int* labels_out, double* cp_out, double* diagnostics_out,
    int* statuses_out, CliPP2Callback callback);
// One shared forest, row-major [count,regions] matrices and [edge_count,2] edges.
// diagnostics = [total active-set sweeps, regions exhausting their sweep budget].
// Independent componentwise KKT admission remains the Python caller's duty.
int CliPP2ForestQP(int count, int regions, int edge_count, const int* edges,
    const double* curvature, const double* rhs, double penalty,
    const double* lower, const double* upper, double* output, int* diagnostics);
}
