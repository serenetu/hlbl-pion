#pragma once

#include <qlat/qlat.h>

#include <cuba.h>

#include <vector>
#include <cstring>
#include <cassert>

template <class F>
inline int cubaFunction(const int* ndim, const double x[],
  const int* ncomp, double f[], void* userdata)
{
  std::vector<double> vx(*ndim, 0.0);
  std::memcpy(vx.data(), x, *ndim * sizeof(double));
  std::vector<double> vf((*((const F*)userdata))(vx));
  assert(vf.size() == (size_t)*ncomp);
  std::memcpy(f, vf.data(), *ncomp * sizeof(double));
  return 0;
}

template <class F>
void integrateDivonne(
    std::vector<double>& integral, std::vector<double>& error, std::vector<double>& prob,
    int& nregions, int& neval, int& fail,
    const int ndim, const int ncomp, const F& f,
    const double epsabs = 0.0, const double epsrel = 1.0e-5,
    const int flags = 0, const int seed = 23,
    const int mineval = 128, const int maxeval = 16 * 1024 * 1024 * 4,
    const int key1 = 7, const int key2 = 7, const int key3 = 1, const int maxpass = 5,
    const double border = 0.0, const double maxchisq = 10.0, const double mindeviation = 0.25,
    const double ngiven = 0, int ldxgiven = 0, double xgiven[] = NULL,
    const int nextra = 0, peakfinder_t peakfinder = NULL,
    const char* statefile = NULL, void* spin = NULL)
{
  TIMER("integrateDivonne");
  if (0 == ldxgiven) {
    ldxgiven = ndim;
  }
  integral.resize(ncomp);
  error.resize(ncomp);
  prob.resize(ncomp);
  cubacores(0, 0);
  Divonne(ndim, ncomp, cubaFunction<F>, (void*)&f, 1,
      epsrel, epsabs, flags, seed, mineval, maxeval,
      key1, key2, key3, maxpass, border, maxchisq, mindeviation,
      ngiven, ldxgiven, xgiven, nextra, peakfinder, statefile, spin,
      &nregions, &neval, &fail,
      integral.data(), error.data(), prob.data());
  // DisplayInfo("", fname, "nregions=%d ; neval=%d ; fail=%d\n", nregions, neval, fail);
  // for (int i = 0; i < ncomp; ++i) {
  //   DisplayInfo("", fname, "i=%d integral=%23.16e ; error=%23.16e ; fail=%f\n", i, integral[i], error[i], prob[i]);
  // }
}

template <class F>
void integrateCuhre(
    std::vector<double>& integral, std::vector<double>& error, std::vector<double>& prob,
    int& nregions, int& neval, int& fail,
    const int ndim, const int ncomp, const F& f,
    const double epsabs = 0.0, const double epsrel = 1.0e-5,
    const int flags = 0, const int mineval = 128, const int maxeval = 16 * 1024 * 1024 * 4,
    const int key = 7, const char* statefile = NULL, void* spin = NULL)
{
  TIMER("integrateCuhre");
  integral.resize(ncomp);
  error.resize(ncomp);
  prob.resize(ncomp);
  cubacores(0, 0);
  Cuhre(ndim, ncomp, cubaFunction<F>, (void*)&f, 1,
      epsrel, epsabs, flags, mineval, maxeval,
      key, statefile, spin,
      &nregions, &neval, &fail,
      integral.data(), error.data(), prob.data());
  // DisplayInfo("", fname, "nregions=%d ; neval=%d ; fail=%d\n", nregions, neval, fail);
  // for (int i = 0; i < ncomp; ++i) {
  //   DisplayInfo("", fname, "i=%d integral=%23.16e ; error=%23.16e ; fail=%f\n", i, integral[i], error[i], prob[i]);
  // }
}

inline std::vector<double> test_integrand4d(const std::vector<double>& vx)
{
  // TIMER_VERBOSE("test_integrand4d");
  using namespace qlat;
  assert(4 == vx.size());
  std::vector<double> ans(1);
  ans[0] = vx[0] * vx[1] * vx[2] * vx[3] + sin(vx[0] * PI) * sin(vx[1] * PI) + sqrt(vx[3]) * sqrt(vx[2]) * sqrt(vx[1]);
  // DisplayInfo("", fname, "%f %f %f %f %f\n", ans[0], vx[0], vx[1], vx[2], vx[3]);
  return ans;
}

inline void test_integrationMultidimensional()
{
  using namespace qlat;
  TIMER_VERBOSE("test_integrationMultidimensional");
  std::vector<double> integral, error, prob;
  int nregions, neval, fail;
  integrateCuhre(integral, error, prob, nregions, neval, fail, 4, 1, test_integrand4d);
  fdisplayln(stdout, ssprintf("%f %f %f", integral[0], error[0], prob[0]));
  integrateDivonne(integral, error, prob, nregions, neval, fail, 4, 1, test_integrand4d);
  fdisplayln(stdout, ssprintf("%f %f %f", integral[0], error[0], prob[0]));
}
