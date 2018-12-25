#pragma once

#include "utils.h"
#include "root-fsolver.h"
#include "interpolation.h"
#include "compute-f.h"
#include "compute-int-seq.h"

#include <qlat/qlat.h>

#include <cmath>
#include <cstdlib>
#include <cassert>

inline std::vector<double> vxFromParam(const qlat::CoordinateD& param)
{
  using namespace qlat;
  std::vector<double> vx(4);
  const double rp = param[0];
  const double v = param[1];
  const double u = param[2];
  const double phi = param[3];
  assert(0.0 <= rp && rp < 1.0);
  assert(0.0 <= v && v <= PI);
  assert(-1.0 <= u && u <= 1.0);
  assert(-PI <= phi && phi <= PI);
  vx[0] = rp;
  vx[1] = v / PI;
  vx[2] = (u + 1.0) / 2.0;
  vx[3] = (phi + PI) / (2.0 * PI);
  return vx;
}

inline qlat::CoordinateD paramFromVx(const std::vector<double>& vx)
{
  using namespace qlat;
  assert(0.0 <= vx[0] && vx[0] <= 1.0);
  assert(0.0 <= vx[1] && vx[1] <= 1.0);
  assert(0.0 <= vx[2] && vx[2] <= 1.0);
  assert(0.0 <= vx[3] && vx[3] <= 1.0);
  const double rp = vx[0];
  const double v = vx[1] * PI;
  const double u = vx[2] * 2.0 - 1.0;
  const double phi = vx[3] * 2.0 * PI - PI;
  const CoordinateD param(rp, v, u, phi);
  return param;
}

inline void recordMuonIntegrand(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const std::vector<double>& vx, const std::vector<double>& ans)
{
  qshow::displayln(qshow::ssprintf("recordMuonCoordinate[%s,%s]", qshow::show(x).c_str(), qshow::show(y).c_str()));
  qshow::display(qshow::ssprintf("recordMuonIntegrand "));
  for (size_t i = 0; i < vx.size(); ++i) {
    qshow::display(qshow::ssprintf("%s ", qshow::show(vx[i]).c_str()));
  }
  qshow::display(" : ");
  for (size_t i = 0; i < ans.size(); ++i) {
    qshow::display(qshow::ssprintf(" %s", qshow::show(ans[i]).c_str()));
  }
  qshow::displayln("");
}

struct MuonLineIntCompact
{
  MuonLineIntegrand mli;
  //
  void init()
  {
    mli.init();
  }
  void init(const qlat::CoordinateD& x, const qlat::CoordinateD& y)
  {
    mli.init(x, y);
  }
  //
  std::vector<double> operator()(const std::vector<double>& vx) const
  {
    // TIMER_VERBOSE("MuonLineIntCompact");
    using namespace qlat;
    const double ratio = PI / 2.0 * 2.0 * 2.0 * PI; // corresponding to volume
    std::vector<double> ans(mli(paramFromVx(vx)));
    for (size_t i = 0; i < ans.size(); ++i) {
      ans[i] *= ratio;
    }
    // recordMuonIntegrand(mli.x, mli.y, vx, ans);
    return ans;
  }
};

inline std::vector<double> integrateMuonLine(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  TIMER("integrateMuonLine");
  MuonLineIntCompact mlic;
  mlic.init(x, y);
  const std::vector<double> xVx(vxFromParam(paramFromEta(x, mlic.mli.r0)));
  const std::vector<double> yVx(vxFromParam(paramFromEta(y, mlic.mli.r0)));
  // qshow::displayln(qshow::ssprintf("x =%s", qshow::show(x).c_str()));
  // qshow::displayln(qshow::ssprintf("y =%s", qshow::show(y).c_str()));
  // qshow::displayln(qshow::ssprintf("integrateMuonLine:x:(%15.5E) %25.17E %25.17E %25.17E %25.17E",
  //      coordinateLen(x), xVx[0], xVx[1], xVx[2], xVx[3]));
  // qshow::displayln(qshow::ssprintf("integrateMuonLine:y:(%15.5E) %25.17E %25.17E %25.17E %25.17E",
  //      coordinateLen(y), yVx[0], yVx[1], yVx[2], yVx[3]));
  std::vector<double> integral, error, prob;
  const int ncomp = 25;
  int nregions, neval, fail;
  // ADJUST ME
  integrateCuhre(
      integral, error, prob,
      nregions, neval, fail,
      4, ncomp, mlic, epsabs, epsrel,
      0, 1024 * 1024, 1024 * 1024 * 1024);
  // qshow::displayln(qshow::ssprintf("%s: nregions=%d neval=%d (%.5lf %%) fail=%d", fname, nregions, neval, 100 * (double)neval/(double)(1024 * 1024 * 1024), fail));
  return integral;
}

inline void profile_computeIntMult()
{
  using namespace qlat;
  TIMER_VERBOSE_FLOPS("profile_computeIntMult");
  RngState rs("profile_computeIntMult");
  const int size = 4;
  const double low = -3.0;
  const double high = 3.0;
  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    CoordinateD x, y;
    for (int m = 0; m < 4; ++m) {
      x[m] = uRandGen(rs, high, low);
      y[m] = uRandGen(rs, high, low);
    }
    DisplayInfo("", fname, "x =%s\n", qshow::show(x).c_str());
    DisplayInfo("", fname, "y =%s\n", qshow::show(y).c_str());
    sum += integrateMuonLine(x, y)[24];
  }
  DisplayInfo("", fname, "sum=%23.16e\n", sum);
  timer.flops += size;
}

inline void test_computeIntMult()
{
  using namespace qlat;
  TIMER_VERBOSE("test_computeIntMult");
  const CoordinateD x(0.1, 0.2, 0.0, 0.5);
  const CoordinateD y(0.3, 0.0, -0.2, 0.1);
  MuonLineIntCompact mlic;
  mlic.init(x, y);
  DisplayInfo("", fname, "param=%s\n", qshow::show(paramFromEta(x + y, mlic.mli.r0)).c_str());
  DisplayInfo("", fname, "param=%s\n", qshow::show(paramFromVx(vxFromParam(paramFromEta(x + y, mlic.mli.r0)))).c_str());
  std::vector<double> vx(4);
  vx[0] = 0.1;
  vx[1] = 0.2;
  vx[2] = 0.3;
  vx[3] = 0.4;
  // DisplayInfo("", fname, "MuonLineIntCompact=%23.16e\n", mlic(vx)[0]);
  integrateMuonLine(x,y);
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.0001), CoordinateD(0.0, 0.0, 0.0, 0.0001));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.001), CoordinateD(0.0, 0.0, 0.0, 0.001));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.01), CoordinateD(0.0, 0.0, 0.0, 0.01));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.1), CoordinateD(0.0, 0.0, 0.0, 0.1));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.2), CoordinateD(0.0, 0.0, 0.0, 0.2));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.3), CoordinateD(0.0, 0.0, 0.0, 0.3));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.4), CoordinateD(0.0, 0.0, 0.0, 0.4));
  // integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, -0.1), CoordinateD(0.0, 0.0, 0.0, 1.0));
  // for (int i = -3; i <= 3; ++i) {
  //   for (int j = -3; j <= 3; ++j) {
  //     const double xt = 30.0 * i;
  //     const double yt = 30.0 * j;
  //     const double v = integrateMuonLine(CoordinateD(0.0, 0.0, 0.0, xt), CoordinateD(0.0, 0.0, 0.0, yt));
  //     DisplayInfo("", fname, "TABLE %5f %5f %23.16e\n", xt, yt, v);
  //   }
  // }
  profile_computeIntMult();
}
