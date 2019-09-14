//#pragma once
#ifndef _MUON-LINE_H
#define _MUON-LINE_H

#include "integration-multidimensional.h"
#include "compute-int-mult.h"
#include "projection.h"
#include "interpolation-bilinear.h"
#include "qlat-analysis.h"

#include <qlat/qlat.h>

#include <cmath>
#include <cstdlib>
#include <cassert>
#include <string>

inline std::vector<std::string> get_lines(std::istream& is)
{
  std::vector<std::string> ret;
  while (is.good()) {
    std::string str;
    std::getline(is, str);
    ret.push_back(str);
  }
  return ret;
}

inline std::vector<std::string> lines(const std::string& s)
{
  std::istringstream is(s);
  return get_lines(is);
}

inline std::string pad_cut_string(const std::string& s, const int width, const char c = ' ')
{
  std::string ret(s);
  ret.resize(width, c);
  return ret;
}

inline std::string compare_multiline_string(const std::string& s1, const std::string& s2, const int width = 80)
{
  std::vector<std::string> ls1(lines(s1)), ls2 (lines(s2));
  std::ostringstream out;
  const size_t size = std::max(ls1.size(), ls2.size());
  ls1.resize(size, "");
  ls2.resize(size, "");
  for (size_t i = 0; i < size; ++i) {
    out << pad_cut_string(ls1[i], width) << " | " << pad_cut_string(ls2[i], width) << std::endl;
  }
  return out.str();
}

typedef Eigen::Matrix<double,3,3,Eigen::RowMajor> SpatialO3Matrix;

inline SpatialO3Matrix makeRotationAroundX(const double phi)
{
  using namespace std;
  SpatialO3Matrix ret;
  ret <<
    1,         0,         0,
    0,  cos(phi), -sin(phi),
    0,  sin(phi),  cos(phi);
  return ret;
}

inline SpatialO3Matrix makeRotationAroundZ(const double phi)
{
  using namespace std;
  SpatialO3Matrix ret;
  ret <<
    cos(phi), -sin(phi),  0,
    sin(phi),  cos(phi),  0,
           0,         0,  1;
  return ret;
}

inline SpatialO3Matrix makeRandomSpatialO3Matrix(qrngstate::RngState& rs)
{
  using namespace qlat;
  const double reflex_x = u_rand_gen(rs) >= 0.5 ? 1.0 : -1.0;
  const double reflex_y = u_rand_gen(rs) >= 0.5 ? 1.0 : -1.0;
  const double reflex_z = u_rand_gen(rs) >= 0.5 ? 1.0 : -1.0;
  SpatialO3Matrix reflex;
  reflex <<
    reflex_x, 0, 0,
    0, reflex_y, 0,
    0, 0, reflex_z;
  const double phi1 = u_rand_gen(rs, PI, -PI);
  const double phi2 = u_rand_gen(rs, PI, -PI);
  const double phi3 = u_rand_gen(rs, PI, -PI);
  return reflex * makeRotationAroundX(phi3) * makeRotationAroundZ(phi2) * makeRotationAroundX(phi1);
}

inline qlat::CoordinateD operator*(const SpatialO3Matrix& m, const qlat::CoordinateD& x)
{
  Eigen::Matrix<double,3,1> vec;
  vec << x[0], x[1], x[2];
  vec = m * vec;
  qlat::CoordinateD ret;
  ret[0] = vec[0];
  ret[1] = vec[1];
  ret[2] = vec[2];
  ret[3] = x[3];
  return ret;
}

typedef std::array<double,92> ManyMagneticMomentsCompressed;

inline ManyMagneticMomentsCompressed operator*(const double a, const ManyMagneticMomentsCompressed& m)
{
  ManyMagneticMomentsCompressed ret;
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = a * m[i];
  }
  return ret;
}

inline ManyMagneticMomentsCompressed operator*(const ManyMagneticMomentsCompressed& m, const double a)
{
  return a * m;
}

inline ManyMagneticMomentsCompressed& operator*=(ManyMagneticMomentsCompressed& m, const double a)
{
  m = a * m;
  return m;
}

inline ManyMagneticMomentsCompressed operator+(
    const ManyMagneticMomentsCompressed& m1, const ManyMagneticMomentsCompressed& m2)
{
  ManyMagneticMomentsCompressed ret;
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = m1[i] + m2[i];
  }
  return ret;
}

inline ManyMagneticMomentsCompressed operator-(
    const ManyMagneticMomentsCompressed& m1, const ManyMagneticMomentsCompressed& m2)
{
  ManyMagneticMomentsCompressed ret;
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = m1[i] - m2[i];
  }
  return ret;
}

inline SpatialO3Matrix makeProperRotation(const qlat::CoordinateD& x, const qlat::CoordinateD& y)
{
  using namespace qlat;
  // TIMER_VERBOSE("makeProperRotation");
  // displayln(ssprintf("y=") + show(y));
  SpatialO3Matrix rot = makeRotationAroundX(0);
  SpatialO3Matrix rotx = makeRotationAroundX(0);
  SpatialO3Matrix rotz = makeRotationAroundX(0);
  SpatialO3Matrix xrotx = makeRotationAroundX(0);
  if (is_very_close(y[2], 0) && is_very_close(y[1], 0)) {
    rotx = makeRotationAroundX(0);
  } else {
    const double phi_x = std::atan2(y[2], y[1]);
    rotx = makeRotationAroundX(-phi_x);
  }
  const CoordinateD y1 = rotx * y;
  // displayln(ssprintf("y1=") + show(y1));
  qassert(is_very_close(y1[2], 0));
  qassert(y1[1] >= 0 || is_very_close(y1[1], 0));
  if (is_very_close(y1[1], 0) && is_very_close(y1[0], 0)) {
    rotz = makeRotationAroundZ(0);
  } else {
    const double phi_z = std::atan2(y1[1], y1[0]);
    rotz = makeRotationAroundZ(-phi_z);
  }
  const CoordinateD y2 = rotz * y1;
  // displayln(ssprintf("y2=") + show(y2));
  qassert(is_very_close(y2[2], 0));
  qassert(is_very_close(y2[1], 0));
  qassert(y2[0] >= 0 || is_very_close(y2[0], 0));
  rot = rotz * rotx;
  const CoordinateD x2 = rot * x;
  if (is_very_close(x2[2], 0) && is_very_close(x2[1], 0)) {
    xrotx = makeRotationAroundX(0);
  } else {
    const double xphi_x = std::atan2(x2[2], x2[1]);
    xrotx = makeRotationAroundX(-xphi_x);
  }
  const CoordinateD x3 = xrotx * x2;
  // displayln(ssprintf("x3=") + show(x3));
  qassert(is_very_close(x3[2], 0));
  qassert(x3[1] >= 0 || is_very_close(x3[1], 0));
  rot = xrotx * rot;
  if (is_very_close(y2[0], 0) && is_very_close(y2[1], 0) && is_very_close(y2[2], 0)) {
    // important
    // if y has no spatial component
    // then eta = 0
    // thus the spatial component of x need to be along y direction
    SpatialO3Matrix rotzz = makeRotationAroundZ(0);
    if (is_very_close(x3[1], 0) && is_very_close(x3[0], 0)) {
      rotzz = makeRotationAroundZ(0);
    } else {
      const double phi_z = std::atan2(x3[1], x3[0]);
      rotzz = makeRotationAroundZ(-phi_z + PI/2.0);
    }
    const CoordinateD x4 = rotzz * x3;
    // displayln(ssprintf("y2=") + show(y2));
    qassert(is_very_close(x4[2], 0));
    qassert(is_very_close(x4[0], 0));
    qassert(x4[1] >= 0 || is_very_close(x4[1], 0));
    rot = rotzz * rot;
  }
  return rot;
}

inline ManyMagneticMoments muonLine(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  TIMER("muonLine");
  std::vector<double> integral = integrateMuonLine(x, y, epsabs, epsrel);
  return computeProjections(integral);
}

inline ManyMagneticMoments muonLineSymR(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  TIMER("muonLineSymR");
  // already performed in the integrand
  return muonLine(x, y, epsabs, epsrel);
}

inline ManyMagneticMoments muonLineSym(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
  // interface function
  // important, return average instead of sum of all six permutations
  // This function return the avg of six different muon-line diagram
  // directly compute the result
{
  TIMER("muonLineSym");
  std::vector<ManyMagneticMoments> mmms(3);
  mmms[0] = muonLineSymR(x, y, epsabs, epsrel);
  mmms[1] = permuteNuRhoMu(muonLineSymR(-x+y, -x, epsabs, epsrel), 2, 0, 1); // x',2 <- y,0 ; y',0 <- z,1 ; z',1 <- x,2
  mmms[2] = permuteNuRhoMu(muonLineSymR(-y, x-y, epsabs, epsrel), 1, 2, 0); // x',2 <- z,1 ; y',0 <- x,2 ; z',1 <- y,0
  return averageManyMagneticMoments(mmms);
}

inline void coordinatesFromParams(qlat::CoordinateD& x, qlat::CoordinateD& y, const std::vector<double>& params)
  // if y[0] == 0 then theta = 0 and eta = PI/2
{
  using namespace qlat;
  assert(5 == params.size());
  const double d = DISTANCE_LIMIT * std::pow(params[0], 2.0);
  const double alpha = std::pow(params[1], 2.0);
  const double theta = params[2] * PI;
  const double phi = params[3] * PI;
  const double eta = params[4] * PI;
  x[0] = d * alpha * std::sin(phi) * std::cos(eta);
  x[1] = d * alpha * std::sin(phi) * std::sin(eta);
  x[2] = 0.0;
  x[3] = d * alpha * std::cos(phi);
  y[0] = d * std::sin(theta);
  y[1] = 0.0;
  y[2] = 0.0;
  y[3] = d * std::cos(theta);
  if (theta == PI || theta == 0.0) {
    y[0] = 0.0;
    x[0] = 0.0;
    x[1] = d * alpha * std::sin(phi);
  }
  if (phi == PI || phi == 0.0) {
    x[0] = 0.0;
    x[1] = 0.0;
  }
}

inline void paramsFromCoordinates(std::vector<double>& params, const qlat::CoordinateD& x, const qlat::CoordinateD& y)
{
  using namespace qlat;
  const double x_len = coordinate_len(x) + 1.0E-99;
  const double d = coordinate_len(y) + 1.0E-99;
  double alpha = x_len / d;
  if (alpha > 1.0) {
    qassert(alpha < 1.0 + 1e-10);
    alpha = 1.0;
  }
  const double cos_theta = y[3] / d;
  const double cos_phi = x[3] / x_len;
  double cos_eta = (x[0]*y[0] + x[1]*y[1] + x[2]*y[2])
    / (std::sqrt((x[0]*x[0] + x[1]*x[1] + x[2]*x[2]) * (y[0]*y[0] + y[1]*y[1] + y[2]*y[2])) + 1.0E-99);
  if (cos_eta > 1.0) {
    cos_eta = 1.0;
  } else if (cos_eta < -1.0) {
    cos_eta = -1.0;
  }
  params.resize(5);
  params[0] = std::pow(d/DISTANCE_LIMIT, 1.0/2.0);
  params[1] = std::pow(alpha, 1.0/2.0);
  params[2] = std::acos(cos_theta) / PI;
  params[3] = std::acos(cos_phi) / PI;
  params[4] = std::acos(cos_eta) / PI;
  if (qisnan(params)) {
    displayln(shows("paramsFromCoordinates ") + show(x));
    displayln(shows("paramsFromCoordinates ") + show(y));
    displayln(shows("paramsFromCoordinates ") + show(params));
    qassert(false);
  }
}

inline ManyMagneticMomentsCompressed compressManyMagneticMoments(const ManyMagneticMoments& m)
{
  ManyMagneticMomentsCompressed ret;
  size_t index = 0;
  bool bi, bj, bk, bl;
  for (int i = 0; i < 4; ++i) {
    bi = 2 == i;
    for (int j = 0; j < 4; ++j) {
      bj = bi ^ (2 == j);
      for (int k = 0; k < 4; ++k) {
        bk = bj ^ (2 == k);
        for (int l = 0; l < 3; ++l) {
          bl = bk ^ (2 == l);
          if (bl) {
            ret[index] = m[i*4*4 + j*4 + k][l];
            index += 1;
          }
        }
      }
    }
  }
  assert(ret.size() == index);
  return ret;
}

inline ManyMagneticMoments uncompressManyMagneticMoments(const ManyMagneticMomentsCompressed& mc)
{
  using namespace qlat;
  ManyMagneticMoments ret;
  set_zero(ret);
  size_t index = 0;
  bool bi, bj, bk, bl;
  for (int i = 0; i < 4; ++i) {
    bi = 2 == i;
    for (int j = 0; j < 4; ++j) {
      bj = bi ^ (2 == j);
      for (int k = 0; k < 4; ++k) {
        bk = bj ^ (2 == k);
        for (int l = 0; l < 3; ++l) {
          bl = bk ^ (2 == l);
          if (bl) {
            ret[i*4*4 + j*4 + k][l] = mc[index];
            index += 1;
          }
        }
      }
    }
  }
  assert(mc.size() == index);
  return ret;
}

inline ManyMagneticMomentsCompressed muonLineSymParamsCompressed(const std::vector<double>& params,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
  // Target function for interpolation
{
  using namespace qlat;
  TIMER("muonLineSymParamsCompressed");
  CoordinateD x, y;
  coordinatesFromParams(x, y, params);
  const ManyMagneticMoments mmm = muonLineSym(x, y, epsabs, epsrel);
  assert(false == qisnan(mmm));
  const ManyMagneticMomentsCompressed ret = compressManyMagneticMoments(mmm);
  const double mmm_len = std::sqrt(qnorm(ret));
  return compressManyMagneticMoments(mmm);
}

typedef InterpolationBilinearNd<ManyMagneticMomentsCompressed> MuonLineInterp;

inline std::vector<MuonLineInterp>& get_muonline_interps()
{
  static std::vector<MuonLineInterp> interps(1);
  return interps;
}

inline size_t& get_default_muonline_interp_idx()
{
  static size_t idx = IS_USING_MUON_LINE_INTERPOLATION ? 0 : -1;
  return idx;
}

inline MuonLineInterp& getMuonLineInterp(const size_t idx = get_default_muonline_interp_idx())
{
  qassert(idx >= 0);
  std::vector<MuonLineInterp>& interps = get_muonline_interps();
  if (idx >= interps.size()) {
    interps.resize(idx + 1);
  }
  return interps[idx];
}

inline ManyMagneticMomentsCompressed muonLineSymParamsCompressedInterpolate(const std::vector<double>& params,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3, const int b_interp = get_default_muonline_interp_idx())
{
  using namespace qlat;
  const MuonLineInterp& interpolation = getMuonLineInterp(b_interp);
  // displayln(ssprintf("muonLineSymParamsCompressedInterpolate: ") + show(params) + " " + show(norm(interpolation(params))));
  return interpolation(params);
}

inline ManyMagneticMoments muonLineSymParams(const std::vector<double>& params,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3, const int b_interp = get_default_muonline_interp_idx())
{
  ManyMagneticMomentsCompressed mmm;
  if (b_interp >= 0) {
    mmm = muonLineSymParamsCompressedInterpolate(params, epsabs, epsrel, b_interp);
  } else {
    mmm = muonLineSymParamsCompressed(params, epsabs, epsrel);
  }
  return uncompressManyMagneticMoments(mmm);
}

inline ManyMagneticMoments muonLineSymThroughParam(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3, const int b_interp = get_default_muonline_interp_idx())
{
  std::vector<double> params;
  paramsFromCoordinates(params, x, y);
  return muonLineSymParams(params, epsabs, epsrel, b_interp);
}

inline MagneticMoment operator*(const SpatialO3Matrix& m, const MagneticMoment& v)
{
  Eigen::Matrix<double,3,1> vec;
  vec << v[0], v[1], v[2];
  MagneticMoment ret;
  vec = m.determinant() * m * vec;
  ret[0] = vec[0];
  ret[1] = vec[1];
  ret[2] = vec[2];
  return ret;
}

inline ManyMagneticMoments operator*(const SpatialO3Matrix& m, const ManyMagneticMoments& v)
{
  // TIMER_VERBOSE("SpatialO3Matrix*ManyMagneticMoments");
  using namespace qlat;
  ManyMagneticMoments ret, tmp;
  for (int i = 0; i < 4*4*4; ++i) {
    ret[i] = m * v[i];
  }
  // displayln(show(sqrt(norm(ret))));
  tmp = ret;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int prefix = i*4*4 + j*4;
      for (int k = 0; k < 3; ++k) {
        MagneticMoment& mm = ret[prefix + k];
        set_zero(mm);
        for (int kk = 0; kk < 3; ++kk) {
          mm = mm + m(k,kk) * tmp[prefix + kk];
        }
      }
    }
  }
  // displayln(show(sqrt(norm(ret))));
  tmp = ret;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int prefix = i*4*4 + j;
      for (int k = 0; k < 3; ++k) {
        MagneticMoment& mm = ret[prefix + k*4];
        set_zero(mm);
        for (int kk = 0; kk < 3; ++kk) {
          mm = mm + m(k,kk) * tmp[prefix + kk*4];
        }
      }
    }
  }
  // displayln(show(sqrt(norm(ret))));
  tmp = ret;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const int prefix = i*4 + j;
      for (int k = 0; k < 3; ++k) {
        MagneticMoment& mm = ret[prefix + k*4*4];
        set_zero(mm);
        for (int kk = 0; kk < 3; ++kk) {
          mm = mm + m(k,kk) * tmp[prefix + kk*4*4];
        }
      }
    }
  }
  // displayln(show(sqrt(norm(ret))));
  return ret;
}

inline ManyMagneticMoments muonLineSymRotate(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3, const int b_interp = get_default_muonline_interp_idx())
{
  using namespace qlat;
  const SpatialO3Matrix rot = makeProperRotation(x, y);
  CoordinateD xr = rot * x;
  CoordinateD yr = rot * y;
  // for (int i = 0; i < 4; ++i) {
  //   if (is_very_close(xr[i], 0)) {
  //     xr[i] = 0.0;
  //   }
  //   if (is_very_close(yr[i], 0)) {
  //     yr[i] = 0.0;
  //   }
  // }
  const SpatialO3Matrix rott = rot.transpose();
  return rott * muonLineSymThroughParam(xr, yr, epsabs, epsrel, b_interp);
}

inline ManyMagneticMoments muonLineSymPermute(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3, const int b_interp = get_default_muonline_interp_idx())
{
  const double xyl = coordinate_len(y - x);
  double xl = coordinate_len(x);
  double yl = coordinate_len(y);
  if (is_very_close(xl, xyl)) {
    xl = xyl;
  }
  if (is_very_close(yl, xyl)) {
    yl = xyl;
  }
  if (is_very_close(yl, xl)) {
    yl = xl;
  }
  std::vector<ManyMagneticMoments> mmms;
  if (xl <= xyl && xyl <= yl) {
    mmms.push_back(muonLineSymRotate(x, y, epsabs, epsrel, b_interp)); // y z x
  }
  if (xyl <= yl && yl <= xl) {
    mmms.push_back(permuteNuRhoMu(muonLineSymRotate(-x+y, -x, epsabs, epsrel, b_interp), 2, 0, 1)); // z x y
  }
  if (yl <= xl && xl <= xyl) {
    mmms.push_back(permuteNuRhoMu(muonLineSymRotate(-y, x-y, epsabs, epsrel, b_interp), 1, 2, 0)); // x y z
  }
  if (yl <= xyl && xyl <= xl) {
    mmms.push_back(permuteNuRhoMu(muonLineSymRotate(y, x, epsabs, epsrel, b_interp), 2, 1, 0)); // x z y
  }
  if (xl <= yl && yl <= xyl) {
    mmms.push_back(permuteNuRhoMu(muonLineSymRotate(-x, -x+y, epsabs, epsrel, b_interp), 0, 2, 1)); // y x z
  }
  if (xyl <= xl && xl <= yl) {
    mmms.push_back(permuteNuRhoMu(muonLineSymRotate(x-y, -y, epsabs, epsrel, b_interp), 1, 0, 2)); // z y x
  }
  qassert(mmms.size() > 0);
  return averageManyMagneticMoments(mmms);
}

inline ManyMagneticMoments muonLineSymTransform(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3, const int b_interp = get_default_muonline_interp_idx())
  // interface function
  // This function return the avg of six different muon-line diagram
  // proper rotation transformation is made in order to reduce to 5 parameters
  // interpolation is performed if IS_USING_MUON_LINE_INTERPOLATION = true
  //
  // ManyMagneticMoments format [y-pol,z-pol,x-pol][mag-dir]
  // argument x and y assume z = 0
{
  using namespace qlat;
  return muonLineSymPermute(x, y, epsabs, epsrel, b_interp);
}

inline void paramsFromCoordinatesPermute(std::vector<double>& params, const qlat::CoordinateD& x, const qlat::CoordinateD& y)
{
  const double xl = coordinate_len(x);
  const double yl = coordinate_len(y);
  const double xyl = coordinate_len(y - x);
  if (xl <= xyl && xyl <= yl) {
    paramsFromCoordinates(params, x, y);
  } else if (xyl <= yl && yl <= xl) {
    paramsFromCoordinates(params, -x+y, -x);
  } else if (yl <= xl && xl <= xyl) {
    paramsFromCoordinates(params, -y, x-y);
  } else if (yl <= xyl && xyl <= xl) {
    paramsFromCoordinates(params, y, x);
  } else if (xl <= yl && yl <= xyl) {
    paramsFromCoordinates(params, -x, -x+y);
  } else if (xyl <= xl && xl <= yl) {
    paramsFromCoordinates(params, x-y, -y);
  } else {
    assert(false);
  }
}

inline void compare_many_magnetic_moments(const std::string& tag,
    const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const ManyMagneticMoments& mmm, const ManyMagneticMoments& mmmp)
{
  using namespace qlat;
  std::vector<double> params;
  paramsFromCoordinatesPermute(params, x, y);
  const double diff_percent = 100.0*sqrt(qnorm(mmmp-mmm) / qnorm(mmm));
  const bool is_print = diff_percent > 0.0001 || true;
  if (is_print) {
#pragma omp critical
    {
      displayln(compare_multiline_string(showManyMagneticMoments(mmm), showManyMagneticMoments(mmmp), 48));
      displayln(tag + ": " + ssprintf("CHECKING: %10.2e %10.2e %10.4f%%",
            sqrt(qnorm(mmm)), sqrt(qnorm(mmmp-mmm)), diff_percent));
      displayln(tag + ": " + shows("params= ") + show(params));
      displayln(tag + ": " + ssprintf(" x  = %8.4f %s", coordinate_len(x) , show(x).c_str()));
      displayln(tag + ": " + ssprintf(" y  = %8.4f %s", coordinate_len(y) , show(y).c_str()));
      displayln(tag + ": " + ssprintf("y-x = %8.4f %s", coordinate_len(y-x) , show(y-x).c_str()));
      displayln(tag + ": " + ssprintf("x  y  ") + show(sqrt(qnorm(mmm))));
      displayln(tag + ": " +
          ssprintf("DATA: %24.17E %24.17E %24.17E %24.17E %24.17E   %24.17E %24.17E %24.17E  %24.17E",
            params[0], params[1], params[2], params[3], params[4],
            sqrt(qnorm(mmm)), sqrt(qnorm(mmmp)), sqrt(qnorm(mmmp-mmm)), sqrt(qnorm(mmmp-mmm)/qnorm(mmm))));
    }
  }
}

inline ManyMagneticMoments muonLineSymParamsCheck(const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  using namespace qlat;
  TIMER_VERBOSE("muonLineSymParamsCheck");
  ManyMagneticMoments mmm = muonLineSym(x, y, epsabs, epsrel);
  ManyMagneticMoments mmmp = muonLineSymTransform(x, y, epsabs, epsrel);
  compare_many_magnetic_moments("params-check", x, y, mmm, mmmp);
  return mmm;
}

inline ManyMagneticMoments muonLineSymRotateCheck(const SpatialO3Matrix& rot,
    const qlat::CoordinateD& x, const qlat::CoordinateD& y,
    const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  using namespace qlat;
  TIMER_VERBOSE("muonLineSymRotateCheck");
  ManyMagneticMoments mmm = muonLineSymTransform(x, y, epsabs, epsrel);
  const CoordinateD xr = rot * x;
  const CoordinateD yr = rot * y;
  const SpatialO3Matrix rott = rot.transpose();
  ManyMagneticMoments mmmp = rott * muonLineSymTransform(xr, yr, epsabs, epsrel);
  compare_many_magnetic_moments("rotate-check", x, y, mmm, mmmp);
  return mmm;
}

inline void initializeMuonLineInterpolation(const std::vector<int>& dims, const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
  // computing the muon-line interpolation database
  // take quite some time
{
  using namespace qlat;
  TIMER_VERBOSE("initializeMuonLineInterpolation");
  MuonLineInterp& interpolation = getMuonLineInterp();
  interpolation.init();
  assert(dims.size() == 5);
  interpolation.add_dimension(dims[0], 1.0, 0.0); // d
  interpolation.add_dimension(dims[1], 1.0, 0.0); // alpha
  interpolation.add_dimension(dims[2], 1.0, 0.0); // theta
  interpolation.add_dimension(dims[3], 1.0, 0.0); // phi
  interpolation.add_dimension(dims[4], 1.0, 0.0); // eta
  const size_t jobs_total = interpolation.size();
  const size_t num_nodes = get_num_node();
  const size_t jobs_per_nodes = jobs_total / num_nodes;
  const size_t jobs_parallel = jobs_per_nodes * num_nodes;
  const size_t jobs_left = jobs_total - jobs_parallel;
  const size_t my_start = jobs_left + jobs_per_nodes * get_id_node();
  std::vector<ManyMagneticMomentsCompressed> workplace(jobs_per_nodes);
#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < jobs_per_nodes; ++i) {
    const size_t idx = my_start + i;
    TIMER_VERBOSE("interp-initial-iter");
#pragma omp critical
    {
      displayln(ssprintf("jobs-par: %5d %10ld %10ld/%ld", get_id_node(), my_start, i, jobs_per_nodes));
    }
    workplace[i] = muonLineSymParamsCompressed(interpolation.get_coor(idx), epsabs, epsrel);
  }
  Vector<ManyMagneticMomentsCompressed> all_workspace(&interpolation[jobs_left], jobs_parallel);
  all_gather(all_workspace, Vector<ManyMagneticMomentsCompressed>(workplace));
#pragma omp parallel for schedule(dynamic)
  for (size_t idx = 0; idx < jobs_left; ++idx) {
    TIMER_VERBOSE("interp-initial-iter");
#pragma omp critical
    if (0 == get_id_node()) {
      displayln(ssprintf("jobs-left: %10ld/%ld", idx, jobs_left));
      displayln(show(idx));
    }
    interpolation[idx] = muonLineSymParamsCompressed(interpolation.get_coor(idx), epsabs, epsrel);
  }
}

inline void saveMuonLineInterpolation(const std::string& path)
{
  using namespace qlat;
  TIMER_VERBOSE("saveMuonLineInterpolation");
  if (0 == get_id_node()) {
    MuonLineInterp& interp= getMuonLineInterp();
    qmkdir(path);
    FILE* fdims = qopen(path + "/dims.txt" + ".partial", "w");
    const std::vector<InterpolationDim>& dims = interp.dims;
    fprintf(fdims, "# i dims[i].n dims[i].xhigh dims[i].xlow\n");
    for (size_t i = 0; i < dims.size(); ++i) {
      fprintf(fdims, "%3d %5d %24.17E %24.17E\n", i, dims[i].n, dims[i].xhigh, dims[i].xlow);
    }
    qclose(fdims);
    qrename_partial(path + "/dims.txt");
    FILE* fdata = qopen(path + "/data.txt" + ".partial", "w");
    const std::vector<ManyMagneticMomentsCompressed>& data = interp.data;
    fprintf(fdata, "# idx params[0-4] ManyMagneticMomentsCompressed[0-91]\n");
    for (size_t idx = 0; idx < data.size(); ++idx) {
      fprintf(fdata, "%10ld", idx);
      const std::vector<double> params = interp.get_coor(idx);
      for (size_t i = 0; i < params.size(); ++i) {
        fprintf(fdata, " %24.17E", params[i]);
      }
      fprintf(fdata, "  ");
      const ManyMagneticMomentsCompressed& mmm = data[idx];
      for (size_t i = 0; i < mmm.size(); ++i) {
        fprintf(fdata, " %24.17E", mmm[i]);
      }
      fprintf(fdata, "\n");
    }
    qclose(fdata);
    qrename_partial(path + "/data.txt");
    qtouch(path + "/checkpoint");
  }
  sync_node();
}

inline bool loadMuonLineInterpolation(const std::string& path, const size_t interp_idx = 0)
{
  using namespace qlat;
  TIMER_VERBOSE("loadMuonLineInterpolation");
  if (!does_file_exist_sync_node(path + "/checkpoint")) {
    return false;
  }
  MuonLineInterp& interp = getMuonLineInterp(interp_idx);
  interp.init();
  std::vector<std::vector<double> > dims;
  if (0 == get_id_node()) {
    qassert(does_file_exist(path + "/dims.txt"));
    dims = qload_datatable(path + "/dims.txt");
    for (size_t i = 0; i < dims.size(); ++i) {
      qassert(i == (size_t)dims[i][0]);
      interp.add_dimension((int)dims[i][1], dims[i][2], dims[i][3]);
    }
  }
  // sync MuonLineInterp dims across all nodes
  bcast(dims);
  if (0 != get_id_node()) {
    for (size_t i = 0; i < dims.size(); ++i) {
      qassert(i == (size_t)dims[i][0]);
      interp.add_dimension((int)dims[i][1], dims[i][2], dims[i][3]);
    }
  }
  long limit = 0;
  if (0 == get_id_node()) {
    // Can be obtained from
    // split -da 10 -l 10000 data.txt data.txt.
    while (does_file_exist(path + ssprintf("/data.txt.%010d", limit))) {
      limit += 1;
    }
  }
  glb_sum(limit);
  std::vector<DataTable> tables(limit);
  const int num_node = get_num_node();
  const int id_node = get_id_node();
#pragma omp parallel for schedule(dynamic)
  for (size_t i = id_node; i < limit; i += num_node) {
    std::string fn = path + ssprintf("/data.txt.%010d", i);
    qassert(does_file_exist(fn));
    tables[i] = qload_datatable(fn);
  }
  if (limit == 0 && get_id_node() == 0) {
    qassert(does_file_exist(path + "/data.txt"));
    tables.push_back(qload_datatable_par(path + "/data.txt"));
  }
  DataTable data;
  for (size_t i = 0; i < tables.size(); ++i) {
    for (size_t j = 0; j < tables[i].size(); ++j) {
      data.push_back(tables[i][j]);
    }
  }
  set_zero(get_data(interp.data));
#pragma omp parallel for
  for (size_t i = 0; i < data.size(); ++i) {
    const std::vector<double>& data_vec = data[i];
    qassert(data_vec.size() == 1 + 5 + 92);
    const size_t idx = (size_t)data_vec[0];
    std::vector<double> params = interp.get_coor(idx);
    qassert(params.size() == 5);
    for (size_t k = 0; k < params.size(); ++k) {
      qassert(params[k] == data_vec[1 + k]);
    }
    ManyMagneticMomentsCompressed mmmc;
    qassert(mmmc.size() == 92);
    for (size_t k = 0; k < mmmc.size(); ++k) {
      mmmc[k] = data_vec[6 + k];
    }
    interp[idx] = mmmc;
  }
  // sync MuonLineInterp data across all nodes
  glb_sum_long_vec(get_data(interp.data));
  return true;
}

inline void load_or_compute_muonline_interpolation(const std::string& path, const std::vector<int>& dims, const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  if (!loadMuonLineInterpolation(path)) {
    test_fCalc();
    initializeMuonLineInterpolation(dims, epsabs, epsrel);
    saveMuonLineInterpolation(path);
  }
}

inline void save_part_muonline_interpolation_data(const std::string& path, const size_t fn_idx, const size_t start_idx, const qlat::Vector<ManyMagneticMomentsCompressed> data)
{
  using namespace qlat;
  TIMER_VERBOSE("save_part_muonline_interpolation_data");
  MuonLineInterp& interp = getMuonLineInterp();
  std::string fn = path + ssprintf("/data.txt.%010d", fn_idx);
  FILE* fdata = qopen(fn + ".partial", "w");
  fprintf(fdata, "# idx params[0-4] ManyMagneticMomentsCompressed[0-91]\n");
  for (size_t k = 0; k < data.size(); ++k) {
    size_t idx = start_idx + k;
    fprintf(fdata, "%10ld", idx);
    const std::vector<double> params = interp.get_coor(idx);
    for (size_t i = 0; i < params.size(); ++i) {
      fprintf(fdata, " %24.17E", params[i]);
    }
    fprintf(fdata, "  ");
    const ManyMagneticMomentsCompressed& mmm = data[k];
    for (size_t i = 0; i < mmm.size(); ++i) {
      fprintf(fdata, " %24.17E", mmm[i]);
    }
    fprintf(fdata, "\n");
  }
  qclose(fdata);
  qrename_partial(fn);
}

inline bool is_part_muonline_interpolation_data_done(const std::string& path, const size_t fn_idx)
{
  using namespace qlat;
  std::string fn = path + ssprintf("/data.txt.%010d", fn_idx);
  return does_file_exist(fn);
}

inline bool compute_save_muonline_interpolation_cc(const std::string& path, const std::vector<int>& dims, const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
  // preferred way to generate interpolation field
{
  using namespace qlat;
  TIMER_VERBOSE("compute_save_muonline_interpolation_cc");
  std::vector<std::vector<double> > ldims;
  if (0 == get_id_node() && does_file_exist(path + "/dims.txt")) {
    std::vector<std::vector<double> > ldims = qload_datatable(path + "/dims.txt");
    qassert(ldims.size() == dims.size());
    for (size_t i = 0; i < ldims.size(); ++i) {
      qassert(i == (size_t)ldims[i][0]);
      qassert(dims[i] == (int)ldims[i][1]);
      qassert(1.0 == ldims[i][2]);
      qassert(0.0 == ldims[i][3]);
    }
  }
  assert(dims.size() == 5);
  MuonLineInterp& interp = getMuonLineInterp();
  interp.init();
  for (size_t i = 0; i < dims.size(); ++i) {
    interp.add_dimension(dims[i], 1.0, 0.0);
  }
  set_zero(get_data(interp.data));
  if (!does_file_exist_sync_node(path + "/checkpoint")) {
    qmkdir_info(path);
    if (!obtain_lock(path + "/lock")) {
      return false;
    }
    if (0 == get_id_node() && !does_file_exist(path + "/dims.txt")) {
      FILE* fdims = qopen(path + "/dims.txt" + ".partial", "w");
      const std::vector<InterpolationDim>& dims = interp.dims;
      fprintf(fdims, "# i dims[i].n dims[i].xhigh dims[i].xlow\n");
      for (size_t i = 0; i < dims.size(); ++i) {
        fprintf(fdims, "%3d %5d %24.17E %24.17E\n", i, dims[i].n, dims[i].xhigh, dims[i].xlow);
      }
      qclose(fdims);
      qrename_partial(path + "/dims.txt");
    }
    test_fCalc();
    const size_t jobs_total = interp.size();
    // ADJUST ME
    const size_t job_chunk_size = 64;
    std::vector<size_t> jobs;
    for (size_t start = 0; start <= jobs_total - job_chunk_size; start += job_chunk_size) {
      jobs.push_back(start);
    }
    std::vector<std::array<ManyMagneticMomentsCompressed, job_chunk_size> > data;
    qassert(get_num_node() > 1);
    if (0 == get_id_node()) {
      const size_t size = jobs.size();
      data.resize(size);
      size_t num_running_jobs = 0;
      size_t idx = 0;
      for (int dest = 1; dest < get_num_node(); ++dest) {
        while (is_part_muonline_interpolation_data_done(path, idx)) {
          idx += 1;
        }
        if (idx >= jobs.size()) {
          break;
        }
        displayln(ssprintf("send-job: %5d %10ld/%ld", dest, jobs[idx], jobs_total));
        send_job(idx, jobs[idx], dest);
        idx += 1;
        num_running_jobs += 1;
      }
      while (idx < jobs.size()) {
        check_time_limit(false);
        TIMER_VERBOSE("muonline_interpolation_cc-traj");
        int source;
        int64_t flag;
        std::array<ManyMagneticMomentsCompressed, job_chunk_size> result;
        receive_result(source, flag, result);
        num_running_jobs -= 1;
        data[flag] = result;
        save_part_muonline_interpolation_data(path, flag, jobs[flag], get_data(result));
        while (is_part_muonline_interpolation_data_done(path, idx)) {
          idx += 1;
        }
        if (idx >= jobs.size()) {
          break;
        }
        displayln(ssprintf("send-job: %5d %10ld/%ld", source, jobs[idx], jobs_total));
        send_job(idx, jobs[idx], source);
        idx += 1;
        num_running_jobs += 1;
      }
      while (num_running_jobs > 0) {
        check_time_limit(false);
        TIMER_VERBOSE("muonline_interpolation_cc-traj");
        int source;
        int64_t flag;
        std::array<ManyMagneticMomentsCompressed, job_chunk_size> result;
        receive_result(source, flag, result);
        data[flag] = result;
        save_part_muonline_interpolation_data(path, flag, jobs[flag], get_data(result));
        num_running_jobs -= 1;
      }
      for (int dest = 1; dest < get_num_node(); ++dest) {
        size_t job = 0;
        displayln(ssprintf("send-job: %5d %10ld/%ld", dest, (long)-1, jobs_total));
        send_job(-1, job, dest);
      }
      idx = 0;
      for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < job_chunk_size; ++j) {
          interp[idx] = data[i][j];
          idx += 1;
        }
      }
    } else {
      while (true) {
        int64_t flag;
        size_t job;
        receive_job(flag, job);
        if (-1 == flag) {
          displayln(ssprintf("par: %5d done", get_id_node()));
          break;
        }
        std::array<ManyMagneticMomentsCompressed, job_chunk_size> result;
#pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < job_chunk_size; ++i) {
          const size_t idx = job + i;
          result[i] = muonLineSymParamsCompressed(interp.get_coor(idx), epsabs, epsrel);
#pragma omp critical
          {
            displayln(ssprintf("par: %5d %10ld/%ld %10ld/%ld", get_id_node(), idx, jobs_total, i, job_chunk_size));
          }
        }
        send_result(flag, result);
      }
    }
    if (0 == get_id_node()) {
      const size_t last_flag = jobs_total / job_chunk_size;
      const size_t last_start_idx = last_flag * job_chunk_size;
#pragma omp parallel for schedule(dynamic)
      for (size_t idx = last_start_idx; idx < jobs_total; ++idx) {
        interp[idx] = muonLineSymParamsCompressed(interp.get_coor(idx), epsabs, epsrel);
#pragma omp critical
        {
          displayln(ssprintf("left: %10ld/%ld", idx, jobs_total));
        }
      }
      save_part_muonline_interpolation_data(path, last_flag, last_start_idx, Vector<ManyMagneticMomentsCompressed>(&interp[last_start_idx], jobs_total - last_start_idx));
      qtouch(path + "/checkpoint");
    }
    release_lock();
  }
  // ADJUST ME
  return true;
  // return loadMuonLineInterpolation(path);
}

inline bool load_multiple_muonline_interpolations(const std::string& path)
{
  using namespace qlat;
  TIMER_VERBOSE("load_multiple_muonline_interpolations");
  long limit = 0;
  if (0 == get_id_node()) {
    while (does_file_exist(path + ssprintf("/%010d/checkpoint", limit))) {
      limit += 1;
    }
  }
  glb_sum(limit);
  if (0 == limit) {
    return loadMuonLineInterpolation(path);
  }
  for (size_t idx = 0; idx < limit; ++idx) {
    loadMuonLineInterpolation(path + ssprintf("/%010d", idx), idx);
  }
  return true;
}

inline void load_compute_save_muonline_interpolation(const std::string& path, const std::vector<int>& dims, const double epsabs = 1.0e-8, const double epsrel = 1.0e-3)
{
  if (!load_multiple_muonline_interpolations(path)) {
    // ADJUST ME
    // load_or_compute_muonline_interpolation(path, dims, epsabs, epsrel);
    compute_save_muonline_interpolation_cc(path, dims, epsabs, epsrel);
    loadMuonLineInterpolation(path);
  }
}

inline void test_muonline_transformation()
{
  using namespace qlat;
  TIMER_VERBOSE_FLOPS("test_muonline_transformation");
  qrngstate::RngState rs(getGlobalRngState(), "test_muonline_transformation");
  const size_t size = 128 * 64 * 2;
  const double high = 3.0;
  const double low = -3.0;
  const size_t jobs_total = size;
  const size_t num_nodes = get_num_node();
  const size_t jobs_per_nodes = jobs_total / num_nodes;
  const size_t jobs_parallel = jobs_per_nodes * num_nodes;
  const size_t jobs_left = jobs_total - jobs_parallel;
  size_t my_start = jobs_left + jobs_per_nodes * get_id_node();
  const size_t my_end = my_start + jobs_per_nodes;
  if (0 == get_id_node()) {
    my_start = 0;
  }
#pragma omp parallel for schedule(dynamic)
  for (int i = my_start; i < my_end; ++i) {
    const int a = i / 9;
    const int b = i % 9;
    const int bx = b / 3;
    const int by = b % 3;
    qrngstate::RngState rsi(rs, a);
    CoordinateD x, y;
    for (int m = 0; m < 4; ++m) {
      x[m] = u_rand_gen(rsi, high, low);
      y[m] = u_rand_gen(rsi, high, low);
    }
    for (int m = 0; m < 4; ++m) {
      if (1 == bx) {
        x[m] /= 5.0;
      } else if (2 == bx) {
        x[m] /= 20.0;
      }
      if (1 == by) {
        y[m] /= 5.0;
      } else if (2 == by) {
        y[m] /= 20.0;
      }
    }
    muonLineSymParamsCheck(x, y, 1e-10, 1e-4);
  }
  timer.flops += size;
}

inline void test_muonline_transform_scaling()
{
  using namespace qlat;
  TIMER_VERBOSE("test_muonline_transform_scaling");
  qrngstate::RngState rs(get_global_rng_state(), "test_muonline_transform_scaling");
  const int size = 1024;
  const double high = 1.0;
  const double low = -1.0;
  const double ratio = 2.0;
  const double ratio2 = 10.0;
  const size_t jobs_total = size;
  const size_t num_nodes = get_num_node();
  const size_t jobs_per_nodes = jobs_total / num_nodes;
  const size_t jobs_parallel = jobs_per_nodes * num_nodes;
  const size_t jobs_left = jobs_total - jobs_parallel;
  size_t my_start = jobs_left + jobs_per_nodes * get_id_node();
  const size_t my_end = my_start + jobs_per_nodes;
  if (0 == get_id_node()) {
    my_start = 0;
  }
#pragma omp parallel for schedule(dynamic)
  for (int i = my_start; i < my_end; ++i) {
    qrngstate::RngState rsi(rs, i);
    CoordinateD x, y;
    for (int m = 0; m < 4; ++m) {
      x[m] = u_rand_gen(rsi, high, low);
      x[m] *= pow(u_rand_gen(rsi, 1.0, 0.0), 10);
      y[m] = u_rand_gen(rsi, high, low);
      y[m] *= pow(u_rand_gen(rsi, 1.0, 0.0), 10);
    }
    ManyMagneticMoments mmm = muonLineSym(x, y, 1e-12, 1e-4);
    ManyMagneticMoments mmmp = muonLineSymTransform(x, y, 1e-7, 1e-2, false);
    // ManyMagneticMoments mmmp = ratio * muonLineSymTransform(x/ratio, y/ratio, 1e-12, 1e-3, false);
    // ManyMagneticMoments mmmpp = ratio2 * muonLineSymTransform(x/ratio2, y/ratio2, 1e-12, 1e-3, false);
    // ManyMagneticMoments mmmppp = ratio*ratio2 * muonLineSymTransform(x/ratio2/ratio, y/ratio2/ratio, 1e-12, 1e-3, false);
    compare_many_magnetic_moments("scaling", x, y, mmm, mmmp);
    // compare_many_magnetic_moments("scaling1", x, y, mmmp, mmmpp);
    // compare_many_magnetic_moments("scaling2", x, y, mmmpp, mmmppp);
  }
}

inline void test_muonline_interp()
{
  using namespace qlat;
  TIMER_VERBOSE("test_muonline_interp")
  qrngstate::RngState rs(get_global_rng_state(), "test_muonline_interp");
  const int size = 1024;
  const double high = 1.0;
  const double low = -1.0;
  const double ratio = 2.0;
  const double ratio2 = 10.0;
  const size_t jobs_total = size;
  const size_t num_nodes = get_num_node();
  const size_t jobs_per_nodes = jobs_total / num_nodes;
  const size_t jobs_parallel = jobs_per_nodes * num_nodes;
  const size_t jobs_left = jobs_total - jobs_parallel;
  size_t my_start = jobs_left + jobs_per_nodes * get_id_node();
  const size_t my_end = my_start + jobs_per_nodes;
  if (0 == get_id_node()) {
    my_start = 0;
  }
  const MuonLineInterp& interp = getMuonLineInterp();
  const size_t interp_size = interp.size();
#pragma omp parallel for schedule(dynamic)
  for (int i = my_start; i < my_end; ++i) {
    qrngstate::RngState rsi(rs, i);
    const size_t idx = rand_gen(rsi) % interp_size;
    const std::vector<double> params = interp.get_coor(idx);
    CoordinateD x, y;
    coordinatesFromParams(x, y, params);
    for (int i = 0; i < 4; ++i) {
      if (is_very_close(x[i], 0)) {
        x[i] = 0.0;
      }
      if (is_very_close(y[i], 0)) {
        y[i] = 0.0;
      }
    }
    ManyMagneticMoments mmm = muonLineSym(x, y, 1e-8, 1e-3);
    ManyMagneticMoments mmmp = muonLineSymTransform(x, y, 1e-8, 1e-3, false);
    ManyMagneticMoments mmmpp = muonLineSymTransform(x, y, 1e-8, 1e-3, true);
    {
      compare_many_magnetic_moments("checking", x, y, mmm, mmmp);
      compare_many_magnetic_moments("checking2", x, y, mmm, mmmpp);
      displayln(ssprintf("test_muonline_interp idx=%d params=%s", idx, show(params).c_str()));
    }
  }
}

inline void test_muonline_rotate()
{
  using namespace qlat;
  TIMER_VERBOSE_FLOPS("test_muonline_rotate");
  qrngstate::RngState rs(getGlobalRngState(), "test_muonline_rotate");
  const size_t size = 128 * 64 * 2;
  const double high = 3.0;
  const double low = -3.0;
  const size_t jobs_total = size;
  const size_t num_nodes = get_num_node();
  const size_t jobs_per_nodes = jobs_total / num_nodes;
  const size_t jobs_parallel = jobs_per_nodes * num_nodes;
  const size_t jobs_left = jobs_total - jobs_parallel;
  size_t my_start = jobs_left + jobs_per_nodes * get_id_node();
  const size_t my_end = my_start + jobs_per_nodes;
  if (0 == get_id_node()) {
    my_start = 0;
  }
#pragma omp parallel for schedule(dynamic)
  for (int i = my_start; i < my_end; ++i) {
    qrngstate::RngState rsi(rs, i);
    CoordinateD x, y;
    for (int m = 0; m < 4; ++m) {
      x[m] = u_rand_gen(rsi, high, low);
      x[m] *= pow(u_rand_gen(rsi, 1.0, 0.0), 5);
      y[m] = u_rand_gen(rsi, high, low);
      y[m] *= pow(u_rand_gen(rsi, 1.0, 0.0), 5);
    }
    muonLineSymRotateCheck(makeRandomSpatialO3Matrix(rsi), x, y);
  }
  timer.flops += size;
}

inline void test_muonline_int()
{
  using namespace qlat;
  TIMER_VERBOSE_FLOPS("test_muonline_int");
  qrngstate::RngState rs(getGlobalRngState(), "test_muonline_int");
  const size_t size = 128 * 64 * 2;
  const double high = 4.0;
  const double low = -3.0;
  const size_t jobs_total = size;
  const size_t num_nodes = get_num_node();
  const size_t jobs_per_nodes = jobs_total / num_nodes;
  const size_t jobs_parallel = jobs_per_nodes * num_nodes;
  const size_t jobs_left = jobs_total - jobs_parallel;
  size_t my_start = jobs_left + jobs_per_nodes * get_id_node();
  const size_t my_end = my_start + jobs_per_nodes;
  if (0 == get_id_node()) {
    my_start = 0;
  }
#pragma omp parallel for schedule(dynamic)
  for (int i = my_start; i < my_end; ++i) {
    qrngstate::RngState rsi(rs, i);
    CoordinateD x, y;
    for (int m = 0; m < 4; ++m) {
      x[m] = 0.1 * (int)u_rand_gen(rsi, high, low);
      y[m] = 0.1 * (int)u_rand_gen(rsi, high, low);
    }
    muonLineSymParamsCheck(x, y, 1e-8, 1e-3);
    // muonLineSymParamsCheck(x, y, 1e-10, 1e-4);
    // muonLineSymRotateCheck(makeRandomSpatialO3Matrix(rsi), x, y);
    // muonLineSymRotateCheck(makeRandomSpatialO3Matrix(rsi), x, y);
    // muonLineSymRotateCheck(makeRandomSpatialO3Matrix(rsi), x, y);
    // muonLineSymRotateCheck(makeRandomSpatialO3Matrix(rsi), x, y);
  }
  timer.flops += size;
}

inline void test_muonLine()
{
  using namespace qlat;
  sync_node();
  TIMER_VERBOSE("test_muonLine");
  if (IS_USING_MUON_LINE_INTERPOLATION) {
    load_compute_save_muonline_interpolation("huge-data-muon-line-interpolation", std::vector<int>(5, 12), 1.0e-7, 1.0e-2);
  }
  // const CoordinateD x(0.1, 0.2, 0.0, 0.5);
  // const CoordinateD y(0.3, 0.0, -0.2, 0.1);
  // ManyMagneticMoments mmm;
  // mmm = muonLineSym(x,y, 1e-8, 1e-3);
  // displayln(showManyMagneticMoments(mmm));
  // mmm = muonLineSym(x,y);
  // displayln(showManyMagneticMoments(mmm));
  // test_muonline_transform_scaling();
  test_muonline_interp();
  // test_muonline_int();
  // test_muonline_rotate();
  // test_muonline_transformation();
}
#endif
