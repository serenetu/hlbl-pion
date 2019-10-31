//#pragma once
#ifndef _LATTICE-UTILS_H
#define _LATTICE-UTILS_H

#include <qlat/qlat.h>
#include <qlat/qlat-analysis.h>
#include <qlat/field-utils.h>
#include <gsl/gsl_sf_bessel.h>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <dirent.h>
#include "muon-line.h"

#include <map>
#include <vector>

QLAT_START_NAMESPACE

const SpinMatrix& gamma5 = SpinMatrixConstants::get_gamma5();
const std::array<SpinMatrix,4>& gammas = SpinMatrixConstants::get_cps_gammas();
const CoordinateD CoorD_0 = CoordinateD(0, 0, 0, 0);
const Coordinate Coor_0 = Coordinate(0, 0, 0, 0);

double pion_prop(const qlat::CoordinateD& x, const qlat::CoordinateD& y, const double& m)
// x, y and m must be in lattice unit
{
  CoordinateD dist = x - y;
  double s = std::pow(std::pow(dist[0], 2.) + std::pow(dist[1], 2.) + std::pow(dist[2], 2.) + std::pow(dist[3], 2.), 1./2.);
  double sm = s * m;
  return m * gsl_sf_bessel_K1(sm) / (4. * std::pow(PI, 2.) * s);
}

inline ManyMagneticMoments get_muon_line_m(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z, const int idx = get_default_muonline_interp_idx())
{
  // ADJUST ME
  return muonLineSymTransform(x - z, y - z, 1e-8, 1e-3, idx);
  // return muonLineSym(x - z, y - z, 1e-8, 1e-3);
}

inline std::string show_mmm(const ManyMagneticMoments& mmm)
{
  std::string out("");
  for (int i = 0; i < 3; ++i) {
    out += ssprintf("mmm i=%d: ", i);
    for (int ii = 0; ii < 64; ++ii) {
      out += ssprintf("%e, ", mmm[ii][i]);
    }
    out += "\n";
  }
  return out;
}

inline void init_muon_line()
{
  TIMER_VERBOSE("init_muon_line");
  const std::string path = "huge-data-muon-line-interpolation";
  load_multiple_muonline_interpolations(path);
}

inline ManyMagneticMoments get_muon_line_m_extra(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z,
    const int tag)
  // tag = 0 sub
  // tag = 1 nosub
{
  ManyMagneticMoments m1, m2, m3;
  if (tag == 0) {
    m1 = get_muon_line_m(x, y, z, 5);
    m2 = get_muon_line_m(x, y, z, 3);
    m3 = get_muon_line_m(x, y, z, 1);
  } else if (tag == 1) {
    m1 = get_muon_line_m(x, y, z, 11);
    m2 = get_muon_line_m(x, y, z, 9);
    m3 = get_muon_line_m(x, y, z, 7);
  } else {
    qassert(false);
  }
  return 3.0476190476190476 * m1 - 2.3142857142857143 * m2 + 0.26666666666666667 * m3;
}

typedef Cache<std::string, Propagator4d> PropCache;

inline PropCache& get_prop_cache()
{
  // ADJUST ME
  static PropCache cache("PropCache", 16);
  return cache;
}

const Propagator4d& get_prop(const std::string& path)
{
  PropCache& cache = get_prop_cache();
  if (!cache.has(path)) {
    TIMER_VERBOSE("get_prop-read");
    Propagator4d& prop = cache[path];
    read_field_double_from_float(prop, path);
  }
  return cache[path];
}

inline PropCache& get_wall_prop_cache()
{
  // ADJUST ME
  static PropCache cache("PropCache", 16);
  return cache;
}

Propagator4d& get_wall_prop(const std::string& path)
{
  PropCache& cache = get_wall_prop_cache();
  if (!cache.has(path)) {
    TIMER_VERBOSE("get_wall_prop-read");
    Propagator4d& prop = cache[path];
    read_field_double_from_float(prop, path);
  }
  return cache[path];
}

QLAT_END_NAMESPACE

#endif
