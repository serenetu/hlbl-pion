#pragma once

#include "muon-line.h"

#include <sstream>

inline void init_muon_line_pi_decay()
{
  TIMER_VERBOSE("init_muon_line_pi_decay");
  const std::string path = "huge-data-muon-line-interpolation";
  load_multiple_muonline_interpolations(path);
}

inline ManyMagneticMoments get_muon_line_m(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z, const int idx = get_default_muonline_interp_idx())
{
  // ADJUST ME
  return muonLineSymTransform(x - z, y - z, 1e-8, 1e-3, idx);
  // return muonLineSym(x - z, y - z, 1e-8, 1e-3);
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
  using namespace qlat;
  return 3.0476190476190476 * m1 - 2.3142857142857143 * m2 + 0.26666666666666667 * m3;
}

inline double get_m_comp(
    const ManyMagneticMoments& mmm,
    const int i, const int rho, const int sigma, const int nu)
{
  return mmm[16 * sigma + 4 * nu + rho][i];
}

QLAT_START_NAMESPACE

struct EpsilonTensorTable
{
  int tensor[4][4][4][4];
  //
  EpsilonTensorTable()
  {
    init();
  }
  //
  void init()
  {
    std::memset(this, 0, sizeof(tensor));
    setv(0,1,2,3);
    setv(0,2,3,1);
    setv(0,3,1,2);
  }
  //
  void setv(const int a, const int b, const int c, const int d)
  {
    set(a,b,c,d,1);
    set(a,b,d,c,-1);
  }
  void set(const int a, const int b, const int c, const int d, const int val)
  {
    tensor[a][b][c][d] = val;
    tensor[b][c][d][a] = -val;
    tensor[c][d][a][b] = val;
    tensor[d][a][b][c] = -val;
  }
};

inline int epsilon_tensor(const int a, const int b, const int c, const int d)
{
  static EpsilonTensorTable table;
  return table.tensor[a][b][c][d];
}

inline int epsilon_tensor(const int i, const int j, const int k)
{
  return epsilon_tensor(i, j, k, 3);
}

inline void scalar_inversion(Field<Complex>& sol, const Field<Complex>& src, const double mass,
    const CoordinateD momtwist = CoordinateD())
  // the mass is not necessarily the exponent of the exponential fall off
{
  TIMER("scalar_inversion");
  const Geometry geo = geo_resize(src.geo);
  const Coordinate total_site = geo.total_site();
  sol.init(geo);
  sol = src;
  fft_complex_field(sol, true);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate kl = geo.coordinate_from_index(index);
    Coordinate kg = geo.coordinate_g_from_l(kl);
    CoordinateD kk;
    double s2 = 0.0;
    for (int i = 0; i < DIMN; i++) {
      kg[i] = smod(kg[i], total_site[i]);
      kk[i] = 2.0 * PI * (kg[i] + momtwist[i]) / (double)total_site[i];
      s2 += 4.0 * sqr(std::sin(kk[i] / 2.0));
    }
    const double fac = 1.0 / (s2 + sqr(mass));
    Vector<Complex> v = sol.get_elems(kl);
    for (int m = 0; m < v.size(); ++m) {
      v[m] *= fac;
    }
  }
  fft_complex_field(sol, false);
  sol *= 1.0 / geo.total_volume();
}

inline void scalar_derivative(Field<Complex>& sol, const Field<Complex>& src,
    const CoordinateD momtwist = CoordinateD())
  // v[m*4 + mu] = sv[m] * std::sin(kk[mu]);
{
  TIMER("scalar_derivative");
  const Geometry geo = geo_reform(src.geo, src.geo.multiplicity * 4);
  sol.init(geo);
  qassert(sol.geo == geo);
  const Coordinate total_site = geo.total_site();
  Field<Complex> src_mom;
  src_mom.init(geo_resize(src.geo));
  src_mom = src;
  fft_complex_field(src_mom, true);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate kl = geo.coordinate_from_index(index);
    Coordinate kg = geo.coordinate_g_from_l(kl);
    CoordinateD kk;
    double s2 = 0.0;
    for (int i = 0; i < DIMN; i++) {
      kg[i] = smod(kg[i], total_site[i]);
      kk[i] = 2.0 * PI * (kg[i] + momtwist[i]) / (double)total_site[i];
    }
    const Vector<Complex> sv = src_mom.get_elems_const(kl);
    Vector<Complex> v = sol.get_elems(kl);
    for (int m = 0; m < sv.size(); ++m) {
      for (int mu = 0; mu < 4; ++mu) {
        v[m*4+mu] = sv[m] * std::sin(kk[mu]);
      }
    }
  }
  fft_complex_field(sol, false);
  sol *= 1.0 / geo.total_volume();
}

inline void scalar_divergence(Field<Complex>& sol, const Field<Complex>& src,
    const CoordinateD momtwist = CoordinateD())
  // v[m] += sv[m*4+mu] * std::sin(kk[mu]);
{
  TIMER("scalar_derivative");
  const Geometry geo = geo_reform(src.geo, src.geo.multiplicity / 4);
  sol.init(geo);
  qassert(sol.geo == geo);
  const Coordinate total_site = geo.total_site();
  Field<Complex> src_mom;
  src_mom.init(geo_resize(src.geo));
  src_mom = src;
  fft_complex_field(src_mom, true);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate kl = geo.coordinate_from_index(index);
    Coordinate kg = geo.coordinate_g_from_l(kl);
    CoordinateD kk;
    double s2 = 0.0;
    for (int i = 0; i < DIMN; i++) {
      kg[i] = smod(kg[i], total_site[i]);
      kk[i] = 2.0 * PI * (kg[i] + momtwist[i]) / (double)total_site[i];
    }
    const Vector<Complex> sv = src_mom.get_elems_const(kl);
    Vector<Complex> v = sol.get_elems(kl);
    for (int m = 0; m < v.size(); ++m) {
      v[m] = 0;
      for (int mu = 0; mu < 4; ++mu) {
        v[m] += sv[m*4+mu] * std::sin(kk[mu]);
      }
    }
  }
  fft_complex_field(sol, false);
  sol *= 1.0 / geo.total_volume();
}

// inline double three_if_d1_smallest(long d1, long d2, long d3)
// {
//   if (d1 > d2 || d1 > d3) {
//     return 0.0;
//   } else if (d1 == d2 && d1 == d3) {
//     return 1.0;
//   } else if (d1 == d2 || d1 == d3) {
//     return 3.0 / 2.0;
//   } else {
//     return 3.0;
//   }
// }

inline int boundary_multiplicity(const Coordinate& xg, const Coordinate& lb, const Coordinate& ub)
{
  int ret = 1;
  for (int i = 0; i < 4; ++i) {
    if (xg[i] == lb[i] || xg[i] == ub[i]) {
      ret *= 2;
    }
  }
  return ret;
}

// inline void set_m_z_field_simple_default(FieldM<ManyMagneticMoments,1>& mf, const Coordinate& xg1, const Coordinate& xg2, const double a)
// {
//   TIMER("set_m_z_field_simple_default");
//   const Geometry& geo = mf.geo;
//   const Coordinate total_site = geo.total_site();
//   displayln_info(show(xg1) + " " + show(xg2));
//   std::vector<Coordinate> coordinate_shifts(3*3*3*3);
//   {
//     size_t sindex = 0;
//     for (int a = -1; a <= 1; ++a) {
//       for (int b = -1; b <= 1; ++b) {
//         for (int c = -1; c <= 1; ++c) {
//           for (int d = -1; d <= 1; ++d) {
//             qassert(sindex < coordinate_shifts.size());
//             coordinate_shifts[sindex] = total_site * Coordinate(a, b, c, d);
//             sindex += 1;
//           }
//         }
//       }
//     }
//   }
//   const Coordinate xr = relative_coordinate(xg2 - xg1, total_site);
//   // const long dis_xy = distance_sq_relative_coordinate_g(xr);
//   const Coordinate& sxg1 = xg1;
//   const Coordinate sxg2 = xg1 + xr;
//   Coordinate lb, ub;
//   for (int m = 0; m < 4; ++m) {
//     lb[m] = std::max(sxg1[m], sxg2[m]) - total_site[m] / 2;
//     ub[m] = std::min(sxg1[m], sxg2[m]) + total_site[m] / 2;
//   }
// #pragma omp parallel for
//   for (long index = 0; index < geo.local_volume(); ++index) {
//     const Coordinate xl = geo.coordinate_from_index(index);
//     const Coordinate xg = geo.coordinate_g_from_l(xl); // z location
//     // const long dis_xz = distance_sq_relative_coordinate_g(relative_coordinate(xg1 - xg, total_site));
//     // const long dis_yz = distance_sq_relative_coordinate_g(relative_coordinate(xg2 - xg, total_site));
//     ManyMagneticMoments& mmm = mf.get_elem(xl);
//     set_zero(mmm);
//     for (size_t sindex = 0; sindex < coordinate_shifts.size(); ++sindex) {
//       const Coordinate sxg = xg + coordinate_shifts[sindex];
//       const Coordinate xr1 = sxg1 - sxg;
//       const Coordinate xr2 = sxg2 - sxg;
//       if (is_outside_coordinate(xr1, total_site) || is_outside_coordinate(xr2, total_site)) {
//         // outside the box
//         continue;
//       }
//       const CoordinateD x = a * CoordinateD(xr1);
//       const CoordinateD y = a * CoordinateD(xr2);
//       const double l1 = coordinate_len(x);
//       const double l2 = coordinate_len(y);
//       const double l3 = coordinate_len(x-y);
//       if (l1 >= DISTANCE_LIMIT - 0.01 || l2 >= DISTANCE_LIMIT - 0.01 || l3 >= DISTANCE_LIMIT - 0.01) {
//         // one edge is too long, outside of the interpolation range
//         continue;
//       }
//       const double fmult = 1.0 / (double)boundary_multiplicity(sxg, lb, ub);
//       mmm += fmult * get_muon_line_m(x, y, CoordinateD());
//     }
//   }
// }

// inline void set_m_z_field_sub(FieldM<ManyMagneticMoments,1>& mf, const Coordinate& xg1, const Coordinate& xg2, const double a)
//   // interp range 0-11
//   // 0-5 sub
//   // 6-11 no-sub
//   /*
//      x1 = 1/16^2; x2 = 1/12^2; x3 = 1/8^2;
//      N[b /. Solve[{y1 == b + x1*k + x1^2*kk, y2 == b + x2*k + x2^2*kk,
//      y3 == b + x3*k + x3^2*kk}, {b, k, kk}][[1]], 17] // FullSimplify
//      */
//   // 3.0476190476190476 y1 - 2.3142857142857143 y2 + 0.26666666666666667 y3
// {
//   TIMER_VERBOSE("set_m_z_field_sub");
//   set_zero(mf);
//   FieldM<ManyMagneticMoments,1> tmf;
//   tmf.init(mf.geo);
//   get_default_muonline_interp_idx() = 5;
//   set_m_z_field_simple_default(tmf, xg1, xg2, a);
//   tmf *= 3.0476190476190476;
//   mf += tmf;
//   get_default_muonline_interp_idx() = 3;
//   set_m_z_field_simple_default(tmf, xg1, xg2, a);
//   tmf *= - 2.3142857142857143;
//   mf += tmf;
//   get_default_muonline_interp_idx() = 1;
//   set_m_z_field_simple_default(tmf, xg1, xg2, a);
//   tmf *= 0.26666666666666667;
//   mf += tmf;
// }

// inline void set_m_z_field_nosub(FieldM<ManyMagneticMoments,1>& mf, const Coordinate& xg1, const Coordinate& xg2, const double a)
//   // interp range 0-11
//   // 0-5 sub
//   // 6-11 no-sub
//   /*
//      x1 = 1/16^2; x2 = 1/12^2; x3 = 1/8^2;
//      N[b /. Solve[{y1 == b + x1*k + x1^2*kk, y2 == b + x2*k + x2^2*kk,
//      y3 == b + x3*k + x3^2*kk}, {b, k, kk}][[1]], 17] // FullSimplify
//      */
//   // 3.0476190476190476 y1 - 2.3142857142857143 y2 + 0.26666666666666667 y3
// {
//   TIMER_VERBOSE("set_m_z_field_nosub");
//   set_zero(mf);
//   FieldM<ManyMagneticMoments,1> tmf;
//   tmf.init(mf.geo);
//   get_default_muonline_interp_idx() = 11;
//   set_m_z_field_simple_default(tmf, xg1, xg2, a);
//   tmf *= 3.0476190476190476;
//   mf += tmf;
//   get_default_muonline_interp_idx() = 9;
//   set_m_z_field_simple_default(tmf, xg1, xg2, a);
//   tmf *= - 2.3142857142857143;
//   mf += tmf;
//   get_default_muonline_interp_idx() = 7;
//   set_m_z_field_simple_default(tmf, xg1, xg2, a);
//   tmf *= 0.26666666666666667;
//   mf += tmf;
// }

struct PointPairWeight
{
  CoordinateD rxy, rxz;
  double weight;
};

inline std::vector<PointPairWeight> shift_lat_corr(const Coordinate& x, const Coordinate& y, const Coordinate& z,
    const Coordinate& total_site, const double a)
  // x and y are closest
  // rxy and rxz are returned
{
  const Coordinate rxy = relative_coordinate(y - x, total_site);
  const CoordinateD rdxy = a * CoordinateD(rxy);
  Coordinate lb, ub;
  for (int m = 0; m < 4; ++m) {
    lb[m] = std::max(0, rxy[m]) - total_site[m] / 2;
    ub[m] = std::min(0, rxy[m]) + total_site[m] / 2;
  }
  Coordinate rxz = relative_coordinate(z - x, total_site);
  std::vector<PointPairWeight> ret;
  for (int r0 = rxz[0]; r0 <= ub[0]; r0 += total_site[0]) {
    if (lb[0] > r0) {
      continue;
    }
    for (int r1 = rxz[1]; r1 <= ub[1]; r1 += total_site[1]) {
      if (lb[1] > r1) {
        continue;
      }
      for (int r2 = rxz[2]; r2 <= ub[2]; r2 += total_site[2]) {
        if (lb[2] > r2) {
          continue;
        }
        for (int r3 = rxz[3]; r3 <= ub[3]; r3 += total_site[3]) {
          if (lb[3] > r3) {
            continue;
          }
          PointPairWeight ppw;
          ppw.rxy = rdxy;
          const Coordinate rxz(r0, r1, r2, r3);
          ppw.rxz = a * CoordinateD(rxz);
          const double l1 = coordinate_len(ppw.rxz);
          const double l2 = coordinate_len(ppw.rxz - ppw.rxy);
          const double l3 = coordinate_len(ppw.rxy);
          if (l1 >= DISTANCE_LIMIT - 0.01 or l2 >= DISTANCE_LIMIT - 0.01 or l3 >= DISTANCE_LIMIT - 0.01) {
            // one edge is too long, outside of the interpolation range
            continue;
          }
          ppw.weight = 1.0 / (double)boundary_multiplicity(rxz, lb, ub);
          ret.push_back(ppw);
        }
      }
    }
  }
  return ret;
}

inline ManyMagneticMoments get_muon_line_m_extra_lat(
    const Coordinate& x, const Coordinate& y, const Coordinate& z, const Coordinate& total_site,
    const double a,
    const int tag)
  // interface
  // tag = 0 sub
  // tag = 1 nosub
{
  const Coordinate rxy = relative_coordinate(y - x, total_site);
  const Coordinate ryz = relative_coordinate(z - y, total_site);
  const Coordinate rzx = relative_coordinate(x - z, total_site);
  const long d2xy = distance_sq_relative_coordinate_g(rxy);
  const long d2yz = distance_sq_relative_coordinate_g(ryz);
  const long d2zx = distance_sq_relative_coordinate_g(rzx);
  std::vector<PointPairWeight> ppws;
  if (d2xy <= d2yz and d2xy <= d2zx) {
    ppws = shift_lat_corr(x, y, z, total_site, a);
  } else if (d2yz <= d2xy and d2yz <= d2zx) {
    ppws = shift_lat_corr(y, z, x, total_site, a);
    // y z : z-y -y
    // -z y-z : y z
    for (int i = 0; i < ppws.size(); ++i) {
      const PointPairWeight ppw = ppws[i];
      ppws[i].rxy = -ppw.rxz;
      ppws[i].rxz = ppw.rxy - ppw.rxz;
    }
  } else if (d2zx <= d2xy and d2zx <= d2yz) {
    ppws = shift_lat_corr(z, x, y, total_site, a);
    // y z : -z y-z
    // z-y -y : y z
    for (int i = 0; i < ppws.size(); ++i) {
      const PointPairWeight ppw = ppws[i];
      ppws[i].rxy = ppw.rxz - ppw.rxy;
      ppws[i].rxz = -ppw.rxy;
    }
  } else {
    qassert(false);
  }
  ManyMagneticMoments mmm;
  set_zero(mmm);
  for (int i = 0; i < ppws.size(); ++i) {
    const PointPairWeight& ppw = ppws[i];
    mmm += ppw.weight * get_muon_line_m_extra(CoordinateD(), ppw.rxy, ppw.rxz, tag);
  }
  return mmm;
}

inline void set_m_z_field_tag(FieldM<ManyMagneticMoments,1>& mf, const Coordinate& xg1, const Coordinate& xg2, const double a, const int tag)
  // interface
{
  TIMER("set_m_z_field_tag");
  const Geometry& geo = mf.geo;
  const Coordinate total_site = geo.total_site();
  displayln_info(show(xg1) + " " + show(xg2));
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Coordinate xg = geo.coordinate_g_from_l(xl); // z location
    ManyMagneticMoments& mmm = mf.get_elem(xl);
    mmm = get_muon_line_m_extra_lat(xg1, xg2, xg, total_site, a, tag);
  }
}

// inline void set_m_z_field(FieldM<ManyMagneticMoments,1>& mf, const Coordinate& xg1, const Coordinate& xg2, const double a)
// {
//   TIMER_VERBOSE("set_m_z_field");
//   // ADJUST ME
//   // get_default_muonline_interp_idx() = 5; // simple_default with-sub
//   // get_default_muonline_interp_idx() = 11; // simple_default no-sub
//   // set_m_z_field_simple_default(mf, xg1, xg2, a);
//   // set_m_z_field_sub(mf, xg1, xg2, a, 0);
//   // set_m_z_field_nosub(mf, xg1, xg2, a);
//   set_m_z_field_tag(mf, xg1, xg2, a, 0);
// }

inline void set_external_photon_i(FieldM<Complex,4>& f, const CoordinateD& xref, const int i)
  // need initialize f beforehand
  // i is the magnetic moment direction
{
  TIMER("set_external_photon_i");
  const Geometry& geo = f.geo;
  Coordinate total_site = geo.total_site();
  set_zero(f);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Coordinate xg = geo.coordinate_g_from_l(xl);
    Vector<Complex> v = f.get_elems(xl);
    for (int j = 0; j < 3; ++j) {
      double r = smod(xg[j] - xref[j], (double)total_site[j]);
      if (std::abs(std::abs(r) * 2.0 - total_site[j]) <= 1e-7) {
        r = 0.0;
      }
      for (int k = 0; k < v.size(); ++k) {
        v[k] += epsilon_tensor(i, j, k) * r;
      }
    }
  }
}

inline void set_external_photons(std::array<FieldM<Complex,4>,3>& ext_photons,
    const Geometry& geo, const CoordinateD& xref)
{
  TIMER("set_external_photons");
  for (int i = 0; i < 3; ++i) {
    ext_photons[i].init(geo);
    set_external_photon_i(ext_photons[i], xref, i);
  }
}

inline void set_muon_line_photon(FieldM<Complex,4>& f, const FieldM<ManyMagneticMoments,1>& mf,
    const int i, const int rho, const int sigma)
{
  TIMER("set_muon_line_photon");
  const Geometry& geo = mf.geo;
  Coordinate total_site = geo.total_site();
  f.init(geo);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    Vector<Complex> v = f.get_elems(xl);
    const ManyMagneticMoments& mmm = mf.get_elem(xl);
    for (int nu = 0; nu < 4; ++nu) {
      v[nu] = get_m_comp(mmm, i, rho, sigma, nu);
    }
  }
}

inline void set_point_src(FieldM<Complex,4>& f, const Coordinate& xg, const double value = 1.0)
  // need initialize f beforehand
{
  TIMER("set_point_src");
  const Geometry& geo = f.geo;
  set_zero(f);
  const Coordinate xl = geo.coordinate_l_from_g(xg);
  if (geo.is_local(xl)) {
    Vector<Complex> v = f.get_elems(xl);
    for (int m = 0; m < v.size(); ++m) {
      v[m] = value;
    }
  }
}

inline bool coordinate_distance_sq_comp(const Coordinate& x, const Coordinate& y)
{
  return distance_sq_relative_coordinate_g(x) < distance_sq_relative_coordinate_g(y);
}

inline std::vector<Coordinate> generate_short_dis_coords(const long limit_sq)
{
  TIMER_VERBOSE("generate_short_dis_coords");
  const long limit = std::round(std::sqrt(limit_sq));
  std::vector<Coordinate> coords;
  for (int t = 0; t <= limit; ++t) {
    for (int z = 0; z <= limit; ++z) {
      for (int y = 0; y <= z; ++y) {
        for (int x = 0; x <= y; ++x) {
          const Coordinate c(x, y, z, t);
          if (distance_sq_relative_coordinate_g(c) <= limit_sq) {
            coords.push_back(c);
          }
        }
      }
    }
  }
  std::sort(coords.begin(), coords.end(), coordinate_distance_sq_comp);
  return coords;
}

struct HlblPionModelInfo
{
  Coordinate total_site;
  double m_mu_mev, a_inv, a, f_pi, m_pi, alpha_inv, e_charge; // default unit is m_mu
  double m_vector;
  std::string path;
  long short_limit_sq;
  //
  std::vector<Coordinate> short_dis_coords;
  double prob_sum;
  std::vector<Coordinate> long_dis_coords;
  std::vector<double> long_dis_probs;
  //
  void init_16nt32()
  {
    path = "16nt32_3x";
    total_site = Coordinate(16, 16, 16, 32);
    m_mu_mev = 3.0 * 105.6583745;
    m_pi = 3.0 * 134.9766 / m_mu_mev;
    a_inv = 1730.0 / m_mu_mev;
    f_pi = 92.0 / m_mu_mev;
    m_vector = 770.0 / m_mu_mev;
    a = 1.0 / a_inv;
    alpha_inv = 137.035999139;
    e_charge = std::sqrt(4 * qlat::PI / alpha_inv);
    short_limit_sq = sqr(5);
  }
  void init_24nt64()
  {
    init_16nt32();
    path = "24nt64_3x";
    total_site = Coordinate(24, 24, 24, 64);
  }
  void init_32nt64()
  {
    init_16nt32();
    path = "32nt64_3x";
    total_site = Coordinate(32, 32, 32, 64);
  }
  void init_48nt96()
  {
    init_16nt32();
    path = "48nt96_3x";
    total_site = Coordinate(48, 48, 48, 96);
  }
  void init_24nt48_fine()
  {
    init_16nt32();
    path = "24nt48_fine_3x";
    total_site = Coordinate(24, 24, 24, 48);
    a_inv = 3.0/2.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_32nt64_fine_2()
  {
    init_16nt32();
    path = "32nt64_fine_2_3x";
    total_site = Coordinate(32, 32, 32, 64);
    a_inv = 2.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_48nt96_phys()
  {
    init_16nt32();
    path = "48nt96_phys";
    total_site = Coordinate(48, 48, 48, 96);
    m_mu_mev = 105.6583745;
    m_pi = 134.9766 / m_mu_mev;
    a_inv = 1730.0 / m_mu_mev;
    f_pi = 92.0 / m_mu_mev;
    m_vector = 770.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_40nt80_coarse_0p5_phys()
  {
    init_48nt96_phys();
    path = "40nt80_coarse_0p5_phys";
    total_site = Coordinate(40, 40, 40, 80);
    a_inv = 5.0/6.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_32nt64_coarse_phys()
  {
    init_48nt96_phys();
    path = "32nt64_coarse_phys";
    total_site = Coordinate(32, 32, 32, 64);
    a_inv = 2.0/3.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_48nt96_coarse_phys()
  {
    init_32nt64_coarse_phys();
    path = "48nt96_coarse_phys";
    total_site = Coordinate(48, 48, 48, 96);
  }
  void init_28nt56_coarse_1p5_phys()
  {
    init_48nt96_phys();
    path = "28nt56_coarse_1p5_phys";
    total_site = Coordinate(28, 28, 28, 56);
    a_inv = 7.0/12.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_24nt48_coarse_2_phys()
  {
    init_48nt96_phys();
    path = "24nt48_coarse_2_phys";
    total_site = Coordinate(24, 24, 24, 48);
    a_inv = 0.5 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_32nt64_coarse_2_phys()
  {
    init_24nt48_coarse_2_phys();
    path = "32nt64_coarse_2_phys";
    total_site = Coordinate(32, 32, 32, 64);
  }
  void init_48nt96_coarse_2_phys()
  {
    init_24nt48_coarse_2_phys();
    path = "48nt96_coarse_2_phys";
    total_site = Coordinate(48, 48, 48, 96);
  }
  void init_64nt128_coarse_2_phys()
  {
    init_24nt48_coarse_2_phys();
    path = "64nt128_coarse_2_phys";
    total_site = Coordinate(64, 64, 64, 128);
  }
  void init_20nt40_coarse_2p5_phys()
  {
    init_48nt96_phys();
    path = "20nt40_coarse_2p5_phys";
    total_site = Coordinate(20, 20, 20, 40);
    a_inv = 5.0/12.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_16nt32_coarse_3_phys()
  {
    init_48nt96_phys();
    path = "16nt32_coarse_3_phys";
    total_site = Coordinate(16, 16, 16, 32);
    a_inv = 1.0/3.0 * 1730.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  void init_8nt16()
  {
    init_16nt32();
    path = "8nt16_6x";
    total_site = Coordinate(8, 8, 8, 16);
    m_mu_mev = 6.0 * 105.6583745;
    m_pi = 6.0 * 134.9766 / m_mu_mev;
    a_inv = 1730.0 / m_mu_mev;
    f_pi = 92.0 / m_mu_mev;
    m_vector = 770.0 / m_mu_mev;
    a = 1.0 / a_inv;
  }
  //
  HlblPionModelInfo()
  {
    init_16nt32();
  }
};

inline void display_hlbl_pion_model_info(const HlblPionModelInfo& info)
{
  displayln_info(ssprintf("total_site     = %28s", show(info.total_site).c_str()));
  displayln_info(ssprintf("m_mu_mev       = %28.17f", info.m_mu_mev));
  displayln_info(ssprintf("m_pi_mev       = %28.17f", info.m_pi * info.m_mu_mev));
  displayln_info(ssprintf("a_inv_mev      = %28.17f", info.a_inv * info.m_mu_mev));
  displayln_info(ssprintf("f_pi_mev       = %28.17f", info.f_pi * info.m_mu_mev));
  displayln_info(ssprintf("m_vector_mev   = %28.17f", info.m_vector * info.m_mu_mev));
  displayln_info(ssprintf("a              = %28.17f", info.a));
  displayln_info(ssprintf("alpha_inv      = %28.17f", info.alpha_inv));
  displayln_info(ssprintf("e_charge       = %28.17f", info.e_charge));
  displayln_info(ssprintf("short_limit_sq = %28ld",   info.short_limit_sq));
}

inline void set_pion_photon_photon_vertex_two_end(FieldM<Complex,4*4>& pion,
    const FieldM<Complex,4>& photon1, const FieldM<Complex,4>& photon2,
    const double m_vector, const double f_pi)
{
  TIMER("set_pion_photon_photon_vertex_two_end");
  qassert(photon1.geo == photon2.geo);
  const Geometry geo = geo_reform(photon1.geo, 4*4, 0);
  FieldM<Complex,4*4> pinv1, pinv2, p1, p2;
  scalar_derivative(p1, photon1);
  scalar_derivative(p2, photon2);
  scalar_inversion(pinv1, p1, m_vector);
  scalar_inversion(pinv2, p2, m_vector);
  pion.init(geo);
  set_zero(pion);
  const Complex fac = 1.0 / (4.0 * sqr(PI) * f_pi) * ii * sqr(m_vector)/2.0;
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Vector<Complex> v1 = pinv1.get_elems_const(xl);
    const Vector<Complex> v2 = pinv2.get_elems_const(xl);
    const Vector<Complex> d1 = p1.get_elems_const(xl);
    const Vector<Complex> d2 = p2.get_elems_const(xl);
    Vector<Complex> pv = pion.get_elems(xl);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) {
          continue;
        }
        for (int rho = 0; rho < 4; ++rho) {
          if (rho == mu or rho == nu) {
            continue;
          }
          for (int sigma = 0; sigma < 4; ++sigma) {
            if (sigma == mu or sigma == nu or sigma == rho) {
              continue;
            }
            pv[mu*4+nu] += fac * (Complex)epsilon_tensor(mu, nu, rho, sigma)
              * (v1[mu*4+rho] * d2[nu*4+sigma] + d1[mu*4+rho] * v2[nu*4+sigma]);
          }
        }
      }
    }
  }
}

inline void set_pion_photon_photon_vertex_vmd(FieldM<Complex,4*4>& pion,
    const FieldM<Complex,4>& photon1, const FieldM<Complex,4>& photon2,
    const double m_vector, const double f_pi)
{
  TIMER("set_pion_photon_photon_vertex_vmd");
  qassert(photon1.geo == photon2.geo);
  const Geometry geo = geo_reform(photon1.geo, 4*4, 0);
  FieldM<Complex,4*4> pinv1, pinv2, p1, p2;
  scalar_derivative(p1, photon1);
  scalar_derivative(p2, photon2);
  scalar_inversion(pinv1, p1, m_vector);
  scalar_inversion(pinv2, p2, m_vector);
  pion.init(geo);
  set_zero(pion);
  const Complex fac = 1.0 / (4.0 * sqr(PI) * f_pi) * ii * sqr(sqr(m_vector));
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Vector<Complex> v1 = pinv1.get_elems_const(xl);
    const Vector<Complex> v2 = pinv2.get_elems_const(xl);
    Vector<Complex> pv = pion.get_elems(xl);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) {
          continue;
        }
        for (int rho = 0; rho < 4; ++rho) {
          if (rho == mu or rho == nu) {
            continue;
          }
          for (int sigma = 0; sigma < 4; ++sigma) {
            if (sigma == mu or sigma == nu or sigma == rho) {
              continue;
            }
            pv[mu*4+nu] += fac * (Complex)epsilon_tensor(mu, nu, rho, sigma) * v1[mu*4+rho] * v2[nu*4+sigma];
          }
        }
      }
    }
  }
}

inline void set_pion_photon_photon_vertex(FieldM<Complex,4*4>& pion,
    const FieldM<Complex,4>& photon1, const FieldM<Complex,4>& photon2,
    const double m_vector, const double f_pi)
  // vertex function
{
  // SADJUST ME
  // set_pion_photon_photon_vertex_two_end(pion, photon1, photon2, m_vector, f_pi);
  set_pion_photon_photon_vertex_vmd(pion, photon1, photon2, m_vector, f_pi);
}

inline Complex pion_contraction(
    const FieldM<Complex,4*4>& pion_xy, const int rho, const int sigma,
    const FieldM<Complex,4*4>& pion_zxop, const int nu, const int k)
{
  TIMER("pion_contraction");
  const Geometry& geo = pion_xy.geo;
  qassert(geo == pion_zxop.geo);
  Complex sum = 0.0;
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    sum += pion_xy.get_elem(xl, rho * 4 + sigma) * pion_zxop.get_elem(xl, nu * 4 + k);
  }
  return sum;
}

inline int hlbl_sample_xy_pion_v(const Coordinate& x, const Coordinate& y, const HlblPionModelInfo& info)
{
  const Coordinate& total_site = info.total_site;
  const Coordinate crel = relative_coordinate(y-x, total_site);
  const std::string fn = get_result_path() + "/" + info.path +
    ssprintf("/f2 ; dis_sq=%010d ; rel=(%s).txt",
        distance_sq_relative_coordinate_g(crel), show(crel).c_str());
  if (does_file_exist_sync_node(fn)) {
    return 0;
  }
  TIMER_VERBOSE("hlbl_sample_xy_pion_v");
  const Geometry geo = Geometry(total_site, 1);
  FieldM<Complex,4> photon_x, photon_y;
  photon_x.init(geo);
  photon_y.init(geo);
  set_point_src(photon_x, x);
  set_point_src(photon_y, y);
  FieldM<Complex,4*4> pion_xy, pion_zxop;
  set_pion_photon_photon_vertex(pion_xy, photon_x, photon_y, info.m_vector*info.a, info.f_pi*info.a);
  scalar_inversion(pion_xy, pion_xy, info.m_pi*info.a);
  std::array<FieldM<Complex,4>,3> ext_photons;
  set_external_photons(ext_photons, geo,
      middle_coordinate(CoordinateD(x), CoordinateD(y), CoordinateD(total_site)));
  FieldM<ManyMagneticMoments,1> mf;
  mf.init(geo);
  // ADJUST ME
  set_m_z_field_tag(mf, x, y, info.a, 0);
  FieldM<Complex,4> photon_muon_line;
  const double coef = 1.0E10 * info.a * std::pow(info.e_charge, 6) / 3.0;
  Complex sum = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int rho = 0; rho < 4; ++rho) {
      for (int sigma = 0; sigma < 4; ++sigma) {
        if (sigma == rho) {
          continue;
        }
        set_muon_line_photon(photon_muon_line, mf, i, rho, sigma);
        set_pion_photon_photon_vertex(pion_zxop, photon_muon_line, ext_photons[i], info.m_vector*info.a, info.f_pi*info.a);
        for (int nu = 0; nu < 4; ++nu) {
          for (int k = 0; k < 3; ++k) {
            if (k == i or k == nu) {
              continue;
            }
            const Complex pc = pion_contraction(pion_xy, rho, sigma, pion_zxop, nu, k);
            sum += 3.0 * coef * pc; // 3.0 is for the 3 function
          }
        }
      }
    }
  }
  glb_sum(sum);
  displayln_info(fname + ssprintf(": point-pair-coordinates: x=") + show(x) + " ; y=" + show(y) + " ; sum=" + show(sum));
  qmkdir_info(get_result_path() + "/" + info.path);
  qtouch_info(fn, ssprintf("# %24.17E %24.17E\n", sum.real(), sum.imag()));
  return 1;
}

inline void set_photon_pion_photon_vertex_two_end(FieldM<Complex,4>& photon1,
    const FieldM<Complex,1>& pion, const FieldM<Complex,4>& photon2,
    const double m_vector, const double f_pi)
{
  TIMER("set_photon_pion_photon_vertex_two_end");
  const Geometry geo = geo_reform(photon2.geo, 4, 0);
  qassert(is_matching_geo(pion.geo, photon2.geo));
  photon1.init(geo);
  qassert(photon1.geo == geo);
  set_zero(photon1);
  FieldM<Complex,4*4> pinv1, pinv2, p1, p2;
  scalar_derivative(p2, photon2);
  scalar_inversion(pinv2, p2, m_vector);
  p1.init(geo);
  pinv1.init(geo);
  const Complex fac = 1.0 / (4.0 * sqr(PI) * f_pi) * ii * sqr(m_vector)/2.0;
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    Vector<Complex> d1 = p1.get_elems(xl);
    const Vector<Complex> d2 = p2.get_elems_const(xl);
    Vector<Complex> v1 = pinv1.get_elems(xl);
    const Vector<Complex> v2 = pinv2.get_elems_const(xl);
    const Complex& v = pion.get_elem(xl);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) {
          continue;
        }
        for (int rho = 0; rho < 4; ++rho) {
          if (rho == mu or rho == nu) {
            continue;
          }
          for (int sigma = 0; sigma < 4; ++sigma) {
            if (sigma == mu or sigma == nu or sigma == rho) {
              continue;
            }
            const Complex c = -fac * (Complex)epsilon_tensor(mu, nu, rho, sigma) * v;
            d1[mu*4+rho] += c * v2[nu*4+sigma];
            v1[mu*4+rho] += c * d2[nu*4+sigma];
          }
        }
      }
    }
  }
  scalar_inversion(pinv1, pinv1, m_vector);
  p1 += pinv1;
  scalar_divergence(photon1, p1);
}

inline void set_photon_pion_photon_vertex_vmd(FieldM<Complex,4>& photon1,
    const FieldM<Complex,1>& pion, const FieldM<Complex,4>& photon2,
    const double m_vector, const double f_pi)
{
  TIMER("set_photon_pion_photon_vertex_vmd");
  const Geometry geo = geo_reform(photon2.geo, 4, 0);
  qassert(is_matching_geo(pion.geo, photon2.geo));
  photon1.init(geo);
  qassert(photon1.geo == geo);
  set_zero(photon1);
  FieldM<Complex,4*4> pinv1, pinv2, p1, p2;
  scalar_derivative(p2, photon2);
  scalar_inversion(pinv2, p2, m_vector);
  p1.init(geo);
  pinv1.init(geo);
  const Complex fac = 1.0 / (4.0 * sqr(PI) * f_pi) * ii * sqr(sqr(m_vector));
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    Vector<Complex> v1 = pinv1.get_elems(xl);
    const Vector<Complex> v2 = pinv2.get_elems_const(xl);
    const Complex& v = pion.get_elem(xl);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) {
          continue;
        }
        for (int rho = 0; rho < 4; ++rho) {
          if (rho == mu or rho == nu) {
            continue;
          }
          for (int sigma = 0; sigma < 4; ++sigma) {
            if (sigma == mu or sigma == nu or sigma == rho) {
              continue;
            }
            const Complex c = -fac * (Complex)epsilon_tensor(mu, nu, rho, sigma) * v;
            v1[mu*4+rho] += c * v2[nu*4+sigma];
          }
        }
      }
    }
  }
  scalar_inversion(p1, pinv1, m_vector);
  scalar_divergence(photon1, p1);
}

inline void set_photon_pion_photon_vertex(FieldM<Complex,4>& photon1,
    const FieldM<Complex,1>& pion, const FieldM<Complex,4>& photon2,
    const double m_vector, const double f_pi)
  // vertex function
{
  // SADJUST ME
  // set_photon_pion_photon_vertex_two_end(photon1, pion, photon2, m_vector, f_pi);
  set_photon_pion_photon_vertex_vmd(photon1, pion, photon2, m_vector, f_pi);
}

inline void set_pion_field(FieldM<Complex,1>& pion_zxop, const FieldM<Complex,4*4>& pion_xy, const int rho, const int sigma)
{
  TIMER("set_pion_field");
  const Geometry geo = geo_reform(pion_xy.geo, 1);
  pion_zxop.init(geo);
  assert(pion_zxop.geo == geo);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    pion_zxop.get_elem(xl) = pion_xy.get_elem(xl, rho*4+sigma);
  }
}

inline void init_sl_limit(int& s_limit, int& l_limit, const Coordinate& total_site)
{
  s_limit = std::min(total_site[0]/2 + 9, total_site[0] + 1);
  l_limit = std::ceil(sqrt(distance_sq_relative_coordinate_g(total_site/2))) + 1;
  qassert(s_limit > 0);
  qassert(l_limit > 0);
}

inline std::vector<Complex> init_sl_table(const int s_limit, const int l_limit)
{
  TIMER("init_sl_table");
  std::vector<Complex> sl_table(s_limit * l_limit, 0.0);
  return sl_table;
}

inline void acc_sl_table(std::vector<Complex>& sl_table, const int s_limit, const int l_limit)
{
  TIMER("acc_sl_table");
  qassert(sl_table.size() == s_limit * l_limit);
  std::vector<Complex> l_table(l_limit, 0.0);
  for (int i = 0; i < s_limit; ++i) {
    Complex sum = 0.0;
    for (int j = 0; j < l_limit; ++j) {
      sum += sl_table[i*l_limit+j];
      l_table[j] = l_table[j] + sum;
      sl_table[i*l_limit+j] = l_table[j];
    }
  }
}

inline std::string show_sl_table(const std::vector<Complex>& sl_table, const int s_limit, const int l_limit)
{
  TIMER("show_sl_table");
  std::ostringstream out;
  for (int i = 0; i < s_limit; ++i) {
    for (int j = 0; j < l_limit; ++j) {
      if (j > 0) {
        out << " ";
      }
      out << show(sl_table[i*l_limit+j].real());
    }
    out << std::endl;
  }
  return out.str();
}

inline Complex photon_z_nu_contraction(std::vector<Complex>& sl_table, const Complex& coef,
    const FieldM<Complex,4>& p1, const FieldM<Complex,4>& p2, const Coordinate& x, const Coordinate& y)
{
  TIMER("photon_z_nu_contraction");
  const Geometry& geo = p1.geo;
  assert(geo == p2.geo);
  const Coordinate total_site = geo.total_site();
  int s_limit, l_limit;
  init_sl_limit(s_limit, l_limit, total_site);
  Complex sum = 0.0;
  const long dis_xy = distance_sq_relative_coordinate_g(relative_coordinate(x - y, total_site));
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    Complex c = 0.0;
    for (int nu = 0; nu < 4; ++nu) {
      c += coef * p1.get_elem(xl, nu) * p2.get_elem(xl, nu);
    }
    sum += c;
    const Coordinate xg = geo.coordinate_g_from_l(xl);
    long dis_max = dis_xy;
    long dis_min = dis_xy;
    const long dis_xz = distance_sq_relative_coordinate_g(relative_coordinate(x - xg, total_site));
    const long dis_yz = distance_sq_relative_coordinate_g(relative_coordinate(y - xg, total_site));
    dis_max = std::max(dis_max, dis_xz);
    dis_max = std::max(dis_max, dis_yz);
    dis_min = std::min(dis_min, dis_xz);
    dis_min = std::min(dis_min, dis_yz);
    const int l_len = std::ceil(std::sqrt(dis_max));
    const int s_len = std::ceil(std::sqrt(dis_min));
    if (s_len < s_limit && l_len < l_limit) {
      sl_table[s_len*l_limit+l_len] += c;
    }
  }
  return sum;
}

inline int hlbl_sample_xy_photon_z(const Coordinate& x, const Coordinate& y, const HlblPionModelInfo& info)
{
  const Coordinate& total_site = info.total_site;
  const Coordinate crel = relative_coordinate(y-x, total_site);
  const std::string fn1 = get_result_path() + "/" + info.path +
    ssprintf("/f2 ; sub=1 ; dis_sq=%010d ; rel=(%s).txt",
        distance_sq_relative_coordinate_g(crel), show(crel).c_str());
  const std::string fn2 = get_result_path() + "/" + info.path +
    ssprintf("/f2 ; sub=0 ; dis_sq=%010d ; rel=(%s).txt",
        distance_sq_relative_coordinate_g(crel), show(crel).c_str());
  if (does_file_exist_sync_node(fn1) and does_file_exist_sync_node(fn2)) {
    return 0;
  }
  TIMER_VERBOSE("hlbl_sample_xy_photon_z");
  const Geometry geo = Geometry(total_site, 1);
  FieldM<Complex,4> photon_x, photon_y;
  photon_x.init(geo);
  photon_y.init(geo);
  set_point_src(photon_x, x);
  set_point_src(photon_y, y);
  FieldM<Complex,4*4> pion_xy;
  set_pion_photon_photon_vertex(pion_xy, photon_x, photon_y, info.m_vector*info.a, info.f_pi*info.a);
  scalar_inversion(pion_xy, pion_xy, info.m_pi*info.a);
  std::array<FieldM<Complex,4>,3> ext_photons;
  set_external_photons(ext_photons, geo,
      middle_coordinate(CoordinateD(x), CoordinateD(y), CoordinateD(total_site)));
  FieldM<ManyMagneticMoments,1> mf1, mf2;
  mf1.init(geo);
  mf2.init(geo);
  set_m_z_field_tag(mf1, x, y, info.a, 0);
  set_m_z_field_tag(mf2, x, y, info.a, 1);
  FieldM<Complex,4> photon_muon_line, photon_z_nu;
  FieldM<Complex,1> pion_zxop;
  const double coef = 1.0E10 * info.a * std::pow(info.e_charge, 6) / 3.0;
  int s_limit, l_limit;
  init_sl_limit(s_limit, l_limit, total_site);
  std::vector<Complex> sl_table1 = init_sl_table(s_limit, l_limit);
  Complex sum1 = 0.0;
  std::vector<Complex> sl_table2 = init_sl_table(s_limit, l_limit);
  Complex sum2 = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int rho = 0; rho < 4; ++rho) {
      for (int sigma = 0; sigma < 4; ++sigma) {
        if (sigma == rho) {
          continue;
        }
        set_pion_field(pion_zxop, pion_xy, rho, sigma);
        set_photon_pion_photon_vertex(photon_z_nu, pion_zxop, ext_photons[i], info.m_vector*info.a, info.f_pi*info.a);
        set_muon_line_photon(photon_muon_line, mf1, i, rho, sigma);
        sum1 += photon_z_nu_contraction(sl_table1, 3.0 * coef, photon_muon_line, photon_z_nu, x, y);
        // 3.0 is for the 3 function
        set_muon_line_photon(photon_muon_line, mf2, i, rho, sigma);
        sum2 += photon_z_nu_contraction(sl_table2, 3.0 * coef, photon_muon_line, photon_z_nu, x, y);
        // 3.0 is for the 3 function
      }
    }
  }
  glb_sum(sum1);
  glb_sum(sum2);
  glb_sum_double_vec(get_data(sl_table1));
  glb_sum_double_vec(get_data(sl_table2));
  acc_sl_table(sl_table1, s_limit, l_limit);
  acc_sl_table(sl_table2, s_limit, l_limit);
  displayln_info(fname + ssprintf(": point-pair-coordinates: x=") + show(x) + " ; y=" + show(y) + " ; sum1=" + show(sum1) + " ; sum2=" + show(sum2));
  qtouch_info(fn1, ssprintf("# %24.17E %24.17E\n", sum1.real(), sum1.imag()) + show_sl_table(sl_table1, s_limit, l_limit));
  qtouch_info(fn2, ssprintf("# %24.17E %24.17E\n", sum2.real(), sum2.imag()) + show_sl_table(sl_table2, s_limit, l_limit));
  return 1;
}

inline int hlbl_sample_xy(const Coordinate& x, const Coordinate& y, const HlblPionModelInfo& info)
  // interface
{
  // ADJUST ME
  // hlbl_sample_xy_pion_v(x, y, info);
  return hlbl_sample_xy_photon_z(x, y, info);
}

template <class M>
inline std::vector<M> merge_vector(const std::vector<M>& v1, const std::vector<M>& v2)
{
  std::vector<M> ret(v1.size() + v2.size());
  const long size = std::max(v1.size(), v2.size());
  long idx = 0;
  for (long i = 0; i < size; ++i) {
    if (i < v1.size()) {
      ret[idx] = v1[i];
      idx += 1;
    }
    if (i < v2.size()) {
      ret[idx] = v2[i];
      idx += 1;
    }
  }
  qassert(idx == ret.size());
  return ret;
}

template <class M>
inline std::vector<M> append_vector(const std::vector<M>& v1, const std::vector<M>& v2)
{
  std::vector<M> ret(v1.size() + v2.size());
  long idx = 0;
  for (long i = 0; i < v1.size(); ++i) {
    ret[idx] = v1[i];
    idx += 1;
  }
  for (long i = 0; i < v2.size(); ++i) {
    ret[idx] = v2[i];
    idx += 1;
  }
  qassert(idx == ret.size());
  return ret;
}

inline void save_long_dis_coords_probs(const HlblPionModelInfo& info)
{
  TIMER_VERBOSE("save_long_dis_coords_probs");
  if (get_id_node() == 0) {
    const std::string fn = get_result_path() + "/" + info.path + "/long_dis_coords.txt";
    FILE* fp = qopen(fn + ".partial", "w");
    qassert(fp != NULL);
    const long size = info.long_dis_coords.size();
    qassert(size == info.long_dis_probs.size());
    displayln(ssprintf("# formula total prob: %24.17E", info.prob_sum), fp);
    displayln(ssprintf("# formula a: %24.17E", info.a), fp);
    displayln(ssprintf("# formula m_vector: %24.17E", info.m_vector), fp);
    displayln(ssprintf("# formula short_limit_sq: %24.17E", (double)info.short_limit_sq), fp);
    displayln(ssprintf("# xgrel[0] xgrel[1] xgrel[2] xgrel[3] prob"), fp);
    for (long traj = 0; traj < size; ++traj) {
      const Coordinate& xgrel = info.long_dis_coords[traj];
      const double& prob = info.long_dis_probs[traj];
      displayln(ssprintf("%5d %5d %5d %5d  %24.17E", xgrel[0], xgrel[1], xgrel[2], xgrel[3], prob), fp);
    }
    qclose(fp);
    qrename(fn + ".partial", fn);
  }
}

inline double long_dis_prob_func(const Coordinate xgrel, const HlblPionModelInfo& info)
  // prob function
{
  const long dis_sq = distance_sq_relative_coordinate_g(xgrel);
  if (dis_sq <= info.short_limit_sq) {
    return 0.0;
  } else {
    const double dis = std::sqrt(dis_sq);
    const double dis_short_limit = std::sqrt(info.short_limit_sq);
    // SADJUST ME
    return exp(- info.m_vector * info.a * (dis - dis_short_limit)) / pow(dis / dis_short_limit, 1);
  }
}

inline void set_long_dis_coords_probs(HlblPionModelInfo& info, const long num_long_dis_coords)
{
  TIMER_VERBOSE("set_long_dis_coords_probs");
  const Geometry geo(info.total_site, 1);
  const Coordinate total_site = geo.total_site();
  FieldM<Complex,1> fprob;
  fprob.init(geo);
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Coordinate xg = geo.coordinate_g_from_l(xl);
    const Coordinate xgrel = relative_coordinate(xg, total_site);
    fprob.get_elem(xl) = long_dis_prob_func(xgrel, info);
  }
  const Complex fprob_sum = field_glb_sum_double(fprob)[0];
  info.prob_sum = fprob_sum.real();
  fprob *= 1.0 / fprob_sum;
  const RngState rs(get_global_rng_state(), fname + ssprintf(
      ": a=%24.17E ; m_pi=%24.17E ; f_pi=%24.17E ; m_vector=%24.17E",
      info.a, info.m_pi, info.f_pi, info.m_vector));
  clear(info.long_dis_coords);
  clear(info.long_dis_probs);
  info.long_dis_coords.resize(num_long_dis_coords);
  info.long_dis_probs.resize(num_long_dis_coords);
  for (long traj = 0; traj < num_long_dis_coords; ++traj) {
    RngState rst = rs.newtype(traj);
    Coordinate xg, xgrel;
    double prob;
    do {
      for (int mu = 0; mu < 4; ++mu) {
        xg[mu] = rand_gen(rst) % total_site[mu];
      }
      xgrel = relative_coordinate(xg, total_site);
      prob = long_dis_prob_func(xgrel, info);
    } while (prob <= u_rand_gen(rst));
    for (int mu = 0; mu < 4; ++mu) {
      xgrel[mu] = std::abs(xgrel[mu]);
    }
    std::sort(xgrel.begin(), xgrel.begin() + 3);
    info.long_dis_coords[traj] = xgrel;
    info.long_dis_probs[traj] = prob / info.prob_sum;
    // to use the info.long_dis_probs[traj]
    // analysis code need to consider
    // 1: the total number of points
    // 2: the multiplicity from discrete symmetry
  }
}

inline void run_form_pi_decay_with_info(HlblPionModelInfo& info)
{
  TIMER_VERBOSE("run_form_pi_decay_with_info");
  display_hlbl_pion_model_info(info);
  if (obtain_lock(get_result_path() + "/" + info.path + "-lock")) {
    qmkdir_info(get_result_path() + "/" + info.path);
    info.short_dis_coords = generate_short_dis_coords(info.short_limit_sq);
    set_long_dis_coords_probs(info, 1024);
    save_long_dis_coords_probs(info);
    // ADJUST ME
    // const std::vector<Coordinate> coords = merge_vector(info.short_dis_coords, info.long_dis_coords);
    const std::vector<Coordinate> coords = append_vector(info.short_dis_coords, info.long_dis_coords);
    const Coordinate x(0, 0, 0, 0);
    int sum = 0;
    for (long i = 0; i < coords.size(); ++i) {
      const Coordinate& y = coords[i];
      sum += hlbl_sample_xy(x, y, info);
      if (sum >= 15) {
        break;
      }
    }
    release_lock();
    Timer::display();
  }
}

inline void run_form_pi_decay()
{
  TIMER_VERBOSE("run_form_pi_decay");
  init_muon_line_pi_decay();
  std::vector<HlblPionModelInfo> infos;
  HlblPionModelInfo info;
  // SADJUST ME
  // info.init_8nt16(); infos.push_back(info);
  // info.init_16nt32(); infos.push_back(info);
  // info.init_16nt32_coarse_3_phys(); infos.push_back(info);
  // info.init_20nt40_coarse_2p5_phys(); infos.push_back(info);
  // info.init_24nt64(); infos.push_back(info);
  // info.init_24nt48_fine(); infos.push_back(info);
  // info.init_24nt48_coarse_2_phys(); infos.push_back(info);
  // info.init_28nt56_coarse_1p5_phys(); infos.push_back(info);
  // info.init_32nt64(); infos.push_back(info);
  // info.init_32nt64_fine_2(); infos.push_back(info);
  // info.init_32nt64_coarse_phys(); infos.push_back(info);
  // info.init_32nt64_coarse_2_phys(); infos.push_back(info);
  // info.init_40nt80_coarse_0p5_phys(); infos.push_back(info);
  // info.init_48nt96(); infos.push_back(info);
  info.init_48nt96_phys(); infos.push_back(info);
  // info.init_48nt96_coarse_phys(); infos.push_back(info);
  // info.init_48nt96_coarse_2_phys(); infos.push_back(info);
  info.init_64nt128_coarse_2_phys(); infos.push_back(info);
  RngState rs(get_global_rng_state(), fname);
  for (int i = 0; i < 128; ++i) {
    split_rng_state(rs, rs, show(get_time()));
    long choice = 0;
    if (get_id_node() == 0) {
      choice = rand_gen(rs) % infos.size();
    }
    glb_sum(choice);
    run_form_pi_decay_with_info(infos[choice]);
  }
}

QLAT_END_NAMESPACE

inline qlat::CoordinateD shift_coordinate(const qlat::CoordinateD& x, const int mu, const double eps)
{
  qlat::CoordinateD nx = x;
  nx[mu] += eps;
  return nx;
}

inline ManyMagneticMoments get_ddd_m(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z,
    const int alpha, const int beta, const int gamma, const double eps)
{
  TIMER("compute_ddd_m");
  using namespace qlat;
  ManyMagneticMoments mmm =
      get_muon_line_m(shift_coordinate(x, alpha, eps), shift_coordinate(y, beta, eps), shift_coordinate(z, gamma, eps))
    - get_muon_line_m(shift_coordinate(x, alpha,-eps), shift_coordinate(y, beta, eps), shift_coordinate(z, gamma, eps))
    - get_muon_line_m(shift_coordinate(x, alpha, eps), shift_coordinate(y, beta,-eps), shift_coordinate(z, gamma, eps))
    - get_muon_line_m(shift_coordinate(x, alpha, eps), shift_coordinate(y, beta, eps), shift_coordinate(z, gamma,-eps))
    + get_muon_line_m(shift_coordinate(x, alpha,-eps), shift_coordinate(y, beta,-eps), shift_coordinate(z, gamma, eps))
    + get_muon_line_m(shift_coordinate(x, alpha,-eps), shift_coordinate(y, beta, eps), shift_coordinate(z, gamma,-eps))
    + get_muon_line_m(shift_coordinate(x, alpha, eps), shift_coordinate(y, beta,-eps), shift_coordinate(z, gamma,-eps))
    - get_muon_line_m(shift_coordinate(x, alpha,-eps), shift_coordinate(y, beta,-eps), shift_coordinate(z, gamma,-eps));
  return 1.0/(8.0*eps*eps*eps) * mmm;
}

inline ManyMagneticMoments get_d_m_1(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z,
    const int alpha, const double eps)
{
  TIMER("compute_d_m");
  using namespace qlat;
  ManyMagneticMoments mmm =
      get_muon_line_m(shift_coordinate(x, alpha, eps), y, z)
    - get_muon_line_m(shift_coordinate(x, alpha,-eps), y, z);
  return 1.0/(2.0*eps) * mmm;
}

inline ManyMagneticMoments get_d_m_2(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z,
    const int alpha, const double eps)
{
  TIMER("compute_d_m");
  using namespace qlat;
  ManyMagneticMoments mmm =
      get_muon_line_m(shift_coordinate(x, alpha, eps), shift_coordinate(y, alpha,-eps), z)
    - get_muon_line_m(shift_coordinate(x, alpha,-eps), shift_coordinate(y, alpha,+eps), z);
  return 1.0/(4.0*eps) * mmm;
}

inline ManyMagneticMoments get_d_m(
    const qlat::CoordinateD& x, const qlat::CoordinateD& y, const qlat::CoordinateD& z,
    const int alpha, const double eps)
{
  // return get_d_m_1(x, y, z, alpha, eps);
  return get_d_m_2(x, y, z, alpha, eps);
}

inline double get_pion_propagator(const qlat::CoordinateD& u, const qlat::CoordinateD& v, const double mass)
{
  const double xlen = qlat::coordinate_len(u - v);
  return mass / (4 * qlat::sqr(qlat::PI) * xlen) * specialK1(mass * xlen);
}

inline double get_d_pion_propagator(
    const qlat::CoordinateD& u, const qlat::CoordinateD& v, const double mass,
    const int gamma, const double reps)
{
  const double eps = reps * qlat::coordinate_len(v - u);
  double dp = get_pion_propagator(u, shift_coordinate(v, gamma, eps), mass)
            - get_pion_propagator(u, shift_coordinate(v, gamma,-eps), mass);
  return 1.0/(2.0*eps) * dp;
}

inline double get_dd_pion_propagator(
    const qlat::CoordinateD& u, const qlat::CoordinateD& v, const double mass,
    const int beta, const int gamma, const double reps)
{
  const double eps = reps * qlat::coordinate_len(v - u);
  double dp = get_pion_propagator(shift_coordinate(u, beta, eps), shift_coordinate(v, gamma, eps), mass)
            - get_pion_propagator(shift_coordinate(u, beta,-eps), shift_coordinate(v, gamma, eps), mass)
            - get_pion_propagator(shift_coordinate(u, beta, eps), shift_coordinate(v, gamma,-eps), mass)
            + get_pion_propagator(shift_coordinate(u, beta,-eps), shift_coordinate(v, gamma,-eps), mass);
  return 1.0/(4.0*qlat::sqr(eps)) * dp;
}

inline double get_pion_contribution(const qlat::CoordinateD& u, const qlat::CoordinateD& v, const double d_m_eps = 0.05)
  // point form factor
{
  const double dd_pion_eps = 0.01;
  const double m_mu_mev = 105.6583745;
  const double f_pi = 92.0 / m_mu_mev;
  // ADJUST ME
  const double m_pi = 134.9766 / m_mu_mev;
  // const double m_pi = 900 / m_mu_mev;
  const double alpha_inv = 137.035999139;
  const double e_charge = std::sqrt(4 * qlat::PI / alpha_inv);
  double sum = 0.0;
  for (int alpha = 0; alpha < 4; ++alpha) {
    const ManyMagneticMoments d_m = get_d_m(u, u, v, alpha, d_m_eps);
    for (int beta = 0; beta < 4; ++beta) {
      if (alpha == beta) {
        continue;
      }
      for (int gamma = 0; gamma < 4; ++gamma) {
        const double dd_pion_prop = get_dd_pion_propagator(u, v, m_pi, beta, gamma, dd_pion_eps);
        for (int rho = 0; rho < 4; ++rho) {
          if (rho == alpha || rho == beta) {
            continue;
          }
          for (int sigma = 0; sigma < 4; ++sigma) {
            if (sigma == alpha || sigma == beta || sigma == rho) {
              continue;
            }
            for (int nu = 0; nu < 4; ++nu) {
              if (nu == gamma) {
                continue;
              }
              for (int i = 0; i < 3; ++i) {
                const double d_m_comp = get_m_comp(d_m, i, rho, sigma, nu);
                // TODO
                for (int j = 0; j < 3; ++j) {
                  if (j == i || j == gamma || j == nu) {
                    continue;
                  }
                  for (int k = 0; k < 3; ++k) {
                    if (k == i || k == j || k == gamma || k == nu) {
                      continue;
                    }
                    sum += dd_pion_prop
                      * qlat::epsilon_tensor(i, j, k)
                      * qlat::epsilon_tensor(alpha, rho, beta, sigma)
                      * qlat::epsilon_tensor(gamma, nu, j, k)
                      * d_m_comp;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  const double coef = 1.0E10 * 2.0 * std::pow(e_charge, 6) * (-1.0)/qlat::sqr(2.0 * qlat::sqr(qlat::PI) * 2.0 * f_pi) * 1.0/2.0;
  // result in unit of 1E-10
  return coef * sum;
}

inline double compute_rt(const double rspan, const double tspan)
  // point form factor
{
  using namespace qlat;
  TIMER("compute_rt");
  const RngState rs = RngState(get_global_rng_state(), fname);
  // displayln(ssprintf("%s ; rspan=%lf ; tspan=%lf", fname, rspan, tspan));
  double sum = 0.0;
  long count = 0;
  for (int i = 0; i < 16; ++i) {
    RngState rsi(rs, i);
    double a = 0, b = 0, c = 0;
    double len = 0;
    while (len < 0.1) {
      a = g_rand_gen(rsi);
      b = g_rand_gen(rsi);
      c = g_rand_gen(rsi);
      len = std::sqrt(sqr(a) + sqr(b) + sqr(c));
    }
    a *= rspan/len;
    b *= rspan/len;
    c *= rspan/len;
    const qlat::CoordinateD x;
    const qlat::CoordinateD z(a,b,c,tspan);
    // ADJUST ME
    // const double val = get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/15.0)); displayln(show(val));
    double val;
    // displayln("start");
    // val = get_pion_contribution(x, z, 0.1); displayln(show(val));
    // val = get_pion_contribution(x, z, 0.05); displayln(show(val));
    // val = get_pion_contribution(x, z, 0.01); displayln(show(val));
    // val = get_pion_contribution(x, z, 0.005); displayln(show(val));
    // val = get_pion_contribution(x, z, 0.001); displayln(show(val));
    // val = get_pion_contribution(x, z, 0.0005); displayln(show(val));
    // val = get_pion_contribution(x, z, 0.0001); displayln(show(val));
    // displayln("end");
    val = get_pion_contribution(x, z, 0.01);
    sum += val;
    count += 1;
  }
  return sum / count;
  // for (size_t interp_idx = 0; interp_idx < get_muonline_interps().size(); ++interp_idx) {
  //   displayln(ssprintf("interp_idx = %d", interp_idx));
  //   get_default_muonline_interp_idx() = interp_idx;
  //   // displayln(showManyMagneticMoments(get_ddd_m(x, y, z, 0, 1, 2, 0.2)));
  //   // displayln(showManyMagneticMoments(get_ddd_m(x, y, z, 0, 1, 2, 0.1)));
  //   // displayln(showManyMagneticMoments(get_ddd_m(x, y, z, 0, 1, 2, 0.05)));
  //   // displayln(showManyMagneticMoments(get_ddd_m(x, y, z, 0, 1, 2, 0.01)));
  //   // displayln(showManyMagneticMoments(get_ddd_m(x, y, z, 0, 1, 2, 0.001)));
  //   // displayln(showManyMagneticMoments(get_d_m(x, y, z, 0, 0.2)));
  //   // displayln(showManyMagneticMoments(get_d_m(x, y, z, 0, 0.1)));
  //   // displayln(showManyMagneticMoments(get_d_m(x, y, z, 0, 0.05)));
  //   // displayln(showManyMagneticMoments(get_d_m(x, y, z, 0, 0.01)));
  //   // displayln(showManyMagneticMoments(get_d_m(x, y, z, 0, 0.001)));
  //   // displayln(show(get_pion_contribution(x, z, 0.2)));
  //   // displayln(show(get_pion_contribution(x, z, 0.1)));
  //   // displayln(show(get_pion_contribution(x, z, 0.05)));
  //   // displayln(show(get_pion_contribution(x, z, 0.02)));
  //   // displayln(show(get_pion_contribution(x, z, 0.01)));
  //   // displayln(show(get_pion_contribution(x, z, 0.005)));
  //   // displayln(show(get_pion_contribution(x, z, 0.001)));
  //   // displayln(show(get_pion_contribution(x, z, 0.0005)));
  //   // displayln(show(get_pion_contribution(x, z, 0.0001)));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/5.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/7.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/9.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/11.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/13.0))));
  //   displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/15.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(2.0/11.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(2.0/13.0))));
  //   // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(2.0/15.0))));
  // }
  // displayln(show(get_pion_contribution(x, z, coordinate_len(x-z)*qlat::sqr(1.0/15.0))));
}

inline void run_point_pi_decay()
{
  using namespace qlat;
  TIMER_VERBOSE("run_point_pi_decay");
  init_muon_line_pi_decay();
  // compute_rt(0.5, 1.0);
  const double dt = 0.5 * 0.1056583745 / 1.730;
  const double dr = dt;
  const double dlen = 0.1056583745 / 1.730;
  const double len_limit = 2;
  const size_t sums_size = len_limit / dlen + 1;
  std::vector<double> sums(sums_size);
  for (double t = dt/2-len_limit; t < len_limit; t += dt) {
    for (double r = dr/2; r < len_limit; r += dr) {
      if (sqr(r) + sqr(t) > sqr(len_limit) || sqr(r) + sqr(t) < sqr(0.1)) {
        continue;
      }
      const double avg = compute_rt(r, t);
      displayln(ssprintf("%s-avg ; r=%lf ; t=%lf ; avg=%24.17E", fname, r, t, avg));
      // displayln(ssprintf("%s-int ; r=%lf ; t=%lf ; int=%24.17E", fname, r, t, avg * 4 * PI * sqr(r)));
      const double len = std::sqrt(sqr(r) + sqr(t));
      sums[len/dlen] += dr * dt * avg * 4 * PI * sqr(r);
    }
  }
  double sum = 0;
  for (int i = sums_size - 1; i >= 0; i -= 1) {
    sum += sums[i];
    displayln(ssprintf("%5d %lf %24.17E %24.17E", i, i * dlen, sums[i], sum));
  }
}

inline void run_pi_decay()
{
  // run_point_pi_decay();
  qlat::run_form_pi_decay();
}
