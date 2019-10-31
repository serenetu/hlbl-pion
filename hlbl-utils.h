#ifndef _HLBL-UTILS_H
#define _HLBL-UTILS_H

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

struct PionGGElem
{
  Complex v[4][4]; // v[mu][nu]

  PionGGElem& operator+=(const PionGGElem& x)
  {
#pragma omp parallel for
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        this -> v[mu][nu] += x.v[mu][nu];
      }
    }
    return *this;
  }

  PionGGElem& operator-=(const PionGGElem& x)
  {
#pragma omp parallel for
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        this -> v[mu][nu] -= x.v[mu][nu];
      }
    }
    return *this;
  }

  PionGGElem& operator*=(const Complex& x)
  {
#pragma omp parallel for
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        this -> v[mu][nu] *= x;
      }
    }
    return *this;
  }
  PionGGElem& operator/=(const Complex& x)
  {
#pragma omp parallel for
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        this -> v[mu][nu] /= x;
      }
    }
    return *this;
  }
};

inline std::string show_pgge(const PionGGElem& pgge)
{
  std::string out("");
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      out += ssprintf("(%e, %e) ", (pgge.v[mu][nu]).real(), (pgge.v[mu][nu]).imag());
    }
    out += "\n";
  }
  return out;
}

PionGGElem operator*(const Complex& x, const PionGGElem& p)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      res.v[mu][nu] = x * p.v[mu][nu];
    }
  }
  return res;
}

PionGGElem operator*(const PionGGElem& p, const Complex& x)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      res.v[mu][nu] = x * p.v[mu][nu];
    }
  }
  return res;
}

PionGGElem operator*(const PionGGElem& p, const double& x)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      res.v[mu][nu] = x * p.v[mu][nu];
    }
  }
  return res;
}

PionGGElem operator*(const double& x, const PionGGElem& p)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int mu = 0; mu < 4; ++mu) {
    for (int nu = 0; nu < 4; ++nu) {
      res.v[mu][nu] = x * p.v[mu][nu];
    }
  }
  return res;
}

PionGGElem operator+(const PionGGElem& x, const PionGGElem& y)
{
  PionGGElem res;
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    int mu = i / 4;
    int nu = i % 4;
    res.v[mu][nu] = x.v[mu][nu] + y.v[mu][nu];
  }
  return res;
}

PionGGElem operator-(const PionGGElem& x, const PionGGElem& y)
{
  PionGGElem res;
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    int mu = i / 4;
    int nu = i % 4;
    res.v[mu][nu] = x.v[mu][nu] - y.v[mu][nu];
  }
  return res;
}

PionGGElem operator*(const PionGGElem& x, const ManyMagneticMoments& mmm)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int imu = 0; imu < 12; ++imu) {
    int i = imu / 4;
    int mu = imu % 4;
    for (int nu = 0; nu < 4; ++nu) {
      for (int nup = 0; nup < 4; ++nup) {
        res.v[i][mu] += x.v[nu][nup] * mmm[16 * nup + 4 * mu + nu][i];
      }
    }
  }
  return res;
}

struct PionGGElemField : FieldM<PionGGElem,1>
{
  virtual const std::string& cname()
  {
    static const std::string s = "PionGGElemField";
    return s;
  }

  PionGGElemField& operator/=(const Complex& x)
  {
    const Geometry& geo = this -> geo;
    const Coordinate total_site = geo.total_site();
#pragma omp parallel for
    for (long index = 0; index < geo.local_volume(); ++index)
    {
      const Coordinate lxp = geo.coordinate_from_index(index);
      PionGGElem& pgge = this -> get_elem(lxp);
      pgge /= x;
    }
    sync_node();
    return *this;
  }

  PionGGElemField& operator+=(const PionGGElemField& x)
  {
    const Geometry& geo = this -> geo;
    const Coordinate total_site = geo.total_site();
#pragma omp parallel for
    for (long index = 0; index < geo.local_volume(); ++index)
    {
      const Coordinate lxp = geo.coordinate_from_index(index);
      PionGGElem& pgge = this -> get_elem(lxp);
      pgge += x.get_elem(lxp);
    }
    sync_node();
    return *this;
  }
};

struct Complex_Table
{
  // Complex c[NUM_RMAX][NUM_RMIN];
  std::vector<std::vector<Complex>> c;

  Complex_Table (int r_max, int r_min) {
    c.resize(r_max);
    for (int row = 0; row < r_max; ++row) {
      c[row].resize(r_min);
    }
  }

  void set_zero() {
#pragma omp parallel for
    for (int row = 0; row < c.size(); ++row) {
      for (int col = 0; col < c[row].size(); ++col) {
        qlat::set_zero(c[row][col]);
      }
    }
  }

  Complex_Table& operator*(const Complex& complex_val)
  {
    int num_rmax = c.size();
    int num_rmin = c[0].size();

#pragma omp parallel for
    for (int rmax = 0; rmax < num_rmax; ++rmax) {
      for (int rmin = 0; rmin < num_rmin; ++rmin) {
        Complex& e = this -> c[rmax][rmin];
        e *= complex_val;
      }
    }
    return *this;
  }

  Complex_Table& operator+=(const Complex_Table& other)
  {
    int num_rmax = c.size();
    int num_rmin = c[0].size();
    qassert(num_rmax == other.c.size() && num_rmin == other.c[0].size());
#pragma omp parallel for
    for (int rmax = 0; rmax < num_rmax; ++rmax) {
      for (int rmin = 0; rmin < num_rmin; ++rmin) {
        this -> c[rmax][rmin] += other.c[rmax][rmin];
      }
    }
    return *this;
  }

  void write_to_disk(std::string path) {
    int num_rmax = c.size();
    int num_rmin = c[0].size();
    Complex c_list[num_rmax][num_rmin];
    for (int row = 0; row < num_rmax; ++row) {
      for (int col = 0; col < num_rmin; ++col) {
        c_list[row][col] = c[row][col];
      }
    }
    write_data_from_0_node((Complex *)&c_list[0][0], num_rmax * num_rmin, path);
    return;
  }
};

Complex_Table operator*(const Complex& complex_val, const Complex_Table& table)
{
  int num_rmax = table.c.size();
  int num_rmin = table.c[0].size();
  Complex_Table res(num_rmax, num_rmin);
  res.set_zero();
#pragma omp parallel for
  for (int rmax = 0; rmax < num_rmax; ++rmax) {
    for (int rmin = 0; rmin < num_rmin; ++rmin) {
      const Complex& e = table.c[rmax][rmin];
      Complex& r = res.c[rmax][rmin];
      r = complex_val * e;
    }
  }
  return res;
}

std::string show_complex_table(const Complex_Table& complex_table)
{
  int num_rmax = complex_table.c.size();
  int num_rmin = complex_table.c[0].size();
  std::string res = "";
  for (int rmax = 0; rmax < num_rmax; ++rmax)
  {
    for (int rmin = 0; rmin < num_rmin; ++rmin)
    {
      res += ssprintf("%24.17e %24.17e, ", (complex_table.c[rmax][rmin]).real(), (complex_table.c[rmax][rmin]).imag());
    }
    res += "\n";
  }
  return res;
}

inline std::string get_point_src_props_path(const std::string& job_tag,
                                            const int traj)
{
  std::string path;
  if (job_tag == "24D-0.00107") {
    path =
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results";
    if (does_file_exist_sync_node(
            path +
            ssprintf("/results=%d/checkpoint/computeContractionInf", traj)) and
        does_file_exist_sync_node(
            path + ssprintf("/prop-hvp ; results=%d/on-disk", traj))) {
      return path +
             ssprintf("/prop-hvp ; results=%d/huge-data/prop-point-src", traj);
    }
  } else if (job_tag == "24D-0.0174") {
    path =
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D-0.0174/discon-2/results";
    if (does_file_exist_sync_node(
            path +
            ssprintf("/results=%d/checkpoint/computeContractionInf", traj)) and
        does_file_exist_sync_node(
            path + ssprintf("/prop-hvp ; results=%d/on-disk", traj))) {
      return path +
             ssprintf("/prop-hvp ; results=%d/huge-data/prop-point-src", traj);
    }
  } else if (job_tag == "32D-0.00107") {
    path =
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/32D/discon-1/results";
    if (does_file_exist_sync_node(
            path +
            ssprintf("/results=%d/checkpoint/computeContractionInf", traj)) and
        does_file_exist_sync_node(
            path + ssprintf("/prop-hvp ; results=%d/on-disk", traj))) {
      return path +
             ssprintf("/prop-hvp ; results=%d/huge-data/prop-point-src", traj);
    }
  } else if (job_tag == "32Dfine-0.0001") {
    path =
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/32Dfine/discon-1/"
        "results";
    if (does_file_exist_sync_node(
            path +
            ssprintf("/results=%d/checkpoint/computeContractionInf", traj)) and
        does_file_exist_sync_node(
            path + ssprintf("/prop-hvp ; results=%d/on-disk", traj))) {
      return path +
             ssprintf("/prop-hvp ; results=%d/huge-data/prop-point-src", traj);
    }
  } else if (job_tag == "48I-0.00078") {
    qassert(traj % 5 == 0);
    path =
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/48I/"
        "discon-strange-2-new/results";
    if (does_file_exist_sync_node(
            path + ssprintf("/results=%d/checkpoint/computeContractionInf",
                            traj / 5)) and
        does_file_exist_sync_node(
            path + ssprintf("/prop-hvp ; results=%d/on-disk", traj / 5))) {
      return path + ssprintf("/prop-hvp ; results=%d/huge-data/prop-point-src",
                             traj / 5);
    }
  } else {
    qassert(false);
  }
  return "";
}

inline std::string get_point_src_prop_path(
    const std::string& job_tag, 
    const int traj,
    const Coordinate& xg, 
    const int type, 
    const int accuracy)
{
  std::string path = get_point_src_props_path(job_tag, traj);
  if (path == "") {return "";}
  std::string full_path = path + ssprintf("/xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", xg[0], xg[1], xg[2], xg[3], type, accuracy);
  if (does_file_exist_sync_node(full_path)) {return full_path;}
  return "";
}

inline std::string get_gauge_transform_path(const std::string& job_tag,
                                            const int traj)
{
  std::string path;
  if (job_tag == "24D-0.00107") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/"
        "results=%d/huge-data/gauge-transform",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint")) {
      return path;
    }
  } else if (job_tag == "24D-0.0174") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D-0.0174/results/24D-0.0174/"
        "results=%d/huge-data/gauge-transform",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint")) {
      return path;
    }
  } else if (job_tag == "32D-0.00107") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/"
        "32D-0.00107/results=%d/huge-data/gauge-transform",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint")) {
      return path;
    }
  } else if (job_tag == "32Dfine-0.0001") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/32Dfine/wall-src/"
        "results/32Dfine-0.0001/results=%d/huge-data/gauge-transform",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint")) {
      return path;
    }
  } else if (job_tag == "48I-0.00078") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/48I/wall-src/results/"
        "48I-0.00078/results=%d/huge-data/gauge-transform",
        traj);
    if (does_file_exist_sync_node(path)) {
      return path;
    }
  } else {
    qassert(false);
  }
  return "";
}

const Propagator4d& get_point_prop(const std::string& path, const Coordinate& c)
{
  return get_prop(path + "/xg=" + show_coordinate(c) + " ; type=0 ; accuracy=0");
}

void load_gauge_inv(GaugeTransform& gtinv, const std::string& ensemble, const int traj) {
  TIMER_VERBOSE("load_gauge_inv");
  const std::string gauge_transform_path = get_gauge_transform_path(ensemble, traj);
  qassert(gauge_transform_path != "");
  GaugeTransform gt;
  read_field(gt, gauge_transform_path);
  to_from_big_endian_64(get_data(gt));
  gt_invert(gtinv, gt);
}

inline std::string get_wall_src_props_sloppy_path(const std::string& job_tag, const int traj)
{
  std::string path;
  if (job_tag == "24D-0.00107") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/"
        "results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "24D-0.0174") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D-0.0174/results/24D-0.0174/"
        "results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "32D-0.00107") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/"
        "32D-0.00107/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "32Dfine-0.0001") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/32Dfine/wall-src/"
        "results/32Dfine-0.0001/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "48I-0.00078") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/48I/wall-src/results/"
        "48I-0.00078/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else {
    qassert(false);
  }
  return "";
}

inline std::string get_wall_src_prop_sloppy_path(const std::string& job_tag, const int traj, const int tslice)
{
  const std::string path_config = get_wall_src_props_sloppy_path(job_tag, traj);
  if (path_config == "") {return "";}
  const std::string path =
      path_config + ssprintf("/t=%d", tslice);
  //
  // if (does_file_exist_sync_node(path + "/checkpoint")) {
    // return path;
  // }
  // return "";
  if (does_file_exist_sync_node(path)) {
    return path;
  }
  return "";
}

void load_wall_sloppy_props(std::vector<Propagator4d>& wall_src_list, const GaugeTransform& gtinv, const std::string ensemble, const int traj) {
  TIMER_VERBOSE("load_wall_sloppy_props");
  const Coordinate total_site = gtinv.geo.total_site();
  wall_src_list.resize(total_site[3]);
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string wall_src_t_path = get_wall_src_prop_sloppy_path(ensemble, traj, t_wall);
    qassert(wall_src_t_path != "");
    main_displayln_info("Read Sloppy Wall From: " + wall_src_t_path);
    Propagator4d& wall_src_prop = wall_src_list[t_wall];
    wall_src_prop = get_wall_prop(wall_src_t_path);
    Propagator4d wall_src_prop_gauge;
    prop_apply_gauge_transformation(wall_src_prop_gauge, wall_src_prop, gtinv);
    wall_src_prop = wall_src_prop_gauge;
  }
}

inline std::string get_wall_src_props_exact_path(const std::string& job_tag, const int traj)
{
  std::string path;
  if (job_tag == "24D-0.00107") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src-exact-2/"
        "results/24D-0.00107/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "24D-0.0174") {
    return get_wall_src_props_sloppy_path(job_tag, traj);
  } else if (job_tag == "32D-0.00107") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src-exact-2/"
        "results/32D-0.00107/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "32Dfine-0.0001") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/32Dfine/wall-src/"
        "results/32Dfine-0.0001/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else if (job_tag == "48I-0.00078") {
    path = ssprintf(
        "/home/ljin/application/Public/Qlat-CPS-cc/jobs/48I/wall-src/results/"
        "48I-0.00078/results=%d",
        traj);
    if (does_file_exist_sync_node(path + "/checkpoint.txt")) {
      return path + "/huge-data/wall_src_propagator";
    }
  } else {
    qassert(false);
  }
  return "";
}

inline std::string get_wall_src_prop_exact_path(const std::string& job_tag, const int traj, const int tslice)
{
  const std::string path_config = get_wall_src_props_exact_path(job_tag, traj);
  if (path_config == "") {return "";}
  const std::string path =
      path_config +
      ssprintf("/exact ; t=%d", tslice);
  // if (does_file_exist_sync_node(path + "/checkpoint")) {
    // return path;
  // }
  if (does_file_exist_sync_node(path)) {
    return path;
  }
  return "";
}

void load_wall_exact_props(
    std::vector<int>& exact_wall_t_list, 
    std::vector<Propagator4d>& exact_wall_src_list, 
    const GaugeTransform& gtinv, 
    const std::string ensemble, 
    const int traj) 
{
  TIMER_VERBOSE("load_wall_exact_props");
  const Coordinate total_site = gtinv.geo.total_site();

  int num_exact = 0;
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string exact_wall_src_t_path = get_wall_src_prop_exact_path(ensemble, traj, t_wall);
    if (exact_wall_src_t_path == "") {continue;}
    exact_wall_t_list.push_back(t_wall);
    num_exact += 1;
  }
  exact_wall_src_list.resize(num_exact);
  for (int i = 0; i < num_exact; ++i){
    int t_wall = exact_wall_t_list[i];
    std::string exact_wall_src_t_path = get_wall_src_prop_exact_path(ensemble, traj, t_wall);
    qassert(exact_wall_src_t_path != "");
    Propagator4d& exact_wall_src_prop = exact_wall_src_list[i];
    main_displayln_info("Read Exact Wall From: " + exact_wall_src_t_path);
    exact_wall_src_prop = get_wall_prop(exact_wall_src_t_path);
    Propagator4d exact_wall_src_prop_gauge;
    prop_apply_gauge_transformation(exact_wall_src_prop_gauge, exact_wall_src_prop, gtinv);
    exact_wall_src_prop = exact_wall_src_prop_gauge;
  }
}

QLAT_END_NAMESPACE

#endif
