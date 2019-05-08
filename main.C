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

#define AINV 1.015
#define TEST 0

QLAT_START_NAMESPACE

const SpinMatrix& gamma5 = SpinMatrixConstants::get_gamma5();
const std::array<SpinMatrix,4>& gammas = SpinMatrixConstants::get_cps_gammas();
const CoordinateD CoorD_0 = CoordinateD(0, 0, 0, 0);
const Coordinate Coor_0 = Coordinate(0, 0, 0, 0);
const int NUM_RMAX = 80;
const int NUM_RMIN = 40;

void main_displayln_info(const std::string str) {
  const std::string out_str = "main:: " + str;
  displayln_info(out_str);
  return;
}

void mult_4d_matrix(double C[4][4], const double A[4][4], const double B[4][4]) {
#pragma omp parallel for
  for (int n = 0; n < 4; n++) {
    for (int p = 0; p < 4; p++) {
      double num = 0;
      for (int m = 0; m < 4; m++) {
        num += A[n][m] * B[m][p];
      }
      C[n][p] = num;
    }
  }
}

struct RotationMatrix
{
  double matrix[4][4];
  RotationMatrix(const double theta_xy, const double theta_xt, const double theta_zt)
  {
    init(theta_xy, theta_xt, theta_zt);
  }
  void init(const double theta_xy, const double theta_xt, const double theta_zt)
  {
    std::memset(this, 0, sizeof(matrix));
    const double rot_xy[4][4] = {
      {1., 0., 0., 0.},
      {0., 1., 0., 0.},
      {0., 0., cos(theta_xy), -sin(theta_xy)},
      {0., 0., sin(theta_xy), cos(theta_xy)}
    };
    const double rot_xt[4][4] = {
      {1., 0., 0., 0.},
      {0., cos(theta_xt), -sin(theta_xt), 0.},
      {0., sin(theta_xt), cos(theta_xt), 0.},
      {0., 0., 0., 1.}
    };
    const double rot_zt[4][4] = {
      {cos(theta_zt), -sin(theta_zt), 0., 0.},
      {sin(theta_zt), cos(theta_zt), 0., 0.},
      {0., 0., 1., 0.},
      {0., 0., 0., 1.}
    };
    double matrix_[4][4] = {{0.,0.,0.,0.},{0.,0.,0.,0.},{0.,0.,0.,0.},{0.,0.,0.,0.}};
    mult_4d_matrix(matrix_, rot_xt, rot_xy);
    mult_4d_matrix(matrix, rot_zt, matrix_);
  }
};

template <class M>
void write_data_from_0_node(const M& data, std::string path)
{
  sync_node();
  if (0 == get_rank() && 0 == get_thread_num())
  {
    FILE* fp = qopen(path, "a");
    std::fwrite((void*) &data, sizeof(M), 1, fp);
    qclose(fp);
  }
  sync_node();
}

inline void string_split_acc(std::vector<std::string>& acc, const std::string& str, const std::string& sep)
{
  long pos = 0;
  while (pos < str.length()) {
    const long new_pos = str.find(sep, pos);
    if (new_pos == std::string::npos) {
      const std::string add = str.substr(pos);
      if (add != "") {
        acc.push_back(add);
      }
      break;
    } else if (new_pos > pos) {
      const std::string add = str.substr(pos, new_pos);
      acc.push_back(add);
    }
    pos = new_pos + sep.length();
  }
}

inline std::vector<std::string> string_split(const std::string& str, const std::string& sep)
{
  std::vector<std::string> ret;
  string_split_acc(ret, str, sep);
  return ret;
}

inline Coordinate my_read_coordinate(const std::string& str)
{
  qassert(str.length() > 2);
  std::vector<std::string> strs;
  if (str[0] == '(' and str[str.length()-1] == ')') {
    strs = string_split(str.substr(1, str.length() - 2), ",");
  } else {
    strs = string_split(str, "x");
  }
  qassert(strs.size() == 4);
  return Coordinate(read_long(strs[0]), read_long(strs[1]), read_long(strs[2]), read_long(strs[3]));
}

inline CoordinateD my_read_coordinateD(const std::string& str)
{
  qassert(str.length() > 2);
  std::vector<std::string> strs;
  if (str[0] == '(' and str[str.length()-1] == ')') {
    strs = string_split(str.substr(1, str.length() - 2), ",");
  } else {
    strs = string_split(str, "x");
  }
  qassert(strs.size() == 4);
  return CoordinateD(read_double(strs[0]), read_double(strs[1]), read_double(strs[2]), read_double(strs[3]));
}

inline int read_traj(const std::string& str, const int lenth = 4)
{
  std::vector<std::string> strs;
  strs = string_split(str, "results=");
  return std::stoi(strs[1].substr(0, lenth));
}

inline int read_t_sep(const std::string& str, const int lenth = 4)
{
  std::vector<std::string> strs;
  strs = string_split(str, "t-sep=");
  return std::stoi(strs[1].substr(0, lenth));
}

inline int read_type(const std::string& str, const int lenth = 1)
{
  std::vector<std::string> strs;
  strs = string_split(str, "type=");
  return std::stoi(strs[1].substr(0, lenth));
}

inline int read_accuracy(const std::string& str, const int lenth = 1)
{
  std::vector<std::string> strs;
  strs = string_split(str, "accuracy=");
  return std::stoi(strs[1].substr(0, lenth));
}

Coordinate get_xg_from_path(const std::string& path) {
  const std::vector<std::string> ps = string_split(string_split(path, "/").back(), " ; ");
  const int type = read_long(info_get_prop(ps, "type="));
  const int accuracy = read_long(info_get_prop(ps, "accuracy="));
  const Coordinate xg = my_read_coordinate(info_get_prop(ps, "xg="));
  return xg;
}

std::vector<std::string> list_folders_under_path(const std::string& path_, const int root = 0)
{
  DIR *dir;
  struct dirent *ent;
  const char *path = path_.c_str();
  std::vector<std::string> res;
  qassert((dir = opendir(path)) != NULL);
  dir = opendir(path);

  /* print all the files and directories within directory */
  while ((ent = readdir (dir)) != NULL) {
    // share to other node
    std::string name(ent -> d_name);
    int name_size = static_cast<int>(name.size());
    MPI_Bcast((void*)&name_size, sizeof(int), MPI_BYTE, root, get_comm());
    if (get_id_node() != root){
      name.resize(name_size);
    }
    sync_node();
    MPI_Bcast((void*)name.c_str(), name_size, MPI_BYTE, root, get_comm());

    //save
    if (strcmp(name.c_str(), ".") == 0 or strcmp(name.c_str(), "..") == 0) {continue;}
    if (is_directory(path_ + "/" + name) == false) {continue;}
    res.push_back(name);
  }
  closedir (dir);
  sync_node();
  return res;
}

double r_coor(const Coordinate& coor)
{
  return sqrt(sqr((long)coor[0]) + sqr((long)coor[1]) + sqr((long)coor[2]) + sqr((long)coor[3]));
}

double r_coorD(const CoordinateD& coor)
{
  return sqrt(sqr((double)coor[0]) + sqr((double)coor[1]) + sqr((double)coor[2]) + sqr((double)coor[3]));
}

template <class T>
inline void set_zero(T& x)
{
  memset(&x, 0, sizeof(T));
}

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

inline std::string show_coordinate(const Coordinate& c)
{
  return ssprintf("(%d,%d,%d,%d)", c[0], c[1], c[2], c[3]);
}

inline std::string show_coordinateD(const CoordinateD& c)
{
  return ssprintf("(%f,%f,%f,%f)", c[0], c[1], c[2], c[3]);
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
    dist_read_field_double_from_float(prop, path);
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
    dist_read_field_double_from_float(prop, path);
  }
  return cache[path];
}

const Propagator4d& get_point_prop(const std::string& path, const Coordinate& c)
{
  return get_prop(path + "/xg=" + show_coordinate(c) + " ; type=0 ; accuracy=0");
}

#if 0
const Propagator4d operator-(const Propagator4d& p1, const Propagator4d& p2)
{
  const Geometry& g1 = p1.geo;
  const Geometry& g2 = p2.geo;
  qassert(g1 == g2);
  Propagator4d res(g1);
  set_zero(res);
#pragma omp parallel for
  for (long index = 0; index < g1.local_volume(); ++index)
  {
    const Coordinate lx = g1.coordinate_from_index(index);
    const Coordinate x = g1.coordinate_g_from_l(lx);
    res.get_elem(lx) = p1.get_elem(lx) - p2.get_elem(lx);
  }
  return res;
}

const Propagator4d operator*(const Complex& c, const Propagator4d& p)
{
  const Geometry& g = p.geo;
  Propagator4d res(g);
  set_zero(res);
#pragma omp parallel for
  for (long index = 0; index < g.local_volume(); ++index)
  {
    const Coordinate lx = g.coordinate_from_index(index);
    const Coordinate x = g.coordinate_g_from_l(lx);
    res.get_elem(lx) = c * p.get_elem(lx);
  }
  return res;
}
#endif

// pion g g
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

struct WallWallField : FieldM<Complex,1>
{
  virtual const std::string& cname()
  {
    static const std::string s = "WallWallField";
    return s;
  }

  void wallwallinit(std::vector<Complex> vec) {
    const int total_t = vec.size();
    const Coordinate total_site(1, 1, 1, total_t);
    Geometry geo(total_site, 1);
    init(geo);
#pragma omp parallel for
    for (long index = 0; index < geo.local_volume(); ++index)
    {
      const Coordinate lx = geo.coordinate_from_index(index);
      const Coordinate x = geo.coordinate_g_from_l(lx);
      this -> get_elem(lx) = vec[x[3]];
    }
  }

  WallWallField(std::vector<Complex> vec) {wallwallinit(vec);}
  WallWallField() { init(); }

  std::vector<Complex> save_to_vec()
  {
    const long total_t = this -> geo.total_volume();
    std::vector<Complex> vec(total_t);
#pragma omp parallel for
    for (long index = 0; index < geo.local_volume(); ++index)
    {
      const Coordinate lx = geo.coordinate_from_index(index);
      const Coordinate x = geo.coordinate_g_from_l(lx);
      vec[x[3]] = this -> get_elem(lx);
    }
    glb_sum(vec);
    return vec;
  }
};


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

PionGGElem operator*(const RotationMatrix& rot, const PionGGElem& p)
{
  PionGGElem resp;
  set_zero(resp);
#pragma omp parallel for
  // Lambda[mu'][alpha'] * B'[alpha][alpha']
  for (int alpha = 0; alpha < 4; ++alpha){
    for (int mup = 0; mup < 4; ++mup){
      Complex num = Complex(0.,0.);
      for (int alphap = 0; alphap < 4; ++alphap){
        num += rot.matrix[mup][alphap] * p.v[alpha][alphap];
      }
      resp.v[alpha][mup] = num;
    }
  }
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  // Lambda[mu][alpha] * Lambda[mu'][alpha'] * B'[alpha][alpha']
  for (int mu = 0; mu < 4; ++mu){
    for (int mup = 0; mup < 4; ++mup){
      Complex num = Complex(0.,0.);
      for (int alpha = 0; alpha < 4; ++alpha){
        num += rot.matrix[mu][alpha] * resp.v[alpha][mup];
      }
      res.v[mu][mup] = num;
    }
  }
  return res;
}

CoordinateD operator*(const RotationMatrix& rot, const CoordinateD& coor)
{
  CoordinateD res;
#pragma omp parallel for
  for (int n = 0; n < 4; n++) {
    double num = 0;
    for (int m = 0; m < 4; m++) {
      num += rot.matrix[n][m] * coor[m];
    }
    res[n] = num;
  }
  return res;
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

std::vector<Complex> operator+(const std::vector<Complex>& v1, const std::vector<Complex>& v2)
{
  qassert(v1.size() == v2.size())
  std::vector<Complex> res(v1.size());
#pragma omp parallel for
  for (int i = 0; i < v1.size(); ++i)
  {
    res[i] = v1[i] + v2[i];
  }
  return res;
}

std::vector<Complex> operator/(const std::vector<Complex>& v1, const Complex a)
{
  std::vector<Complex> res(v1.size());
#pragma omp parallel for
  for (int i = 0; i < v1.size(); ++i)
  {
    res[i] = v1[i] / a;
  }
  return res;
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

struct PionGG
{
  Coordinate mom;
  std::vector<PionGGElem> table;
};

Complex pi_pi_contraction(const WilsonMatrix& wm_from_1_to_2, const WilsonMatrix& wm_from_2_to_1)
{
  return -matrix_trace((ii * gamma5) * wm_from_2_to_1 * (ii * gamma5) * wm_from_1_to_2);
}

void three_prop_contraction(PionGGElem& pgge, const WilsonMatrix& wm_21, const WilsonMatrix& wm_32, const WilsonMatrix& wm_13)
  // 1(mu) --wm_21--> 2(5) --wm_32--> 3(nu) --wm_13--> 1(mu)
  // need additional minus sign from the loop
  // need additional minus sign from the two iis for the current
  // need additional charge factor 1/3 and pion source norm factor 1/sqrt(2)
  // need Z_V^2
{
  const WilsonMatrix wm_a = wm_32 * (ii * gamma5) * wm_21;
  for (int nu = 0; nu < 4; ++nu) {
    const WilsonMatrix wm_b = wm_13 * gammas[nu] * wm_a;
    for (int mu = 0; mu < 4; ++mu) {
      pgge.v[mu][nu] += matrix_trace(gammas[mu] * wm_b);
    }
  }
}

void three_prop_contraction_(PionGGElem& pgge, const WilsonMatrix& wm_21, const WilsonMatrix& wm_32, const WilsonMatrix& wm_13)
  // 1(nu) --wm_21--> 2(5) --wm_32--> 3(mu) --wm_13--> 1(nu)
  // need additional minus sign from the loop
  // need additional minus sign from the two iis for the current
  // need additional charge factor 1/3 and pion source norm factor 1/sqrt(2)
  // need Z_V^2
{
  const WilsonMatrix wm_a = wm_32 * (ii * gamma5) * wm_21;
  for (int mu = 0; mu < 4; ++mu) {
    const WilsonMatrix wm_b = wm_13 * gammas[mu] * wm_a;
    for (int nu = 0; nu < 4; ++nu) {
      pgge.v[mu][nu] += matrix_trace(gammas[nu] * wm_b);
    }
  }
}

bool is_under_limit(const Coordinate& x, const Coordinate& y, const double& r)
{
  Coordinate dist = x - y;
  double xy = std::pow(std::pow(dist[0], 2.) + std::pow(dist[1], 2.) + std::pow(dist[2], 2.) + std::pow(dist[3], 2.), 1./2.);
  if (r >= xy) {
    return true;
  } else {
    return false;
  }
}

int get_r_persent(Coordinate& vec1, Coordinate& vec2)
{
  double rmax = std::max(r_coor(vec1), r_coor(vec2));
  double r21  = r_coor(vec1 - vec2);
  return int(ceil(r21 / rmax * 10.));
}

struct BM_TABLE
{
  PionGGElem bm[NUM_RMAX][NUM_RMIN];
};

void partialsum_bmtable(BM_TABLE& bm_table)
{
  for (int r_max = 0; r_max < NUM_RMAX; ++r_max)
  {
    for (int r_min = 0; r_min < NUM_RMIN; ++r_min)
    {
      if (r_max == 0 && r_min == 0)
      {
        bm_table.bm[r_max][r_min] = bm_table.bm[r_max][r_min];
      } else if (r_max == 0) {
        bm_table.bm[r_max][r_min] = bm_table.bm[r_max][r_min - 1] + bm_table.bm[r_max][r_min];
      } else if (r_min == 0) {
        bm_table.bm[r_max][r_min] = bm_table.bm[r_max - 1][r_min] + bm_table.bm[r_max][r_min];
      } else {
        bm_table.bm[r_max][r_min] = bm_table.bm[r_max - 1][r_min] + bm_table.bm[r_max][r_min - 1]
		                  + bm_table.bm[r_max][r_min] - bm_table.bm[r_max - 1][r_min - 1];
      }
    }
  }
  return;
}

struct XB
{
  PionGGElem xB[3];
};

inline std::string show_xb(const XB& xb)
{
  std::string out("");
  for (int i = 0; i < 3; ++i) {
    out += ssprintf("XB i=%d: ", i);
    out += show_pgge(xb.xB[i]);
    out += "\n";
  }
  return out;
}

XB operator+(const XB& xb1, const XB& xb2)
{
  XB res;
  set_zero(res);
  for (int i = 0; i < 3; ++i)
  {
    res.xB[i] = xb1.xB[i] + xb2.xB[i];
  }
  return res;
}

struct XB_TABLE
{
  XB xb_e[NUM_RMIN];
};

void partialsum_xbtable(XB_TABLE& xb_table)
{
  for (int r_min = 1; r_min < NUM_RMIN; ++r_min)
  {
    xb_table.xb_e[r_min] = xb_table.xb_e[r_min - 1] + xb_table.xb_e[r_min];
  }
  return;
}

struct Complex_Table
{
  Complex c[NUM_RMAX][NUM_RMIN];

  Complex_Table& operator*(const Complex& c)
  {
#pragma omp parallel for
    for (int rmax = 0; rmax < NUM_RMAX; ++rmax) {
      for (int rmin = 0; rmin < NUM_RMIN; ++rmin) {
        Complex& e = this -> c[rmax][rmin];
        e *= c;
      }
    }
    return *this;
  }

  Complex_Table& operator+=(const Complex_Table& other)
  {
#pragma omp parallel for
    for (int rmax = 0; rmax < NUM_RMAX; ++rmax) {
      for (int rmin = 0; rmin < NUM_RMIN; ++rmin) {
        this -> c[rmax][rmin] += other.c[rmax][rmin];
      }
    }
    return *this;
  }
};

Complex_Table operator*(const Complex& c, const Complex_Table& table)
{
  Complex_Table res;
  set_zero(res);
#pragma omp parallel for
  for (int rmax = 0; rmax < NUM_RMAX; ++rmax) {
    for (int rmin = 0; rmin < NUM_RMIN; ++rmin) {
      const Complex& e = table.c[rmax][rmin];
      Complex& r = res.c[rmax][rmin];
      r = c * e;
    }
  }
  return res;
}

std::string show_complex_table(const Complex_Table& complex_table)
{
  std::string res = "";
  for (int rmax = 0; rmax < NUM_RMAX; ++rmax)
  {
    for (int rmin = 0; rmin < NUM_RMIN; ++rmin)
    {
      res += ssprintf("%24.17e %24.17e, ", (complex_table.c[rmax][rmin]).real(), (complex_table.c[rmax][rmin]).imag());
    }
    res += "\n";
  }
  return res;
}

struct Y_And_Rotation_Info_Elem
{
  double dist;
  CoordinateD y_large;
  double theta_xy;
  double theta_xt;
  double theta_zt;
};

std::string show_y_and_rotation_info_elem(const Y_And_Rotation_Info_Elem& elem)
{
  std::string info = "Y_And_Rotation_Info_Elem:\n";
  info += "y_large:\n";
  info += show_coordinateD(elem.y_large);
  info += "\n";
  info += ssprintf("dist: %24.17E\n", elem.dist);
  info += ssprintf("theta_xy: %24.17E\n", elem.theta_xy);
  info += ssprintf("theta_xt: %24.17E\n", elem.theta_xt);
  info += ssprintf("theta_zt: %24.17E\n", elem.theta_zt);
  return info;
}

std::vector<Y_And_Rotation_Info_Elem> read_y_and_rotation_info(std::string file)
{
  std::vector<Y_And_Rotation_Info_Elem> info_list;
  std::ifstream fileopen(file);
  std::string line;
  long linenum = 0;
  while (getline(fileopen, line)) {
    if (linenum % 5 == 0) {
      Y_And_Rotation_Info_Elem elem;
      info_list.push_back(elem);
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).y_large = my_read_coordinateD(coor);
    } else if (linenum % 5 == 1) {
      (info_list.back()).dist = std::stod(line);
    } else if (linenum % 5 == 2) {
      (info_list.back()).theta_xy = std::stod(line);
    } else if (linenum % 5 == 3) {
      (info_list.back()).theta_xt = std::stod(line);
    } else if (linenum % 5 == 4) {
      (info_list.back()).theta_zt = std::stod(line);
      // main_displayln_info(show_y_and_rotation_info_elem(info_list.back()));
    }
    linenum++;
  }
  main_displayln_info(ssprintf("%d Pairs Have Been Read From Two Configs.", info_list.size()));
  return info_list;
}

double f_r(double r, double fpi, double mv)
{
    double fte;
    double fvmd;

    fte  = 3. * std::pow(mv, 4.) * std::pow(r, 2.) * gsl_sf_bessel_Kn(2, mv * r) / (16. * std::pow(fpi, 2.) * std::pow(PI, 2.));
    fvmd = 3. * std::pow(mv, 5.) * std::pow(r, 3.) * gsl_sf_bessel_K1(   mv * r) / (32. * std::pow(fpi, 2.) * std::pow(PI, 2.));

    return 8. * std::pow(PI, 2.) * std::pow(fpi, 2.) / (3. * std::pow(mv, 2.)) * fte + (1. - 8. * std::pow(PI, 2.) * std::pow(fpi, 2.) / (3. * std::pow(mv, 2.))) * fvmd;
}

void b_model(PionGGElem& pgge, const CoordinateD& x_xp, const CoordinateD& xp_x_mid_z, const double& pion)
{
    set_zero(pgge);
    double r_x_xp = r_coorD(x_xp);
    double r_xp_x_mid_z = r_coorD(xp_x_mid_z);
    if (r_x_xp <= 0.0 or r_xp_x_mid_z <= 0.0){return;}
    double fpi = 0.093 / AINV;
    double mv = 0.77 / AINV;
    int mu;
    int mup;
    int rho;
    int sigma;
    int sign;
    for (mu = 0; mu < 4; ++mu){
	int base_sign;
	if (mu == 0 || mu == 2) {
          base_sign = 1;
	} else {
	  base_sign = -1;
	}

        // 0 1 2 3 +1
        mup =   (mu + 1) % 4;
        rho =   (mu + 2) % 4;
        sigma = (mu + 3) % 4;
        sign = base_sign * 1;
        pgge.v[mu][mup] += sign * x_xp[rho] * xp_x_mid_z[sigma];

        // 0 1 3 2 -1
        mup =   (mu + 1) % 4;
        rho =   (mu + 3) % 4;
        sigma = (mu + 2) % 4;
        sign = base_sign * -1;
        pgge.v[mu][mup] += sign * x_xp[rho] * xp_x_mid_z[sigma];
        
        // 0 2 1 3 -1
        mup =   (mu + 2) % 4;
        rho =   (mu + 1) % 4;
        sigma = (mu + 3) % 4;
        sign = base_sign * -1;
        pgge.v[mu][mup] += sign * x_xp[rho] * xp_x_mid_z[sigma];

        // 0 2 3 1 +1
        mup =   (mu + 2) % 4;
        rho =   (mu + 3) % 4;
        sigma = (mu + 1) % 4;
        sign = base_sign * 1;
        pgge.v[mu][mup] += sign * x_xp[rho] * xp_x_mid_z[sigma];

        // 0 3 1 2 +1
        mup =   (mu + 3) % 4;
        rho =   (mu + 1) % 4;
        sigma = (mu + 2) % 4;
        sign = base_sign * 1;
        pgge.v[mu][mup] += sign * x_xp[rho] * xp_x_mid_z[sigma];

        // 0 3 2 1 -1
        mup =   (mu + 3) % 4;
        rho =   (mu + 2) % 4;
        sigma = (mu + 1) % 4;
        sign = base_sign * -1;
        pgge.v[mu][mup] += sign * x_xp[rho] * xp_x_mid_z[sigma];
    }
    double xz_mass = pion * r_xp_x_mid_z;
    pgge = - ii * fpi * std::pow(pion, 2.) / (12. * std::pow(PI, 4.) * std::pow(r_x_xp, 4.) * std::pow(r_xp_x_mid_z, 2.)) * f_r(r_x_xp, fpi, mv) * gsl_sf_bessel_Kn(2, xz_mass) * pgge;
    return;
}

void find_bm_table_rotation_pgge(BM_TABLE& bm_table, const PionGGElemField& pgge_field, const CoordinateD& y_large, const RotationMatrix rot, const double& muon, const std::string mod="")
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_table_rotation_pgge");
  qassert(mod == "xy>=xyp" or mod == "xyp>=xy" or mod == "");
  set_zero(bm_table);
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();
  // const RotationMatrix rot180(PI, 0., 0.);
  Coordinate y  = Coordinate(0,0,0,0);
  
  double r_y_large = r_coorD(y_large);
  const CoordinateD muon_y_large = muon * y_large;
  double r_muon_y_large = muon * r_y_large;

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lyp = geo.coordinate_from_index(index);
    const Coordinate yp = geo.coordinate_g_from_l(lyp);

    // find coor info
    CoordinateD yp_y = relative_coordinate(yp-y, total_site);
    // CoordinateD yp_y_rot = rot * (rot180 * yp_y);
    CoordinateD yp_y_rot = rot * yp_y;
    const CoordinateD yp_large = y_large + yp_y_rot;
    const CoordinateD muon_yp_large = muon * CoordinateD(yp_large);
    double r_yp_large = r_coorD(yp_large);
    double r_muon_yp_large = muon * r_yp_large;
    double r_yp_y = r_coorD(yp_y_rot);
    double r_muon_yp_y = muon * r_yp_y;

    // get muon line
    if (r_muon_y_large > 6 || r_muon_yp_large > 6 || r_muon_yp_y > 6) {continue; }

    const ManyMagneticMoments mmm = get_muon_line_m_extra(muon_y_large, muon_yp_large, CoorD_0, 0);

    if (mod == "xy>=xyp" && r_yp_large - r_y_large > std::pow(10., -10.)) {continue;}
    if (mod == "xyp>=xy" && r_y_large - r_yp_large > std::pow(10., -10.)) {continue;}

    int r_max = int(ceil(std::max(r_y_large, r_yp_large)));
    r_max = int(ceil(std::max(double(r_max), r_yp_y)));
    int r_min = int(ceil(std::min(r_y_large, r_yp_large)));
    r_min = int(ceil(std::min(double(r_min), r_yp_y)));
    if (r_max >= NUM_RMAX || r_min >= NUM_RMIN) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];

    PionGGElem pgge;
    set_zero(pgge);
    // pgge = rot * (rot180 * pgge_field.get_elem(lyp));
    pgge = rot * pgge_field.get_elem(lyp);

    if ((mod == "") || ((mod != "") && (std::abs(r_y_large - r_yp_large) < std::pow(10., -10.)))) {
      bm += pgge * mmm;
    } else {
      bm += 2. * (pgge * mmm);
    }
  }
  glb_sum_double(bm_table);
  return;
}

void find_xb_rotation_pgge(XB& xb, const PionGGElemField& pgge_field, const RotationMatrix rot, const double& xxp_limit)
{
  TIMER_VERBOSE("find_xb_rotation_pgge");
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();
  const RotationMatrix rot180(PI, 0., 0.);

  Coordinate x = Coordinate(0,0,0,0);

  for (long index = 0; index < geo.local_volume(); ++index) 
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const Coordinate xp_x = relative_coordinate(xp-x, total_site);

    if (r_coor(xp_x) > xxp_limit or r_coor(xp_x) < std::pow(10., -5)) {continue; }

    PionGGElem pgge;
    set_zero(pgge);
    pgge = rot * (rot180 * pgge_field.get_elem(lxp));
    CoordinateD xp_x_rot = rot * (rot180 * xp_x);
    for (int j = 0; j < 3; j++) {
      xb.xB[j] += xp_x_rot[j] * pgge;
    }
  }
  glb_sum_double(xb);
  return;
}

Complex find_e_xbbm(const XB& xb, const PionGGElem& bm)
{
  Complex res(0., 0.);
  int j, mup;
  for (int i = 0; i < 3; ++i) {
    j = (i + 1) % 3;
    mup = (j + 1) % 3;
    for (int mu = 0; mu < 4; ++mu) {
      res += (xb.xB[j]).v[mu][mup] * bm.v[i][mu];
    }

    j = (i + 3 - 1) % 3;
    mup = (j + 3 - 1) % 3;
    for (int mu = 0; mu < 4; ++mu) {
      res -= (xb.xB[j]).v[mu][mup] * bm.v[i][mu];
    }
  }
  return res;
}

void find_e_xbbm_table(Complex_Table& e_xbbm_table, const XB& xb, const BM_TABLE& bm_table)
{
  TIMER_VERBOSE("find_e_xbbm_table");
  set_zero(e_xbbm_table);
#pragma omp parallel for
  for (int rmax = 0; rmax < NUM_RMAX; ++rmax) {
    for (int rmin = 0; rmin < NUM_RMIN; ++rmin) {
      const PionGGElem& bm = bm_table.bm[rmax][rmin];
      Complex& e_xbbm = e_xbbm_table.c[rmax][rmin];
      e_xbbm = find_e_xbbm(xb, bm);
    }
  }
}

QLAT_END_NAMESPACE

using namespace qlat;

#if 0
void find_onepair_f2_table_rotation_pgge(Complex_Table& f2_model_table, const PionGGElemField& pgge_field, const Y_And_Rotation_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("find_onepair_f2_table_rotation_pgge");
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();

  const double theta_xy = onepair.theta_xy;
  const double theta_xt = onepair.theta_xt;
  const double theta_zt = onepair.theta_zt;
  const RotationMatrix rotation_matrix = RotationMatrix(theta_xy, theta_xt, theta_zt);
  // y_large
  const CoordinateD& y_large = onepair.y_large;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_rotation_pgge(xb, pgge_field, rotation_matrix, xb_limit);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_table_rotation_pgge(bm_table, pgge_field, y_large, rotation_matrix, muon);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Complex prop = pion_prop(y_large, Coor_0, pion);

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;

  // show one pair e_xbbm_table
  std::string info = "";
  info += show_y_and_rotation_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  main_displayln_info(info);

  f2_model_table += e_xbbm_table;

  return;
}
#endif

Complex_Table find_onepair_f2_table_rotation_pgge(const PionGGElemField& pgge_field_1, const PionGGElemField& pgge_field_2, const Y_And_Rotation_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion, const std::string& mod="")
{
  TIMER_VERBOSE("find_onepair_f2_table_rotation_pgge");
  const Geometry& geo = pgge_field_1.geo;
  const Geometry& geo_ = pgge_field_2.geo;
  qassert(geo == geo_);
  const Coordinate total_site = geo.total_site();

  const double theta_xy = onepair.theta_xy;
  const double theta_xt = onepair.theta_xt;
  const double theta_zt = onepair.theta_zt;
  const RotationMatrix rotation_matrix = RotationMatrix(theta_xy, theta_xt, theta_zt);
  // y_large
  const CoordinateD& y_large = onepair.y_large;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_rotation_pgge(xb, pgge_field_1, rotation_matrix, xb_limit);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_table_rotation_pgge(bm_table, pgge_field_2, y_large, rotation_matrix, muon, mod);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Complex prop = pion_prop(y_large, Coor_0, pion);

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;

  // show one pair e_xbbm_table
  std::string info = "";
  info += show_y_and_rotation_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  main_displayln_info(info);

  return e_xbbm_table;
}

#if 0
Complex_Table avg_f2_table_from_three_point_corr_from_y_and_rotation_info(std::string f_pairs, const PionGGElemField& pgge_field, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("avg_f2_table_from_three_point_corr_from_y_and_rotation_info");
  std::vector<Y_And_Rotation_Info_Elem> pairs_info = read_y_and_rotation_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_table;
  set_zero(f2_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    main_displayln_info(ssprintf("Begin pair %d", i));
    Y_And_Rotation_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_table_rotation_pgge(f2_table, pgge_field, one_pair, xxp_limit, muon, pion);
  }
  f2_table = 1. / num_pairs * f2_table;
  main_displayln_info("Final F2 Model Nofac:");
  main_displayln_info(show_complex_table(f2_table));
  return f2_table;
}
#endif

Complex_Table avg_f2_table_from_three_point_corr_from_y_and_rotation_info(std::vector<Y_And_Rotation_Info_Elem> pairs_info, const PionGGElemField& pgge_field_1, const PionGGElemField& pgge_field_2, const double& xxp_limit, const double& muon, const double& pion, std::string one_pair_save_folder="", const std::string& mod="", long num_pairs_=0)
{
  TIMER_VERBOSE("avg_f2_table_from_three_point_corr_from_y_and_rotation_info");
  long num_pairs;
  if (num_pairs_ == 0) {
    num_pairs = pairs_info.size();
  } else {
    num_pairs = num_pairs_;
  }
  Complex_Table f2_table;
  set_zero(f2_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    main_displayln_info(ssprintf("Begin pair %d", i));
    std::string one_pair_save_path = one_pair_save_folder + "/" + ssprintf("%05d", i);
    if (does_file_exist_sync_node(one_pair_save_path)) {continue;}

    Y_And_Rotation_Info_Elem& one_pair = pairs_info[i];
    Complex_Table one_pair_table = find_onepair_f2_table_rotation_pgge(pgge_field_1, pgge_field_2, one_pair, xxp_limit, muon, pion, mod);

    // save
    write_data_from_0_node(one_pair_table, one_pair_save_path);

    f2_table += one_pair_table;
  }
  f2_table = 1. / num_pairs * f2_table;
  main_displayln_info("Final F2 Model Nofac:");
  main_displayln_info(show_complex_table(f2_table));
  return f2_table;
}

void compute_three_point_correlator_from_wall_src_prop(PionGGElemField& three_point_correlator_labeled_xp, const Propagator4d& point_src_prop, const Propagator4d& wall_src_prop)
{
  TIMER_VERBOSE("compute_three_point_correlator_from_wall_src_prop");
  const Geometry& geo = point_src_prop.geo;
  qassert(geo == geo_reform(geo));
  qassert(geo == wall_src_prop.geo);
  qassert(geo == three_point_correlator_labeled_xp.geo);
  const Coordinate total_site = geo.total_site();

  const Coordinate x = Coordinate(0, 0, 0, 0);
  const Coordinate lx = geo.coordinate_l_from_g(x);

  WilsonMatrix wm_from_wall_to_x;
  if (geo.is_local(lx)) {
    wm_from_wall_to_x = wall_src_prop.get_elem(lx);
  } else {
    set_zero(wm_from_wall_to_x);
  }
  glb_sum_double(wm_from_wall_to_x);
  const WilsonMatrix wm_from_x_to_wall = gamma5 * matrix_adjoint(wm_from_wall_to_x) * gamma5;

#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const WilsonMatrix& wm_from_wall_to_xp = wall_src_prop.get_elem(lxp);
    const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
    const WilsonMatrix wm_from_xp_to_wall = gamma5 * matrix_adjoint(wm_from_wall_to_xp) * gamma5;
    const WilsonMatrix wm_from_xp_to_x = gamma5 * matrix_adjoint(wm_from_x_to_xp) * gamma5;

    PionGGElem& pgge = three_point_correlator_labeled_xp.get_elem(lxp);

    three_prop_contraction (pgge, wm_from_x_to_wall, wm_from_wall_to_xp, wm_from_xp_to_x);
    three_prop_contraction_(pgge, wm_from_xp_to_wall, wm_from_wall_to_x, wm_from_x_to_xp);
  }
  return;
}

void compute_three_point_correlator_from_closest_wall_src_prop(const Coordinate& x, const int t_min, PionGGElemField& three_point_correlator_labeled_xp, const Propagator4d& point_src_prop, const std::vector<Propagator4d>& wall_src_list, const double pion)
// three_point_correlator_labeled_xp(xp)[mu][nu] =
//   Tr(S(x, wall) * ii * gamma5 * S(wall, xp) * gamma_nu * S(xp, x) * gamma_mu) / exp(-pion * (t_wall - t_x))
// + Tr(S(xp, wall) * ii * gamma5 * S(wall, x) * gamma_mu * S(x, xp) * gamma_nu) / exp(-pion * (t_wall - t_x))
// t_wall > t_x
{
  TIMER_VERBOSE("compute_three_point_correlator_from_closest_wall_src_prop");
  const Geometry& geo = point_src_prop.geo;
  qassert(geo == geo_reform(geo));
  qassert(geo == three_point_correlator_labeled_xp.geo);
  const Coordinate total_site = geo.total_site();

  // prepare wm from wall to x
  const Coordinate lx = geo.coordinate_l_from_g(x);
  std::vector<WilsonMatrix> wm_from_wall_to_x_t(total_site[3]);
  std::vector<WilsonMatrix> wm_from_x_to_wall_t(total_site[3]);
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall)
  {
    WilsonMatrix& wm_from_wall_to_x = wm_from_wall_to_x_t[t_wall];
    WilsonMatrix& wm_from_x_to_wall = wm_from_x_to_wall_t[t_wall];

    const Propagator4d& wall_src_prop = wall_src_list[t_wall];
    qassert(geo == wall_src_prop.geo);
    if (geo.is_local(lx)) {
      wm_from_wall_to_x = wall_src_prop.get_elem(lx);
    } else {
      set_zero(wm_from_wall_to_x);
    }
    glb_sum_double(wm_from_wall_to_x);
    wm_from_x_to_wall = gamma5 * matrix_adjoint(wm_from_wall_to_x) * gamma5;
  }

#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    int t_wall;
    int t_sep;
    int diff = smod(x[3] - xp[3], total_site[3]);

    // t_wall << t and t'
    if (diff >= 0)
    {
      t_wall = mod(xp[3] - t_min, total_site[3]);
    } else {
      t_wall = mod(x[3] - t_min, total_site[3]);
    }

    // this is for t_wall >> t and t'
    // if (diff >= 0)
    // {
      // t_wall = mod(x[3] + t_min, total_site[3]);
    // } else {
      // t_wall = mod(xp[3] + t_min, total_site[3]);
    // }
    
    t_sep = abs(smod(t_wall - x[3], total_site[3]));
    
    const Propagator4d& wall_src_prop = wall_src_list[t_wall];
    WilsonMatrix& wm_from_wall_to_x = wm_from_wall_to_x_t[t_wall];
    WilsonMatrix& wm_from_x_to_wall = wm_from_x_to_wall_t[t_wall];

    const WilsonMatrix& wm_from_wall_to_xp = wall_src_prop.get_elem(lxp);
    const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
    const WilsonMatrix wm_from_xp_to_wall = gamma5 * matrix_adjoint(wm_from_wall_to_xp) * gamma5;
    const WilsonMatrix wm_from_xp_to_x = gamma5 * matrix_adjoint(wm_from_x_to_xp) * gamma5;

    PionGGElem& pgge = three_point_correlator_labeled_xp.get_elem(lxp);

    three_prop_contraction (pgge, wm_from_x_to_wall, wm_from_wall_to_xp, wm_from_xp_to_x);
    three_prop_contraction_(pgge, wm_from_xp_to_wall, wm_from_wall_to_x, wm_from_x_to_xp);
    pgge /= exp(-pion * t_sep);
  }
  return;
}

void compute_three_point_correlator_ama_from_closest_wall_src_prop(const Coordinate& x, const int t_min, PionGGElemField& three_point_correlator_labeled_xp, const Propagator4d& point_src_prop, const std::vector<Propagator4d>& wall_src_list, const std::vector<Propagator4d>& exact_wall_src_list, const std::vector<int>& exact_wall_t_list, const double pion)
// three_point_correlator_labeled_xp(xp)[mu][nu] =
//   Tr(S(x, wall) * ii * gamma5 * S(wall, xp) * gamma_nu * S(xp, x) * gamma_mu) / exp(-pion * (t_wall - t_x))
// + Tr(S(xp, wall) * ii * gamma5 * S(wall, x) * gamma_mu * S(x, xp) * gamma_nu) / exp(-pion * (t_wall - t_x))
// t_wall > t_x
{
  TIMER_VERBOSE("compute_three_point_correlator_ama_from_closest_wall_src_prop");
  const Geometry& geo = point_src_prop.geo;
  qassert(geo == geo_reform(geo));
  qassert(geo == three_point_correlator_labeled_xp.geo);
  qassert(exact_wall_src_list.size() == exact_wall_t_list.size());
  const Coordinate total_site = geo.total_site();

  // prepare wm from wall to x
  const Coordinate lx = geo.coordinate_l_from_g(x);
  std::vector<WilsonMatrix> wm_from_wall_to_x_t(total_site[3]);
  std::vector<WilsonMatrix> wm_from_x_to_wall_t(total_site[3]);
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall)
  {
    WilsonMatrix& wm_from_wall_to_x = wm_from_wall_to_x_t[t_wall];
    WilsonMatrix& wm_from_x_to_wall = wm_from_x_to_wall_t[t_wall];

    const Propagator4d& wall_src_prop = wall_src_list[t_wall];
    qassert(geo == wall_src_prop.geo);
    if (geo.is_local(lx)) {
      wm_from_wall_to_x = wall_src_prop.get_elem(lx);
    } else {
      set_zero(wm_from_wall_to_x);
    }
    glb_sum_double(wm_from_wall_to_x);
    wm_from_x_to_wall = gamma5 * matrix_adjoint(wm_from_wall_to_x) * gamma5;
  }

  // prepare wm from exact wall to x
  std::vector<WilsonMatrix> exact_wm_from_wall_to_x_t(exact_wall_src_list.size());
  std::vector<WilsonMatrix> exact_wm_from_x_to_wall_t(exact_wall_src_list.size());
  for (int i = 0; i < exact_wall_src_list.size(); ++i)
  {
    WilsonMatrix& exact_wm_from_wall_to_x = exact_wm_from_wall_to_x_t[i];
    WilsonMatrix& exact_wm_from_x_to_wall = exact_wm_from_x_to_wall_t[i];

    const Propagator4d& exact_wall_src_prop = exact_wall_src_list[i];
    qassert(geo == exact_wall_src_prop.geo);
    if (geo.is_local(lx)) {
      exact_wm_from_wall_to_x = exact_wall_src_prop.get_elem(lx);
    } else {
      set_zero(exact_wm_from_wall_to_x);
    }
    glb_sum_double(exact_wm_from_wall_to_x);
    exact_wm_from_x_to_wall = gamma5 * matrix_adjoint(exact_wm_from_wall_to_x) * gamma5;
  }

  double ama_factor = 1.0 * wall_src_list.size() / exact_wall_src_list.size();

#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    int t_wall;
    int t_sep;
    int diff = smod(x[3] - xp[3], total_site[3]);

    // t_wall << t and t'
    if (diff >= 0)
    {
      t_wall = mod(xp[3] - t_min, total_site[3]);
    } else {
      t_wall = mod(x[3] - t_min, total_site[3]);
    }

    // t_wall >> t and t'
    // if (diff >= 0)
    // {
      // t_wall = mod(x[3] + t_min, total_site[3]);
    // } else {
      // t_wall = mod(xp[3] + t_min, total_site[3]);
    // }

    t_sep = abs(smod(t_wall - x[3], total_site[3]));

    PionGGElem& pgge = three_point_correlator_labeled_xp.get_elem(lxp);
    {
      const Propagator4d& wall_src_prop = wall_src_list[t_wall];
      WilsonMatrix& wm_from_wall_to_x = wm_from_wall_to_x_t[t_wall];
      WilsonMatrix& wm_from_x_to_wall = wm_from_x_to_wall_t[t_wall];

      const WilsonMatrix& wm_from_wall_to_xp = wall_src_prop.get_elem(lxp);
      const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
      const WilsonMatrix wm_from_xp_to_wall = gamma5 * matrix_adjoint(wm_from_wall_to_xp) * gamma5;
      const WilsonMatrix wm_from_xp_to_x = gamma5 * matrix_adjoint(wm_from_x_to_xp) * gamma5;

      three_prop_contraction (pgge, wm_from_x_to_wall, wm_from_wall_to_xp, wm_from_xp_to_x);
      three_prop_contraction_(pgge, wm_from_xp_to_wall, wm_from_wall_to_x, wm_from_x_to_xp);
    }
    
    // exact
    PionGGElem pgge_exact;
    {
      set_zero(pgge_exact);
      for (int i = 0; i < exact_wall_src_list.size(); ++i) {
        if (exact_wall_t_list[i] != t_wall) {continue;}
        const Propagator4d& exact_wall_src_prop = exact_wall_src_list[i];
        WilsonMatrix& exact_wm_from_wall_to_x = exact_wm_from_wall_to_x_t[i];
        WilsonMatrix& exact_wm_from_x_to_wall = exact_wm_from_x_to_wall_t[i];

        const WilsonMatrix& exact_wm_from_wall_to_xp = exact_wall_src_prop.get_elem(lxp);
        const WilsonMatrix exact_wm_from_xp_to_wall = gamma5 * matrix_adjoint(exact_wm_from_wall_to_xp) * gamma5;
        const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
        const WilsonMatrix wm_from_xp_to_x = gamma5 * matrix_adjoint(wm_from_x_to_xp) * gamma5;

        three_prop_contraction (pgge_exact, exact_wm_from_x_to_wall, exact_wm_from_wall_to_xp, wm_from_xp_to_x);
        three_prop_contraction_(pgge_exact, exact_wm_from_xp_to_wall, exact_wm_from_wall_to_x, wm_from_x_to_xp);

        pgge_exact -= pgge;
        pgge_exact *= ama_factor;
      }
    }
    pgge += pgge_exact;
    pgge /= exp(-pion * t_sep);
  }
  return;
}

void compute_point_point_wall_correlator_in_one_traj(const int t_min, const std::string& wall_src_path, const std::string& gauge_transform_path, const std::string& point_src_path, const std::string& field_out_traj_path, const double pion, int type_, int accuracy_)
{
  TIMER_VERBOSE("compute_point_point_wall_correlator_in_one_traj");

  qassert(does_file_exist_sync_node(field_out_traj_path));
  const std::string field_out_tmin_path  = field_out_traj_path + "/" + ssprintf("t-min=%04d", t_min);
  qmkdir_sync_node(field_out_tmin_path);
  if (does_file_exist_sync_node(field_out_tmin_path + "/checkpoint")){return;}

  main_displayln_info("Load Gauge Transform And Get Inv: " + gauge_transform_path);
  GaugeTransform gtinv;
  {
    GaugeTransform gt;
    dist_read_field(gt, gauge_transform_path);
    to_from_big_endian_64(get_data(gt));
    gt_inverse(gtinv, gt);
  }
  const Coordinate total_site = gtinv.geo.total_site();

  // pre-load and gauge inv wall_prop
  main_displayln_info(ssprintf("Preload And Gauge Inv Wall Src Prop for All T[%d]: ", total_site[3]) + wall_src_path);
  std::vector<Propagator4d> wall_src_list(total_site[3]);
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string wall_src_t_path = wall_src_path + "/t=" + ssprintf("%d", t_wall);
    Propagator4d& wall_src_prop = wall_src_list[t_wall];
    wall_src_prop = get_wall_prop(wall_src_t_path);
    Propagator4d wall_src_prop_gauge;
    prop_apply_gauge_transformation(wall_src_prop_gauge, wall_src_prop, gtinv);
    wall_src_prop = wall_src_prop_gauge;
  }

  int count = 0;
  const std::vector<std::string> point_src_prop_list = list_folders_under_path(point_src_path);
  for (int i =0; i < point_src_prop_list.size(); ++i){

    if (type_ != read_type(point_src_prop_list[i]) or accuracy_ != read_accuracy(point_src_prop_list[i])) {continue; }
    const Coordinate point_src_coor = get_xg_from_path(point_src_prop_list[i]);
    const std::string coor_name = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, accuracy_);
    std::string field_out_coor_path = field_out_tmin_path + "/" + coor_name;
    if (does_file_exist_sync_node(field_out_coor_path + "/checkpoint")){
      count += 1;
      continue;
    }
    if (coor_name != point_src_prop_list[i]) {continue;}

    // load point src prop
    main_displayln_info("point_src_prop path: " + point_src_path + "/" + point_src_prop_list[i]);
    const Propagator4d& point_src_prop = get_prop(point_src_path + "/" + point_src_prop_list[i]);
    const Geometry& point_src_geo = point_src_prop.geo;
    const Coordinate total_site_ = point_src_geo.total_site();
    qassert(total_site == total_site_);

    // setup PionGGElemField
    const Geometry& geo = point_src_prop.geo;
    PionGGElemField three_point_correlator_labeled_xp;
    three_point_correlator_labeled_xp.init(geo);
    set_zero(three_point_correlator_labeled_xp.field);

    compute_three_point_correlator_from_closest_wall_src_prop(point_src_coor, t_min, three_point_correlator_labeled_xp, point_src_prop, wall_src_list, pion);

    // shift
    PionGGElemField three_point_correlator_labeled_xp_shift;
    field_shift(three_point_correlator_labeled_xp_shift, three_point_correlator_labeled_xp, -point_src_coor);

    // save
    const Coordinate new_geom(1, 1, 1, 8);
    qmkdir_sync_node(field_out_coor_path);
    dist_write_field(three_point_correlator_labeled_xp_shift, new_geom, field_out_coor_path);
    sync_node();

    count += 1;
    main_displayln_info(ssprintf("Save PionGGElem Field to [%04d]: ", count) + field_out_coor_path);
  }
  qtouch_info(field_out_tmin_path + "/checkpoint");
}

void compute_point_point_wall_correlator_ama_in_one_traj(const int t_min, const std::string& wall_src_path, const std::string& exact_wall_src_path, const std::string& gauge_transform_path, const std::string& point_src_path, const std::string& field_out_traj_path, const double pion, int type_)
{
  TIMER_VERBOSE("compute_point_point_wall_correlator_ama_in_one_traj");

  qassert(does_file_exist_sync_node(field_out_traj_path));
  const std::string field_out_tmin_path = field_out_traj_path + "/" + ssprintf("ama ; t-min=%04d", t_min);
  qmkdir_sync_node(field_out_tmin_path);
  if (does_file_exist_sync_node(field_out_tmin_path + "/checkpoint")){return;}

  main_displayln_info("Load Gauge Transform And Get Inv: " + gauge_transform_path);
  GaugeTransform gtinv;
  {
    GaugeTransform gt;
    dist_read_field(gt, gauge_transform_path);
    to_from_big_endian_64(get_data(gt));
    gt_inverse(gtinv, gt);
  }
  const Coordinate total_site = gtinv.geo.total_site();

  // pre-load and gauge inv wall_prop
  main_displayln_info(ssprintf("Preload And Gauge Inv Wall Src Prop for All T[%d]: ", total_site[3]) + wall_src_path);
  std::vector<Propagator4d> wall_src_list(total_site[3]);
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string wall_src_t_path = wall_src_path + "/t=" + ssprintf("%d", t_wall);
    main_displayln_info("Read Sloppy Wall From: " + wall_src_t_path);
    Propagator4d& wall_src_prop = wall_src_list[t_wall];
    wall_src_prop = get_wall_prop(wall_src_t_path);
    Propagator4d wall_src_prop_gauge;
    prop_apply_gauge_transformation(wall_src_prop_gauge, wall_src_prop, gtinv);
    wall_src_prop = wall_src_prop_gauge;
  }
  main_displayln_info("Preload And Gauge Inv Exact Wall Src Prop");
  int num_exact = 0;
  std::vector<int> exact_wall_t_list;
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string exact_wall_src_t_path = exact_wall_src_path + "/exact ; t=" + ssprintf("%d", t_wall);
    if (!does_file_exist_sync_node(exact_wall_src_t_path)) {continue;}
    exact_wall_t_list.push_back(t_wall);
    num_exact += 1;
  }
  std::vector<Propagator4d> exact_wall_src_list(num_exact);
  for (int i = 0; i < num_exact; ++i){
    std::string exact_wall_src_t_path = exact_wall_src_path + "/exact ; t=" + ssprintf("%d", exact_wall_t_list[i]);
    Propagator4d& exact_wall_src_prop = exact_wall_src_list[i];
    main_displayln_info("Read Exact Wall From: " + exact_wall_src_t_path);
    exact_wall_src_prop = get_wall_prop(exact_wall_src_t_path);
    Propagator4d exact_wall_src_prop_gauge;
    prop_apply_gauge_transformation(exact_wall_src_prop_gauge, exact_wall_src_prop, gtinv);
    exact_wall_src_prop = exact_wall_src_prop_gauge;
  }
  main_displayln_info(ssprintf("Wall Src Num of Sloppy %d, Exact %d", wall_src_list.size(), exact_wall_src_list.size()));
  
  // check num of sloppy and exact
  int num_0 = 0;
  int num_1 = 0;
  int num_2 = 0;
  const std::vector<std::string> point_src_prop_list = list_folders_under_path(point_src_path);
  for (int i =0; i < point_src_prop_list.size(); ++i){
    const Coordinate point_src_coor = get_xg_from_path(point_src_prop_list[i]);
    const std::string fname_0 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, 0);
    if (fname_0 != point_src_prop_list[i]) {continue; }
    num_0 += 1;

    const std::string fname_1 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, 1);
    if (!does_file_exist_sync_node(point_src_path + "/" + fname_1)) {continue;}
    num_1 += 1;

    const std::string fname_2 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, 2);
    if (!does_file_exist_sync_node(point_src_path + "/" + fname_2)) {continue;}
    num_2 += 1;
  }
  main_displayln_info(ssprintf("Point Src Num of Accuracy0 %d, Accuracy1 %d, Accuracy2 %d", num_0, num_1, num_2));

  int count = 0;
  for (int i =0; i < point_src_prop_list.size(); ++i){

    if (type_ != read_type(point_src_prop_list[i]) or 0 != read_accuracy(point_src_prop_list[i])) {continue; }
    const Coordinate point_src_coor = get_xg_from_path(point_src_prop_list[i]);
    const std::string field_out_coor_path = field_out_tmin_path + "/" + ssprintf("xg=(%d,%d,%d,%d) ; type=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_);
    if (does_file_exist_sync_node(field_out_coor_path + "/checkpoint")){
      count += 1;
      continue; 
    }

    // load point src prop
    const std::string fname_0 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, 0);
    if (fname_0 != point_src_prop_list[i]) {continue;}
    main_displayln_info("point_src_prop path (accuracy == 0): " + point_src_path + "/" + fname_0);
    const Propagator4d& point_src_prop_0 = get_prop(point_src_path + "/" + fname_0);
    Geometry point_src_geo = point_src_prop_0.geo;
    Coordinate total_site_ = point_src_geo.total_site();
    qassert(total_site == total_site_);
    Propagator4d point_src_prop;
    point_src_prop = point_src_prop_0;

    // load point src prop accuracy == 1
    const std::string fname_1 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, 1);
    if (does_file_exist_sync_node(point_src_path + "/" + fname_1)) {
      main_displayln_info("point_src_prop path (accuracy == 1): " + point_src_path + "/" + fname_1);
      const Propagator4d& point_src_prop_1 = get_prop(point_src_path + "/" + fname_1);
      point_src_geo = point_src_prop_1.geo;
      total_site_ = point_src_geo.total_site();
      qassert(total_site == total_site_);

      Propagator4d point_src_prop_;
      point_src_prop_ = point_src_prop_1;
      point_src_prop_ -= point_src_prop_0;
      point_src_prop_ *= (1.0 * num_0 / num_1);
      point_src_prop += point_src_prop_;

      // load point src prop accuracy == 2
      const std::string fname_2 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], type_, 2);
      if (does_file_exist_sync_node(point_src_path + "/" + fname_2)) {
        main_displayln_info("point_src_prop path (accuracy == 2): " + point_src_path + "/" + fname_2);
        const Propagator4d& point_src_prop_2 = get_prop(point_src_path + "/" + fname_2);
        point_src_geo = point_src_prop_2.geo;
        total_site_ = point_src_geo.total_site();
        qassert(total_site == total_site_);

        Propagator4d point_src_prop__;
        point_src_prop__ = point_src_prop_2;
        point_src_prop__ -= point_src_prop_1;
        point_src_prop__ *= (1.0 * num_0 / num_2);
        point_src_prop += point_src_prop__;
      }
    }

    // setup PionGGElemField
    const Geometry& geo = point_src_prop.geo;
    PionGGElemField three_point_correlator_labeled_xp;
    three_point_correlator_labeled_xp.init(geo);
    set_zero(three_point_correlator_labeled_xp.field);

    compute_three_point_correlator_ama_from_closest_wall_src_prop(point_src_coor, t_min, three_point_correlator_labeled_xp, point_src_prop, wall_src_list, exact_wall_src_list, exact_wall_t_list, pion);

    // shift
    PionGGElemField three_point_correlator_labeled_xp_shift;
    field_shift(three_point_correlator_labeled_xp_shift, three_point_correlator_labeled_xp, -point_src_coor);

    // save
    const Coordinate new_geom(1, 1, 1, 8);
    qmkdir_sync_node(field_out_coor_path);
    dist_write_field(three_point_correlator_labeled_xp_shift, new_geom, field_out_coor_path);
    sync_node();

    count += 1;
    main_displayln_info(ssprintf("Save PionGGElem Field to [%04d/%04d]: ", count, num_0) + field_out_coor_path);
  }
  qtouch_info(field_out_tmin_path + "/checkpoint");
  return;
}

void read_pionggelemfield_with_accuracy_and_avg(PionGGElemField& pgge_field_avg, const std::string& field_path, int type_, int accuracy_)
{
  TIMER_VERBOSE("read_pionggelemfield_with_accuracy_and_avg");
  const std::vector<std::string> field_list = list_folders_under_path(field_path);
  qassert(field_list.size() >= 1);

  int num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string fname = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", coor[0], coor[1], coor[2], coor[3], type_, accuracy_);
    if (fname != field_list[i]) {continue;}
    num += 1;
    PionGGElemField pgge_field;
    main_displayln_info("Read PionGGElemField from: " + one_field_path);
    dist_read_field(pgge_field, one_field_path);
    if (num == 1) {
      pgge_field_avg = pgge_field;
    } else {
      pgge_field_avg += pgge_field;
    }
  }
  main_displayln_info(ssprintf("Read PionGGElemField Num: %d", num));
  qassert(num != 0);
  pgge_field_avg /= double(num);
  return;
}

void avg_pionggelemfield_with_accuracy_and_rm(const std::string& field_path, int type_, int accuracy_)
{
  TIMER_VERBOSE("avg_pionggelemfield_with_accuracy_and_rm");
  if (!does_file_exist_sync_node(field_path + "/checkpoint")) {
    main_displayln_info("PionGGElemField Has Not Completed In: " + field_path);
    return;
  }
  if (does_file_exist_sync_node(field_path + "/avg_checkpoint")) {
    main_displayln_info("Avg PionGGElemField Has Already Completed In: " + field_path);
    return;
  }
  const std::vector<std::string> field_list = list_folders_under_path(field_path);
  PionGGElemField pgge_field_avg;

  int num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string fname = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", coor[0], coor[1], coor[2], coor[3], type_, accuracy_);
    if (fname != field_list[i]) {continue;}
    num += 1;
    PionGGElemField pgge_field;
    main_displayln_info("Read PionGGElemField from: " + one_field_path);
    dist_read_field(pgge_field, one_field_path);
    if (num == 1) {
      pgge_field_avg = pgge_field;
    } else {
      pgge_field_avg += pgge_field;
    }
  }
  main_displayln_info(ssprintf("Read PionGGElemField Num: %d", num));
  qassert(num != 0);
  pgge_field_avg /= double(num);
  
  // save avg
  std::string field_out_path = field_path + ssprintf("/avg ; type=%d ; accuracy=%d", type_, accuracy_);
  qmkdir_sync_node(field_out_path);
  const Coordinate new_geom(1, 1, 1, 8);
  dist_write_field(pgge_field_avg, new_geom, field_out_path);
  sync_node();
  qtouch_info(field_path + "/avg_checkpoint");
  main_displayln_info(ssprintf("Save Avg PionGGElem Field to: ") + field_out_path);

  // rm folders
  num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string fname = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", coor[0], coor[1], coor[2], coor[3], type_, accuracy_);
    if (fname != field_list[i]) {continue;}
    qremove_all_info(one_field_path);
    num += 1;
    main_displayln_info("Delete PionGGElemField Folder: " + one_field_path);
  }
  main_displayln_info(ssprintf("Delete PionGGElemField Folders Num: %d", num));
  return;
}

void avg_pionggelemfield_without_accuracy_and_rm(const std::string& field_path, int type_)
{
  TIMER_VERBOSE("avg_pionggelemfield_without_accuracy_and_rm");
  if (!does_file_exist_sync_node(field_path + "/checkpoint")) {
    main_displayln_info("PionGGElemField Has Not Completed In: " + field_path);
    return;
  }
  if (does_file_exist_sync_node(field_path + "/avg_checkpoint")) {
    main_displayln_info("Avg PionGGElemField Has Already Completed In: " + field_path);
    return;
  }
  const std::vector<std::string> field_list = list_folders_under_path(field_path);
  PionGGElemField pgge_field_avg;

  int num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string fname = ssprintf("xg=(%d,%d,%d,%d) ; type=%d", coor[0], coor[1], coor[2], coor[3], type_);
    if (fname != field_list[i]) {continue;}
    num += 1;
    PionGGElemField pgge_field;
    main_displayln_info("Read PionGGElemField from: " + one_field_path);
    dist_read_field(pgge_field, one_field_path);
    if (num == 1) {
      pgge_field_avg = pgge_field;
    } else {
      pgge_field_avg += pgge_field;
    }
  }
  main_displayln_info(ssprintf("Read PionGGElemField Num: %d", num));
  qassert(num != 0);
  pgge_field_avg /= double(num);
  
  // save avg
  std::string field_out_path = field_path + ssprintf("/avg ; type=%d", type_);
  qmkdir_sync_node(field_out_path);
  const Coordinate new_geom(1, 1, 1, 8);
  dist_write_field(pgge_field_avg, new_geom, field_out_path);
  sync_node();
  qtouch_info(field_path + "/avg_checkpoint");
  main_displayln_info(ssprintf("Save Avg PionGGElem Field to: ") + field_out_path);

  // rm folders
  num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string fname = ssprintf("xg=(%d,%d,%d,%d) ; type=%d", coor[0], coor[1], coor[2], coor[3], type_);
    if (fname != field_list[i]) {continue;}
    qremove_all_info(one_field_path);
    num += 1;
    main_displayln_info("Delete PionGGElemField Folder: " + one_field_path);
  }
  main_displayln_info(ssprintf("Delete PionGGElemField Folders Num: %d", num));
  return;
}

void read_pionggelemfield_without_accuracy_and_avg(PionGGElemField& pgge_field_avg, const std::string& field_path, int type_)
{
  TIMER_VERBOSE("read_pionggelemfield_without_accuracy_and_avg");
  const std::vector<std::string> field_list = list_folders_under_path(field_path);
  qassert(field_list.size() >= 1);

  int num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string fname = ssprintf("xg=(%d,%d,%d,%d) ; type=%d", coor[0], coor[1], coor[2], coor[3], type_);
    if (fname != field_list[i]) {continue;}
    num += 1;
    PionGGElemField pgge_field;
    main_displayln_info("Read PionGGElemField from: " + one_field_path);
    dist_read_field(pgge_field, one_field_path);
    if (num == 1) {
      pgge_field_avg = pgge_field;
    } else {
      pgge_field_avg += pgge_field;
    }
  }
  main_displayln_info(ssprintf("Read PionGGElemField Num: %d", num));
  qassert(num != 0);
  pgge_field_avg /= double(num);
  return;
}

void sum_sink_over_space_from_prop(std::vector<WilsonMatrix>& wm_t, const Propagator4d& prop)
{
  TIMER_VERBOSE("sum_sink_over_space_from_prop");
  const Geometry& geo = prop.geo;
  const int total_t = geo.total_site()[3];
  qassert(total_t == wm_t.size());
  for (int i = 0; i < wm_t.size(); ++i)
  {
    set_zero(wm_t[i]);
  }

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lx = geo.coordinate_from_index(index);
    const Coordinate x = geo.coordinate_g_from_l(lx);
    const int t = x[3];

    wm_t[t] += prop.get_elem(lx);
  }

  for (int i = 0; i < wm_t.size(); ++i)
  {
    glb_sum_double(wm_t[i]);
  }
}

void sum_src_over_space_from_prop(std::vector<WilsonMatrix>& wm_t, const Propagator4d& prop)
{
  TIMER_VERBOSE("sum_src_over_space_from_prop");
  const Geometry& geo = prop.geo;
  const int total_t = geo.total_site()[3];
  qassert(total_t == wm_t.size());
  for (int i = 0; i < wm_t.size(); ++i)
  {
    set_zero(wm_t[i]);
  }

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lx = geo.coordinate_from_index(index);
    const Coordinate x = geo.coordinate_g_from_l(lx);
    const int t = x[3];

    wm_t[t] += gamma5 * matrix_adjoint(prop.get_elem(lx)) * gamma5;
  }

  for (int i = 0; i < wm_t.size(); ++i)
  {
    glb_sum_double(wm_t[i]);
  }
}


std::vector<Complex> compute_wall_wall_correlation_function(const std::string& wall_src_path, const int total_t)
{
  TIMER_VERBOSE("compute_wall_wall_correlation_function");

  main_displayln_info("sum src and sink over space for all wall src props start");
  std::vector<std::vector<WilsonMatrix>> v_v_wm_sink(total_t);
  std::vector<std::vector<WilsonMatrix>> v_v_wm_src(total_t);
  for (int t = 0; t < total_t; ++t)
  {
    v_v_wm_sink[t].resize(total_t);
    v_v_wm_src[t].resize(total_t);
  }

  for (int t = 0; t < total_t; ++t)
  {
    main_displayln_info("sum src and sink, t=" + ssprintf("%d", t));
    const std::string wall_src_t_path = wall_src_path + "/t=" + ssprintf("%d", t);
    const Propagator4d& wall_src_t = get_prop(wall_src_t_path);
    std::vector<WilsonMatrix>& v_wm_sink = v_v_wm_sink[t];
    std::vector<WilsonMatrix>& v_wm_src = v_v_wm_src[t];

    if (v_wm_sink.size() != wall_src_t.geo.total_site()[3])
    {
      main_displayln_info("Bad wall_src_t_path: " + wall_src_t_path);
      main_displayln_info(ssprintf("v_wm_sink.size=%d, wall_src_t.total_site()[3]=%d", v_wm_sink.size(), wall_src_t.geo.total_site()[3]));
      continue;
    }

    sum_sink_over_space_from_prop(v_wm_sink, wall_src_t);
    sum_src_over_space_from_prop(v_wm_src, wall_src_t);
  }
  main_displayln_info("sum src and sink over space for all wall src props end");

  main_displayln_info("compute wall to wall");
  std::vector<Complex> wall_wall_corr(total_t / 2);
  set_zero(wall_wall_corr);
  for (int tstart = 0; tstart < total_t; ++tstart)
  {
    for (int tsep = 0; tsep < total_t / 2; ++tsep)
    {
      int tend = mod(tstart + tsep, total_t);
      main_displayln_info(ssprintf("compute wall to wall, tstart=%d, tsep=%d, tend=%d", tstart, tsep, tend));
      // wall_wall_corr[tsep] += pi_pi_contraction(v_v_wm_sink[tstart][tend], v_v_wm_sink[tend][tstart]);
      // wall_wall_corr[tsep] += pi_pi_contraction(v_v_wm_src[tstart][tend], v_v_wm_src[tend][tstart]);
      wall_wall_corr[tsep] += pi_pi_contraction(v_v_wm_sink[tstart][tend], v_v_wm_src[tstart][tend]);
    }
  }

  for (int tsep = 0; tsep < total_t / 2; ++tsep)
  {
    wall_wall_corr[tsep] /= total_t;
  }

  return wall_wall_corr;
}

std::string show_vec_complex(std::vector<Complex> vec)
{
  std::string res = "";
  for (int i = 0; i < vec.size(); ++i)
  {
    res += ssprintf("%e %e, ", vec[i].real(), vec[i].imag());
  }
  return res;
}

std::vector<Complex> compute_zw_from_wall_wall_corr(const std::vector<Complex>& vec, const double pion)
{
  std::vector<Complex> res(vec.size());
  for (int tsep = 0; tsep < vec.size(); ++tsep)
  {
    res[tsep] = vec[tsep] * exp(tsep * pion);
  }
  return res;
}


void compute_three_point_corr_model(PionGGElemField& three_point_correlator_labeled_xp, const int tsep, const double pion)
{
  TIMER_VERBOSE("compute_three_point_corr_model");
  Geometry& geo = three_point_correlator_labeled_xp.geo;
  const Coordinate total_site = geo.total_site();

  Coordinate x = Coordinate(0,0,0,0);
  Coordinate z = Coordinate(0,0,0,tsep);

#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) 
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const Coordinate xp_x = relative_coordinate(xp-x, total_site);
    // const CoordinateD xp_x_mid_z = relative_coordinate(middle_coordinate(x, xp, CoordinateD(total_site)) - CoordinateD(z), CoordinateD(total_site));
    const CoordinateD xp_x_mid_z = relative_coordinate(middle_coordinate(x, xp, CoordinateD(total_site)), CoordinateD(total_site)) - CoordinateD(z);

    PionGGElem& pgge = three_point_correlator_labeled_xp.get_elem(lxp);
    set_zero(pgge);
    b_model(pgge, CoordinateD(-xp_x), xp_x_mid_z, pion);
  }
  return;
}

void compute_f2_24D_sloppy_all_traj()
{
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975;
  const std::string ENSEMBLE = "24D";
  const std::string THREE_POINT_ENSEMBLE_PATH = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + "/sloppy";
  double XXP_LIMIT = 10;
  const int TYPE = 0;
  const int ACCURACY = 0;
  const int TMIN = 10;
  const std::string MOD = "";
  // const std::string MOD = "xyp>=xy";
  // const std::string MOD = "xy>=xyp";
  const int NUM_PAIRS_IN_CONFIG = 1024;
  const int NUM_PAIRS_JUMP = 4096;
  std::string pair_info_file = "/home/tucheng/qcdlib-python/hlbl-pi0.out/";
  pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";
  const std::string F2_FOLDER = "f2";
  qmkdir_sync_node(F2_FOLDER);
  qmkdir_sync_node(F2_FOLDER + "/" + ENSEMBLE);

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info("THREE_POINT_ENSEMBLE_PATH: " + THREE_POINT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("ACCURACY: %d", ACCURACY));
  main_displayln_info(ssprintf("TMIN: %d", TMIN));
  main_displayln_info("MOD: " + MOD);
  main_displayln_info(ssprintf("NUM_PAIRS_IN_CONFIG: %d", NUM_PAIRS_IN_CONFIG));
  main_displayln_info(ssprintf("NUM_PAIRS_JUMP: %d", NUM_PAIRS_JUMP));
  main_displayln_info(ssprintf("Pair Info File: ") + pair_info_file);

  // read all pair info
  std::vector<Y_And_Rotation_Info_Elem> pairs_info_all = read_y_and_rotation_info(pair_info_file);

  // read pigg_field all traj
  std::vector<int> traj_list = {1800, 1800};

  // f2
  for (int i_pair = 0; i_pair < traj_list.size() / 2; ++i_pair)
  {
    int i_pgge_1 = i_pair;
    int i_pgge_2 = traj_list.size() / 2 + i_pair;
    main_displayln_info(ssprintf("Compute f2 from Configs: %04d, %04d", traj_list[i_pgge_1], traj_list[i_pgge_2]));

    // set saving folder and set pairs
    const std::string one_pair_save_folder = F2_FOLDER + "/" + ENSEMBLE + ssprintf("/traj=%04d,%04d;t-min=%04d;xxp-limit=%d;mod=%s;type=%d;accuracy=%d", traj_list[i_pgge_1], traj_list[i_pgge_2], TMIN, int(XXP_LIMIT), MOD.c_str(), TYPE, ACCURACY);
    int i_pairs_start = 4096 + i_pair * NUM_PAIRS_JUMP;
    qmkdir_sync_node(one_pair_save_folder);
    if (does_file_exist_sync_node(one_pair_save_folder + "/" + ssprintf("%05d", NUM_PAIRS_IN_CONFIG - 1))) {continue;}

    // set pairs
    std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + NUM_PAIRS_JUMP);
    main_displayln_info(ssprintf("Y_And_Rotation_Info_Elem Index: %08d - %08d", i_pairs_start, i_pairs_start + NUM_PAIRS_JUMP));

    // read pgge
    PionGGElemField pgge_field_1;
    PionGGElemField pgge_field_2;
    const std::string PGGE_FIELD_PATH_1 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_1], TMIN);
    const std::string PGGE_FIELD_PATH_2 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_2], TMIN);
    if (!does_file_exist_sync_node(PGGE_FIELD_PATH_1 + "/avg_checkpoint") or !does_file_exist_sync_node(PGGE_FIELD_PATH_2 + "/avg_checkpoint")){continue; }
    std::string AVG_PGGE_FIELD_PATH_1 = PGGE_FIELD_PATH_1 + ssprintf("/avg ; type=%d ; accuracy=%d", TYPE, ACCURACY);
    main_displayln_info("Read AVG_PGGE_FIELD_1 from: " + AVG_PGGE_FIELD_PATH_1);
    dist_read_field(pgge_field_1, AVG_PGGE_FIELD_PATH_1);
    std::string AVG_PGGE_FIELD_PATH_2 = PGGE_FIELD_PATH_2 + ssprintf("/avg ; type=%d ; accuracy=%d", TYPE, ACCURACY);
    main_displayln_info("Read AVG_PGGE_FIELD_2 from: " + AVG_PGGE_FIELD_PATH_2);
    dist_read_field(pgge_field_2, AVG_PGGE_FIELD_PATH_2);
    // const std::string PGGE_FIELD_PATH_1 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_1], TMIN);
    // read_pionggelemfield_with_accuracy_and_avg(pgge_field_1, PGGE_FIELD_PATH_1, TYPE, ACCURACY);
    // const std::string PGGE_FIELD_PATH_2 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_2], TMIN);
    // read_pionggelemfield_with_accuracy_and_avg(pgge_field_2, PGGE_FIELD_PATH_2, TYPE, ACCURACY);

    // compute f2
    avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, pgge_field_1, pgge_field_2, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD, NUM_PAIRS_IN_CONFIG);
  }
}

void compute_f2_24D_ama_all_traj()
{
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975;
  const std::string ENSEMBLE = "24D";
  double XXP_LIMIT = 10;
  const int TYPE = 0;
  const int TMIN = 10;
  const std::string MOD = "";
  // const std::string MOD = "xyp>=xy";
  // const std::string MOD = "xy>=xyp";
  const int NUM_PAIRS_IN_CONFIG = 1024;
  const int NUM_PAIRS_JUMP = 4096;
  std::string pair_info_file = "/home/tucheng/qcdlib-python/hlbl-pi0.out/";
  pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";
  const std::string F2_FOLDER = "f2";
  qmkdir_sync_node(F2_FOLDER);
  qmkdir_sync_node(F2_FOLDER + "/" + ENSEMBLE);

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("TMIN: %d", TMIN));
  main_displayln_info("MOD: " + MOD);
  main_displayln_info(ssprintf("NUM_PAIRS_IN_CONFIG: %d", NUM_PAIRS_IN_CONFIG));
  main_displayln_info(ssprintf("NUM_PAIRS_JUMP: %d", NUM_PAIRS_JUMP));
  main_displayln_info(ssprintf("Pair Info File: ") + pair_info_file);

  // read all pair info
  std::vector<Y_And_Rotation_Info_Elem> pairs_info_all = read_y_and_rotation_info(pair_info_file);

  // read pigg_field all traj
  // std::vector<int> traj_list = {1010, 1030, 1050, 1070, 1090, 1110, 1140, 1160, 1180, 1220, 1240, 1260, 1280, 1300, 1320, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280};
  std::vector<int> traj_list = {1000, 1000};

  // f2
  for (int i_pair = 0; i_pair < traj_list.size() / 2; ++i_pair)
  {
    int i_pgge_1 = i_pair;
    int i_pgge_2 = traj_list.size() / 2 + i_pair;
    main_displayln_info(ssprintf("Compute f2 from Configs: %04d, %04d", traj_list[i_pgge_1], traj_list[i_pgge_2]));

    // set saving folder and set pairs
    const std::string one_pair_save_folder = F2_FOLDER + "/" + ENSEMBLE + ssprintf("/traj=%04d,%04d;ama;t-min=%04d;xxp-limit=%d;mod=%s;type=%d", traj_list[i_pgge_1], traj_list[i_pgge_2], TMIN, int(XXP_LIMIT), MOD.c_str(), TYPE);
    int i_pairs_start = 4096 + i_pair * NUM_PAIRS_JUMP;
    qmkdir_sync_node(one_pair_save_folder);
    if (does_file_exist_sync_node(one_pair_save_folder + "/" + ssprintf("%05d", NUM_PAIRS_IN_CONFIG - 1))) {continue;}

    // set pairs
    std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + NUM_PAIRS_JUMP);
    main_displayln_info(ssprintf("Y_And_Rotation_Info_Elem Index: %08d - %08d", i_pairs_start, i_pairs_start + NUM_PAIRS_JUMP));

    // read pgge
    PionGGElemField pgge_field_1;
    PionGGElemField pgge_field_2;
    const std::string PGGE_FIELD_PATH_1 = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + ssprintf("/results=%04d/ama ; t-min=%04d", traj_list[i_pgge_1], TMIN);
    read_pionggelemfield_without_accuracy_and_avg(pgge_field_1, PGGE_FIELD_PATH_1, TYPE);
    const std::string PGGE_FIELD_PATH_2 = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + ssprintf("/results=%04d/ama ; t-min=%04d", traj_list[i_pgge_2], TMIN);
    read_pionggelemfield_without_accuracy_and_avg(pgge_field_2, PGGE_FIELD_PATH_2, TYPE);

    // compute f2
    avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, pgge_field_1, pgge_field_2, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD, NUM_PAIRS_IN_CONFIG);
  }
}

void compute_f2_32D_sloppy_all_traj()
{
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.139474;
  const std::string ENSEMBLE = "32D-0.00107";
  const std::string THREE_POINT_ENSEMBLE_PATH = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + "/sloppy";
  double XXP_LIMIT = 10;
  const int TYPE = 0;
  const int ACCURACY = 0;
  const int TMIN = 10;
  const std::string MOD = "";
  // const std::string MOD = "xyp>=xy";
  // const std::string MOD = "xy>=xyp";
  const int NUM_PAIRS_IN_CONFIG = 1024;
  const int NUM_PAIRS_JUMP = 1024;
  std::string pair_info_file = "/home/tucheng/qcdlib-python/hlbl-pi0.out/";
  pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";
  const std::string F2_FOLDER = "f2";
  qmkdir_sync_node(F2_FOLDER);
  qmkdir_sync_node(F2_FOLDER + "/" + ENSEMBLE);

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info("THREE_POINT_ENSEMBLE_PATH: " + THREE_POINT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("ACCURACY: %d", ACCURACY));
  main_displayln_info(ssprintf("TMIN: %d", TMIN));
  main_displayln_info("MOD: " + MOD);
  main_displayln_info(ssprintf("NUM_PAIRS_IN_CONFIG: %d", NUM_PAIRS_IN_CONFIG));
  main_displayln_info(ssprintf("NUM_PAIRS_JUMP: %d", NUM_PAIRS_JUMP));
  main_displayln_info(ssprintf("Pair Info File: ") + pair_info_file);

  // read all pair info
  std::vector<Y_And_Rotation_Info_Elem> pairs_info_all = read_y_and_rotation_info(pair_info_file);

  std::vector<int> traj_list;
  for (int traj = 680; traj < 1371; traj += 10) {
    traj_list.push_back(traj);
  }

  // f2
  for (int i_pair = 0; i_pair < traj_list.size() / 2; ++i_pair)
  {
    int i_pgge_1 = i_pair;
    int i_pgge_2 = traj_list.size() / 2 + i_pair;
    main_displayln_info(ssprintf("Compute f2 from Configs: %04d, %04d", traj_list[i_pgge_1], traj_list[i_pgge_2]));

    // set saving folder and set pairs
    const std::string one_pair_save_folder = F2_FOLDER + "/" + ENSEMBLE + ssprintf("/traj=%04d,%04d;t-min=%04d;xxp-limit=%d;mod=%s;type=%d;accuracy=%d", traj_list[i_pgge_1], traj_list[i_pgge_2], TMIN, int(XXP_LIMIT), MOD.c_str(), TYPE, ACCURACY);
    int i_pairs_start = 4096 + i_pair * NUM_PAIRS_JUMP;
    qmkdir_sync_node(one_pair_save_folder);
    if (does_file_exist_sync_node(one_pair_save_folder + "/" + ssprintf("%05d", NUM_PAIRS_IN_CONFIG - 1))) {continue;}

    // set pairs
    std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + NUM_PAIRS_JUMP);
    main_displayln_info(ssprintf("Y_And_Rotation_Info_Elem Index: %08d - %08d", i_pairs_start, i_pairs_start + NUM_PAIRS_JUMP));

    // read pgge
    PionGGElemField pgge_field_1;
    PionGGElemField pgge_field_2;
    const std::string PGGE_FIELD_PATH_1 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_1], TMIN);
    const std::string PGGE_FIELD_PATH_2 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_2], TMIN);
    if (!does_file_exist_sync_node(PGGE_FIELD_PATH_1 + "/avg_checkpoint") or !does_file_exist_sync_node(PGGE_FIELD_PATH_2 + "/avg_checkpoint")){continue; }
    std::string AVG_PGGE_FIELD_PATH_1 = PGGE_FIELD_PATH_1 + ssprintf("/avg ; type=%d ; accuracy=%d", TYPE, ACCURACY);
    main_displayln_info("Read AVG_PGGE_FIELD_1 from: " + AVG_PGGE_FIELD_PATH_1);
    dist_read_field(pgge_field_1, AVG_PGGE_FIELD_PATH_1);
    std::string AVG_PGGE_FIELD_PATH_2 = PGGE_FIELD_PATH_2 + ssprintf("/avg ; type=%d ; accuracy=%d", TYPE, ACCURACY);
    main_displayln_info("Read AVG_PGGE_FIELD_2 from: " + AVG_PGGE_FIELD_PATH_2);
    dist_read_field(pgge_field_2, AVG_PGGE_FIELD_PATH_2);
    // const std::string PGGE_FIELD_PATH_1 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_1], TMIN);
    // read_pionggelemfield_with_accuracy_and_avg(pgge_field_1, PGGE_FIELD_PATH_1, TYPE, ACCURACY);
    // const std::string PGGE_FIELD_PATH_2 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_2], TMIN);
    // read_pionggelemfield_with_accuracy_and_avg(pgge_field_2, PGGE_FIELD_PATH_2, TYPE, ACCURACY);

    // compute f2
    avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, pgge_field_1, pgge_field_2, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD, NUM_PAIRS_IN_CONFIG);
  }
}

void compute_f2_32D_ama_all_traj(const double XXP_LIMIT_, const int TMIN_, const std::string MOD_)
{
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.139474;
  const std::string ENSEMBLE = "32D-0.00107";
  const std::string THREE_POINT_ENSEMBLE_PATH = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + "/ama";
  double XXP_LIMIT = XXP_LIMIT_;
  const int TYPE = 0;
  const int TMIN = TMIN_;
  const std::string MOD = MOD_;
  // const std::string MOD = "xyp>=xy";
  // const std::string MOD = "xy>=xyp";
  const int NUM_PAIRS_IN_CONFIG = 1024;
  const int NUM_PAIRS_JUMP = 1024;
  std::string pair_info_file = "/home/tucheng/qcdlib-python/hlbl-pi0.out/";
  pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";
  const std::string F2_FOLDER = "f2";
  qmkdir_sync_node(F2_FOLDER);
  qmkdir_sync_node(F2_FOLDER + "/" + ENSEMBLE);

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info("THREE_POINT_ENSEMBLE_PATH: " + THREE_POINT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("TMIN: %d", TMIN));
  main_displayln_info("MOD: " + MOD);
  main_displayln_info(ssprintf("NUM_PAIRS_IN_CONFIG: %d", NUM_PAIRS_IN_CONFIG));
  main_displayln_info(ssprintf("NUM_PAIRS_JUMP: %d", NUM_PAIRS_JUMP));
  main_displayln_info(ssprintf("Pair Info File: ") + pair_info_file);

  // read all pair info
  std::vector<Y_And_Rotation_Info_Elem> pairs_info_all = read_y_and_rotation_info(pair_info_file);

  std::vector<int> traj_list;
  for (int traj = 680; traj < 1371; traj += 10) {
    traj_list.push_back(traj);
  }

  // f2
  for (int i_pair = 0; i_pair < traj_list.size() / 2; ++i_pair)
  {
    int i_pgge_1 = i_pair;
    int i_pgge_2 = traj_list.size() / 2 + i_pair;
    main_displayln_info(ssprintf("Compute f2 from Configs: %04d, %04d", traj_list[i_pgge_1], traj_list[i_pgge_2]));

    // set saving folder and set pairs
    const std::string one_pair_save_folder = F2_FOLDER + "/" + ENSEMBLE + ssprintf("/traj=%04d,%04d;ama;t-min=%04d;xxp-limit=%d;mod=%s;type=%d", traj_list[i_pgge_1], traj_list[i_pgge_2], TMIN, int(XXP_LIMIT), MOD.c_str(), TYPE);
    int i_pairs_start = 50000 + i_pair * NUM_PAIRS_JUMP;
    if (does_file_exist_sync_node(one_pair_save_folder + "/" + ssprintf("%05d", NUM_PAIRS_IN_CONFIG - 1))) {
      main_displayln_info(ssprintf("f2 of Configs %04d, %04d have already completed", traj_list[i_pgge_1], traj_list[i_pgge_2]));
      continue;
    }

    // set pairs
    std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + NUM_PAIRS_JUMP);
    main_displayln_info(ssprintf("Y_And_Rotation_Info_Elem Index: %08d - %08d", i_pairs_start, i_pairs_start + NUM_PAIRS_JUMP));

    // read pgge
    PionGGElemField pgge_field_1;
    PionGGElemField pgge_field_2;
    const std::string PGGE_FIELD_PATH_1 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/ama ; t-min=%04d", traj_list[i_pgge_1], TMIN);
    const std::string PGGE_FIELD_PATH_2 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/ama ; t-min=%04d", traj_list[i_pgge_2], TMIN);
    if (!does_file_exist_sync_node(PGGE_FIELD_PATH_1 + "/avg_checkpoint") or !does_file_exist_sync_node(PGGE_FIELD_PATH_2 + "/avg_checkpoint")){
      main_displayln_info(ssprintf("PionGG Field are not completed, %04d, %04d", traj_list[i_pgge_1], traj_list[i_pgge_2]));
      continue;
    }
    std::string AVG_PGGE_FIELD_PATH_1 = PGGE_FIELD_PATH_1 + ssprintf("/avg ; type=%d", TYPE);
    main_displayln_info("Read AVG_PGGE_FIELD_1 from: " + AVG_PGGE_FIELD_PATH_1);
    dist_read_field(pgge_field_1, AVG_PGGE_FIELD_PATH_1);
    std::string AVG_PGGE_FIELD_PATH_2 = PGGE_FIELD_PATH_2 + ssprintf("/avg ; type=%d", TYPE);
    main_displayln_info("Read AVG_PGGE_FIELD_2 from: " + AVG_PGGE_FIELD_PATH_2);
    dist_read_field(pgge_field_2, AVG_PGGE_FIELD_PATH_2);
    // read_pionggelemfield_without_accuracy_and_avg(pgge_field_1, PGGE_FIELD_PATH_1, TYPE);
    // read_pionggelemfield_without_accuracy_and_avg(pgge_field_2, PGGE_FIELD_PATH_2, TYPE);

    // compute f2
    qmkdir_sync_node(one_pair_save_folder);
    avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, pgge_field_1, pgge_field_2, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD, NUM_PAIRS_IN_CONFIG);
  }
}

void compute_point_point_wall_correlator_sloppy_24D_all_traj()
{
  // prepare traj_list
  // std::vector<int> traj_list = {1030};
  // std::vector<int> traj_list = {1010, 1030, 1050, 1070, 1090, 1110, 1140, 1160, 1180, 1220, 1240, 1260, 1280, 1300, 1320, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280};
  const std::string ENSEMBLE = "24D";
  const int T_MIN = 10;
  const std::string FIELD_OUT_ENSEMBLE_PATH = "ThreePointCorrField/24D/sloppy";
  double PION = 0.13975;
  const int ACCURACY= 0;
  const int TYPE = 0;

  qmkdir_sync_node("ThreePointCorrField");
  qmkdir_sync_node("ThreePointCorrField/24D");
  qmkdir_sync_node(FIELD_OUT_ENSEMBLE_PATH);

  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info(ssprintf("T_MIN: %d", T_MIN));
  main_displayln_info("FIELD_OUT_ENSEMBLE_PATH: " + FIELD_OUT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("ACCURACY: %d", ACCURACY));

  // check complecity
  std::vector<int> traj_list;
  for (int i = 1000; i < 2281; i += 10) {
    const std::string point_file = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/results=%d/checkpoint/computeContractionInf", i);
    const std::string wall_file = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/results=%d/checkpoint", i);
    if (!does_file_exist_sync_node(point_file) or !does_file_exist_sync_node(wall_file)) {continue;}
    traj_list.push_back(i);
  }
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));


  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    const std::string FIELD_OUT_TRAJ_PATH = FIELD_OUT_ENSEMBLE_PATH + "/" + ssprintf("results=%04d", traj_list[i]);
    qmkdir_sync_node(FIELD_OUT_TRAJ_PATH);
    compute_point_point_wall_correlator_in_one_traj(T_MIN, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_TRAJ_PATH, PION, TYPE, ACCURACY);

    // avg and remove
    const std::string PGGE_FIELD_PATH = FIELD_OUT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i], T_MIN);
    avg_pionggelemfield_with_accuracy_and_rm(PGGE_FIELD_PATH, TYPE, ACCURACY);
  }
}

void compute_point_point_wall_correlator_ama_24D_all_traj()
{
  // prepare traj_list
  std::vector<int> traj_list = {1130};
  // std::vector<int> traj_list = {1010, 1030, 1050, 1070, 1090, 1110, 1140, 1160, 1180, 1220, 1240, 1260, 1280, 1300, 1320, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280};
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  const std::string ENSEMBLE = "24D";
  const int T_MIN = 10;
  const std::string FIELD_OUT_ENSEMBLE_PATH = "ThreePointCorrField/24D/ama";
  double PION = 0.13975;
  const int TYPE = 0;

  qmkdir_sync_node("ThreePointCorrField");
  qmkdir_sync_node("ThreePointCorrField/24D");
  qmkdir_sync_node(FIELD_OUT_ENSEMBLE_PATH);

  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info(ssprintf("T_MIN: %d", T_MIN));
  main_displayln_info("FIELD_OUT_ENSEMBLE_PATH: " + FIELD_OUT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));

  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string EXACT_WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src-exact/results/24D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("EXACT_WALL_SRC_PATH: " + EXACT_WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    const std::string FIELD_OUT_TRAJ_PATH = FIELD_OUT_ENSEMBLE_PATH + "/" + ssprintf("results=%04d", traj_list[i]);
    qmkdir_sync_node(FIELD_OUT_TRAJ_PATH);
    compute_point_point_wall_correlator_ama_in_one_traj(T_MIN, WALL_SRC_PATH, EXACT_WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_TRAJ_PATH, PION, TYPE);
  }
}

void compute_point_point_wall_correlator_sloppy_32D_all_traj()
{
  const std::string ENSEMBLE = "32D";
  const int T_MIN = 10;
  const std::string FIELD_OUT_ENSEMBLE_PATH = "ThreePointCorrField/32D-0.00107/sloppy";
  double PION = 0.139474;
  const int ACCURACY = 0;
  const int TYPE = 0;

  qmkdir_sync_node("ThreePointCorrField");
  qmkdir_sync_node("ThreePointCorrField/32D-0.00107");
  qmkdir_sync_node(FIELD_OUT_ENSEMBLE_PATH);

  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info(ssprintf("T_MIN: %d", T_MIN));
  main_displayln_info("FIELD_OUT_ENSEMBLE_PATH: " + FIELD_OUT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("ACCURACY: %d", ACCURACY));

  // check complecity
  std::vector<int> traj_list;
  for (int i = 600; i < 1500; i += 10) {
    const std::string point_file = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/results=%d/checkpoint/computeContractionInf", i);
    const std::string wall_file = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/checkpoint.txt", i);
    if (!does_file_exist_sync_node(point_file) or !does_file_exist_sync_node(wall_file)) {continue;}
    traj_list.push_back(i);
    main_displayln_info(ssprintf("Valid Configuration: %d", i));
  }
  // traj_list = {1370};
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    const std::string FIELD_OUT_TRAJ_PATH = FIELD_OUT_ENSEMBLE_PATH + "/" + ssprintf("results=%04d", traj_list[i]);
    qmkdir_sync_node(FIELD_OUT_TRAJ_PATH);
    compute_point_point_wall_correlator_in_one_traj(T_MIN, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_TRAJ_PATH, PION, TYPE, ACCURACY);

    // avg and remove
    const std::string PGGE_FIELD_PATH = FIELD_OUT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i], T_MIN);
    avg_pionggelemfield_with_accuracy_and_rm(PGGE_FIELD_PATH, TYPE, ACCURACY);
  }
}

void compute_point_point_wall_correlator_ama_32D_all_traj()
{
  const std::string ENSEMBLE = "32D";
  const int T_MIN = 10;
  const std::string FIELD_OUT_ENSEMBLE_PATH = "ThreePointCorrField/32D-0.00107/ama";
  double PION = 0.139474;
  const int TYPE = 0;

  qmkdir_sync_node("ThreePointCorrField");
  qmkdir_sync_node("ThreePointCorrField/32D-0.00107");
  qmkdir_sync_node(FIELD_OUT_ENSEMBLE_PATH);

  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info(ssprintf("T_MIN: %d", T_MIN));
  main_displayln_info("FIELD_OUT_ENSEMBLE_PATH: " + FIELD_OUT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));

  // check complecity
  std::vector<int> traj_list;
  // for (int i = 680; i < 1021; i += 10) {
  for (int i = 1020; i > 679; i -= 10) {
    std::string point_file = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/results=%d/checkpoint/computeContractionInf", i);
    std::string wall_file = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/checkpoint.txt", i);
    std::string exact_wall_file = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src-exact-2/results/32D-0.00107/results=%d/checkpoint.txt", i);
    if (!does_file_exist_sync_node(point_file) or !does_file_exist_sync_node(wall_file) or !does_file_exist_sync_node(exact_wall_file)) {continue;}

    int ii = i + 350;
    point_file = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/results=%d/checkpoint/computeContractionInf", ii);
    wall_file = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/checkpoint.txt", ii);
    exact_wall_file = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src-exact-2/results/32D-0.00107/results=%d/checkpoint.txt", ii);
    if (!does_file_exist_sync_node(point_file) or !does_file_exist_sync_node(wall_file) or !does_file_exist_sync_node(exact_wall_file)) {continue;}

    traj_list.push_back(i);
    traj_list.push_back(ii);
    main_displayln_info(ssprintf("Valid Configuration: %d", i));
    main_displayln_info(ssprintf("Valid Configuration: %d", ii));
  }
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string EXACT_WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src-exact-2/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("EXACT_WALL_SRC_PATH: " + EXACT_WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    const std::string FIELD_OUT_TRAJ_PATH = FIELD_OUT_ENSEMBLE_PATH + "/" + ssprintf("results=%04d", traj_list[i]);
    qmkdir_sync_node(FIELD_OUT_TRAJ_PATH);
    const std::string PGGE_FIELD_PATH = FIELD_OUT_ENSEMBLE_PATH + ssprintf("/results=%04d/ama ; t-min=%04d", traj_list[i], T_MIN);
    if (does_file_exist_sync_node(PGGE_FIELD_PATH + "/avg_checkpoint")) {
      main_displayln_info("Avg PionGGElemField Has Already Completed In: " + PGGE_FIELD_PATH);
      continue;
    }

    if (obtain_lock(FIELD_OUT_TRAJ_PATH + "-lock")) {
      compute_point_point_wall_correlator_ama_in_one_traj(T_MIN, WALL_SRC_PATH, EXACT_WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_TRAJ_PATH, PION, TYPE);

      // avg and remove
      avg_pionggelemfield_without_accuracy_and_rm(PGGE_FIELD_PATH, TYPE);
      release_lock();
    }
  }
}

void compute_point_point_wall_correlator_ama_48I_all_traj()
{
  const std::string ENSEMBLE = "48I";
  const int T_MIN = 10;
  const std::string FIELD_OUT_ENSEMBLE_PATH = "ThreePointCorrField/48I-0.00078/ama";
  double PION;
  const int TYPE = 0;

  qmkdir_sync_node("ThreePointCorrField");
  qmkdir_sync_node("ThreePointCorrField/48I-0.00078");
  qmkdir_sync_node(FIELD_OUT_ENSEMBLE_PATH);

  main_displayln_info("ENSEMBLE: " + ENSEMBLE);
  main_displayln_info(ssprintf("T_MIN: %d", T_MIN));
  main_displayln_info("FIELD_OUT_ENSEMBLE_PATH: " + FIELD_OUT_ENSEMBLE_PATH);
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("TYPE: %d", TYPE));

  // check complecity
  std::vector<int> traj_list;
  for (int i = 990; i < 2000; i += 20) {
    std::string point_file = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/results=%d/checkpoint/computeContractionInf", i);
    std::string wall_file = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/48I/wall-src/results/48I-0.00078/results=%d/checkpoint.txt", i);
    if (!does_file_exist_sync_node(point_file) or !does_file_exist_sync_node(wall_file)) {continue;}

    traj_list.push_back(i);
    main_displayln_info(ssprintf("Valid Configuration: %d", i));
  }
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string EXACT_WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src-exact-2/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/" + ENSEMBLE + ssprintf("/wall-src/results/32D-0.00107/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/" + ENSEMBLE + ssprintf("/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("EXACT_WALL_SRC_PATH: " + EXACT_WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    const std::string FIELD_OUT_TRAJ_PATH = FIELD_OUT_ENSEMBLE_PATH + "/" + ssprintf("results=%04d", traj_list[i]);
    qmkdir_sync_node(FIELD_OUT_TRAJ_PATH);
    compute_point_point_wall_correlator_ama_in_one_traj(T_MIN, WALL_SRC_PATH, EXACT_WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_TRAJ_PATH, PION, TYPE);

    // avg and remove
    const std::string PGGE_FIELD_PATH = FIELD_OUT_ENSEMBLE_PATH + ssprintf("/results=%04d/ama ; t-min=%04d", traj_list[i], T_MIN);
    avg_pionggelemfield_without_accuracy_and_rm(PGGE_FIELD_PATH, TYPE);
  }
}

void compute_wall_wall_corr_24D()
{
  // prepare traj_list
  std::vector<int> traj_list = {1010, 1030, 1050, 1070, 1090, 1110, 1140, 1160, 1180, 1220, 1240, 1260, 1280, 1300, 1320, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020, 2040, 2060, 2080, 2120, 2140, 2160, 2180, 2200, 2220, 2240, 2260, 2280};
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  const int TOTAL_T = 64;
  main_displayln_info(ssprintf("TOTAL_T: %d", TOTAL_T));

  std::vector<Complex> wall_wall_corr_avg(TOTAL_T / 2);
  set_zero(wall_wall_corr_avg);
  main_displayln_info("Compute Wall to Wall Corr for All Configurations Start");
  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Wall to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);

    std::vector<Complex> wall_wall_corr;
    wall_wall_corr = compute_wall_wall_correlation_function(WALL_SRC_PATH, TOTAL_T);
    main_displayln_info(ssprintf("wall_wall_corr[traj=%d] result:", traj_list[i]));
    main_displayln_info(show_vec_complex(wall_wall_corr));

    wall_wall_corr_avg = wall_wall_corr_avg + wall_wall_corr;
  }
  main_displayln_info("Compute Wall to Wall Corr for All Configurations End");
  wall_wall_corr_avg = wall_wall_corr_avg / traj_list.size();
  main_displayln_info("wall_wall_corr_avg result:");
  main_displayln_info(show_vec_complex(wall_wall_corr_avg));

  // std::vector<Complex> zw = compute_zw_from_wall_wall_corr(wall_wall_corr_avg, PION);
  // main_displayln_info("zw result:");
  // main_displayln_info(show_vec_complex(zw));

}

void compute_model()
{
  const int TSEP = 2000;
  double PION = 0.13975;
  const Coordinate TOTAL_SITE = Coordinate(64, 64, 64, 64);
  const Geometry geo(TOTAL_SITE, 1);
  PionGGElemField three_point_correlator_labeled_xp;
  three_point_correlator_labeled_xp.init(geo);
  set_zero(three_point_correlator_labeled_xp.field);

  compute_three_point_corr_model(three_point_correlator_labeled_xp, TSEP, PION);

  double XXP_LIMIT = 10;
  double MUON = 0.1056583745 / AINV;
  std::string pair_info_file = "/home/tucheng/qcdlib-python/hlbl-pi0.out/";
  // pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096";
  pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TSEP: %d", TSEP));
  main_displayln_info(ssprintf("PION INFO FILE: ") + pair_info_file);

  init_muon_line();
  std::vector<Y_And_Rotation_Info_Elem> pairs_info_all = read_y_and_rotation_info(pair_info_file);
  int i_pairs_start = 8000;
  int num_pairs = 1024;
  std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + num_pairs);

  const std::string F2_FOLDER = "f2";
  qmkdir_sync_node(F2_FOLDER);
  qmkdir_sync_node(F2_FOLDER + "/model");

  std::string MOD;
  std::string one_pair_save_folder;
  // MOD = ""
  MOD = "";
  one_pair_save_folder = F2_FOLDER + "/model" + ssprintf("/pion=%f;t-sep=%d;xxp-limit=%d;mod=%s", PION, TSEP, int(XXP_LIMIT), MOD.c_str());
  qmkdir_sync_node(one_pair_save_folder);
  avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, three_point_correlator_labeled_xp, three_point_correlator_labeled_xp, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD);

  // MOD = "xyp>=xy"
  MOD = "xyp>=xy";
  one_pair_save_folder = F2_FOLDER + "/model" + ssprintf("/pion=%f;t-sep=%d;xxp-limit=%d;mod=%s", PION, TSEP, int(XXP_LIMIT), MOD.c_str());
  qmkdir_sync_node(one_pair_save_folder);
  avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, three_point_correlator_labeled_xp, three_point_correlator_labeled_xp, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD);

  // MOD = "xy>=xyp"
  MOD = "xy>=xyp";
  one_pair_save_folder = F2_FOLDER + "/model" + ssprintf("/pion=%f;t-sep=%d;xxp-limit=%d;mod=%s", PION, TSEP, int(XXP_LIMIT), MOD.c_str());
  qmkdir_sync_node(one_pair_save_folder);
  avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, three_point_correlator_labeled_xp, three_point_correlator_labeled_xp, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD);
}

void compute_wall_wall_correlator_all_traj_32D_sloppy()
{
  // prepare traj_list
  std::vector<int> traj_list;
  for (int traj = 680; traj < 1371; traj += 10) {
    traj_list.push_back(traj);
    std::string wall_file = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/32D-0.00107/results=%d/checkpoint.txt", traj);
    if (!does_file_exist_sync_node(wall_file)) {continue;}
  }
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  const int TOTAL_T = 64;
  const std::string field_out_ensemble_path = "WallWallCorr/32D-0.00107/sloppy";
  main_displayln_info(ssprintf("TOTAL_T: %d", TOTAL_T));
  main_displayln_info("WallWallCorr Ensemble Path: " + field_out_ensemble_path);

  std::vector<Complex> wall_wall_corr_avg(TOTAL_T / 2);
  set_zero(wall_wall_corr_avg);
  main_displayln_info("Compute Wall to Wall Corr for All Configurations Start");
  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Wall to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);

    std::vector<Complex> wall_wall_corr;
    wall_wall_corr = compute_wall_wall_correlation_function(WALL_SRC_PATH, TOTAL_T);
    main_displayln_info(ssprintf("wall_wall_corr[traj=%d] result:", traj_list[i]));
    main_displayln_info(show_vec_complex(wall_wall_corr));

#if 0
    //save wall wall corr as Field
    WallWallField wwfield(wall_wall_corr);
    const Coordinate new_geom(1, 1, 1, 8);
    const std::string field_out_traj_path = field_out_ensemble_path + ssprintf("/results=%04d", i);
    main_displayln_info("WallWallCorr Ensemble Path: " + field_out_traj_path);
    qmkdir_sync_node(field_out_traj_path);
    dist_write_field(wwfield, new_geom, field_out_traj_path);
    sync_node();

    // check
    WallWallField read_wwfield;
    dist_read_field(read_wwfield, field_out_traj_path);
    std::vector<Complex> vec = read_wwfield.save_to_vec();
    main_displayln_info(ssprintf("check traj=%d:", traj_list[i]));
    main_displayln_info(show_vec_complex(vec));
#endif

    wall_wall_corr_avg = wall_wall_corr_avg + wall_wall_corr;
  }
  main_displayln_info("Compute Wall to Wall Corr for All Configurations End");
  wall_wall_corr_avg = wall_wall_corr_avg / traj_list.size();
  main_displayln_info("wall_wall_corr_avg result:");
  main_displayln_info(show_vec_complex(wall_wall_corr_avg));
}

void test()
{
  const std::string ENSEMBLE = "24D";
  const std::string THREE_POINT_ENSEMBLE_PATH = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + "/sloppy";
  const int TYPE = 0;
  const int ACCURACY = 0;
  const int TMIN = 10;
  std::vector<int> traj_list = {1800};
  for (int i = 0; i < traj_list.size(); ++i)
  {
    // read pgge
    const std::string PGGE_FIELD_PATH = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i], TMIN);
    avg_pionggelemfield_with_accuracy_and_rm(PGGE_FIELD_PATH, TYPE, ACCURACY);
  }
}

int main(int argc, char* argv[])
{
  begin(&argc, &argv);
  initialize();

  // compute_wall_wall_correlator_all_traj_32D_sloppy();

  // compute_model();
  // compute_point_point_wall_correlator_ama_24D_all_traj();
  // compute_f2_24D_ama_all_traj();
  // compute_point_point_wall_correlator_sloppy_24D_all_traj();
  // compute_f2_24D_sloppy_all_traj();

  // compute_point_point_wall_correlator_sloppy_32D_all_traj();
  compute_point_point_wall_correlator_ama_32D_all_traj();
  // int XXP_LIMIT = 10;
  // int TMIN = 10;
  // std::string MOD = "";
  // compute_f2_32D_ama_all_traj(XXP_LIMIT, TMIN, MOD);
  // MOD = "xyp>=xy";
  // XXP_LIMIT = 14;
  // compute_f2_32D_ama_all_traj(XXP_LIMIT, TMIN, MOD);
  // MOD = "";
  // XXP_LIMIT = 16;
  // compute_f2_32D_ama_all_traj(XXP_LIMIT, TMIN, MOD);
  
  // compute_point_point_wall_correlator_sloppy_32D_all_traj();
  // compute_f2_32D_sloppy_all_traj();

#if 0 // model
  const int TSEP = 1000;
  double PION = 0.13975;
  const Coordinate TOTAL_SITE = Coordinate(64, 64, 64, 64);
  const Geometry geo(TOTAL_SITE, 1);
  PionGGElemField three_point_correlator_labeled_xp;
  three_point_correlator_labeled_xp.init(geo);
  set_zero(three_point_correlator_labeled_xp.field);

  compute_three_point_corr_model(three_point_correlator_labeled_xp, TSEP, PION);

  double XXP_LIMIT = 20;
  double MUON = 0.1056583745 / AINV;
  std::string pair_info_file = "/home/tucheng/qcdlib-python/hlbl-pi0.out/";
  // pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096";
  pair_info_file += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TSEP: %d", TSEP));
  main_displayln_info(ssprintf("PION INFO FILE: ") + pair_info_file);

  init_muon_line();
  std::vector<Y_And_Rotation_Info_Elem> pairs_info_all = read_y_and_rotation_info(pair_info_file);
  int i_pairs_start = 8000;
  int num_pairs = 4096;
  std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + num_pairs - 1);

  const std::string F2_FOLDER = "f2";
  qmkdir_sync_node(F2_FOLDER);
  qmkdir_sync_node(F2_FOLDER + "/model");
  const std::string one_pair_save_folder = F2_FOLDER + "/model" + ssprintf("/pion=%f;t-sep=%d;xxp-limit=%d", PION, TSEP, int(XXP_LIMIT));
  qmkdir_sync_node(one_pair_save_folder);
  avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, three_point_correlator_labeled_xp, three_point_correlator_labeled_xp, TSEP, XXP_LIMIT, MUON, PION, one_pair_save_folder);
#endif

#if 0 // compute point-point-wall correlator all traj
  // prepare traj_list
  std::vector<int> traj_list = {1030};

  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  const int T_SEP = 10;
  const std::string FIELD_OUT_PATH = "ThreePointCorrField/24D";
  const int TYPE = 0;
  const int ACCURACY= 0;

  qmkdir_sync_node("ThreePointCorrField");
  qmkdir_sync_node(FIELD_OUT_PATH);

  main_displayln_info(ssprintf("T_SEP: %d", T_SEP));
  main_displayln_info("FIELD_OUT_PATH: " + FIELD_OUT_PATH);
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("ACCURACY: %d", ACCURACY));

  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = ssprintf("/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    // pair_wall_src_prop_and_point_src_prop(T_SEP, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_PATH, TYPE, ACCURACY);
    compute_point_point_wall_correlator_in_one_traj(T_SEP, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_PATH, TYPE, ACCURACY);
  }
#endif

#if 0 // compute point-point-wall correlator all traj 32D
  // prepare traj_list
  std::vector<int> traj_list = {1050};

  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  const int T_SEP = 20;
  const std::string FIELD_OUT_PATH = "PionGGElemField/32D-0.00107";
  const int TYPE = 0;
  const int ACCURACY= 0;

  qmkdir_sync_node("PionGGElemField");
  qmkdir_sync_node(FIELD_OUT_PATH);

  main_displayln_info(ssprintf("T_SEP: %d", T_SEP));
  main_displayln_info("FIELD_OUT_PATH: " + FIELD_OUT_PATH);
  main_displayln_info(ssprintf("TYPE: %d", TYPE));
  main_displayln_info(ssprintf("ACCURACY: %d", ACCURACY));

  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Point Point to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    const std::string GAUGE_TRANSFORM_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/32D-0.00107/results=%d/huge-data/gauge-transform", traj_list[i]);
    const std::string POINT_SRC_PATH = ssprintf("/home/ljin/application/Public/Muon-GM2-cc/jobs/32D/discon-1/results/prop-hvp ; results=%d/huge-data/prop-point-src", traj_list[i]);

    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);
    main_displayln_info("GAUGE_TRANSFORM_PATH: " + GAUGE_TRANSFORM_PATH);
    main_displayln_info("POINT_SRC_PATH: " + POINT_SRC_PATH);

    pair_wall_src_prop_and_point_src_prop(T_SEP, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_PATH, TYPE, ACCURACY);
  }
#endif

#if 0 // compute wall-wall correlator all traj 32D
  // prepare traj_list
  std::vector<int> traj_list = {1050, 1060, 1070, 1080, 1090};
  main_displayln_info(ssprintf("Num of Configurations: %d", traj_list.size()));

  const int TOTAL_T = 64;
  main_displayln_info(ssprintf("TOTAL_T: %d", TOTAL_T));

  std::vector<Complex> wall_wall_corr_avg(TOTAL_T / 2);
  set_zero(wall_wall_corr_avg);
  main_displayln_info("Compute Wall to Wall Corr for All Configurations Start");
  for (int i = 0; i < traj_list.size(); ++i)
  {
    main_displayln_info(ssprintf("Compute Wall to Wall Corr [traj=%d]", traj_list[i]));
    const std::string WALL_SRC_PATH = ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/32D/wall-src/results/32D-0.00107/results=%d/huge-data/wall_src_propagator", traj_list[i]);
    main_displayln_info("WALL_SRC_PATH: " + WALL_SRC_PATH);

    std::vector<Complex> wall_wall_corr;
    wall_wall_corr = compute_wall_wall_correlation_function(WALL_SRC_PATH, TOTAL_T);
    main_displayln_info(ssprintf("wall_wall_corr[traj=%d] result:", traj_list[i]));
    main_displayln_info(show_vec_complex(wall_wall_corr));

    wall_wall_corr_avg = wall_wall_corr_avg + wall_wall_corr;
  }
  main_displayln_info("Compute Wall to Wall Corr for All Configurations End");
  wall_wall_corr_avg = wall_wall_corr_avg / traj_list.size();
  main_displayln_info("wall_wall_corr_avg result:");
  main_displayln_info(show_vec_complex(wall_wall_corr_avg));

#endif

#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975;
  double XXP_LIMIT = 15;
  const Coordinate TOTAL_SITE = Coordinate(64, 64, 64, 64);
  std::string f_two_configs = "/home/tucheng/qcdlib/hlbl-pi0.out/";
  f_two_configs += "distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:1024,r_pion_to_gamma:30";
  //f_two_configs += "distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs=1024";
  //f_two_configs += "distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512,r_pion_to_gamma:30";

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TOTAL_SITE in small area:"));
  main_displayln_info(show_coordinate(TOTAL_SITE));
  main_displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  avg_f2_model_table(f_two_configs, XXP_LIMIT, MUON, PION, TOTAL_SITE);
#endif

#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975;
  double XXP_LIMIT = 10;
  const Coordinate TOTAL_SITE = Coordinate(24, 24, 24, 64);
  std::string f_two_configs = "/home/tucheng/qcdlib/hlbl-pi0.out/";
  f_two_configs += "24D_config=2280_config=2240_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512,r_pion_to_gamma:20";

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TOTAL_SITE in small area:"));
  main_displayln_info(show_coordinate(TOTAL_SITE));
  main_displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  avg_f2_model_table(f_two_configs, XXP_LIMIT, MUON, PION, TOTAL_SITE);
  sum_f2_nofac_xb_lat_bm_model_table(f_two_configs, XXP_LIMIT, MUON, PION);
  sum_f2_nofac_xb_model_bm_lat_table(f_two_configs, XXP_LIMIT, MUON, PION);
  sum_f2_nofac_table(f_two_configs, XXP_LIMIT, MUON, PION);
#endif

#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975;
  const int TSEP = 30;
  double XXP_LIMIT = 15;
  const Coordinate TOTAL_SITE = Coordinate(64, 64, 64, 64);

  std::string f_two_configs = "/home/tucheng/qcdlib/hlbl-pi0.out/";
  f_two_configs += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512";

  main_displayln_info(ssprintf("MUON: %f", MUON));
  main_displayln_info(ssprintf("PION: %f", PION));
  main_displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  main_displayln_info(ssprintf("TOTAL_SITE in small area:"));
  main_displayln_info(show_coordinate(TOTAL_SITE));
  main_displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  //avg_f2_rotation_model_table(f_two_configs, TSEP, XXP_LIMIT, MUON, PION, TOTAL_SITE);
  avg_f2_model_table_from_y_and_rotation_info(f_two_configs, TSEP, XXP_LIMIT, MUON, PION, TOTAL_SITE);
#endif

  return 0;
}

