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
const int NUM_RMIN = 20;

#if 0
template<int N, int M, int P>
void mult_matrix(double C[N][P], const double A[N][M], const double B[M][P]) {
  static_assert(N > 1, "N must be greater than 1");
  static_assert(M > 1, "M must be greater than 1");
  static_assert(P > 1, "P must be greater than 1");

  for (int n = 0; n < N; n++) {
    for (int p = 0; p < P; p++) {
      double num = 0;
      for (int m = 0; m < M; m++) {
        num += A[n][m] * B[m][p];
      }
      C[n][p] = num;
    }
  }
}
#endif

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

#if 0
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
#endif

#if 0
inline int epsilon_tensor(const int a, const int b, const int c, const int d)
{
  static EpsilonTensorTable table;
  return table.tensor[a][b][c][d];
}
#endif

#if 0
inline int epsilon_tensor(const int i, const int j, const int k)
{
  return epsilon_tensor(i, j, k, 3);
}
#endif

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

#if 1
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
#endif

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

// check
double r_coor(const Coordinate& coor)
{
  return sqrt(sqr((long)coor[0]) + sqr((long)coor[1]) + sqr((long)coor[2]) + sqr((long)coor[3]));
#if 0
  double r = 0;
  for (int i = 0; i < 4; ++i) {
    r += pow(coor[i], 2.);
  }
  return pow(r, 1. / 2.);
#endif
}

// check
double r_coorD(const CoordinateD& coor)
{
  return sqrt(sqr((double)coor[0]) + sqr((double)coor[1]) + sqr((double)coor[2]) + sqr((double)coor[3]));
#if 0
  double r = 0;
  for (int i = 0; i < 4; ++i) {
    r += pow(coor[i], 2.);
  }
  return pow(r, 1. / 2.);
#endif
}

// check
template <class T>
inline void set_zero(T& x)
{
  memset(&x, 0, sizeof(T));
}

// check
double pion_prop(const qlat::Coordinate& x, const qlat::Coordinate& y, const double& m)
// x, y and m must be in lattice unit
{
  Coordinate dist = x - y;
  double s = pow(pow(dist[0], 2.) + pow(dist[1], 2.) + pow(dist[2], 2.) + pow(dist[3], 2.), 1./2.);
  double sm = s * m;
  return m * gsl_sf_bessel_K1(sm) / (4. * pow(PI, 2.) * s);
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

const Propagator4d& get_point_prop(const std::string& path, const Coordinate& c)
{
  return get_prop(path + "/xg=" + show_coordinate(c) + " ; type=0 ; accuracy=0");
}

// pion g g
// check
struct PionGGElem
{
  Complex v[4][4]; // v[mu][nu]

#if 0
  Complex* data()
  {
    return &(v[0][0]);
  }

  PionGGElem& operator+=(PionGGElem x)
  {
#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
      *(this -> data() + i) += *(x.data() + i);
    }
    return *this;
  }

  PionGGElem& operator*=(const double& x)
  {
#pragma omp parallel for
    for (int i = 0; i < 16; i++) {
      *(this -> data() + i) *= x;
    }
    return *this;
  }
#endif

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

struct PionGGElemField : FieldM<PionGGElem,1>
{
  virtual const std::string& cname()
  {
    static const std::string s = "PionGGElem";
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

// check
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

#if 0
// check
PionGGElem operator*(const double& x, PionGGElem& p)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    *(res.data() + i) = x * *(p.data() + i);
  }
  return res;
}
#endif

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

// check
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

#if 0 
// check
PionGGElem operator*(PionGGElem& p, const double& x)
{
  PionGGElem res;
  set_zero(res);
#pragma omp parallel for
  for (int i = 0; i < 16; i++) {
    *(res.data() + i) = x * *(p.data() + i);
  }
  return res;
}
#endif

// check
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

// check
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

// check
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

// check
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

// check
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

// check
bool is_under_limit(const Coordinate& x, const Coordinate& y, const double& r)
{
  Coordinate dist = x - y;
  double xy = pow(pow(dist[0], 2.) + pow(dist[1], 2.) + pow(dist[2], 2.) + pow(dist[3], 2.), 1./2.);
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

// check
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

// check
void find_bm_table(BM_TABLE& bm_table, const Propagator4d& propy, const Propagator4d& propzp, const Coordinate& y, const Coordinate& zp, const Coordinate& y_large, const double& muon)
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_table");
  const Geometry& geo = propy.geo;
  qassert(geo == geo_reform(geo));
  qassert(geo == propzp.geo);
  const Coordinate total_site = geo.total_site();
  const Coordinate ly = geo.coordinate_l_from_g(y);
  const Coordinate lzp = geo.coordinate_l_from_g(zp);

  set_zero(bm_table);

  WilsonMatrix wm_yzp; // from zp to y
  if (geo.is_local(ly)) {
    wm_yzp = propzp.get_elem(ly);
  } else {
    set_zero(wm_yzp);
  }
  glb_sum_double(wm_yzp);

  WilsonMatrix wm_zpy; // from y to zp
  if (geo.is_local(lzp)) {
    wm_zpy = propy.get_elem(lzp);
  } else {
    set_zero(wm_zpy);
  }
  glb_sum_double(wm_zpy);
  displayln_info(ssprintf("norm ratio = %24.17E (should be very small)", norm(gamma5 * matrix_adjoint(wm_yzp) * gamma5 - wm_zpy) / norm(wm_zpy)));
  
  double r_y_large = r_coor(y_large);
  const CoordinateD muon_y_large = muon*CoordinateD(y_large);
  double r_muon_y_large = muon * r_y_large;

  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate lyp = geo.coordinate_from_index(index);
    const Coordinate yp = geo.coordinate_g_from_l(lyp);

    // find coor info
    const Coordinate yp_y = relative_coordinate(yp-y, total_site);
    const Coordinate yp_large = y_large + yp_y;
    const CoordinateD muon_yp_large = muon * CoordinateD(yp_large);
    double r_yp_large = r_coor(yp_large);
    double r_muon_yp_large = muon * r_yp_large;
    double r_yp_y = r_coor(yp_y);
    double r_muon_yp_y = muon * r_yp_y;

    // get muon line
    if (r_muon_y_large > 6 || r_muon_yp_large > 6 || r_muon_yp_y > 6) {continue; }
    const ManyMagneticMoments mmm = get_muon_line_m_extra(muon_y_large, muon_yp_large, CoorD_0, 0);

    int r_max = int(ceil(std::max(r_y_large, r_yp_large)));
    r_max = int(ceil(std::max(double(r_max), r_yp_y)));
    int r_min = int(ceil(std::min(r_y_large, r_yp_large)));
    r_min = int(ceil(std::min(double(r_min), r_yp_y)));
    if (r_max >= NUM_RMAX || r_min >= NUM_RMIN) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];
    const WilsonMatrix& wm_ypy = propy.get_elem(lyp);
    const WilsonMatrix& wm_ypzp = propzp.get_elem(lyp);
    const WilsonMatrix wm_yyp = gamma5 * matrix_adjoint(wm_ypy) * gamma5;
    const WilsonMatrix wm_zpyp = gamma5 * matrix_adjoint(wm_ypzp) * gamma5;

    PionGGElem pgge;
    set_zero(pgge);
    three_prop_contraction(pgge, wm_zpy, wm_ypzp, wm_yyp);
    three_prop_contraction_(pgge, wm_zpyp, wm_yzp, wm_ypy);

    bm += pgge * mmm;
  }
  glb_sum_double(bm_table);
  // partialsum_bmtable(bm_table);
  return;
}

double f_r(double r, double fpi, double mv)
{
    double fte;
    double fvmd;

    fte  = 3. * pow(mv, 4.) * pow(r, 2.) * gsl_sf_bessel_Kn(2, mv * r) / (16. * pow(fpi, 2.) * pow(PI, 2.));
    fvmd = 3. * pow(mv, 5.) * pow(r, 3.) * gsl_sf_bessel_K1(   mv * r) / (32. * pow(fpi, 2.) * pow(PI, 2.));

    return 8. * pow(PI, 2.) * pow(fpi, 2.) / (3. * pow(mv, 2.)) * fte + (1. - 8. * pow(PI, 2.) * pow(fpi, 2.) / (3. * pow(mv, 2.))) * fvmd;
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
    pgge = - ii * fpi * pow(pion, 2.) / (12. * pow(PI, 4.) * pow(r_x_xp, 4.) * pow(r_xp_x_mid_z, 2.)) * f_r(r_x_xp, fpi, mv) * gsl_sf_bessel_Kn(2, xz_mass) * pgge;
    return;
}

void find_bm_model_table(BM_TABLE& bm_table, const Coordinate& y, const Coordinate& zp, const double& muon, const double& pion, const Coordinate& total_site)
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_model_table");
  set_zero(bm_table);
  Geometry geo = Geometry(total_site, 1);
  
  const CoordinateD muon_y = muon * CoordinateD(y);
  const double r_y = r_coor(y);
  const double r_muon_y = muon * r_y;

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lyp = geo.coordinate_from_index(index);
    const Coordinate yp = geo.coordinate_g_from_l(lyp);

    // find coor info
    const CoordinateD muon_yp = muon * CoordinateD(yp);
    const double r_yp = r_coor(yp);
    const double r_muon_yp = muon * r_yp;
    const Coordinate yp_y = relative_coordinate(yp-y, total_site);
    const double r_yp_y = r_coor(yp_y);
    const double r_muon_yp_y = muon * r_yp_y;

    // get muon line
    if (r_muon_y > 6 || r_muon_yp > 6 || r_muon_yp_y > 6) {continue; }
    const ManyMagneticMoments mmm = get_muon_line_m_extra(muon_y, muon_yp, CoorD_0, 0);

    int r_max = int(ceil(std::max(r_y, r_yp)));
    r_max = int(ceil(std::max(double(r_max), r_yp_y)));
    int r_min = int(ceil(std::min(r_y, r_yp)));
    r_min = int(ceil(std::min(double(r_min), r_yp_y)));
    if (r_max >= NUM_RMAX || r_min >= NUM_RMIN) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];

    const CoordinateD yp_y_mid_zp = relative_coordinate(middle_coordinate(CoordinateD(y), CoordinateD(yp), CoordinateD(total_site)) - CoordinateD(zp), CoordinateD(total_site));
    PionGGElem pgge;
    set_zero(pgge);
    b_model(pgge, CoordinateD(-yp_y), yp_y_mid_zp, pion);

    bm += pgge * mmm;
  }
  glb_sum_double(bm_table);
  // partialsum_bmtable(bm_table);
  return;
}

void find_bm_model_table(BM_TABLE& bm_table, const Coordinate& y, const Coordinate& zp, const Coordinate& y_large, const double& muon, const double& pion, const Coordinate& total_site)
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_model_table");
  set_zero(bm_table);
  Geometry geo = Geometry(total_site, 1);
  
  double r_y_large = r_coor(y_large);
  const CoordinateD muon_y_large = muon*CoordinateD(y_large);
  double r_muon_y_large = muon * r_y_large;

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lyp = geo.coordinate_from_index(index);
    const Coordinate yp = geo.coordinate_g_from_l(lyp);

    // find coor info
    const Coordinate yp_y = relative_coordinate(yp-y, total_site);
    const Coordinate yp_large = y_large + yp_y;
    const CoordinateD muon_yp_large = muon * CoordinateD(yp_large);
    double r_yp_large = r_coor(yp_large);
    double r_muon_yp_large = muon * r_yp_large;
    double r_yp_y = r_coor(yp_y);
    double r_muon_yp_y = muon * r_yp_y;

    // get muon line
    if (r_muon_y_large > 6 || r_muon_yp_large > 6 || r_muon_yp_y > 6) {continue; }
    const ManyMagneticMoments mmm = get_muon_line_m_extra(muon_y_large, muon_yp_large, CoorD_0, 0);

    int r_max = int(ceil(std::max(r_y_large, r_yp_large)));
    r_max = int(ceil(std::max(double(r_max), r_yp_y)));
    int r_min = int(ceil(std::min(r_y_large, r_yp_large)));
    r_min = int(ceil(std::min(double(r_min), r_yp_y)));
    if (r_max >= NUM_RMAX || r_min >= NUM_RMIN) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];

    const CoordinateD yp_y_mid_zp = relative_coordinate(middle_coordinate(CoordinateD(y), CoordinateD(yp), CoordinateD(total_site)) - CoordinateD(zp), CoordinateD(total_site));
    PionGGElem pgge;
    set_zero(pgge);
    b_model(pgge, CoordinateD(-yp_y), yp_y_mid_zp, pion);

    bm += pgge * mmm;

  }
  glb_sum_double(bm_table);
  // partialsum_bmtable(bm_table);
  return;
}

void find_bm_table_rotation_pgge(BM_TABLE& bm_table, const PionGGElemField& pgge_field, const Coordinate& y_large, const int tsep, const RotationMatrix rot, const double& muon, const double& pion)
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_table_rotation_pgge");
  set_zero(bm_table);
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();

  // Coordinate zp = Coordinate(0,0,0,0);
  Coordinate y  = Coordinate(0,0,0,tsep);
  
  double r_y_large = r_coor(y_large);
  const CoordinateD muon_y_large = muon*CoordinateD(y_large);
  double r_muon_y_large = muon * r_y_large;

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lyp = geo.coordinate_from_index(index);
    const Coordinate yp = geo.coordinate_g_from_l(lyp);

    // find coor info
    CoordinateD yp_y = relative_coordinate(yp-y, total_site);
    CoordinateD yp_y_rot = rot * yp_y;
    const CoordinateD yp_large = CoordinateD(y_large) + yp_y_rot;
    const CoordinateD muon_yp_large = muon * CoordinateD(yp_large);
    double r_yp_large = r_coorD(yp_large);
    double r_muon_yp_large = muon * r_yp_large;
    double r_yp_y = r_coorD(yp_y_rot);
    double r_muon_yp_y = muon * r_yp_y;

    // get muon line
    if (r_muon_y_large > 6 || r_muon_yp_large > 6 || r_muon_yp_y > 6) {continue; }
    const ManyMagneticMoments mmm = get_muon_line_m_extra(muon_y_large, muon_yp_large, CoorD_0, 0);

    int r_max = int(ceil(std::max(r_y_large, r_yp_large)));
    r_max = int(ceil(std::max(double(r_max), r_yp_y)));
    int r_min = int(ceil(std::min(r_y_large, r_yp_large)));
    r_min = int(ceil(std::min(double(r_min), r_yp_y)));
    if (r_max >= NUM_RMAX || r_min >= NUM_RMIN) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];

    PionGGElem pgge;
    set_zero(pgge);
    pgge = rot * pgge_field.get_elem(lyp);

    bm += pgge * mmm;
  }
  glb_sum_double(bm_table);
  // partialsum_bmtable(bm_table);
  return;
}

void find_bm_rotation_model_table(BM_TABLE& bm_table, const Coordinate& y_large, const int tsep, const RotationMatrix rot, const double& muon, const double& pion, const Coordinate& total_site)
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_rotation_model_table");
  set_zero(bm_table);
  Geometry geo = Geometry(total_site, 1);

  Coordinate zp = Coordinate(0,0,0,0);
  Coordinate y  = Coordinate(0,0,0,tsep);
  
  double r_y_large = r_coor(y_large);
  const CoordinateD muon_y_large = muon*CoordinateD(y_large);
  double r_muon_y_large = muon * r_y_large;

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lyp = geo.coordinate_from_index(index);
    const Coordinate yp = geo.coordinate_g_from_l(lyp);

    // find coor info
    CoordinateD yp_y = relative_coordinate(yp-y, total_site);
    CoordinateD yp_y_rot = rot * yp_y;
    const CoordinateD yp_large = CoordinateD(y_large) + yp_y_rot;
    const CoordinateD muon_yp_large = muon * CoordinateD(yp_large);
    double r_yp_large = r_coorD(yp_large);
    double r_muon_yp_large = muon * r_yp_large;
    double r_yp_y = r_coorD(yp_y_rot);
    double r_muon_yp_y = muon * r_yp_y;

    // get muon line
    if (r_muon_y_large > 6 || r_muon_yp_large > 6 || r_muon_yp_y > 6) {continue; }
    const ManyMagneticMoments mmm = get_muon_line_m_extra(muon_y_large, muon_yp_large, CoorD_0, 0);

    int r_max = int(ceil(std::max(r_y_large, r_yp_large)));
    r_max = int(ceil(std::max(double(r_max), r_yp_y)));
    int r_min = int(ceil(std::min(r_y_large, r_yp_large)));
    r_min = int(ceil(std::min(double(r_min), r_yp_y)));
    if (r_max >= NUM_RMAX || r_min >= NUM_RMIN) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];

    const CoordinateD yp_y_mid_zp = relative_coordinate(middle_coordinate(CoordinateD(y), CoordinateD(yp), CoordinateD(total_site)) - CoordinateD(zp), CoordinateD(total_site));
    PionGGElem pgge;
    set_zero(pgge);
    b_model(pgge, CoordinateD(-yp_y), yp_y_mid_zp, pion);
    pgge = rot * pgge;

    bm += pgge * mmm;

  }
  glb_sum_double(bm_table);
  // partialsum_bmtable(bm_table);
  return;
}

// need check
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

// need check
void find_xb(XB& xb, const Propagator4d& propx, const Propagator4d& propz, const Coordinate& x, const Coordinate& z, const double& xxp_limit)
{
  TIMER_VERBOSE("find_xb");
  const Geometry& geo = propx.geo;
  qassert(geo == geo_reform(geo));
  qassert(geo == propz.geo);
  const Coordinate total_site = geo.total_site();
  const Coordinate lx = geo.coordinate_l_from_g(x);
  const Coordinate lz = geo.coordinate_l_from_g(z);

  WilsonMatrix wm_xz; // from z to x
  if (geo.is_local(lx)) {
    wm_xz = propz.get_elem(lx);
  } else {
    set_zero(wm_xz);
  }
  glb_sum_double(wm_xz);

  WilsonMatrix wm_zx; // from x to z
  if (geo.is_local(lz)) {
    wm_zx = propx.get_elem(lz);
  } else {
    set_zero(wm_zx);
  }
  glb_sum_double(wm_zx);
  displayln_info(ssprintf("norm ratio = %24.17E (should be very small)", norm(gamma5 * matrix_adjoint(wm_xz) * gamma5 - wm_zx) / norm(wm_zx)));
  
  const Coordinate x_z = relative_coordinate(x-z, total_site);
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const Coordinate xp_x = relative_coordinate(xp-x, total_site);
    if (r_coor(xp_x) > xxp_limit or r_coor(xp_x) < pow(10., -5)) {continue; }

    const WilsonMatrix& wm_xpx = propx.get_elem(lxp);
    const WilsonMatrix& wm_xpz = propz.get_elem(lxp);
    const WilsonMatrix wm_xxp = gamma5 * matrix_adjoint(wm_xpx) * gamma5;
    const WilsonMatrix wm_zxp = gamma5 * matrix_adjoint(wm_xpz) * gamma5;

    PionGGElem pgge;
    set_zero(pgge);
    three_prop_contraction(pgge, wm_zx, wm_xpz, wm_xxp);
    three_prop_contraction_(pgge, wm_zxp, wm_xz, wm_xpx);
    for (int j = 0; j < 3; j++) {
      xb.xB[j] += xp_x[j] * pgge;
    }
  }
  glb_sum_double(xb);
  return;
}

void find_xb_model(XB& xb, const Coordinate& x, const Coordinate& z, const double& xxp_limit, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("find_xb_model");
  // const Coordinate total_site = geo.total_site();
  Geometry geo = Geometry(total_site, 1);

  for (long index = 0; index < geo.local_volume(); ++index) 
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const Coordinate xp_x = relative_coordinate(xp-x, total_site);
    const CoordinateD xp_x_mid_z = relative_coordinate(middle_coordinate(x, xp, CoordinateD(total_site)) - CoordinateD(z), CoordinateD(total_site));

    if (r_coor(xp_x) > xxp_limit or r_coor(xp_x) < pow(10., -5)) {continue; }

    PionGGElem pgge;
    set_zero(pgge);
    b_model(pgge, CoordinateD(-xp_x), xp_x_mid_z, pion);
    for (int j = 0; j < 3; j++) {
      xb.xB[j] += xp_x[j] * pgge;
    }
  }
  glb_sum_double(xb);
  return;
}

void find_xb_rotation_pgge(XB& xb, const PionGGElemField& pgge_field,const int tsep, const RotationMatrix rot, const double& xxp_limit, const double& pion)
{
  TIMER_VERBOSE("find_xb_rotation_pgge");
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();

  Coordinate x = Coordinate(0,0,0,0);
  // Coordinate z = Coordinate(0,0,0,tsep);

  for (long index = 0; index < geo.local_volume(); ++index) 
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const Coordinate xp_x = relative_coordinate(xp-x, total_site);

    if (r_coor(xp_x) > xxp_limit or r_coor(xp_x) < pow(10., -5)) {continue; }

    PionGGElem pgge;
    set_zero(pgge);
    pgge = rot * pgge_field.get_elem(lxp);
    CoordinateD xp_x_rot = rot * xp_x;
    for (int j = 0; j < 3; j++) {
      xb.xB[j] += xp_x_rot[j] * pgge;
    }
  }
  glb_sum_double(xb);
  return;
}

void find_xb_rotation_model(XB& xb, const int tsep, const RotationMatrix rot, const double& xxp_limit, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("find_xb_rotation_model");
  // const Coordinate total_site = geo.total_site();
  Geometry geo = Geometry(total_site, 1);

  Coordinate x = Coordinate(0,0,0,0);
  Coordinate z = Coordinate(0,0,0,tsep);

  for (long index = 0; index < geo.local_volume(); ++index) 
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    const Coordinate xp_x = relative_coordinate(xp-x, total_site);
    const CoordinateD xp_x_mid_z = relative_coordinate(middle_coordinate(x, xp, CoordinateD(total_site)) - CoordinateD(z), CoordinateD(total_site));

    if (r_coor(xp_x) > xxp_limit or r_coor(xp_x) < pow(10., -5)) {continue; }

    PionGGElem pgge;
    set_zero(pgge);
    b_model(pgge, CoordinateD(-xp_x), xp_x_mid_z, pion);
    pgge = rot * pgge;
    CoordinateD xp_x_rot = rot * xp_x;
    for (int j = 0; j < 3; j++) {
      xb.xB[j] += xp_x_rot[j] * pgge;
    }
  }
  glb_sum_double(xb);
  return;
}

// need check
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

struct Complex_Table
{
#if 0
  std::vector< std::vector<Complex> > c;
  long NUM_RMAX;
  long NUM_RMIN;

  Complex_Table()
  {
    c.resize(NUM_RMAX);
    for (long i = 0; i < c.size(); ++i)
    {
      c[i].resize(NUM_RMIN);
    }
  }

  Complex_Table(long NUM_RMAX_, long NUM_RMIN_)
  {
    c.resize(NUM_RMAX_);
    for (long i = 0; i < NUM_RMAX_; ++i)
    {
      c[i].resize(NUM_RMIN_);
    }
    NUM_RMAX = NUM_RMAX_
    NUM_RMIN = NUM_RMIN_
  }

  void set_zeros()
  {
    for (long i = 0; i < c.size(); ++i)
    {
      set_zero(c[i]);
    }
  }
#else
  Complex c[NUM_RMAX][NUM_RMIN];
#endif

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
  std::string res = "Complex_Table:\n";
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

// need check
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
void test_PionGGElem()
{
  PionGGElem p0;
  set_zero(p0);
  for (int mu = 0; mu < 4; ++mu ) {
    for (int nu = 0; nu < 4; ++nu) {
      p0.v[mu][nu] = Complex(1, 1);
    }
  }

  PionGGElem p1;
  for (int mu = 0; mu < 4; ++mu ) {
    for (int nu = 0; nu < 4; ++nu) {
      p1.v[mu][nu] = Complex(4 * mu + nu, -(4 * mu + nu));
    }
  }

  displayln_info("show p0 p1");
  displayln_info(show_pgge(p0));
  displayln_info(show_pgge(p1));

  displayln_info("show p1 += p0");
  p1 += p0;
  displayln_info(show_pgge(p0));
  displayln_info(show_pgge(p1));

  displayln_info("show p0 *= 2.7");
  p0 *= 2.7;
  displayln_info(show_pgge(p0));

  displayln_info("p3 = p0 * 2.6");
  PionGGElem p3;
  p3 = p0 * 2.6;
  displayln_info(show_pgge(p3));
  displayln_info("p3 = 2.4 * p0");
  p3 = 2.4 * p0;
  displayln_info(show_pgge(p3));
  displayln_info("p3 = 2.4 * p0");

  displayln_info("p3 = p0 + p1");
  p3 = p0 + p1;
  displayln_info(show_pgge(p3));
  displayln_info("p3 = p0 + p1");

  return;
}

struct Two_Configs_Info_Elem
{
  Coordinate Y_R;
  double dist;
  std::string X;
  std::string Z;
  std::string ZP;
  std::string Y;
};

struct Four_Points_Info_Elem
{
  double dist;
  Coordinate y_large;
  Coordinate x;
  Coordinate z;
  Coordinate zp;
  Coordinate y;
};

struct Y_And_Rotation_Info_Elem
{
  double dist;
  Coordinate y_large;
  double theta_xy;
  double theta_xt;
  double theta_zt;
};

std::string show_two_configs_info_elem(const Two_Configs_Info_Elem& elem)
{
  std::string info = "Two_Configs_Info_Elem:\n";
  info += "Y_R:\n";
  info += show_coordinate(elem.Y_R);
  info += "\n";
  info += ssprintf("dist: %24.17E\n", elem.dist);
  info += "X: " + elem.X + "\n";
  info += "Z: " + elem.Z + "\n";
  info += "ZP: " + elem.ZP + "\n";
  info += "Y: " + elem.Y + "\n";
  return info;
}

std::string show_four_points_info_elem(const Four_Points_Info_Elem& elem)
{
  std::string info = "Four_Points_Info_Elem:\n";
  info += ssprintf("dist: %24.17E\n", elem.dist);
  info += "y_large:\n";
  info += show_coordinate(elem.y_large);
  info += "\n";
  info += "x:\n";
  info += show_coordinate(elem.x);
  info += "\n";
  info += "z:\n";
  info += show_coordinate(elem.z);
  info += "\n";
  info += "zp:\n";
  info += show_coordinate(elem.zp);
  info += "\n";
  info += "y:\n";
  info += show_coordinate(elem.y);
  info += "\n";
  return info;
}

std::string show_y_and_rotation_info_elem(const Y_And_Rotation_Info_Elem& elem)
{
  std::string info = "Y_And_Rotation_Info_Elem:\n";
  info += "y_large:\n";
  info += show_coordinate(elem.y_large);
  info += "\n";
  info += ssprintf("dist: %24.17E\n", elem.dist);
  info += ssprintf("theta_xy: %24.17E\n", elem.theta_xy);
  info += ssprintf("theta_xt: %24.17E\n", elem.theta_xt);
  info += ssprintf("theta_zt: %24.17E\n", elem.theta_zt);
  return info;
}

std::vector<Two_Configs_Info_Elem> read_two_configs_info(std::string file)
{
  std::vector<Two_Configs_Info_Elem> info_list;
  std::ifstream fileopen(file);
  std::string line;
  long linenum = 0;
  while (getline(fileopen, line)) {
    if (linenum % 6 == 0) {
      Two_Configs_Info_Elem elem;
      info_list.push_back(elem);
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).Y_R = my_read_coordinate(coor);
    } else if (linenum % 6 == 1) {
      (info_list.back()).dist = std::stod(line);
    } else if (linenum % 6 == 2) {
      (info_list.back()).X = line;
    } else if (linenum % 6 == 3) {
      (info_list.back()).Z = line;
    } else if (linenum % 6 == 4) {
      (info_list.back()).ZP = line;
    } else if (linenum % 6 == 5) {
      (info_list.back()).Y = line;
      displayln_info(show_two_configs_info_elem(info_list.back()));
    }
    linenum++;
  }
  displayln_info(ssprintf("%d Pairs Have Been Read From Two Configs.", info_list.size()));
  return info_list;
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
      (info_list.back()).y_large = my_read_coordinate(coor);
    } else if (linenum % 5 == 1) {
      (info_list.back()).dist = std::stod(line);
    } else if (linenum % 5 == 2) {
      (info_list.back()).theta_xy = std::stod(line);
    } else if (linenum % 5 == 3) {
      (info_list.back()).theta_xt = std::stod(line);
    } else if (linenum % 5 == 4) {
      (info_list.back()).theta_zt = std::stod(line);
      displayln_info(show_y_and_rotation_info_elem(info_list.back()));
    }
    linenum++;
  }
  displayln_info(ssprintf("%d Pairs Have Been Read From Two Configs.", info_list.size()));
  return info_list;
}

std::vector<Four_Points_Info_Elem> read_four_points_info(std::string file)
{
  TIMER_VERBOSE("read_four_points_info");
  std::vector<Four_Points_Info_Elem> info_list;
  std::ifstream fileopen(file);
  std::string line;
  long linenum = 0;
  while (getline(fileopen, line)) {
    if (linenum % 6 == 0) {
      Four_Points_Info_Elem elem;
      info_list.push_back(elem);
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).y_large = my_read_coordinate(coor);
    } else if (linenum % 6 == 1) {
      (info_list.back()).dist = std::stod(line);
    } else if (linenum % 6 == 2) {
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).x = my_read_coordinate(coor);
    } else if (linenum % 6 == 3) {
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).z = my_read_coordinate(coor);
    } else if (linenum % 6 == 4) {
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).zp = my_read_coordinate(coor);
    } else if (linenum % 6 == 5) {
      std::string coor = string_split(line, "Coordinate").back();
      (info_list.back()).y = my_read_coordinate(coor);
    }
    linenum++;
  }
  displayln_info(ssprintf("%d Pairs Have Been Read From Two Configs.", info_list.size()));
  return info_list;
}

void find_onepair_f2_nofac_table(Complex_Table& f2_nofac_table, const Two_Configs_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("find_onepair_f2_nofac_table");
  // x
  const Coordinate x = get_xg_from_path(onepair.X);
  const Propagator4d& propx = get_prop(onepair.X);
  // z
  const Coordinate z = get_xg_from_path(onepair.Z);
  const Propagator4d& propz = get_prop(onepair.Z);
  // zp
  const Coordinate zp = get_xg_from_path(onepair.ZP);
  const Propagator4d& propzp = get_prop(onepair.ZP);
  // y
  const Coordinate y = get_xg_from_path(onepair.Y);
  const Propagator4d& propy = get_prop(onepair.Y);
  // y_large
  const Coordinate& y_large = onepair.Y_R;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb(xb, propx, propz, x, z, xb_limit);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_table(bm_table, propy, propzp, y, zp, y_large, muon);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  const Geometry& geo = propy.geo;
  qassert(geo == geo_reform(geo));
  const Coordinate total_site = geo.total_site();
  Coordinate x_z = relative_coordinate(x-z, total_site);
  Coordinate zp_y = relative_coordinate(zp-y, total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_two_configs_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_nofac_table += e_xbbm_table;

  return;
}

void find_onepair_f2_nofac_xb_model_bm_lat_table(Complex_Table& f2_nofac_table, const Two_Configs_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("find_onepair_f2_nofac_xb_model_bm_lat_table");
  // x
  const Coordinate x = get_xg_from_path(onepair.X);
  const Propagator4d& propx = get_prop(onepair.X);
  // z
  const Coordinate z = get_xg_from_path(onepair.Z);
  const Propagator4d& propz = get_prop(onepair.Z);
  // zp
  const Coordinate zp = get_xg_from_path(onepair.ZP);
  const Propagator4d& propzp = get_prop(onepair.ZP);
  // y
  const Coordinate y = get_xg_from_path(onepair.Y);
  const Propagator4d& propy = get_prop(onepair.Y);
  // y_large
  const Coordinate& y_large = onepair.Y_R;

  const Geometry& geo = propy.geo;
  qassert(geo == geo_reform(geo));
  const Coordinate total_site = geo.total_site();

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_model(xb, x, z, xb_limit, pion, total_site);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_table(bm_table, propy, propzp, y, zp, y_large, muon);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(x-z, total_site);
  Coordinate zp_y = relative_coordinate(zp-y, total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_two_configs_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_nofac_table += e_xbbm_table;

  return;
}

void find_onepair_f2_nofac_xb_lat_bm_model_table(Complex_Table& f2_nofac_table, const Two_Configs_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("find_onepair_f2_nofac_xb_lat_bm_model_table");
  // x
  const Coordinate x = get_xg_from_path(onepair.X);
  const Propagator4d& propx = get_prop(onepair.X);
  // z
  const Coordinate z = get_xg_from_path(onepair.Z);
  const Propagator4d& propz = get_prop(onepair.Z);
  // zp
  const Coordinate zp = get_xg_from_path(onepair.ZP);
  const Propagator4d& propzp = get_prop(onepair.ZP);
  // y
  const Coordinate y = get_xg_from_path(onepair.Y);
  const Propagator4d& propy = get_prop(onepair.Y);
  // y_large
  const Coordinate& y_large = onepair.Y_R;

  const Geometry& geo = propy.geo;
  qassert(geo == geo_reform(geo));
  const Coordinate total_site = geo.total_site();

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb(xb, propx, propz, x, z, xb_limit);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_model_table(bm_table, y, zp, y_large, muon, pion, total_site);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(x-z, total_site);
  Coordinate zp_y = relative_coordinate(zp-y, total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_two_configs_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_nofac_table += e_xbbm_table;

  return;
}

void find_onepair_f2_model_table(Complex_Table& f2_model_table, const Four_Points_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("find_onepair_f2_model_table");
  // y_large
  const Coordinate y_large = onepair.y_large;
  // x
  const Coordinate x = onepair.x;
  // z
  const Coordinate z = onepair.z;
  // zp
  const Coordinate zp = onepair.zp;
  // y
  const Coordinate y = onepair.y;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_model(xb, x, z, xb_limit, pion, total_site);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_model_table(bm_table, y, zp, y_large, muon, pion, total_site);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(x-z, total_site);
  Coordinate zp_y = relative_coordinate(zp-y, total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_four_points_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_model_table += e_xbbm_table;

  return;
}

void find_onepair_f2_model_table(Complex_Table& f2_model_table, const Two_Configs_Info_Elem& onepair, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("find_onepair_f2_model_table");
  // x
  const Coordinate x = get_xg_from_path(onepair.X);
  // z
  const Coordinate z = get_xg_from_path(onepair.Z);
  // zp
  const Coordinate zp = get_xg_from_path(onepair.ZP);
  // y
  const Coordinate y = get_xg_from_path(onepair.Y);
  // y_large
  const Coordinate& y_large = onepair.Y_R;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_model(xb, x, z, xb_limit, pion, total_site);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_model_table(bm_table, y, zp, y_large, muon, pion, total_site);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(x-z, total_site);
  Coordinate zp_y = relative_coordinate(zp-y, total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_two_configs_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_model_table += e_xbbm_table;

  return;
}

void from_y_large_find_z(CoordinateD& z, const Coordinate& y_large, const double& r_z_x)
{
  double fact = r_z_x / r_coor(y_large);
#pragma omp parallel for
  for (int i = 0; i < 4; ++i){
    z[i] = y_large[i] * fact;
  }
}

void from_y_large_find_z(Coordinate& z, const Coordinate& y_large, const double& r_z_x)
{
  double fact = r_z_x / r_coor(y_large);
#pragma omp parallel for
  for (int i = 0; i < 4; ++i){
    z[i] = int(std::round(y_large[i] * fact));
  }
}

void find_onepair_f2_model_table(Complex_Table& f2_model_table, const Coordinate& y_large, const double& dist, const double& r_z_x, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("find_onepair_f2_model_table");
  // x
  const Coordinate x = Coordinate(0,0,0,0);
  // z
  Coordinate z;
  from_y_large_find_z(z, y_large, r_z_x);
  // zp
  const Coordinate zp = Coordinate(0,0,0,0);
  // y
  const Coordinate y = z;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_model(xb, x, z, xb_limit, pion, total_site);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_model_table(bm_table, y, zp, y_large, muon, pion, total_site);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(x-z, total_site);
  Coordinate zp_y = relative_coordinate(zp-y, total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_model_table += e_xbbm_table;

  return;
}

void find_onepair_f2_table_rotation_pgge(Complex_Table& f2_model_table, const PionGGElemField& pgge_field, const Y_And_Rotation_Info_Elem& onepair, const int& tsep, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("find_onepair_f2_rotation_model_table");
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();

  const double theta_xy = onepair.theta_xy;
  const double theta_xt = onepair.theta_xt;
  const double theta_zt = onepair.theta_zt;
  const RotationMatrix rotation_matrix = RotationMatrix(theta_xy, theta_xt, theta_zt);
  // y_large
  const Coordinate& y_large = onepair.y_large;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_rotation_pgge(xb, pgge_field, tsep, rotation_matrix, xb_limit, pion);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_table_rotation_pgge(bm_table, pgge_field, y_large, tsep, rotation_matrix, muon, pion);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(Coordinate(0.,0.,0.,-tsep), total_site);
  Coordinate zp_y = relative_coordinate(Coordinate(0.,0.,0.,-tsep), total_site);
  // Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));
  Complex prop = pion_prop(y_large, Coor_0, pion) / (1. / (2. * pion) * exp(-pion * tsep) * 1. / (2. * pion) * exp(-pion * tsep));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_y_and_rotation_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_model_table += e_xbbm_table;

  return;
}

void find_onepair_f2_rotation_model_table(Complex_Table& f2_model_table, const Y_And_Rotation_Info_Elem& onepair, const int& tsep, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("find_onepair_f2_rotation_model_table");
  const double theta_xy = onepair.theta_xy;
  const double theta_xt = onepair.theta_xt;
  const double theta_zt = onepair.theta_zt;
  const RotationMatrix rotation_matrix = RotationMatrix(theta_xy, theta_xt, theta_zt);
  // y_large
  const Coordinate& y_large = onepair.y_large;

  // xb
  XB xb;
  set_zero(xb);
  double xb_limit = xxp_limit;
  find_xb_rotation_model(xb, tsep, rotation_matrix, xb_limit, pion, total_site);

  // bm_table
  BM_TABLE bm_table;
  set_zero(bm_table);
  find_bm_rotation_model_table(bm_table, y_large, tsep, rotation_matrix, muon, pion, total_site);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  set_zero(e_xbbm_table);
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Coordinate x_z = relative_coordinate(Coordinate(0.,0.,0.,-tsep), total_site);
  Coordinate zp_y = relative_coordinate(Coordinate(0.,0.,0.,-tsep), total_site);
  Complex prop = pion_prop(y_large, Coor_0, pion) / (pion_prop(x_z, Coor_0, pion) * pion_prop(zp_y, Coor_0, pion));

  e_xbbm_table = (prop / onepair.dist) * e_xbbm_table;
  // show one pair e_xbbm_table
#if 1
  std::string info = "";
  info += show_y_and_rotation_info_elem(onepair);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  displayln_info(info);
#endif

  f2_model_table += e_xbbm_table;

  return;
}

Complex_Table sum_f2_nofac_table(std::string f_pairs, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("sum_f2_nofac_table");
  std::vector<Two_Configs_Info_Elem> pairs_info = read_two_configs_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_nofac_table;
  set_zero(f2_nofac_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Two_Configs_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_nofac_table(f2_nofac_table, one_pair, xxp_limit, muon, pion);
  }
  f2_nofac_table = 1. / num_pairs * f2_nofac_table;
  displayln_info("Final F2 Nofac:");
  displayln_info(show_complex_table(f2_nofac_table));
  return f2_nofac_table;
}

Complex_Table sum_f2_nofac_xb_lat_bm_model_table(std::string f_pairs, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("sum_f2_nofac_xb_lat_bm_model_table");
  std::vector<Two_Configs_Info_Elem> pairs_info = read_two_configs_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_nofac_table;
  set_zero(f2_nofac_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Two_Configs_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_nofac_xb_lat_bm_model_table(f2_nofac_table, one_pair, xxp_limit, muon, pion);
  }
  f2_nofac_table = 1. / num_pairs * f2_nofac_table;
  displayln_info("Final F2 Nofac:");
  displayln_info(show_complex_table(f2_nofac_table));
  return f2_nofac_table;
}

Complex_Table sum_f2_nofac_xb_model_bm_lat_table(std::string f_pairs, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("sum_f2_nofac_xb_model_bm_lat_table");
  std::vector<Two_Configs_Info_Elem> pairs_info = read_two_configs_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_nofac_table;
  set_zero(f2_nofac_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Two_Configs_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_nofac_xb_model_bm_lat_table(f2_nofac_table, one_pair, xxp_limit, muon, pion);
  }
  f2_nofac_table = 1. / num_pairs * f2_nofac_table;
  displayln_info("Final F2 Nofac:");
  displayln_info(show_complex_table(f2_nofac_table));
  return f2_nofac_table;
}

#if 1
Complex_Table avg_f2_model_table(std::string f_pairs, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("avg_f2_model_table");
  std::vector<Four_Points_Info_Elem> pairs_info = read_four_points_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_model_table;
  set_zero(f2_model_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Four_Points_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_model_table(f2_model_table, one_pair, xxp_limit, muon, pion, total_site);
  }
  f2_model_table = 1. / num_pairs * f2_model_table;
  displayln_info("Final F2 Model Nofac:");
  displayln_info(show_complex_table(f2_model_table));
  return f2_model_table;
}
#else

Complex_Table avg_f2_model_table(std::string f_pairs, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("avg_f2_model_table");
  std::vector<Two_Configs_Info_Elem> pairs_info = read_two_configs_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_model_table;
  set_zero(f2_model_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Two_Configs_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_model_table(f2_model_table, one_pair, xxp_limit, muon, pion, total_site);
  }
  f2_model_table = 1. / num_pairs * f2_model_table;
  displayln_info("Final F2 Model Nofac:");
  displayln_info(show_complex_table(f2_model_table));
  return f2_model_table;
}
#endif

Complex_Table avg_f2_model_table_from_y_and_rotation_info(std::string f_pairs, const double& r_z_x, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("avg_f2_model_table_from_y_and_rotation_info");
  std::vector<Y_And_Rotation_Info_Elem> pairs_info = read_y_and_rotation_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_model_table;
  set_zero(f2_model_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Y_And_Rotation_Info_Elem& one_pair = pairs_info[i];
    const Coordinate y_large = one_pair.y_large;
    const double dist = one_pair.dist;
    find_onepair_f2_model_table(f2_model_table, y_large, dist, r_z_x, xxp_limit, muon, pion, total_site);
  }
  f2_model_table = 1. / num_pairs * f2_model_table;
  displayln_info("Final F2 Model Nofac:");
  displayln_info(show_complex_table(f2_model_table));
  return f2_model_table;
}

Complex_Table avg_f2_table_from_wall_src_from_y_and_rotation_info(std::string f_pairs, const PionGGElemField& pgge_field, const int tsep, const double& xxp_limit, const double& muon, const double& pion)
{
  TIMER_VERBOSE("avg_f2_table_from_wall_src_from_y_and_rotation_info");
  std::vector<Y_And_Rotation_Info_Elem> pairs_info = read_y_and_rotation_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_model_table;
  set_zero(f2_model_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Y_And_Rotation_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_table_rotation_pgge(f2_model_table, pgge_field, one_pair, tsep, xxp_limit, muon, pion);
  }
  f2_model_table = 1. / num_pairs * f2_model_table;
  displayln_info("Final F2 Model Nofac:");
  displayln_info(show_complex_table(f2_model_table));
  return f2_model_table;
}

Complex_Table avg_f2_rotation_model_table(std::string f_pairs, const int& tsep, const double& xxp_limit, const double& muon, const double& pion, const Coordinate& total_site)
{
  TIMER_VERBOSE("avg_f2_rotation_model_table");
  std::vector<Y_And_Rotation_Info_Elem> pairs_info = read_y_and_rotation_info(f_pairs);
  long num_pairs = pairs_info.size();
  Complex_Table f2_model_table;
  set_zero(f2_model_table);
  for (int i = 0; i < num_pairs; ++i)
  {
    displayln_info(ssprintf("Begin pair %d", i));
    Y_And_Rotation_Info_Elem& one_pair = pairs_info[i];
    find_onepair_f2_rotation_model_table(f2_model_table, one_pair, tsep, xxp_limit, muon, pion, total_site);
  }
  f2_model_table = 1. / num_pairs * f2_model_table;
  displayln_info("Final F2 Model Nofac:");
  displayln_info(show_complex_table(f2_model_table));
  return f2_model_table;
}

std::vector<std::string> list_files_under_path(const std::string& path_, const int root = 0)
{
  displayln_info("List Files and Dirs Under: " + path_);
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
    res.push_back(name);
    displayln_info(name);
  }
  closedir (dir);
  return res;
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
}

void pair_wall_src_prop_and_point_src_prop(const int t_sep, const std::string& wall_src_path, const std::string& gauge_transform_path, const std::string& point_src_path, const std::string& field_out_path, int type_, int accuracy_)
{
  TIMER_VERBOSE("pair_wall_src_prop_and_point_src_prop");
  GaugeTransform gtinv;
  {
    GaugeTransform gt;
    dist_read_field(gt, gauge_transform_path);
    to_from_big_endian_64(get_data(gt));
    gt_inverse(gtinv, gt);
  }

  int traj = read_traj(point_src_path);
  qassert(opendir(&field_out_path[0]) != NULL);
  qmkdir_sync_node(field_out_path + "/" + ssprintf("results=%04d", traj));
  qmkdir_sync_node(field_out_path + "/" + ssprintf("results=%04d/t-sep=%04d", traj, t_sep));

  int count = 0;

  const std::vector<std::string> point_src_prop_list = list_files_under_path(point_src_path);
  for (int i =0; i < point_src_prop_list.size(); ++i){
    if (type_ != read_type(point_src_prop_list[i]) or accuracy_ != read_accuracy(point_src_prop_list[i])) {continue;}

    std::string full_path = field_out_path + "/" + ssprintf("results=%04d/t-sep=%04d", traj, t_sep) + "/" + point_src_prop_list[i];

    // load point src prop
    displayln_info("point_src_prop path: " + point_src_path + "/" + point_src_prop_list[i]);
    const Coordinate point_src_coor = get_xg_from_path(point_src_prop_list[i]);
    const Propagator4d& point_src_prop = get_prop(point_src_path + "/" + point_src_prop_list[i]);
    const Geometry& point_src_geo = point_src_prop.geo;
    const Coordinate total_site = point_src_geo.total_site();

    // setup PionGGElemField
    const Geometry& geo = point_src_prop.geo;
    PionGGElemField three_point_correlator_labeled_xp;
    three_point_correlator_labeled_xp.init(geo);
    set_zero(three_point_correlator_labeled_xp.field);

    for (int i = -1; i < 2; i += 2){  // t_sep is + and -
      // load wall src prop
      int t_point = point_src_coor[3];
      int t_wall = mod(t_point + i * t_sep, total_site[3]);
      const std::string wall_src_t_path = wall_src_path + "/t=" + ssprintf("%d", t_wall);
      displayln_info("wall_src_prop path: " + wall_src_t_path);

      // gauge inv and shift
      Propagator4d point_src_prop_shift;
      Propagator4d wall_src_prop_shift;

      {
        const Propagator4d& wall_src_prop = get_prop(wall_src_t_path);
        qassert(point_src_geo == wall_src_prop.geo);

        Propagator4d wall_src_prop_gauge;
        prop_apply_gauge_transformation(wall_src_prop_gauge, wall_src_prop, gtinv);

        field_shift(point_src_prop_shift, point_src_prop, -point_src_coor);
        field_shift(wall_src_prop_shift, wall_src_prop_gauge, -point_src_coor);
      }

      compute_three_point_correlator_from_wall_src_prop(three_point_correlator_labeled_xp, point_src_prop_shift, wall_src_prop_shift);
      sync_node();
    }
    three_point_correlator_labeled_xp /= 2.;
    sync_node();

    const Coordinate new_geom(1, 1, 1, 8);
    qmkdir_sync_node(full_path);
    dist_write_field(three_point_correlator_labeled_xp, new_geom, full_path);
    sync_node();

    count += 1;
    displayln_info(ssprintf("Save PionGGElem Field to [%04d]: ", count) + full_path);
  }
}

void read_pionggelemfield_and_avg(PionGGElemField& pgge_field_avg, const std::string& field_path, int type_, int accuracy_)
{
  const std::vector<std::string> field_list = list_files_under_path(field_path);
  qassert(field_list.size() >= 1);

  // read the first one
  dist_read_field(pgge_field_avg, field_path + "/" + field_list[0]);
  displayln_info("Read PionGGElemField from: " + field_path + "/" + field_list[0]);
  int num = 1;
  // test
  displayln_info("Testme::show first pgge");
  displayln_info(show_pgge(pgge_field_avg.get_elem(0)));
  displayln_info(show_pgge(pgge_field_avg.get_elem(10)));

  for (int i = 1; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    if (type_ != read_type(one_field_path) or accuracy_ != read_accuracy(one_field_path)) {continue;}
    PionGGElemField pgge_field;
    dist_read_field(pgge_field, one_field_path);
    displayln_info("Read PionGGElemField from: " + one_field_path);
    pgge_field_avg += pgge_field;
    num += 1;
  }
  pgge_field_avg /= double(num);

  // test
  displayln_info("test::show_pgge_avg");
  displayln_info(show_pgge(pgge_field_avg.get_elem(0)));
  displayln_info(show_pgge(pgge_field_avg.get_elem(10)));
}

void test()
{
#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975 / AINV;
  const int TSEP = 20;
  double XXP_LIMIT = 15;

  std::string f_two_configs = "/projects/HadronicLight_4/ctu/qcdlib/hlbl-pi0.out/";
  f_two_configs += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:4096";

  displayln_info(ssprintf("MUON: %f", MUON));
  displayln_info(ssprintf("PION: %f", PION));
  displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  const std::string PGGE_FIELD_PATH = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/PionGGElemField/24D/results=1030/t-sep=0020";
  const int TYPE = 0;
  const int ACCURACY = 0;
  PionGGElemField pgge_field;
  read_pionggelemfield_and_avg(pgge_field, PGGE_FIELD_PATH, TYPE, ACCURACY);

  avg_f2_table_from_wall_src_from_y_and_rotation_info(f_two_configs, pgge_field, TSEP, XXP_LIMIT, MUON, PION);
  exit(0);
#endif

#if 0
  const std::string PGGE_FIELD_PATH = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/PionGGElemField/24D/results=1030/t-sep=0020";
  const int TYPE = 0;
  const int ACCURACY= 0;
  PionGGElemField pgge_field;
  read_pionggelemfield_and_avg(pgge_field, PGGE_FIELD_PATH, TYPE, ACCURACY);
  exit(0);
#endif

#if 1
  const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=1030/huge-data/wall_src_propagator";
  const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/25D/wall-src/results/results=1030/huge-data/gauge-transform";
  const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp\ \;\ results\=1030/huge-data/prop-point-src";
  const int T_SEP = 20;
  const std::string FIELD_OUT_PATH = "PionGGElemField/24D";

  qmkdir_sync_node("PionGGElemField");
  qmkdir_sync_node(FIELD_OUT_PATH);

  const int TYPE = 0;
  const int ACCURACY= 0;

  pair_wall_src_prop_and_point_src_prop(T_SEP, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_PATH, TYPE, ACCURACY);
  exit(0);
#endif

#if 0
  list_files_under_path("/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp\ \;\ results\=1030/huge-data/prop-point-src/");
  qmkdir_sync_node("PionGGElemField");

  // pair_wall_src_prop_and_point_src_prop(30, Coordinate(24,24,24,64), "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=1030/huge-data/wall_src_propagator/", "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=1030/huge-data/gauge-transform", "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp\ \;\ results\=1030/huge-data/prop-point-src/", 15);

  const Coordinate total_site = Coordinate(80, 80, 80, 80);
  double pion = 0.13975 / AINV;
  
  CoordinateD a = CoordinateD(0,0,0,30);
  RotationMatrix rot = RotationMatrix(1.31013244102, 2.05142463828, -1.10266444292);
  CoordinateD a_rot = rot * a;
  displayln_info(show_coordinateD(a_rot));

  Coordinate x = Coordinate(0,0,0,0);
  CoordinateD xp = rot * Coordinate(1,3,2,0);
  CoordinateD z = rot * Coordinate(0,0,0,30);
  CoordinateD xp_x = relative_coordinate(xp-x, total_site);
  CoordinateD xp_x_mid_z = relative_coordinate(middle_coordinate(x, xp, CoordinateD(total_site)) - CoordinateD(z), CoordinateD(total_site));
  PionGGElem pgge;
  set_zero(pgge);
  b_model(pgge, CoordinateD(-xp_x), xp_x_mid_z, pion);
  displayln_info(show_pgge(pgge));

  x = Coordinate(0,0,0,0);
  xp = Coordinate(1,3,2,0);
  z = Coordinate(0,0,0,30);
  xp_x = relative_coordinate(xp-x, total_site);
  xp_x_mid_z = relative_coordinate(middle_coordinate(x, xp, CoordinateD(total_site)) - CoordinateD(z), CoordinateD(total_site));
  set_zero(pgge);
  b_model(pgge, CoordinateD(-xp_x), xp_x_mid_z, pion);
  pgge = rot * pgge;
  displayln_info(show_pgge(pgge));
#endif
}

int main(int argc, char* argv[])
{
  begin(&argc, &argv);
  initialize();

  test();

#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975 / AINV;
  double XXP_LIMIT = 15;
  const Coordinate TOTAL_SITE = Coordinate(64, 64, 64, 64);
  std::string f_two_configs = "/home/tucheng/qcdlib/hlbl-pi0.out/";
  f_two_configs += "distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:1024,r_pion_to_gamma:30";
  //f_two_configs += "distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs=1024";
  //f_two_configs += "distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512,r_pion_to_gamma:30";

  displayln_info(ssprintf("MUON: %f", MUON));
  displayln_info(ssprintf("PION: %f", PION));
  displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  displayln_info(ssprintf("TOTAL_SITE in small area:"));
  displayln_info(show_coordinate(TOTAL_SITE));
  displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  avg_f2_model_table(f_two_configs, XXP_LIMIT, MUON, PION, TOTAL_SITE);
#endif

#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975 / AINV;
  double XXP_LIMIT = 10;
  const Coordinate TOTAL_SITE = Coordinate(24, 24, 24, 64);
  std::string f_two_configs = "/home/tucheng/qcdlib/hlbl-pi0.out/";
  f_two_configs += "24D_config=2280_config=2240_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512,r_pion_to_gamma:20";

  displayln_info(ssprintf("MUON: %f", MUON));
  displayln_info(ssprintf("PION: %f", PION));
  displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  displayln_info(ssprintf("TOTAL_SITE in small area:"));
  displayln_info(show_coordinate(TOTAL_SITE));
  displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  avg_f2_model_table(f_two_configs, XXP_LIMIT, MUON, PION, TOTAL_SITE);
  sum_f2_nofac_xb_lat_bm_model_table(f_two_configs, XXP_LIMIT, MUON, PION);
  sum_f2_nofac_xb_model_bm_lat_table(f_two_configs, XXP_LIMIT, MUON, PION);
  sum_f2_nofac_table(f_two_configs, XXP_LIMIT, MUON, PION);
#endif

#if 0
  init_muon_line();
  double MUON = 0.1056583745 / AINV;
  double PION = 0.13975 / AINV;
  const int TSEP = 30;
  double XXP_LIMIT = 15;
  const Coordinate TOTAL_SITE = Coordinate(64, 64, 64, 64);

  std::string f_two_configs = "/home/tucheng/qcdlib/hlbl-pi0.out/";
  f_two_configs += "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:512";

  displayln_info(ssprintf("MUON: %f", MUON));
  displayln_info(ssprintf("PION: %f", PION));
  displayln_info(ssprintf("XXP_LIMIT: %f", XXP_LIMIT));
  displayln_info(ssprintf("TOTAL_SITE in small area:"));
  displayln_info(show_coordinate(TOTAL_SITE));
  displayln_info(ssprintf("Compute Config: ") + f_two_configs);

  //avg_f2_rotation_model_table(f_two_configs, TSEP, XXP_LIMIT, MUON, PION, TOTAL_SITE);
  avg_f2_model_table_from_y_and_rotation_info(f_two_configs, TSEP, XXP_LIMIT, MUON, PION, TOTAL_SITE);
#endif

#if 0
  const std::string WALL_SRC_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=1030/huge-data/wall_src_propagator";
  const std::string GAUGE_TRANSFORM_PATH = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/24D/wall-src/results/results=1030/huge-data/gauge-transform";
  const std::string POINT_SRC_PATH = "/home/ljin/application/Public/Muon-GM2-cc/jobs/24D/discon-1/results/prop-hvp\ \;\ results\=1030/huge-data/prop-point-src";
  const int T_SEP = 30;
  const std::string FIELD_OUT_PATH = "PionGGElemField/24D";

  qmkdir_sync_node("PionGGElemField");
  qmkdir_sync_node(FIELD_OUT_PATH);

  pair_wall_src_prop_and_point_src_prop(T_SEP, WALL_SRC_PATH, GAUGE_TRANSFORM_PATH, POINT_SRC_PATH, FIELD_OUT_PATH);
#endif

  return 0;
}

