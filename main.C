#include <qlat/qlat.h>
#include <qlat/qlat-analysis.h>
#include <qlat/field-utils.h>
#include <gsl/gsl_sf_bessel.h>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <dirent.h>
#include "muon-line.h"
#include "my-utils.h"
#include "f2.h"
#include "wallwall.h"
#include "point-point-wall.h"
#include "piongg.h"

#include <map>
#include <vector>

#define TEST 0

// double AINV = 1.015;
QLAT_START_NAMESPACE

#if 0
const SpinMatrix& gamma5 = SpinMatrixConstants::get_gamma5();
const std::array<SpinMatrix,4>& gammas = SpinMatrixConstants::get_cps_gammas();
const CoordinateD CoorD_0 = CoordinateD(0, 0, 0, 0);
const Coordinate Coor_0 = Coordinate(0, 0, 0, 0);
// const int NUM_RMAX = 80;
// const int NUM_RMIN = 40;
#endif

#if 0
inline std::string showWilsonMatrix(const WilsonMatrix& mat,
                                        bool human_readable = false)
{
  double* p = (double*)&mat;
  const int sizewm = sizeof(WilsonMatrix) / sizeof(WilsonVector);
  const int sizewv = sizeof(WilsonVector) / sizeof(double);
  std::ostringstream out;
  if (human_readable) {
    out.precision(1);
    out << std::fixed;
  } else {
    out.precision(16);
    out << std::scientific;
  }
  for (int i = 0; i < sizewm; i++) {
    out << p[i * sizewv];
    for (int j = 1; j < sizewv; j++) {
      out << " " << p[i * sizewv + j];
    }
    out << std::endl;
  }
  return out.str();
}

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
#endif

#if 0
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

template <class M>
void write_data_from_0_node(const M* data, const int size, std::string path)
{
  sync_node();
  if (0 == get_rank() && 0 == get_thread_num())
  {
    FILE* fp = qopen(path, "a");
    std::fwrite((void*) data, sizeof(M), size, fp);
    qclose(fp);
  }
  sync_node();
}
#endif

#if 0
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
#endif

#if 0
double r_coor(const Coordinate& coor)
{
  return sqrt(sqr((long)coor[0]) + sqr((long)coor[1]) + sqr((long)coor[2]) + sqr((long)coor[3]));
}

double r_coorD(const CoordinateD& coor)
{
  return sqrt(sqr((double)coor[0]) + sqr((double)coor[1]) + sqr((double)coor[2]) + sqr((double)coor[3]));
}
#endif

#if 0
template <class T>
inline void set_zero(T& x)
{
  memset(&x, 0, sizeof(T));
}
#endif

#if 0
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
#endif

#if 0
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

const Propagator4d& get_point_prop(const std::string& path, const Coordinate& c)
{
  return get_prop(path + "/xg=" + show_coordinate(c) + " ; type=0 ; accuracy=0");
}
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
struct PionGG
{
  Coordinate mom;
  std::vector<PionGGElem> table;
};
#endif

#if 0
Complex pi_pi_contraction(const WilsonMatrix& wm_from_1_to_2, const WilsonMatrix& wm_from_2_to_1)
{
  return -matrix_trace((SpinMatrix)(ii * gamma5) * (WilsonMatrix)(wm_from_2_to_1 * (SpinMatrix)(ii * gamma5) * wm_from_1_to_2));
}
#endif

#if 0
void three_prop_contraction(PionGGElem& pgge, const WilsonMatrix& wm_21, const WilsonMatrix& wm_32, const WilsonMatrix& wm_13)
  // 1(mu) --wm_21--> 2(5) --wm_32--> 3(nu) --wm_13--> 1(mu)
  // need additional minus sign from the loop
  // need additional minus sign from the two iis for the current
  // need additional charge factor 1/3 and pion source norm factor 1/sqrt(2)
  // need Z_V^2
{
  const WilsonMatrix wm_a = wm_32 * (SpinMatrix)(ii * gamma5) * wm_21;
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
  const WilsonMatrix wm_a = wm_32 * (SpinMatrix)(ii * gamma5) * wm_21;
  for (int mu = 0; mu < 4; ++mu) {
    const WilsonMatrix wm_b = wm_13 * gammas[mu] * wm_a;
    for (int nu = 0; nu < 4; ++nu) {
      pgge.v[mu][nu] += matrix_trace(gammas[nu] * wm_b);
    }
  }
}
#endif

#if 0
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
#endif

#if 0
struct BM_TABLE
{
  // PionGGElem bm[NUM_RMAX][NUM_RMIN];
  std::vector<std::vector<PionGGElem>> bm;
  BM_TABLE(int r_max, int r_min) {
    bm.resize(r_max);
    for (int row = 0; row < r_max; ++row) {
      bm[row].resize(r_min);
    }
  }

  void set_zero() {
#pragma omp parallel for
    for (int row = 0; row < bm.size(); ++row) {
      for (int col = 0; col < bm[row].size(); ++col) {
        qlat::set_zero(bm[row][col]);
      }
    }
  }

  void glb_sum_double() {
    for (int row = 0; row < bm.size(); ++row) {
      for (int col = 0; col < bm[row].size(); ++col) {
        qlat::glb_sum_double(bm[row][col]);
      }
    }
  }
};

void partialsum_bmtable(BM_TABLE& bm_table)
{
  int num_rmax = bm_table.bm.size();
  int num_rmin = bm_table.bm[0].size();
  for (int r_max = 0; r_max < num_rmax; ++r_max)
  {
    for (int r_min = 0; r_min < num_rmin; ++r_min)
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
      // main)_displayln_info(show_y_and_rotation_info_elem(info_list.back()));
    }
    linenum++;
  }
  return info_list;
}

struct YAndRotationInfo
{
  std::string yinfo_dir = "/home/tucheng/qcdlib-python/hlbl-pi0.out";
  std::string yinfo_f;
  std::vector<Y_And_Rotation_Info_Elem> info_list;

  int start_index;
  int chunk_size = 1024;
  int step = 1024;

  void show_info() const {
    main_displayln_info("YAndRotationInfo:");
    main_displayln_info("yinfo_dir: " + yinfo_dir);
    main_displayln_info("yinfo_f: " + yinfo_f);
    main_displayln_info(ssprintf("info size: %d", info_list.size()));
    main_displayln_info(ssprintf("start_index: %d", start_index));
    main_displayln_info(ssprintf("chunk_size: %d", chunk_size));
    main_displayln_info(ssprintf("step: %d", step));
  }

  YAndRotationInfo(const std::string& ensemble_) {
    const std::string ENSEMBLE = ensemble_;
    if (ENSEMBLE == "24D-0.00107") {
      yinfo_f = "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:500000";
      start_index = 10000;
    } else if (ENSEMBLE == "24D-0.0174") {
      yinfo_f = "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:450000";
      start_index = 10000;
    } else if (ENSEMBLE == "32D-0.00107") {
      yinfo_f = "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:130000";
      start_index = 10000;
    } else if (ENSEMBLE == "32Dfine-0.0001") {
      yinfo_f = "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:400000";
      start_index = 10000;
    } else if (ENSEMBLE == "48I-0.00078") {
      yinfo_f = "y_and_rotation_angles_distr:m=0,a=2,r_left=5,r_mid=40,r_right=60,npairs:550000";
      start_index = 10000;
    } else {
      qassert(false);
    }
    qassert(step >= chunk_size);
    info_list = read_y_and_rotation_info(yinfo_dir + "/" + yinfo_f);
    qassert(start_index < info_list.size());
    show_info();
  }

  std::vector<Y_And_Rotation_Info_Elem> get_next_y_info_list() const {
    static int curr_index = start_index;
    qassert(start_index + chunk_size <= info_list.size());
    main_displayln_info("YAndRotationInfo::get_next_y_info_list():: " + ssprintf("curr_index %d, chunk_size %d, step %d", curr_index, chunk_size, step));
    std::vector<Y_And_Rotation_Info_Elem> info_sub_list(info_list.begin() + curr_index, info_list.begin() + curr_index + chunk_size);
    curr_index += step;
    return info_sub_list;
  }
};
#endif

#if 0
double f_r(double r, double fpi, double mv)
{
    double fte;
    double fvmd;

    fte  = 3. * std::pow(mv, 4.) * std::pow(r, 2.) * gsl_sf_bessel_Kn(2, mv * r) / (16. * std::pow(fpi, 2.) * std::pow(PI, 2.));
    fvmd = 3. * std::pow(mv, 5.) * std::pow(r, 3.) * gsl_sf_bessel_K1(   mv * r) / (32. * std::pow(fpi, 2.) * std::pow(PI, 2.));

    return 8. * std::pow(PI, 2.) * std::pow(fpi, 2.) / (3. * std::pow(mv, 2.)) * fte + (1. - 8. * std::pow(PI, 2.) * std::pow(fpi, 2.) / (3. * std::pow(mv, 2.))) * fvmd;
}
#endif

#if 0
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
#endif

#if 0
void find_bm_table_rotation_pgge(BM_TABLE& bm_table, const PionGGElemField& pgge_field, const CoordinateD& y_large, const RotationMatrix rot, const double& muon, const std::string mod="")
// bm_table is initialized in this func
{
  TIMER_VERBOSE("find_bm_table_rotation_pgge");
  qassert(mod == "xy>=xyp" or mod == "xyp>=xy" or mod == "");
  bm_table.set_zero();
  const Geometry& geo = pgge_field.geo;
  const Coordinate total_site = geo.total_site();
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

    int num_rmax = bm_table.bm.size();
    int num_rmin = bm_table.bm[0].size();
    if (r_max >= num_rmax || r_min >= num_rmin) {continue; }

    PionGGElem& bm = bm_table.bm[r_max][r_min];

    PionGGElem pgge;
    set_zero(pgge);
    pgge = rot * pgge_field.get_elem(lyp);

    if ((mod == "") || ((mod != "") && (std::abs(r_y_large - r_yp_large) < std::pow(10., -10.)))) {
      bm += pgge * mmm;
    } else {
      bm += 2. * (pgge * mmm);
    }
  }
  bm_table.glb_sum_double();
  return;
}

void find_xb_rotation_pgge(XB& xb, const PionGGElemField& pgge_field, const RotationMatrix rot, const int xxp_limit)
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

    if (r_coor(xp_x) > (double) xxp_limit or r_coor(xp_x) < std::pow(10., -5)) {continue; }

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
  e_xbbm_table.set_zero();
  int num_rmax = e_xbbm_table.c.size();
  int num_rmin = e_xbbm_table.c[0].size();
  qassert(num_rmax == bm_table.bm.size() && num_rmax == bm_table.bm[0].size());
#pragma omp parallel for
  for (int rmax = 0; rmax < num_rmax; ++rmax) {
    for (int rmin = 0; rmin < num_rmin; ++rmin) {
      const PionGGElem& bm = bm_table.bm[rmax][rmin];
      Complex& e_xbbm = e_xbbm_table.c[rmax][rmin];
      e_xbbm = find_e_xbbm(xb, bm);
    }
  }
}
#endif

#if 0
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

inline std::string get_point_src_prop_path(const std::string& job_tag, 
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
#endif

#if 0
inline int get_tmin(const std::string& job_tag)
{
  if (job_tag == "24D-0.00107" or job_tag == "32D-0.00107") {
    return 10;
  } else if (job_tag == "24D-0.0174") {
    return 10;
  } else if (job_tag == "32Dfine-0.0001") {
    return 14;
  } else if (job_tag == "48I-0.00078") {
    return 16;
  } else {
    qassert(false);
  }
  return 8;
}

struct EnsembleInfo {
  std::string ENSEMBLE;
  double AINV;
  double MUON;
  double PION;

  int TRAJ_START;
  int TRAJ_END;

  void init(const std::string& ensemble_) {
    ENSEMBLE = ensemble_;
    if (ENSEMBLE == "24D-0.00107") {
      PION = 0.13975;
      AINV = 1.015;

      TRAJ_START = 1000;
      TRAJ_END = 3000;
    } else if (ENSEMBLE == "24D-0.0174") {
      PION = 0.3357;
      AINV = 1.015;

      TRAJ_START = 200;
      TRAJ_END = 1000;
    } else if (ENSEMBLE == "32D-0.00107") {
      PION = 0.139474;
      AINV = 1.015;

      TRAJ_START = 680;
      TRAJ_END = 2000;
    } else if (ENSEMBLE == "32Dfine-0.0001") {
      PION = 0.10468;
      AINV = 1.378;

      TRAJ_START = 200;
      TRAJ_END = 2000;
    } else if (ENSEMBLE == "48I-0.00078") {
      PION = 0.08049;
      AINV = 1.73;

      TRAJ_START = 500;
      TRAJ_END = 3000;
    } else {
      qassert(false);
    }
    MUON = 0.1056583745 / AINV;
  }

  void show_info() const {
    main_displayln_info("EnsembleInfo:");
    main_displayln_info("ENSEMBLE: " + ENSEMBLE);
    main_displayln_info(ssprintf("AINV: %.20f", AINV));
    main_displayln_info(ssprintf("MUON: %.20f", MUON));
    main_displayln_info(ssprintf("PION: %.20f", PION));

    main_displayln_info(ssprintf("TRAJ_START %d, TRAJ_END %d", TRAJ_START, TRAJ_END));
  }

  EnsembleInfo(const std::string& ensemble_) {
    init(ensemble_);
    show_info();
  }
};
#endif

#if 0
struct WallWallEnsembleInfo : public EnsembleInfo {
  std::string ACCURACY;

  std::string CORR_OUT_PATH = "WallWallCorr";
  std::string CORR_ENSEMBLE_ACCURACY_OUT_PATH;

  int NUM_WALL_SLOPPY;
  int NUM_WALL_EXACT;
  double WALL_SLOPPY_EXACT_RATIO;

  void init(const std::string& ensemble_, const std::string& accuracy_) {
    ACCURACY = accuracy_;
    qassert(ACCURACY == "ama" || ACCURACY == "sloppy");
    CORR_ENSEMBLE_ACCURACY_OUT_PATH = CORR_OUT_PATH + "/" + ENSEMBLE + "/" + ACCURACY;

    if (ENSEMBLE == "24D-0.00107") {
      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "24D-0.0174") {
      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "32D-0.00107") {
      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "32Dfine-0.0001") {
      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "48I-0.00078") {
      NUM_WALL_SLOPPY = 96;
      NUM_WALL_EXACT = 2;
    } else {
      qassert(false);
    }
    compute_wall_ratio();
  }

  void compute_wall_ratio() {
    qassert(NUM_WALL_SLOPPY != 0 && NUM_WALL_EXACT != 0);
    const double prob = 1. - std::pow(1. - 1. / (double) NUM_WALL_SLOPPY, (double) NUM_WALL_EXACT);
    WALL_SLOPPY_EXACT_RATIO = 1. / prob;
  }

  void show_info() const {
    main_displayln_info("WallWallEnsembleInfo:");
    main_displayln_info("ACCURACY: " + ACCURACY);

    main_displayln_info("CORR_OUT_PATH: " + CORR_OUT_PATH);
    main_displayln_info("CORR_ENSEMBLE_ACCURACY_OUT_PATH: " + CORR_ENSEMBLE_ACCURACY_OUT_PATH);

    main_displayln_info(ssprintf("NUM_WALL_SLOPPY: %d", NUM_WALL_SLOPPY));
    main_displayln_info(ssprintf("NUM_WALL_EXACT: %d", NUM_WALL_EXACT));
    main_displayln_info(ssprintf("WALL_SLOPPY_EXACT_RATIO: %.20f", WALL_SLOPPY_EXACT_RATIO));
  }

  WallWallEnsembleInfo(const std::string& ensemble_, const std::string& accuracy_) : EnsembleInfo(ensemble_) {
    init(ensemble_, accuracy_);
    show_info();
  }

  std::string get_traj_path(const int traj) {
    return CORR_ENSEMBLE_ACCURACY_OUT_PATH + ssprintf("/results=%04d", traj);
  }

  bool is_traj_computed(const int traj) {
    return does_file_exist_sync_node(get_traj_path(traj));
  }

  void make_corr_ensemble_accuracy_dir() {
    qassert(CORR_OUT_PATH != "" && ENSEMBLE != "" && ACCURACY != "");
    qmkdir_sync_node(CORR_OUT_PATH);
    qmkdir_sync_node(CORR_OUT_PATH + "/" + ENSEMBLE);
    qmkdir_sync_node(CORR_OUT_PATH + "/" + ENSEMBLE + "/" + ACCURACY);
  }
};
#endif

#if 0
struct TwoPointWallEnsembleInfo : public EnsembleInfo{
  std::string ACCURACY;
  int T_MIN;
  int TYPE = 0;

  std::string FIELD_OUT_PATH = "TwoPointWallCorrField";
  std::string FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH;

  int NUM_WALL_SLOPPY;
  int NUM_WALL_EXACT;
  double WALL_SLOPPY_EXACT_RATIO;

  int NUM_POINT_SLOPPY;
  int NUM_POINT_EXACT_1;
  int NUM_POINT_EXACT_2;

  void init(const std::string& ensemble_, const std::string& accuracy_) {
    ACCURACY = accuracy_;
    qassert(ACCURACY == "ama" || ACCURACY == "sloppy");
    T_MIN = get_tmin(ENSEMBLE);
    FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH = FIELD_OUT_PATH + "/" + ENSEMBLE + "/" + ACCURACY + ssprintf("/t-min=%04d", T_MIN);

    if (ENSEMBLE == "24D-0.00107") {
      NUM_POINT_SLOPPY = 1024;
      NUM_POINT_EXACT_1 = 32;
      NUM_POINT_EXACT_2 = 8;

      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "24D-0.0174") {
      NUM_POINT_SLOPPY = 1024;
      NUM_POINT_EXACT_1 = 32;
      NUM_POINT_EXACT_2 = 8;

      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "32D-0.00107") {
      NUM_POINT_SLOPPY = 2048;
      NUM_POINT_EXACT_1 = 64;
      NUM_POINT_EXACT_2 = 16;

      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "32Dfine-0.0001") {
      // not sure
      NUM_POINT_SLOPPY = 1024;
      NUM_POINT_EXACT_1 = 32;
      NUM_POINT_EXACT_2 = 8;

      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "48I-0.00078") {
      // not sure
      NUM_POINT_SLOPPY = 1024;
      NUM_POINT_EXACT_1 = 32;
      NUM_POINT_EXACT_2 = 8;

      NUM_WALL_SLOPPY = 96;
      NUM_WALL_EXACT = 2;
    } else {
      qassert(false);
    }
    compute_wall_ratio();
  }

  void compute_wall_ratio() {
    qassert(NUM_WALL_SLOPPY != 0 && NUM_WALL_EXACT != 0);
    const double prob = 1. - std::pow(1. - 1. / (double) NUM_WALL_SLOPPY, (double) NUM_WALL_EXACT);
    WALL_SLOPPY_EXACT_RATIO = 1. / prob;
  }

  void show_info() const {
    main_displayln_info("TwoPointWallEnsembleInfo:");
    main_displayln_info("ACCURACY: " + ACCURACY);
    main_displayln_info(ssprintf("T_MIN: %d", T_MIN));
    main_displayln_info(ssprintf("TYPE: %d", TYPE));

    main_displayln_info("FIELD_OUT_PATH: " + FIELD_OUT_PATH);
    main_displayln_info("FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH: " + FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH);

    main_displayln_info(ssprintf("NUM_WALL_SLOPPY: %d", NUM_WALL_SLOPPY));
    main_displayln_info(ssprintf("NUM_WALL_EXACT: %d", NUM_WALL_EXACT));
    main_displayln_info(ssprintf("WALL_SLOPPY_EXACT_RATIO: %.20f", WALL_SLOPPY_EXACT_RATIO));

    main_displayln_info(ssprintf("NUM_POINT_SLOPPY: %d", NUM_POINT_SLOPPY));
    main_displayln_info(ssprintf("NUM_POINT_EXACT_1: %d", NUM_POINT_EXACT_1));
    main_displayln_info(ssprintf("NUM_POINT_EXACT_2: %d", NUM_POINT_EXACT_2));
  }

  TwoPointWallEnsembleInfo(const std::string& ensemble_, const std::string& accuracy_) : EnsembleInfo(ensemble_) {
    init(ensemble_, accuracy_);
    show_info();
  }

  void make_field_ensemble_accuracy_tmin_dir() const {
    qassert(FIELD_OUT_PATH != "" && ENSEMBLE != "" && ACCURACY != "");
    qmkdir_sync_node(FIELD_OUT_PATH);
    qmkdir_sync_node(FIELD_OUT_PATH + "/" + ENSEMBLE);
    qmkdir_sync_node(FIELD_OUT_PATH + "/" + ENSEMBLE + "/" + ACCURACY);
    qmkdir_sync_node(FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH);
  }

  std::string get_field_traj_dir(const int traj) const {
    if (ENSEMBLE == "32Dfine-0.0001") {
      return ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/32Dfine-0.0001/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    } else if (ENSEMBLE == "24D-0.0174") {
      return ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/24D-0.0174/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    } else if (ENSEMBLE == "48I-0.00078") {
      return ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/48I-0.00078/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    }
    return FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH + ssprintf("/results=%04d", traj);
  }

  std::string get_field_traj_avg_dir(const int traj) const {
    std::string traj_dir = get_field_traj_dir(traj);
    if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "24D-0.0174" || ENSEMBLE == "48I-0.00078") {
      return get_field_traj_dir(traj);
    }
    return traj_dir + ssprintf("/avg ; type=%d", TYPE);
  }

  void load_field_traj_avg(PionGGElemField& field, const int traj) const {
    const std::string path = get_field_traj_avg_dir(traj);
    if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "24D-0.0174" || ENSEMBLE == "48I-0.00078") {
      FieldM<Complex, 16> flc;
      read_field(flc, path);
      to_from_big_endian_64(get_data(flc));
      const Geometry& geo = flc.geo;
      field.init(geo, 1);
      qassert(is_matching_geo(geo, field.geo));
#pragma omp parallel for
      for (long index = 0; index < geo.local_volume(); ++index) {
        Coordinate x = geo.coordinate_from_index(index);

        for (int mu = 0; mu < 4; ++mu) {
          for (int nu = 0; nu < 4; ++nu) {
            field.get_elem(x).v[mu][nu] = flc.get_elem(x, 4 * mu + nu);
          }
        }
      }
      return;
    }
    read_field(field, path);
    return;
  }
  
  void make_field_traj_dir(const int traj) const {
    const std::string field_traj_path = get_field_traj_dir(traj);
    qmkdir_sync_node(field_traj_path);
  }

  bool is_traj_computed(const int traj) const {
    if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "24D-0.0174" || ENSEMBLE == "48I-0.00078") {
      std::string decay_cheng = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/" + ENSEMBLE + ssprintf("/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
      if (does_file_exist_sync_node(decay_cheng) || does_file_exist_sync_node(decay_cheng + "/checkpoint")) {
        return true;
      } else {
        return false;
      }
    }
    return does_file_exist_sync_node(get_field_traj_dir(traj) + "/avg_checkpoint");
  }

  bool is_traj_all_xg_computed(const int traj) const {
    return does_file_exist_sync_node(FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH + ssprintf("/results=%04d/checkpoint", traj));
  }

  bool props_is_ready(const int traj) const {
    const std::string point_path = get_point_src_props_path(ENSEMBLE, traj);
    const std::string gt_path = get_gauge_transform_path(ENSEMBLE, traj);
    const std::string wall_path = get_wall_src_props_sloppy_path(ENSEMBLE, traj);
    if (ACCURACY == "sloppy") {return point_path != "" && gt_path != "" && wall_path != "";}
    const std::string exact_wall_path = get_wall_src_props_exact_path(ENSEMBLE, traj);
    if (ACCURACY == "ama") {return point_path != "" && gt_path != "" && wall_path != "" && exact_wall_path != "";}
    return false;
  }

  std::vector<int> get_num_point_prop_sloppy_exact(const int traj) const {
    const std::string point_src_path = get_point_src_props_path(ENSEMBLE, traj);
    const std::vector<std::string> point_src_prop_list = list_folders_under_path(point_src_path);
    int num_0 = 0, num_1 = 0, num_2 = 0;
    for (int i =0; i < point_src_prop_list.size(); ++i){
      const Coordinate point_src_coor = get_xg_from_path(point_src_prop_list[i]);
      const std::string fname_0 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], TYPE, 0);
      if (fname_0 != point_src_prop_list[i]) {continue; }
      num_0 += 1;

      const std::string fname_1 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], TYPE, 1);
      if (!does_file_exist_sync_node(point_src_path + "/" + fname_1)) {continue;}
      num_1 += 1;

      const std::string fname_2 = ssprintf("xg=(%d,%d,%d,%d) ; type=%d ; accuracy=%d", point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3], TYPE, 2);
      if (!does_file_exist_sync_node(point_src_path + "/" + fname_2)) {continue;}
      num_2 += 1;
    }
    std::vector<int> res = {num_0, num_1, num_2};
    return res;
  }

  std::vector<Coordinate> get_point_xg_list(const int traj) const {
    const std::string point_src_path = get_point_src_props_path(ENSEMBLE, traj);
    const std::vector<std::string> point_src_prop_list = list_folders_under_path(point_src_path);
    std::vector<Coordinate> res;
    for (int i = 0; i < point_src_prop_list.size(); ++i){
      if (read_type(point_src_prop_list[i]) != TYPE or read_accuracy(point_src_prop_list[i]) != 0) {continue; }
      res.push_back(get_xg_from_path(point_src_prop_list[i]));
    }
    return res;
  }

  std::string get_field_traj_xg_folder_name(const Coordinate& xg) const {
    return ssprintf("xg=(%d,%d,%d,%d) ; type=%d", xg[0], xg[1], xg[2], xg[3], TYPE);
  }

  std::string get_field_traj_xg_dir(const int traj, const Coordinate& xg) const {
    const std::string field_out_coor_path = get_field_traj_dir(traj) + "/" + get_field_traj_xg_folder_name(xg);
    return field_out_coor_path;
  }

  void make_field_traj_xg_dir(const int traj, const Coordinate& xg) const {
    const std::string field_out_coor_path = get_field_traj_xg_dir(traj, xg);
    qmkdir_sync_node(field_out_coor_path);
  }

  bool is_traj_xg_computed(const int traj, const Coordinate& xg) const {
    const std::string field_out_coor_path = get_field_traj_xg_dir(traj, xg);
    return does_file_exist_sync_node(field_out_coor_path + "/checkpoint");
  }

  void load_field_traj_xg(PionGGElemField& pgge_field, const int traj, const Coordinate& xg) const {
    qassert(is_traj_xg_computed(traj, xg));
    const std::string path = get_field_traj_xg_dir(traj, xg);
    read_field(pgge_field, path);
  }

  int count_field_traj_num_xg(const int traj) const {
    const std::string field_traj_path = get_field_traj_dir(traj);
    const std::vector<std::string> folder_list = list_folders_under_path(field_traj_path);
    int count = 0;
    for (int i = 0; i < folder_list.size(); ++i) {
      const Coordinate coor = get_xg_from_path(folder_list[i]);
      const std::string folder_name = get_field_traj_xg_folder_name(coor);
      if (folder_name == folder_list[i]) {count += 1;}
    }
    return count;
  }
};
#endif

#if 0
struct F2EnsembleInfo : public EnsembleInfo, public YAndRotationInfo{
  int T_MIN;
  int TYPE = 0;
  
  std::string ACCURACY;
  std::string MOD;
  int XXP_LIMIT;

  std::string F2_PATH = "f2";
  std::string F2_ENSEMBLE_PATH;
  
  std::string TRAJ_PAIRS_PATH = "TrajPairs";
  std::string TRAJ_PAIRS_FILE_PATH;
  int TRAJ_JUMP;

  int NUM_RMAX;
  int NUM_RMIN;

  void init(const std::string& accuracy_, const std::string& mod_, const int xxp_limit) {
    ACCURACY = accuracy_;
    qassert(ACCURACY == "ama" || ACCURACY == "sloppy");
    MOD = mod_;
    qassert(MOD == "" || MOD == "xyp>=xy" || MOD == "xy>=xyp");
    XXP_LIMIT = xxp_limit;
    F2_ENSEMBLE_PATH = F2_PATH + "/" + ENSEMBLE;
    T_MIN = get_tmin(ENSEMBLE);
    if (ENSEMBLE == "24D-0.00107") {
      TRAJ_JUMP = 50;
      NUM_RMAX = 80;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "24D-0.0174") {
      TRAJ_JUMP = 50;
      NUM_RMAX = 80;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "32D-0.00107") {
      TRAJ_JUMP = 50;
      NUM_RMAX = 80;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "32Dfine-0.0001") {
      TRAJ_JUMP = 50;
      NUM_RMAX = 100;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "48I-0.00078") {
      TRAJ_JUMP = 60;
      NUM_RMAX = 120;
      NUM_RMIN = 60;
    } else {
      qassert(false);
    }
    std::string file_name = "ensemble:" + ENSEMBLE + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    TRAJ_PAIRS_FILE_PATH = TRAJ_PAIRS_PATH + "/" + file_name;
  }

  void show_info() const {
    main_displayln_info("F2EnsembleInfo:");
    main_displayln_info("ACCURACY: " + ACCURACY);
    main_displayln_info("MOD: " + MOD);
    main_displayln_info(ssprintf("XXP_LIMIT: %d", XXP_LIMIT));

    main_displayln_info("F2_PATH: " + F2_PATH);
    main_displayln_info("F2_ENSEMBLE_PATH: " + F2_ENSEMBLE_PATH);

    main_displayln_info("TRAJ_PAIRS_FILE_PATH: " + TRAJ_PAIRS_FILE_PATH);
    main_displayln_info(ssprintf("TRAJ_JUMP: %d", TRAJ_JUMP));

    main_displayln_info(ssprintf("NUM_RMAX: %d", NUM_RMAX));
    main_displayln_info(ssprintf("NUM_RMIN: %d", NUM_RMIN));
  }

  F2EnsembleInfo(const std::string& ensemble_, const std::string& accuracy_, const std::string& mod_, const int xxp_limit) : EnsembleInfo(ensemble_), YAndRotationInfo(ensemble_) {
    init(accuracy_, mod_, xxp_limit);
    show_info();
  }

  void make_f2_ensemble_dir() const {
    qmkdir_sync_node(F2_PATH);
    qmkdir_sync_node(F2_ENSEMBLE_PATH);
  }

  std::string get_traj_pair_dir(std::vector<int> traj_pair) const {
    qassert(traj_pair.size() == 2);
    return F2_ENSEMBLE_PATH + ssprintf("/traj=%04d,%04d;accuracy=%s;t-min=%04d;xxp-limit=%d;mod=%s;type=%d", traj_pair[0], traj_pair[1], ACCURACY.c_str(), T_MIN, XXP_LIMIT, MOD.c_str(), TYPE);
  }

  void make_traj_pair_dir(std::vector<int> traj_pair) const {
    qmkdir_sync_node(get_traj_pair_dir(traj_pair));
  }
  
  std::string get_traj_pair_i_path(std::vector<int> traj_pair, int i) const {
    return get_traj_pair_dir(traj_pair) + ssprintf("/%05d", i);
  }

  bool is_traj_pair_i_computed(std::vector<int> traj_pair, int i) const {
    std::string path = get_traj_pair_i_path(traj_pair, i);
    return does_file_exist_sync_node(path);
  }

  bool is_traj_pair_computed(std::vector<int> traj_pair) const {
    return is_traj_pair_i_computed(traj_pair, chunk_size - 1);
  }

  std::vector<std::vector<int>> get_traj_pair_list() const {
#if 0
    std::vector<std::vector<int>> traj_pair_list;
    int traj_batch = TRAJ_START;
    while (traj_batch < TRAJ_END) {
      for (int traj_1 = traj_batch; traj_1 < traj_batch + TRAJ_JUMP; traj_1 += 10) {
        if (traj_1 + TRAJ_JUMP > TRAJ_END) {break;}
        std::vector<int> traj_pair = {traj_1, traj_1 + TRAJ_JUMP};
        traj_pair_list.push_back(traj_pair);
      }
      traj_batch += TRAJ_JUMP * 2;
    }
    main_displayln_info("Traj Pair List:");
    for (int i = 0; i < traj_pair_list.size(); ++i) {
      main_displayln_info(ssprintf("%d, %d", traj_pair_list[i][0], traj_pair_list[i][1]));
    }
    return traj_pair_list;
#endif
    std::vector<std::vector<int>> traj_pair_list;
    std::ifstream fileopen(TRAJ_PAIRS_FILE_PATH);
    std::string line;
    while (getline(fileopen, line)) {
      std::vector<std::string> strs;
      strs = string_split(line, " ");
      qassert(strs.size() == 2);
      std::vector<int> traj_pair(2);
      traj_pair[0] = std::stoi(strs[0]);
      traj_pair[1] = std::stoi(strs[1]);
      traj_pair_list.push_back(traj_pair);
    }
    main_displayln_info(ssprintf("Traj Pair List Size: %d", traj_pair_list.size()));
    return traj_pair_list;
  }
};
#endif

QLAT_END_NAMESPACE

using namespace qlat;

#if 0
Complex_Table find_onepair_f2_table_rotation_pgge(
    const PionGGElemField& pgge_field_1, 
    const PionGGElemField& pgge_field_2, 
    const Y_And_Rotation_Info_Elem& onepair, 
    const int xxp_limit, 
    const double& muon, 
    const double& pion, 
    const std::string& mod="")
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
  int xb_limit = xxp_limit;
  find_xb_rotation_pgge(xb, pgge_field_1, rotation_matrix, xb_limit);

  // bm_table
  BM_TABLE bm_table;
  bm_table.set_zero();
  find_bm_table_rotation_pgge(bm_table, pgge_field_2, y_large, rotation_matrix, muon, mod);

  // e_xbbm_table
  Complex_Table e_xbbm_table;
  e_xbbm_table.set_zero();
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
#endif

#if 0
Complex_Table find_one_y_f2_table_rotation_pgge(
    const F2EnsembleInfo& f2_ensemble_info, 
    const Y_And_Rotation_Info_Elem& one_y, 
    const PionGGElemField& pgge_field_1, 
    const PionGGElemField& pgge_field_2)
{
  TIMER_VERBOSE("find_one_y_f2_table_rotation_pgge");
  const Geometry& geo = pgge_field_1.geo;
  const Geometry& geo_ = pgge_field_2.geo;
  qassert(geo == geo_);

  const double muon = f2_ensemble_info.MUON;
  const double pion = f2_ensemble_info.PION;
  const int xxp_limit = f2_ensemble_info.XXP_LIMIT;
  const std::string mod = f2_ensemble_info.MOD;

  const int num_rmax = f2_ensemble_info.NUM_RMAX;
  const int num_rmin = f2_ensemble_info.NUM_RMIN;

  const double theta_xy = one_y.theta_xy;
  const double theta_xt = one_y.theta_xt;
  const double theta_zt = one_y.theta_zt;
  const RotationMatrix rotation_matrix = RotationMatrix(theta_xy, theta_xt, theta_zt);
  // y_large
  const CoordinateD& y_large = one_y.y_large;

  // xb
  XB xb;
  set_zero(xb);
  int xb_limit = xxp_limit;
  find_xb_rotation_pgge(xb, pgge_field_1, rotation_matrix, xb_limit);

  // bm_table
  BM_TABLE bm_table(num_rmax, num_rmin);
  bm_table.set_zero();
  find_bm_table_rotation_pgge(bm_table, pgge_field_2, y_large, rotation_matrix, muon, mod);

  // e_xbbm_table
  Complex_Table e_xbbm_table(num_rmax, num_rmin);
  e_xbbm_table.set_zero();
  find_e_xbbm_table(e_xbbm_table, xb, bm_table);

  // prop
  Complex prop = pion_prop(y_large, Coor_0, pion);

  e_xbbm_table = (prop / one_y.dist) * e_xbbm_table;

  // show one pair e_xbbm_table
  std::string info = "";
  info += show_y_and_rotation_info_elem(one_y);
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  main_displayln_info(info);

  return e_xbbm_table;
}
#endif

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

#if 0
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
#endif

#if 0
Complex_Table compute_f2_one_traj_pair_and_save(
    const std::vector<int>& traj_pair, 
    const F2EnsembleInfo& f2_ensemble_info, 
    const std::vector<Y_And_Rotation_Info_Elem>& y_info, 
    const PionGGElemField& pgge_field_1, 
    const PionGGElemField& pgge_field_2)
{
  TIMER_VERBOSE("compute_f2_one_traj_pair_and_save");
  qassert(traj_pair.size() == 2);
  main_displayln_info(fname + ssprintf(": Compute f2 From Traj Pair: %04d, %04d", traj_pair[0], traj_pair[1]));
  f2_ensemble_info.make_traj_pair_dir(traj_pair);
  const int num_rmax = f2_ensemble_info.NUM_RMAX;
  const int num_rmin = f2_ensemble_info.NUM_RMIN;
  Complex_Table f2_table(num_rmax, num_rmin);
  f2_table.set_zero();
  for (int i = 0; i < y_info.size(); ++i)
  {
    // check
    if (f2_ensemble_info.is_traj_pair_i_computed(traj_pair, i)) {
      main_displayln_info(fname + ssprintf(": Traj Pair %04d, %04d, One Y i=%d Have Been Computed", traj_pair[0], traj_pair[1], i));
      continue;
    }

    // compute
    main_displayln_info(fname + ssprintf(": Compute One Y %d/%d From Traj Pair %04d, %04d", i, y_info.size(), traj_pair[0], traj_pair[1]));
    const Y_And_Rotation_Info_Elem& one_y = y_info[i];
    Complex_Table one_pair_table = find_one_y_f2_table_rotation_pgge(f2_ensemble_info, one_y, pgge_field_1, pgge_field_2);

    // save
    std::string one_y_save_path = f2_ensemble_info.get_traj_pair_i_path(traj_pair, i);
    write_data_from_0_node(one_pair_table, one_y_save_path);

    f2_table += one_pair_table;
  }
  f2_table = 1. / y_info.size() * f2_table;
  main_displayln_info(std::string(fname) + ": f2 avg");
  main_displayln_info(show_complex_table(f2_table));
  return f2_table;
}
#endif

#if 0
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
#endif

#if 0
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
    wm_from_x_to_wall = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_wall_to_x) * gamma5;
  }

#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    int t_wall;
    int t_sep;
    int diff = smod(xp[3] - x[3], total_site[3]);

    // t_wall << t and t'
    if (diff < 0)
    {
      t_wall = mod(xp[3] - t_min, total_site[3]);
      t_sep = t_min + abs(diff);

    } else {
      t_wall = mod(x[3] - t_min, total_site[3]);
      t_sep = t_min;
    }

    const Propagator4d& wall_src_prop = wall_src_list[t_wall];
    WilsonMatrix& wm_from_wall_to_x = wm_from_wall_to_x_t[t_wall];
    WilsonMatrix& wm_from_x_to_wall = wm_from_x_to_wall_t[t_wall];

    const WilsonMatrix& wm_from_wall_to_xp = wall_src_prop.get_elem(lxp);
    const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
    const WilsonMatrix wm_from_xp_to_wall = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_wall_to_xp) * gamma5;
    const WilsonMatrix wm_from_xp_to_x = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_x_to_xp) * gamma5;

    PionGGElem& pgge = three_point_correlator_labeled_xp.get_elem(lxp);

    three_prop_contraction (pgge, wm_from_x_to_wall, wm_from_wall_to_xp, wm_from_xp_to_x);
    three_prop_contraction_(pgge, wm_from_xp_to_wall, wm_from_wall_to_x, wm_from_x_to_xp);
    pgge /= exp(-pion * (double)t_sep);
  }
  return;
}
#endif

#if 0
void compute_three_point_correlator_ama_from_closest_wall_src_prop(const Coordinate& x, const int t_min, PionGGElemField& three_point_correlator_labeled_xp, const Propagator4d& point_src_prop, const std::vector<Propagator4d>& wall_src_list, const std::vector<Propagator4d>& exact_wall_src_list, const std::vector<int>& exact_wall_t_list, const double sloppy_exact_ratio, const double pion)
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
    wm_from_x_to_wall = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_wall_to_x) * gamma5;
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
    exact_wm_from_x_to_wall = gamma5 * (WilsonMatrix)matrix_adjoint(exact_wm_from_wall_to_x) * gamma5;
  }

  double ama_factor = sloppy_exact_ratio;

#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lxp = geo.coordinate_from_index(index);
    const Coordinate xp = geo.coordinate_g_from_l(lxp);

    int t_wall;
    int t_sep;
    int diff = smod(xp[3] - x[3], total_site[3]);

    // t_wall << t and t'
    if (diff < 0)
    {
      t_wall = mod(xp[3] - t_min, total_site[3]);
      t_sep = t_min + abs(diff);
    } else {
      t_wall = mod(x[3] - t_min, total_site[3]);
      t_sep = t_min;
    }

    PionGGElem& pgge = three_point_correlator_labeled_xp.get_elem(lxp);
    {
      const Propagator4d& wall_src_prop = wall_src_list[t_wall];
      WilsonMatrix& wm_from_wall_to_x = wm_from_wall_to_x_t[t_wall];
      WilsonMatrix& wm_from_x_to_wall = wm_from_x_to_wall_t[t_wall];

      const WilsonMatrix& wm_from_wall_to_xp = wall_src_prop.get_elem(lxp);
      const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
      const WilsonMatrix wm_from_xp_to_wall = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_wall_to_xp) * gamma5;
      const WilsonMatrix wm_from_xp_to_x = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_x_to_xp) * gamma5;

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
        const WilsonMatrix exact_wm_from_xp_to_wall = gamma5 * (WilsonMatrix)matrix_adjoint(exact_wm_from_wall_to_xp) * gamma5;
        const WilsonMatrix& wm_from_x_to_xp = point_src_prop.get_elem(lxp);
        const WilsonMatrix wm_from_xp_to_x = gamma5 * (WilsonMatrix)matrix_adjoint(wm_from_x_to_xp) * gamma5;

        three_prop_contraction (pgge_exact, exact_wm_from_x_to_wall, exact_wm_from_wall_to_xp, wm_from_xp_to_x);
        three_prop_contraction_(pgge_exact, exact_wm_from_xp_to_wall, exact_wm_from_wall_to_x, wm_from_x_to_xp);

        pgge_exact -= pgge;
        pgge_exact *= ama_factor;
      }
    }
    pgge += pgge_exact;
    pgge /= exp(-pion * (double)t_sep);
  }
  return;
}
#endif

#if 0
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
#endif

#if 0
void load_point_xg_prop(Propagator4d& point_src_prop, const TwoPointWallEnsembleInfo& ensemble_info, const int traj, const Coordinate& point_src_coor) {
  const std::string ensemble = ensemble_info.ENSEMBLE;
  const int type = ensemble_info.TYPE;
  const std::string accuracy = ensemble_info.ACCURACY;

  // accuracy == 0
  std::string point_src_0_path = get_point_src_prop_path(ensemble, traj, point_src_coor, type, 0);
  qassert(point_src_0_path != "");
  const Propagator4d& point_src_prop_0 = get_prop(point_src_0_path);
  point_src_prop = point_src_prop_0;
  
  if (accuracy == "sloppy") {return;}

  std::vector<int> num_point_prop_sloppy_exact = {ensemble_info.NUM_POINT_SLOPPY, ensemble_info.NUM_POINT_EXACT_1, ensemble_info.NUM_POINT_EXACT_2};
  // accuracy == 1
  std::string point_src_1_path = get_point_src_prop_path(ensemble, traj, point_src_coor, type, 1);
  if (point_src_1_path != "") {
    const Propagator4d& point_src_prop_1 = get_prop(point_src_1_path);
    Propagator4d point_src_prop_;
    point_src_prop_ = point_src_prop_1;
    point_src_prop_ -= point_src_prop_0;
    point_src_prop_ *= (1.0 * num_point_prop_sloppy_exact[0] / num_point_prop_sloppy_exact[1]);
    point_src_prop += point_src_prop_;

    // accuracy == 2
    std::string point_src_2_path = get_point_src_prop_path(ensemble, traj, point_src_coor, type, 2);
    if (point_src_1_path != "" && point_src_2_path != "") {
      const Propagator4d& point_src_prop_2 = get_prop(point_src_2_path);
      Propagator4d point_src_prop__;
      point_src_prop__ = point_src_prop_2;
      point_src_prop__ -= point_src_prop_1;
      point_src_prop__ *= (1.0 * num_point_prop_sloppy_exact[0] / num_point_prop_sloppy_exact[2]);
      point_src_prop += point_src_prop__;
    }
  }

  if (accuracy == "ama") {
    return;
  } else {
    qassert(false);
  }
  return;
}
#endif

#if 0
void load_gauge_inv(GaugeTransform& gtinv, const std::string& ensemble, const int traj) {
  TIMER_VERBOSE("load_gauge_inv");
  const std::string gauge_transform_path = get_gauge_transform_path(ensemble, traj);
  qassert(gauge_transform_path != "");
  GaugeTransform gt;
  read_field(gt, gauge_transform_path);
  to_from_big_endian_64(get_data(gt));
  gt_invert(gtinv, gt);
}
#endif

#if 0
void load_wall_sloppy_props_no_gauge(std::vector<Propagator4d>& wall_src_list, const GaugeTransform& gtinv, const std::string ensemble, const int traj) {
  TIMER_VERBOSE("load_wall_sloppy_props_no_gauge");
  const Coordinate total_site = gtinv.geo.total_site();
  wall_src_list.resize(total_site[3]);
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string wall_src_t_path = get_wall_src_prop_sloppy_path(ensemble, traj, t_wall);
    qassert(wall_src_t_path != "");
    main_displayln_info("Read Sloppy Wall From: " + wall_src_t_path);
    Propagator4d& wall_src_prop = wall_src_list[t_wall];
    Propagator4d wall_src_prop_ = get_wall_prop(wall_src_t_path);
    wall_src_prop = wall_src_prop_;
  }
}
#endif

#if 0
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
#endif

#if 0
void load_wall_exact_props_no_gauge(std::vector<int>& exact_wall_t_list, std::vector<Propagator4d>& exact_wall_src_list, const GaugeTransform& gtinv, const std::string ensemble, const int traj) {
  TIMER_VERBOSE("load_wall_exact_props_no_gauge");
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
    Propagator4d exact_wall_src_prop_ = get_wall_prop(exact_wall_src_t_path);
    exact_wall_src_prop = exact_wall_src_prop_;
  }
}
#endif

#if 0
void load_wall_exact_props(std::vector<int>& exact_wall_t_list, std::vector<Propagator4d>& exact_wall_src_list, 
                          const GaugeTransform& gtinv, const std::string ensemble, const int traj) {
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
#endif

#if 0
void compute_point_point_wall_correlator_ama_in_one_traj(const TwoPointWallEnsembleInfo& ensemble_info, const int traj)
{
  TIMER_VERBOSE("compute_point_point_wall_correlator_ama_in_one_traj");

  if (ensemble_info.is_traj_computed(traj)) {
    main_displayln_info(fname + ssprintf(": Two-Point to Wall Has Already Been Computed In Traj=%d", traj));
    return;
  }
  if (!ensemble_info.props_is_ready(traj)) {
    main_displayln_info(fname + ssprintf(": Props Are Not Ready In Traj=%d", traj));
    return;
  }

  // check point src num of sloppy and exact
  std::vector<int> num_point_prop_sloppy_exact = ensemble_info.get_num_point_prop_sloppy_exact(traj);
  main_displayln_info(fname + ssprintf("Point Src Num of Accuracy0 %d, Accuracy1 %d, Accuracy2 %d", num_point_prop_sloppy_exact[0], num_point_prop_sloppy_exact[1], num_point_prop_sloppy_exact[2]));

  std::vector<Coordinate> xg_list = ensemble_info.get_point_xg_list(traj);
  qassert(num_point_prop_sloppy_exact[0] == ensemble_info.NUM_POINT_SLOPPY &&
          num_point_prop_sloppy_exact[0] == xg_list.size() &&
          num_point_prop_sloppy_exact[1] == ensemble_info.NUM_POINT_EXACT_1 &&
          num_point_prop_sloppy_exact[2] == ensemble_info.NUM_POINT_EXACT_2);

  // load ensemble info
  const std::string ensemble = ensemble_info.ENSEMBLE;
  const int type = ensemble_info.TYPE;
  const double pion = ensemble_info.PION;
  const int t_min = ensemble_info.T_MIN;
  const double sloppy_exact_ratio = ensemble_info.WALL_SLOPPY_EXACT_RATIO;

  // mkdir field traj
  ensemble_info.make_field_traj_dir(traj);
  const std::string field_out_path = ensemble_info.get_field_traj_dir(traj);

  // load gauge
  GaugeTransform gtinv;
  load_gauge_inv(gtinv, ensemble, traj);
  const Coordinate total_site = gtinv.geo.total_site();

  // load and gauge inv wall_prop
  main_displayln_info(std::string(fname) + ": Load And Gauge Inv Wall Src Prop for All T");
  std::vector<Propagator4d> wall_src_list;
  load_wall_sloppy_props(wall_src_list, gtinv, ensemble, traj);

  // load and gauge inv exact wall_prop
  main_displayln_info(std::string(fname) + ": Load And Gauge Inv Exact Wall Src Prop");
  std::vector<int> exact_wall_t_list;
  std::vector<Propagator4d> exact_wall_src_list;
  load_wall_exact_props(exact_wall_t_list, exact_wall_src_list, gtinv, ensemble, traj);
  main_displayln_info(fname + ssprintf(": Wall Src Num of Sloppy %d, Exact %d", wall_src_list.size(), exact_wall_src_list.size()));
  main_displayln_info(fname + ssprintf(": WALL_SLOPPY_EXACT_RATIO: %.20f", sloppy_exact_ratio));
  
  int count = 0;
  for (int i = 0; i < xg_list.size(); ++i) {
    // one point src
    const Coordinate point_src_coor = xg_list[i];
    if (ensemble_info.is_traj_xg_computed(traj, point_src_coor)) {
      main_displayln_info(fname + ssprintf(": Two Point to Wall Has Been Computed in traj=%d, xg=(%d,%d,%d,%d)", traj, point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3]));
      count += 1;
      continue;
    }

    main_displayln_info(fname + ssprintf(": Compute Two Point to Wall in traj=%d, xg=(%d,%d,%d,%d)", traj, point_src_coor[0], point_src_coor[1], point_src_coor[2], point_src_coor[3]));
    Propagator4d point_src_prop;
    load_point_xg_prop(point_src_prop, ensemble_info, traj, point_src_coor);

    // setup PionGGElemField
    const Geometry& geo = point_src_prop.geo;
    PionGGElemField three_point_correlator_labeled_xp;
    three_point_correlator_labeled_xp.init(geo);
    set_zero(three_point_correlator_labeled_xp.field);

    compute_three_point_correlator_ama_from_closest_wall_src_prop(point_src_coor, t_min, three_point_correlator_labeled_xp, point_src_prop, wall_src_list, exact_wall_src_list, exact_wall_t_list, sloppy_exact_ratio, pion);

    // shift
    PionGGElemField three_point_correlator_labeled_xp_shift;
    field_shift(three_point_correlator_labeled_xp_shift, three_point_correlator_labeled_xp, -point_src_coor);

    // save
    const Coordinate new_geom(1, 1, 1, 8);
    std::string field_out_coor_path = ensemble_info.get_field_traj_xg_dir(traj, point_src_coor);
    ensemble_info.make_field_traj_xg_dir(traj, point_src_coor);
    dist_write_field(three_point_correlator_labeled_xp_shift, new_geom, field_out_coor_path);
    sync_node();

    count += 1;
    main_displayln_info(ssprintf("Save PionGGElem Field to [%04d/%04d]: ", count, xg_list.size()) + field_out_coor_path);
  }
  qtouch_info(field_out_path + "/checkpoint");
  return;
}
#endif

#if 0
void compute_point_point_wall_correlator_ama_in_one_traj(const int t_min, const std::string& wall_src_path, const std::string& exact_wall_src_path, const std::string& gauge_transform_path, const double sloppy_exact_ratio, const std::string& point_src_path, const std::string& field_out_traj_path, const double pion, int type_)
{
  TIMER_VERBOSE("compute_point_point_wall_correlator_ama_in_one_traj");

  qassert(does_file_exist_sync_node(field_out_traj_path));
  const std::string field_out_tmin_path = field_out_traj_path + "/" + ssprintf("t-min=%04d", t_min);
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

    compute_three_point_correlator_ama_from_closest_wall_src_prop(point_src_coor, t_min, three_point_correlator_labeled_xp, point_src_prop, wall_src_list, exact_wall_src_list, exact_wall_t_list, sloppy_exact_ratio, pion);

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
#endif

#if 0
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
    read_field(pgge_field, one_field_path);
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
#endif

#if 0
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
#endif

#if 0
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
  main_displayln_info(ssprintf("Save Avg PionGGElem Field to: ") + field_out_path);
  qtouch_info(field_path + "/avg_checkpoint");

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
#endif

#if 0
void avg_pionggelemfield_and_rm(const TwoPointWallEnsembleInfo& tpw_info, const int traj)
{
  TIMER_VERBOSE("avg_pionggelemfield_and_rm");
  if (!tpw_info.is_traj_all_xg_computed(traj)) {
    main_displayln_info(fname + ssprintf(": Field of All xg Has Not Completed In Traj: %d", traj));
    return;
  }
  if (tpw_info.is_traj_computed(traj)) {
    main_displayln_info(fname + ssprintf(": Avg Field Has Already Completed In Traj: %d", traj));
    return;
  }
  const std::string field_path = tpw_info.get_field_traj_dir(traj);
  const std::vector<std::string> field_list = list_folders_under_path(field_path);
  PionGGElemField pgge_field_avg;

  int num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string folder_name = tpw_info.get_field_traj_xg_folder_name(coor);
    if (folder_name != field_list[i]) {continue;}
    num += 1;
    PionGGElemField pgge_field;
    main_displayln_info(std::string(fname) + ": Read PionGGElemField from: " + tpw_info.get_field_traj_xg_dir(traj, coor));
    tpw_info.load_field_traj_xg(pgge_field, traj, coor);
    if (num == 1) {
      pgge_field_avg = pgge_field;
    } else {
      pgge_field_avg += pgge_field;
    }
  }
  main_displayln_info(fname + ssprintf(": Read PionGGElemField Num: %d", num));
  qassert(num == tpw_info.count_field_traj_num_xg(traj));
  pgge_field_avg /= double(num);
  
  // save avg
  std::string field_out_path = tpw_info.get_field_traj_avg_dir(traj);
  qmkdir_sync_node(field_out_path);
  const Coordinate new_geom(1, 1, 1, 8);
  dist_write_field(pgge_field_avg, new_geom, field_out_path);
  sync_node();
  main_displayln_info(fname + ssprintf(": Save Avg PionGGElem Field to: ") + field_out_path);
  qtouch_info(field_path + "/avg_checkpoint");

  // rm folders
  num = 0;
  for (int i = 0; i < field_list.size(); ++i)
  {
    const std::string one_field_path = field_path + "/" + field_list[i];
    const Coordinate coor = get_xg_from_path(field_list[i]);
    const std::string folder_name = tpw_info.get_field_traj_xg_folder_name(coor);
    if (folder_name != field_list[i]) {continue;}
    qremove_all_info(one_field_path);
    num += 1;
    main_displayln_info(std::string(fname) + ": Delete PionGGElemField Folder: " + one_field_path);
  }
  main_displayln_info(fname + ssprintf(": Delete PionGGElemField Folders Num: %d", num));

  return;
}
#endif

#if 0
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
#endif

#if 0
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

    wm_t[t] += gamma5 * (WilsonMatrix)matrix_adjoint(prop.get_elem(lx)) * gamma5;
  }

  for (int i = 0; i < wm_t.size(); ++i)
  {
    glb_sum_double(wm_t[i]);
  }
}
#endif

#if 0
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
  for (int i = 0; i < wall_wall_corr.size(); ++i) {
    set_zero(wall_wall_corr[i]);
  }
  for (int tstart = 0; tstart < total_t; ++tstart)
  {
    for (int tsep = 0; tsep < total_t / 2; ++tsep)
    {
      int tend = mod(tstart + tsep, total_t);
      main_displayln_info(ssprintf("compute wall to wall, tstart=%d, tsep=%d, tend=%d", tstart, tsep, tend));
      // wall_wall_corr[tsep] += pi_pi_contraction(v_v_wm_sink[tstart][tend], v_v_wm_sink[tend][tstart]);
      // wall_wall_corr[tsep] += pi_pi_contraction(v_v_wm_src[tstart][tend], v_v_wm_src[tend][tstart]);
      wall_wall_corr[tsep] += pi_pi_contraction(v_v_wm_sink[tstart][tend], v_v_wm_src[tstart][tend]);
#if 0
      Complex w_w_contraction = pi_pi_contraction(v_v_wm_sink[tstart][tend], v_v_wm_src[tstart][tend]);
      main_displayln_info(ssprintf("w_w_contraction %f %f", w_w_contraction.real(), w_w_contraction.imag()));
      return wall_wall_corr;
#endif
    }
  }

  for (int tsep = 0; tsep < total_t / 2; ++tsep)
  {
    wall_wall_corr[tsep] /= total_t;
  }

  return wall_wall_corr;
}
#endif

#if 0
std::vector<Complex> compute_wall_wall_correlator_one_traj(const WallWallEnsembleInfo& ensemble_info, const int traj)
{
  TIMER_VERBOSE("compute_wall_wall_correlator_one_traj");
  std::string traj_label = ssprintf("[Traj=%d]", traj);

  // load ensemble info
  const std::string ensemble = ensemble_info.ENSEMBLE;
  const double sloppy_exact_ratio = ensemble_info.WALL_SLOPPY_EXACT_RATIO;
  const bool is_ama = (ensemble_info.ACCURACY == "ama");

  // load gauge
  // no need, just for total_t
  GaugeTransform gtinv;
  load_gauge_inv(gtinv, ensemble, traj);
  const Coordinate total_site = gtinv.geo.total_site();
  const int total_t = total_site[3];

#if 0
  // load wall_prop
  main_displayln_info(std::string(fname) + traj_label + ": Load And Gauge Inv Wall Src Prop for All T");
  std::vector<Propagator4d> wall_src_list;
  load_wall_sloppy_props_no_gauge(wall_src_list, gtinv, ensemble, traj);

  // load exact wall_prop
  main_displayln_info(std::string(fname) + traj_label + ": Load And Gauge Inv Exact Wall Src Prop");
  std::vector<int> exact_wall_t_list;
  std::vector<Propagator4d> exact_wall_src_list;
  load_wall_exact_props_no_gauge(exact_wall_t_list, exact_wall_src_list, gtinv, ensemble, traj);
  main_displayln_info(fname + traj_label + ssprintf(": Wall Src Num of Sloppy %d, Exact %d", wall_src_list.size(), exact_wall_src_list.size()));
  main_displayln_info(fname + traj_label + ssprintf(": WALL_SLOPPY_EXACT_RATIO: %.20f", sloppy_exact_ratio));
#endif

  // find t_wall_list
  std::vector<int> exact_wall_t_list;
  for (int t_wall = 0; t_wall < total_site[3]; ++t_wall){
    std::string exact_wall_src_t_path = get_wall_src_prop_exact_path(ensemble, traj, t_wall);
    if (exact_wall_src_t_path == "") {continue;}
    exact_wall_t_list.push_back(t_wall);
  }

  // prepare wm[src_t][sink_t]
  std::vector<std::vector<WilsonMatrix>> v_v_wm_sink(total_t);
  std::vector<std::vector<WilsonMatrix>> v_v_wm_src(total_t);
  std::vector<std::vector<WilsonMatrix>> v_v_wm_sink_exact(exact_wall_t_list.size());
  std::vector<std::vector<WilsonMatrix>> v_v_wm_src_exact(exact_wall_t_list.size());
  for (int t = 0; t < total_t; ++t)
  {
    v_v_wm_sink[t].resize(total_t);
    v_v_wm_src[t].resize(total_t);
  }
  for (int i = 0; i < exact_wall_t_list.size(); ++i)
  {
    v_v_wm_sink_exact[i].resize(total_t);
    v_v_wm_src_exact[i].resize(total_t);
  }

  // sum src and sink
  main_displayln_info(std::string(fname) + traj_label + ": Sum Src And Sink Over Space For All Wall Src Props Start");
  for (int t = 0; t < total_t; ++t)
  {
    main_displayln_info(std::string(fname) + traj_label + ": Sum Src And Sink In t=" + ssprintf("%d", t));
    // const Propagator4d& wall_src_t = wall_src_list[t];
    std::string wall_src_t_path = get_wall_src_prop_sloppy_path(ensemble, traj, t);
    qassert(wall_src_t_path != "");
    const Propagator4d& wall_src_t = get_prop(wall_src_t_path);
    std::vector<WilsonMatrix>& v_wm_sink = v_v_wm_sink[t];
    std::vector<WilsonMatrix>& v_wm_src = v_v_wm_src[t];
    qassert(v_wm_sink.size() == wall_src_t.geo.total_site()[3]);

    sum_sink_over_space_from_prop(v_wm_sink, wall_src_t);
    sum_src_over_space_from_prop(v_wm_src, wall_src_t);
  }
  for (int i = 0; i < exact_wall_t_list.size(); ++i)
  {
    main_displayln_info(std::string(fname) + traj_label + ": Sum Src And Sink (Exact) In t=" + ssprintf("%d", exact_wall_t_list[i]));
    // const Propagator4d& wall_src_t = exact_wall_src_list[i];
    std::string wall_src_t_path = get_wall_src_prop_exact_path(ensemble, traj, exact_wall_t_list[i]);
    qassert(wall_src_t_path != "");
    const Propagator4d& wall_src_t = get_prop(wall_src_t_path);;
    std::vector<WilsonMatrix>& v_wm_sink = v_v_wm_sink_exact[i];
    std::vector<WilsonMatrix>& v_wm_src = v_v_wm_src_exact[i];
    qassert(v_wm_sink.size() == wall_src_t.geo.total_site()[3]);

    sum_sink_over_space_from_prop(v_wm_sink, wall_src_t);
    sum_src_over_space_from_prop(v_wm_src, wall_src_t);
  }
  main_displayln_info(std::string(fname) + traj_label + ": Sum Src And Sink Over Space For All Wall Src Props Start");

  // wall to wall
  main_displayln_info(std::string(fname) + traj_label + ": Compute Wall To Wall");
  std::vector<Complex> wall_wall_corr(total_t / 2);
  for (int i = 0; i < wall_wall_corr.size(); ++i) {
    set_zero(wall_wall_corr[i]);
  }
  for (int tstart = 0; tstart < total_t; ++tstart)
  {
    for (int tsep = 0; tsep < total_t / 2; ++tsep)
    {
      int tend = mod(tstart + tsep, total_t);
      main_displayln_info(fname + traj_label + ssprintf(": Compute Wall To Wall, tstart=%d, tsep=%d, tend=%d", tstart, tsep, tend));
      Complex w_w_contraction = pi_pi_contraction(v_v_wm_sink[tstart][tend], v_v_wm_src[tstart][tend]);
      wall_wall_corr[tsep] += w_w_contraction;

#if 0
      main_displayln_info(ssprintf("w_w_contraction %f %f", w_w_contraction.real(), w_w_contraction.imag()));
      return wall_wall_corr;
#endif 

      // exact
      if (!is_ama) { continue; }
      for (int i = 0; i < exact_wall_t_list.size(); ++i) {
        if (exact_wall_t_list[i] != tstart) {continue;}
        main_displayln_info(fname + traj_label + ssprintf(": Compute Exact Wall To Wall, tstart=%d, tsep=%d, tend=%d", tstart, tsep, tend));
        Complex exact_w_w_contraction = pi_pi_contraction(v_v_wm_sink_exact[i][tend], v_v_wm_src_exact[i][tend]);
        exact_w_w_contraction = (exact_w_w_contraction - w_w_contraction) * sloppy_exact_ratio;
        wall_wall_corr[tsep] += exact_w_w_contraction;
      }
    }
  }

  // avg over tsep
  for (int tsep = 0; tsep < total_t / 2; ++tsep)
  {
    wall_wall_corr[tsep] /= total_t;
  }

  return wall_wall_corr;
}
#endif

#if 0
std::string show_vec_complex(std::vector<Complex> vec)
{
  std::string res = "";
  for (int i = 0; i < vec.size(); ++i)
  {
    res += ssprintf("%e %e, ", vec[i].real(), vec[i].imag());
  }
  return res;
}
#endif

#if 0
std::vector<Complex> compute_zw_from_wall_wall_corr(const std::vector<Complex>& vec, const double pion)
{
  std::vector<Complex> res(vec.size());
  for (int tsep = 0; tsep < vec.size(); ++tsep)
  {
    res[tsep] = vec[tsep] * exp(tsep * pion);
  }
  return res;
}
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
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
    const std::string PGGE_FIELD_PATH_1 = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_1], TMIN);
    read_pionggelemfield_without_accuracy_and_avg(pgge_field_1, PGGE_FIELD_PATH_1, TYPE);
    const std::string PGGE_FIELD_PATH_2 = "/projects/HadronicLight_4/ctu//hlbl/hlbl-pion/ThreePointCorrField/" + ENSEMBLE + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_2], TMIN);
    read_pionggelemfield_without_accuracy_and_avg(pgge_field_2, PGGE_FIELD_PATH_2, TYPE);

    // compute f2
    avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, pgge_field_1, pgge_field_2, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD, NUM_PAIRS_IN_CONFIG);
  }
}
#endif

#if 0 
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
#endif

#if 0
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

    if (obtain_lock(one_pair_save_folder + "-lock")) {
      // set pairs
      std::vector<Y_And_Rotation_Info_Elem> pairs_info(pairs_info_all.begin() + i_pairs_start, pairs_info_all.begin() + i_pairs_start + NUM_PAIRS_JUMP);
      main_displayln_info(ssprintf("Y_And_Rotation_Info_Elem Index: %08d - %08d", i_pairs_start, i_pairs_start + NUM_PAIRS_JUMP));

      // read pgge
      PionGGElemField pgge_field_1;
      PionGGElemField pgge_field_2;
      const std::string PGGE_FIELD_PATH_1 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_1], TMIN);
      const std::string PGGE_FIELD_PATH_2 = THREE_POINT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i_pgge_2], TMIN);
      if (!does_file_exist_sync_node(PGGE_FIELD_PATH_1 + "/avg_checkpoint") or !does_file_exist_sync_node(PGGE_FIELD_PATH_2 + "/avg_checkpoint")){
        main_displayln_info(ssprintf("PionGG Field are not completed, %04d, %04d", traj_list[i_pgge_1], traj_list[i_pgge_2]));
        release_lock();
        continue;
      }
      std::string AVG_PGGE_FIELD_PATH_1 = PGGE_FIELD_PATH_1 + ssprintf("/avg ; type=%d", TYPE);
      main_displayln_info("Read AVG_PGGE_FIELD_1 from: " + AVG_PGGE_FIELD_PATH_1);
      dist_read_field(pgge_field_1, AVG_PGGE_FIELD_PATH_1);
      std::string AVG_PGGE_FIELD_PATH_2 = PGGE_FIELD_PATH_2 + ssprintf("/avg ; type=%d", TYPE);
      main_displayln_info("Read AVG_PGGE_FIELD_2 from: " + AVG_PGGE_FIELD_PATH_2);
      dist_read_field(pgge_field_2, AVG_PGGE_FIELD_PATH_2);

      // compute f2
      qmkdir_sync_node(one_pair_save_folder);
      avg_f2_table_from_three_point_corr_from_y_and_rotation_info(pairs_info, pgge_field_1, pgge_field_2, XXP_LIMIT, MUON, PION, one_pair_save_folder, MOD, NUM_PAIRS_IN_CONFIG);
      release_lock();
    }
  }
}
#endif

#if 0
void compute_point_point_wall_correlator_sloppy_24D_all_traj()
{
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
#endif

#if 0
void compute_point_point_wall_correlator_ama_24D_all_traj()
{
  TwoPointWallEnsembleInfo ensemble_info_24d_ama("24D-0.00107", "ama");
  ensemble_info_24d_ama.make_field_ensemble_accuracy_tmin_dir();

  // for (int traj = 2280; traj >= 1000; traj -= 10)
  for (int traj = 1900; traj >= 1900; traj -= 10)
  {
    ensemble_info_24d_ama.make_field_traj_dir(traj);
    const std::string FIELD_OUT_FULL_PATH = ensemble_info_24d_ama.get_field_traj_dir(traj);

    if (obtain_lock(FIELD_OUT_FULL_PATH + "-lock")) {
      compute_point_point_wall_correlator_ama_in_one_traj(ensemble_info_24d_ama, traj);
      // avg and remove
      avg_pionggelemfield_without_accuracy_and_rm(FIELD_OUT_FULL_PATH, ensemble_info_24d_ama.TYPE);
      release_lock();
    }
  }
}
#endif

#if 0
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
  traj_list = {1370};
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
    // const std::string PGGE_FIELD_PATH = FIELD_OUT_ENSEMBLE_PATH + ssprintf("/results=%04d/t-min=%04d", traj_list[i], T_MIN);
    // avg_pionggelemfield_with_accuracy_and_rm(PGGE_FIELD_PATH, TYPE, ACCURACY);
  }
}
#endif

#if 0
void compute_point_point_wall_correlator_ama_32D_all_traj()
{
  TwoPointWallEnsembleInfo ensemble_info_32d_ama("32D-0.00107", "ama");
  ensemble_info_32d_ama.make_field_ensemble_accuracy_tmin_dir();

  for (int traj = 1020; traj >= 680; traj -= 10)
  {
    ensemble_info_32d_ama.make_field_traj_dir(traj);
    const std::string FIELD_OUT_FULL_PATH = ensemble_info_32d_ama.get_field_traj_dir(traj);

    if (obtain_lock(FIELD_OUT_FULL_PATH + "-lock")) {
      compute_point_point_wall_correlator_ama_in_one_traj(ensemble_info_32d_ama, traj);
      // avg and remove
      avg_pionggelemfield_without_accuracy_and_rm(FIELD_OUT_FULL_PATH, ensemble_info_32d_ama.TYPE);
      release_lock();
    }
  }
}
#endif

#if 0
void compute_wall_wall_corr_24D()
{
  // prepare traj_list
  std::vector<int> traj_list = {2640};
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
#if 1
    break;
#endif
  }
  main_displayln_info("Compute Wall to Wall Corr for All Configurations End");
  wall_wall_corr_avg = wall_wall_corr_avg / traj_list.size();
  main_displayln_info("wall_wall_corr_avg result:");
  main_displayln_info(show_vec_complex(wall_wall_corr_avg));

  // std::vector<Complex> zw = compute_zw_from_wall_wall_corr(wall_wall_corr_avg, PION);
  // main_displayln_info("zw result:");
  // main_displayln_info(show_vec_complex(zw));

}
#endif

# if 0
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
#endif

#if 0
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
#endif

#if 0
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
#endif

#if 0
void compute_point_point_wall_correlator(const std::string& ensemble, const std::string& accuracy)
{
  TwoPointWallEnsembleInfo ensemble_info(ensemble, accuracy);
  ensemble_info.make_field_ensemble_accuracy_tmin_dir();

  for (int traj = ensemble_info.TRAJ_END; traj >= ensemble_info.TRAJ_START; traj -= 10)
  {
    const std::string FIELD_OUT_FULL_PATH = ensemble_info.get_field_traj_dir(traj);

    if (obtain_lock(FIELD_OUT_FULL_PATH + "-lock")) {
      if (ensemble_info.ACCURACY == "ama") {
        compute_point_point_wall_correlator_ama_in_one_traj(ensemble_info, traj);
        // avg and remove
        avg_pionggelemfield_and_rm(ensemble_info, traj);
      }
      release_lock();
    }
  }
}
#endif

#if 0
void compute_f2_all_traj_pairs(const std::string ensemble, const std::string accuracy, const std::string mod, const int xxp_limit)
{
  TIMER_VERBOSE("compute_f2_all_traj_pairs");
  init_muon_line();
  const TwoPointWallEnsembleInfo tpw_info(ensemble, accuracy);
  const F2EnsembleInfo f2_ensemble_info(ensemble, accuracy, mod, xxp_limit);

  f2_ensemble_info.make_f2_ensemble_dir();
  std::vector<std::vector<int>> traj_pair_list = f2_ensemble_info.get_traj_pair_list();

  for (int i = 0; i < traj_pair_list.size(); ++i) {
    std::vector<int> traj_pair = traj_pair_list[i];
    std::vector<Y_And_Rotation_Info_Elem> y_info = f2_ensemble_info.get_next_y_info_list();
    if (f2_ensemble_info.is_traj_pair_computed(traj_pair)) {
      main_displayln_info(ssprintf("F2 Has Already Been Computed In Traj Pair %04d, %04d", traj_pair[0], traj_pair[1]));
      continue;
    }
    std::string one_pair_save_folder = f2_ensemble_info.get_traj_pair_dir(traj_pair);
    if (obtain_lock(one_pair_save_folder + "-lock")) {
      if (!tpw_info.is_traj_computed(traj_pair[0]) || !tpw_info.is_traj_computed(traj_pair[1])) {
        main_displayln_info(fname + ssprintf(": Field Are Not Completed In Traj Pair %04d, %04d", traj_pair[0], traj_pair[1]));
        release_lock();
        continue;
      }

      main_displayln_info(ssprintf("Load Two Two-Point-Wall Field From Traj %d, %d", traj_pair[0], traj_pair[1]));
      PionGGElemField two_point_wall_1;
      PionGGElemField two_point_wall_2;
      tpw_info.load_field_traj_avg(two_point_wall_1, traj_pair[0]);
      tpw_info.load_field_traj_avg(two_point_wall_2, traj_pair[1]);

      compute_f2_one_traj_pair_and_save(traj_pair, f2_ensemble_info, y_info, two_point_wall_1, two_point_wall_2);
      release_lock();
    }
  }
}
#endif

#if 0
void compute_wall_wall_correlator_all_traj(std::string ensemble, std::string accuracy)
{
  TIMER_VERBOSE("compute_wall_wall_correlator_all_traj");

  WallWallEnsembleInfo ensemble_info(ensemble, accuracy);
  ensemble_info.make_corr_ensemble_accuracy_dir();
  std::vector<Complex> wall_wall_corr_avg;

  main_displayln_info(std::string(fname) + ": Compute Wall to Wall Corr for All Traj Start");
  // int cnt = 0;
  for (int traj = ensemble_info.TRAJ_END; traj >= ensemble_info.TRAJ_START; traj -= 10)
  {
    if (obtain_lock(ensemble_info.get_traj_path(traj) + "-lock")) {

      // check computed
      if (ensemble_info.is_traj_computed(traj)) {
        main_displayln_info(fname + ssprintf(": Wall to Wall Corr for Traj=%d Has Already Been Computed", traj));
        release_lock();
        continue;
      }

      // check props
      if (get_gauge_transform_path(ensemble, traj) == "" || get_wall_src_props_sloppy_path(ensemble, traj) == "") {
        main_displayln_info(fname + ssprintf(": Wall Props for Traj=%d Are Not Ready", traj));
        release_lock();
        continue;
      }

      // compute
      main_displayln_info(fname + ssprintf(": Compute Wall to Wall Corr for Traj=%d", traj));
      std::vector<Complex> wall_wall_corr = compute_wall_wall_correlator_one_traj(ensemble_info, traj);
      if (wall_wall_corr_avg.size() != wall_wall_corr.size()) {
        wall_wall_corr_avg.resize(wall_wall_corr.size());
        for (int i = 0; i < wall_wall_corr_avg.size(); ++i) {
          set_zero(wall_wall_corr_avg[i]);
        }
      }
      main_displayln_info(fname + ssprintf(": Wall to Wall Corr Result for Traj=%d:", traj));
      main_displayln_info(show_vec_complex(wall_wall_corr));

      // add to avg
      wall_wall_corr_avg = wall_wall_corr_avg + wall_wall_corr;
      // cnt++;

      // save
      write_data_from_0_node(&wall_wall_corr[0], wall_wall_corr.size(), ensemble_info.get_traj_path(traj));

      release_lock();
    }
  }
  main_displayln_info(std::string(fname) + ": Compute Wall to Wall Corr for All Traj End");

  // wall_wall_corr_avg = wall_wall_corr_avg / cnt;
  // main_displayln_info(std::string(fname) + ssprintf(": Wall Wall Corr Avg Result Over %d Trajs:", cnt));
  // main_displayln_info(show_vec_complex(wall_wall_corr_avg));
}
#endif

void test_gen_coor() {
  int m = 0;
  int a = 2;
  int left = 5;
  int mid = 50;
  int right = 100;
  DistFunc dist_func(m, a, left, mid, right);
  double norm = dist_func.norm;
  main_displayln_info(ssprintf("norm %24.17E", norm));

  std::string str_for_rs = "test";
  Coordinate init_coor(11, 11, 11, 11);
  Coordinate range(11, 11, 11, 11);
  int sleep = 100;
  int stats[100];

  for (int i = 0; i < 100; ++i) {
    stats[i] = 0;
  }
  GenCoorDist gen_coor_dist(dist_func, str_for_rs, init_coor, range, sleep);
  for (int i = 0; i < 10000; ++i) {
    CoordinateD curr_coor = gen_coor_dist.get_next_coor();
    stats[int(r_coorD(curr_coor))] += 1;
  }
  for (int i = 0; i < 100; ++i) {
    main_displayln_info(ssprintf("[%d], %d, %f, %f", i, stats[i], 1. * stats[i] / 10000, dist_func.compute_r_p((double) i)));
  }

  std::vector<double> theta;
  theta = rotate_thetas_from_0001_to(Coordinate(0, 0, 0, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(Coordinate(1, 0, 0, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(Coordinate(-1, 0, 0, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(Coordinate(0, 1, 0, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(Coordinate(0, -1, 0, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(Coordinate(0, 0, 1, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(Coordinate(0, 0, -1, 0));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(CoordinateD(0, 0, 0, 0.0000001));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(CoordinateD(0.287, -379.281, 2392.573, -23146763129478.238478));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(CoordinateD(2.32, -3.46, 5.01, 10.5));
  main_displayln_info(show_vec_double(theta));
  theta = rotate_thetas_from_0001_to(CoordinateD(27.03, -47.82, -46.17, 11.26));
  main_displayln_info(show_vec_double(theta));

  sync_node();
  return;
}

int main(int argc, char* argv[])
{
  begin(&argc, &argv);

  init_muon_line();
  compute_f2_model("physical-24nt96-1.0", "", 12);
  compute_f2_model("physical-32nt128-1.0", "", 16);
  compute_f2_model("physical-32nt128-1.3333", "", 16);
  compute_f2_model("physical-48nt192-1.0", "", 24);
  compute_f2_model("physical-48nt192-2.0", "", 24);

  compute_f2_model("heavy-24nt96-1.0", "", 12);
  compute_f2_model("heavy-32nt128-1.0", "", 16);
  compute_f2_model("heavy-32nt128-1.3333", "", 16);
  compute_f2_model("heavy-48nt192-1.0", "", 24);
  compute_f2_model("heavy-48nt192-2.0", "", 24);
  exit(0);

  initialize();
  compute_model_fr_interpolation("physical-24nt96-1.0", 4., 200, 100, 100);
  compute_model_fr_interpolation("physical-32nt128-1.0", 4., 200, 100, 100);
  compute_model_fr_interpolation("physical-32nt128-1.3333", 4., 200, 100, 100);
  compute_model_fr_interpolation("physical-48nt192-1.0", 4., 200, 100, 100);
  compute_model_fr_interpolation("physical-48nt192-2.0", 4., 200, 100, 100);

  compute_model_fr_interpolation("heavy-24nt96-1.0", 4., 200, 100, 100);
  compute_model_fr_interpolation("heavy-32nt128-1.0", 4., 200, 100, 100);
  compute_model_fr_interpolation("heavy-32nt128-1.3333", 4., 200, 100, 100);
  compute_model_fr_interpolation("heavy-48nt192-1.0", 4., 200, 100, 100);
  compute_model_fr_interpolation("heavy-48nt192-2.0", 4., 200, 100, 100);
  exit(0);

  // compute_f2_all_traj_pairs("32D-0.00107-physical-pion", "ama", "", 16);
  // compute_f2_all_traj_pairs("48I-0.00078-physical-pion", "ama", "", 24);
  // compute_f2_all_traj_pairs("24D-0.00107-physical-pion", "ama", "", 12);
  // compute_f2_all_traj_pairs("32Dfine-0.0001-physical-pion", "ama", "", 16);
  // compute_f2_all_traj_pairs("24D-0.0174-physical-pion", "ama", "", 12);
  exit(0);

  compute_fr_interpolation_all_traj("48I-0.00078", "ama", 4., 80, 20, 20);
  compute_fr_interpolation_all_traj("24D-0.00107", "ama", 4., 80, 20, 20);
  compute_fr_interpolation_all_traj("32D-0.00107", "ama", 4., 80, 20, 20);
  compute_fr_interpolation_all_traj("32Dfine-0.0001", "ama", 4., 80, 20, 20);

  compute_fr_all_traj("48I-0.00078", "ama", 24 * 24);
  compute_fr_all_traj("24D-0.00107", "ama", 144);
  compute_fr_all_traj("32D-0.00107", "ama", 256);
  compute_fr_all_traj("32Dfine-0.0001", "ama", 256);
  exit(0);

  initialize();
  compute_f2_all_traj_pairs("24D-0.00107-refine-field", "ama", "", 12);
  exit(0);

  // test_gen_coor();
  initialize();
  compute_f2_all_traj_pairs("48I-0.00078-physical-pion", "ama", "", 24);
  exit(0);

  initialize();
  compute_f2_all_traj_pairs("24D-0.0174-physical-pion", "ama", "", 12);
  exit(0);
  compute_f2_all_traj_pairs("24D-0.00107-physical-pion", "ama", "", 12);

  compute_f2_all_traj_pairs("32Dfine-0.0001-physical-pion", "ama", "", 16);
  exit(0);
  compute_f2_all_traj_pairs("32Dfine-0.0001", "ama", "", 16);

  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 24);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 22);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 20);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 18);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 16);
  exit(0);

  initialize();
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 24);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 22);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 20);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 18);
  compute_f2_all_traj_pairs("48I-0.00078", "ama", "", 16);
  exit(0);

  initialize();
  compute_wall_wall_correlator_all_traj("48I-0.00078", "ama");
  exit(0);

  initialize();
  compute_f2_all_traj_pairs("32Dfine-0.0001", "ama", "", 16);
  exit(0);

  const TwoPointWallEnsembleInfo tpw_info("32Dfine-0.0001", "ama");
  if (tpw_info.is_traj_computed(260)) {
    main_displayln_info("Yes");
  } else {
    main_displayln_info("No");
  }
  exit(0);

  compute_wall_wall_correlator_all_traj("24D-0.0174", "ama");
  exit(0);

  initialize();
  compute_f2_all_traj_pairs("24D-0.0174", "ama", "", 10);
  compute_f2_all_traj_pairs("24D-0.0174", "ama", "", 12);
  exit(0);

  initialize();
  compute_f2_all_traj_pairs("32Dfine-0.0001", "ama", "", 16);
  exit(0);

  compute_wall_wall_correlator_all_traj("32D-0.00107", "ama");
  compute_wall_wall_correlator_all_traj("24D-0.00107", "ama");
  compute_wall_wall_correlator_all_traj("32Dfine-0.0001", "ama");
  exit(0);

  compute_point_point_wall_correlator("32D-0.00107", "ama");
  compute_point_point_wall_correlator("24D-0.00107", "ama");

  initialize();

  compute_f2_all_traj_pairs("32Dfine-0.0001", "ama", "", 14);
  compute_f2_all_traj_pairs("32Dfine-0.0001", "ama", "", 12);
  compute_f2_all_traj_pairs("32Dfine-0.0001", "ama", "", 10);

  compute_f2_all_traj_pairs("32D-0.00107", "ama", "", 14);
  compute_f2_all_traj_pairs("32D-0.00107", "ama", "", 12);
  compute_f2_all_traj_pairs("32D-0.00107", "ama", "", 10);

  compute_f2_all_traj_pairs("24D-0.00107", "ama", "", 12);
  compute_f2_all_traj_pairs("24D-0.00107", "ama", "", 10);

  return 0;
}

