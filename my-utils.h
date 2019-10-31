#ifndef _MY-UTILS_H
#define _MY-UTILS_H

#include <qlat/qlat.h>
#include <qlat/qlat-analysis.h>
#include <qlat/field-utils.h>
#include <gsl/gsl_sf_bessel.h>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <dirent.h>
#include "muon-line.h"
#include <gsl/gsl_integration.h>

#include <map>
#include <vector>

QLAT_START_NAMESPACE

void main_displayln_info(const std::string str) {
  const std::string out_str = "main:: " + str;
  displayln_info(out_str);
  return;
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

template <class T>
inline void set_zero(T& x)
{
  memset(&x, 0, sizeof(T));
}

template <class M>
void write_data_from_0_node(const M& data, std::string path)
{
  sync_node();
  if (0 == get_rank() && 0 == get_thread_num())
  {
    FILE* fp = qopen(path, "w");
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
    FILE* fp = qopen(path, "w");
    std::fwrite((void*) data, sizeof(M), size, fp);
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

std::string show_vec_complex(std::vector<Complex> vec)
{
  std::string res = "";
  for (int i = 0; i < vec.size(); ++i)
  {
    res += ssprintf("%24.17E %24.17E, ", vec[i].real(), vec[i].imag());
  }
  return res;
}

std::string show_vec_double(std::vector<double> vec)
{
  std::string res = "";
  for (int i = 0; i < vec.size(); ++i)
  {
    res += ssprintf("%24.17E, ", vec[i]);
  }
  return res;
}

inline std::string show_coordinate(const Coordinate& c)
{
  return ssprintf("(%d,%d,%d,%d)", c[0], c[1], c[2], c[3]);
}

inline std::string show_coordinateD(const CoordinateD& c)
{
  return ssprintf("(%f,%f,%f,%f)", c[0], c[1], c[2], c[3]);
}

double r_coor(const Coordinate& coor)
{
  return sqrt(sqr((long)coor[0]) + sqr((long)coor[1]) + sqr((long)coor[2]) + sqr((long)coor[3]));
}

double r_coorD(const CoordinateD& coor)
{
  return sqrt(sqr((double)coor[0]) + sqr((double)coor[1]) + sqr((double)coor[2]) + sqr((double)coor[3]));
}

double dist_r_func (double x, void * params) {
  double m = *(double *) params;
  double a = *((double *) params + 1);
  double left = *((double *) params + 2);
  double mid = *((double *) params + 3);
  double right = *((double *) params + 4);
  if (left <= x && x < mid) {
    return 1.0 * exp(-(double)m * (double)mid) / pow((double)mid, (double)a);
  } else if (mid <= x && x < right) {
    return 1.0 * exp(-(double)m * (double)x) / pow((double)x, (double)a);
  } else {
    return 0.;
  }
}

double dist_x_func (double x, void * params) {
  return dist_r_func(x, params) / (pow(x, 3.) * 2. * pow(PI, 2.));
}

class DistFunc {

  public:
    double params[5];
    double norm;

    DistFunc() {
      return;
    }

    DistFunc(double m, double a, double left, double mid, double right) {
      params[0] = m;
      params[1] = a;
      params[2] = left;
      params[3] = mid;
      params[4] = right;
      norm = compute_sphere_dist_norm();
      return;
    }

    DistFunc(const DistFunc& other) {
      params[0] = other.params[0];
      params[1] = other.params[1];
      params[2] = other.params[2];
      params[3] = other.params[3];
      params[4] = other.params[4];
      norm = other.norm;
      return;
    }

    DistFunc& operator=(const DistFunc& other) {
      params[0] = other.params[0];
      params[1] = other.params[1];
      params[2] = other.params[2];
      params[3] = other.params[3];
      params[4] = other.params[4];
      norm = other.norm;
      return *this;
    }

    double compute_sphere_dist_norm() {
      gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
      double result, error;

      gsl_function F;
      F.function = &dist_r_func;
      F.params = params;
      gsl_integration_qags (&F, params[2], params[4], 0, 1e-13, 10000, w, &result, &error);
      return result;
    }

    double compute_r_p(double r) {
      return dist_r_func(r, params) / norm;
    }

    double compute(double x) {
      return dist_x_func(x, params) / norm;
    }
};

CoordinateD gen_rand_coor(const CoordinateD& coor, const CoordinateD& range, RngState& rs)
{
  CoordinateD step(
      u_rand_gen(rs, range[0], -range[0]),
      u_rand_gen(rs, range[1], -range[1]),
      u_rand_gen(rs, range[2], -range[2]),
      u_rand_gen(rs, range[3], -range[3])
      );
  return coor + step;
}

class GenCoorDist {
  private:
    DistFunc DIST;
    RngState RS;
    CoordinateD RANGE;
    int SLEEP;
    int HEAT_STEP = 100;
    CoordinateD curr_coor;

  public:
    GenCoorDist() {
      return;
    }

    GenCoorDist(
        const DistFunc& dist, 
        const std::string& str_for_rs,
        const CoordinateD& init_coor,
        const CoordinateD& range, 
        const int sleep) {
      DIST = dist;
      RS = RngState(str_for_rs);;
      curr_coor = init_coor;
      RANGE = range;
      SLEEP = sleep;
      heat();
      return;
    }

    bool is_accept(const CoordinateD& old_coor, const CoordinateD& new_coor) {
      double pnew_pold = DIST.compute(r_coorD(new_coor)) / DIST.compute(r_coorD(old_coor));
      if (pnew_pold >= 1.) {
        return true;
      } else {
        double rand = u_rand_gen(RS, 1.0, 0.0);
        if (rand <= pnew_pold) {
          return true;
        } else {
          return false;
        }
      }
    }

    void update() {
      for (int i = 0; i < SLEEP; ++i) {
        CoordinateD new_coor = gen_rand_coor(curr_coor, RANGE, RS);
        if (is_accept(curr_coor, new_coor)) {
          curr_coor = new_coor;
        }
      }
      return;
    }

    void heat() {
      for (int i = 0; i < HEAT_STEP; ++i) {
        update();
      }
      return;
    }

    CoordinateD get_next_coor() {
      update();
      return curr_coor;
    }
};

double cut_1(double x) {
  if (1.0 < x) {
    return 1.0;
  } else if (x < -1.0) {
    return -1.0;
  } else {
    return x;
  }
}

std::vector<double> rotate_thetas_from_0001_to(const CoordinateD& coor) {
  std::vector<double> res (3, 0.);
  if (r_coorD(coor) < 1e-16) {
    return res;
  }
  CoordinateD vec = coor / r_coorD(coor);

  double theta_xy = acos(cut_1(vec[3]));
  if (theta_xy == 0.) {
    return res;
  }
  res[0] = theta_xy;

  double theta_xt = acos(cut_1(-vec[2] / sin(theta_xy)));
  if (theta_xt == 0.) {
    return res;
  }
  res[1] = theta_xt;

  double cos_theta_zt = vec[1] / sin(theta_xy) / sin(theta_xt);
  double sin_theta_zt = -vec[0] / sin(theta_xy) / sin(theta_xt);
  double theta_zt;
  if (sin_theta_zt > 0.) {
    theta_zt = acos(cut_1(cos_theta_zt));
  } else if (sin_theta_zt < 0.) {
    theta_zt = -acos(cut_1(cos_theta_zt));
  } else {
    theta_zt = 0.;
  }
  res[2] = theta_zt;
  return res;
}

QLAT_END_NAMESPACE

#endif
