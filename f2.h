#ifndef _F2_H
#define _F2_H

#include <qlat/qlat.h>
#include <qlat/qlat-analysis.h>
#include <qlat/field-utils.h>
#include <gsl/gsl_sf_bessel.h>
#include <dirent.h>
#include <fstream>
#include <math.h>
#include <dirent.h>
#include "muon-line.h"
#include "hlbl-utils.h"
#include "point-point-wall.h"
#include "ensemble.h"

#include <map>
#include <vector>

QLAT_START_NAMESPACE

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

  RotationMatrix(const std::vector<double>& thetas)
  {
    qassert(thetas.size() == 3);
    double theta_xy = thetas[0];
    double theta_xt = thetas[1];
    double theta_zt = thetas[2];
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

struct YAndRotationInfo
{
  int chunk_size = 1024;
  int step = 1024;

  int m;
  int a;
  int left;
  int mid;
  int right;
  DistFunc dist_func;

  std::string str_for_rs;
  CoordinateD init_coor;
  CoordinateD range;
  int sleep;
  GenCoorDist gen_coor_dist;

  void show_info() const {
    main_displayln_info("YAndRotationInfo:");
    main_displayln_info(ssprintf("chunk_size: %d", chunk_size));
    main_displayln_info(ssprintf("step: %d", step));
    main_displayln_info(ssprintf("dist func params: m %d, a %d, left %d, mid %d, right %d", m, a, left, mid, right));
    main_displayln_info("str_for_rs: " + str_for_rs);
    main_displayln_info("init_coor: " + show_coordinateD(init_coor));
    main_displayln_info("range: " + show_coordinateD(range));
    main_displayln_info(ssprintf("sleep: %d", sleep));
    return;
  }

  YAndRotationInfo(const std::string& ensemble_) {
    const std::string ENSEMBLE = ensemble_;
    qassert(step >= chunk_size);

    if (ENSEMBLE == "24D-0.00107" || ENSEMBLE == "24D-0.00107-physical-pion" || ENSEMBLE == "24D-0.00107-refine-field") {
      m = 0; a = 2; left = 5; mid = 40; right = 60;
    } else if (ENSEMBLE == "24D-0.0174" || ENSEMBLE == "24D-0.0174-physical-pion" || ENSEMBLE == "24D-0.0174-refine-field") {
      m = 0; a = 2; left = 5; mid = 40; right = 60;
    } else if (ENSEMBLE == "32D-0.00107" || ENSEMBLE == "32D-0.00107-physical-pion" || ENSEMBLE == "32D-0.00107-refine-field") {
      m = 0; a = 2; left = 5; mid = 40; right = 60;
    } else if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "32Dfine-0.0001-physical-pion" || ENSEMBLE == "32Dfine-0.0001-refine-field") {
      m = 0; a = 2; left = 5; mid = 55; right = 85;
    } else if (ENSEMBLE == "48I-0.00078" || ENSEMBLE == "48I-0.00078-physical-pion" || ENSEMBLE == "48I-0.00078-refine-field") {
      m = 0; a = 2; left = 5; mid = 70; right = 110;
    } else {
      qassert(false);
    }
    dist_func = DistFunc(m, a, left, mid, right);

    str_for_rs = ENSEMBLE;
    init_coor = CoordinateD(10, 10, 10, 10);
    range = CoordinateD(5, 5, 5, 5);
    sleep = 1000;
    gen_coor_dist = GenCoorDist(dist_func, str_for_rs, init_coor, range, sleep);
    show_info();
  }

  std::vector<Y_And_Rotation_Info_Elem> get_next_y_info_list() {
    std::vector<Y_And_Rotation_Info_Elem> info_sub_list(chunk_size);
    for (int i = 0; i < step; ++i) {
      CoordinateD coor = gen_coor_dist.get_next_coor();
      if (i < chunk_size) {
        double p = dist_func.compute(r_coorD(coor));
        std::vector<double> thetas = rotate_thetas_from_0001_to(coor);
        Y_And_Rotation_Info_Elem elem;
        elem.y_large = coor;
        elem.dist = p;
        elem.theta_xy = thetas[0];
        elem.theta_xt = thetas[1];
        elem.theta_zt = thetas[2];
        info_sub_list[i] = elem;
      }
    }
    return info_sub_list;
  }
};

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
  // int TRAJ_JUMP;

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
    if (ENSEMBLE == "24D-0.00107" || ENSEMBLE == "24D-0.00107-physical-pion" || ENSEMBLE == "24D-0.00107-refine-field") {
      // TRAJ_JUMP = 50;
      NUM_RMAX = 80;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "24D-0.0174" || ENSEMBLE == "24D-0.0174-physical-pion" || ENSEMBLE == "24D-0.0174-refine-field") {
      // TRAJ_JUMP = 50;
      NUM_RMAX = 80;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "32D-0.00107" || ENSEMBLE == "32D-0.00107-physical-pion" || ENSEMBLE == "32D-0.00107-refine-field") {
      // TRAJ_JUMP = 50;
      NUM_RMAX = 80;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "32Dfine-0.0001-physical-pion" || ENSEMBLE == "32Dfine-0.0001-refine-field") {
      // TRAJ_JUMP = 50;
      NUM_RMAX = 100;
      NUM_RMIN = 40;
    } else if (ENSEMBLE == "48I-0.00078" || ENSEMBLE == "48I-0.00078-physical-pion" || ENSEMBLE == "48I-0.00078-refine-field") {
      // TRAJ_JUMP = 60;
      NUM_RMAX = 120;
      NUM_RMIN = 60;
    } else {
      qassert(false);
    }
    std::string file_name = "ensemble:" + ENSEMBLE + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    if (ENSEMBLE == "24D-0.00107-physical-pion" || ENSEMBLE == "24D-0.00107-refine-field") {
      file_name = "ensemble:24D-0.00107" + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    } else if (ENSEMBLE == "24D-0.0174-physical-pion" || ENSEMBLE == "24D-0.0174-refine-field") {
      file_name = "ensemble:24D-0.0174" + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    } else if (ENSEMBLE == "32D-0.00107-physical-pion" || ENSEMBLE == "32D-0.00107-refine-field") {
      file_name = "ensemble:32D-0.00107" + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    } else if (ENSEMBLE == "32Dfine-0.0001-physical-pion" || ENSEMBLE == "32Dfine-0.0001-refine-field") {
      file_name = "ensemble:32Dfine-0.0001" + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    } else if (ENSEMBLE == "48I-0.00078-physical-pion" || ENSEMBLE == "48I-0.00078-refine-field") {
      file_name = "ensemble:48I-0.00078" + ssprintf("_start:%d_end:%d_step:10_numpairs:10000_seplimit:50", TRAJ_START, TRAJ_END);
    }
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
    // main_displayln_info(ssprintf("TRAJ_JUMP: %d", TRAJ_JUMP));

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
  qassert(num_rmax == bm_table.bm.size() && num_rmin == bm_table.bm[0].size());
#pragma omp parallel for
  for (int rmax = 0; rmax < num_rmax; ++rmax) {
    for (int rmin = 0; rmin < num_rmin; ++rmin) {
      const PionGGElem& bm = bm_table.bm[rmax][rmin];
      Complex& e_xbbm = e_xbbm_table.c[rmax][rmin];
      e_xbbm = find_e_xbbm(xb, bm);
    }
  }
}

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
  info += ssprintf("prop: %24.17E %24.17E\n", prop.real(), prop.imag());
  info += "show one pair e_xbbm_table with prop and dist:\n";
  info += show_complex_table(e_xbbm_table);
  main_displayln_info(info);

  return e_xbbm_table;
}

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
    const Y_And_Rotation_Info_Elem& one_y = y_info[i];
    main_displayln_info(show_y_and_rotation_info_elem(one_y));

    // check
    if (f2_ensemble_info.is_traj_pair_i_computed(traj_pair, i)) {
      main_displayln_info(fname + ssprintf(": Traj Pair %04d, %04d, One Y i=%d Have Been Computed", traj_pair[0], traj_pair[1], i));
      continue;
    }

    // compute
    main_displayln_info(fname + ssprintf(": Compute One Y %d/%d From Traj Pair %04d, %04d", i, y_info.size(), traj_pair[0], traj_pair[1]));
    Complex_Table one_pair_table = find_one_y_f2_table_rotation_pgge(f2_ensemble_info, one_y, pgge_field_1, pgge_field_2);

    // save
    std::string one_y_save_path = f2_ensemble_info.get_traj_pair_i_path(traj_pair, i);
    one_pair_table.write_to_disk(one_y_save_path);

    f2_table += one_pair_table;
  }
  f2_table = 1. / y_info.size() * f2_table;
  main_displayln_info(std::string(fname) + ": f2 avg");
  main_displayln_info(show_complex_table(f2_table));
  return f2_table;
}

void refine_field(PionGGElemField& field) {
  TIMER_VERBOSE("refine_field");
  const Geometry& geo = field.geo;
  const Coordinate total_site = geo.total_site();

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lx = geo.coordinate_from_index(index);
    Coordinate x = geo.coordinate_g_from_l(lx);
    x = relative_coordinate(x, total_site);

    PionGGElem& pgge = field.get_elem(lx);

    Complex f(0., 0.);
#if 0
    int cnt = 0;
    for (int rho = 0; rho < 4; ++rho) {
      if (x[rho] == 0) {continue;}

      int aa = 0;
      for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
          aa += epsilon_tensor(mu, nu, rho, 3) * epsilon_tensor(mu, nu, rho, 3);
        }
      }
      if (aa == 0) {continue;}

      Complex la = Complex(0., 0.);
      for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
          la += pgge.v[mu][nu] * (double) epsilon_tensor(mu, nu, rho, 3);
        }
      }

      f += la / (double) aa / (double) x[rho];
      cnt += 1;
    }

    if (cnt == 0) {continue;}

    f /= (double) cnt;
#endif

    int a[4][4];
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        a[mu][nu] = 0;
        for (int rho = 0; rho < 4; ++rho) {
          a[mu][nu] += epsilon_tensor(mu, nu, rho, 3) * x[rho];
        }
      }
    }

    int aa = 0;
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        aa += a[mu][nu] * a[mu][nu];
      }
    }
    if (aa == 0) {continue;}

    Complex la = Complex(0., 0.);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        la += pgge.v[mu][nu] * (double) a[mu][nu];
      }
    }
    f = la / (double) aa;

    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        pgge.v[mu][nu] = Complex(0., 0.);
        for (int rho = 0; rho < 4; ++rho) {
          pgge.v[mu][nu] += epsilon_tensor(mu, nu, rho, 3) * (double) x[rho] * f;
        }
      }
    }
  }
  sync_node();
}

void compute_f2_all_traj_pairs(const std::string ensemble, const std::string accuracy, const std::string mod, const int xxp_limit)
{
  TIMER_VERBOSE("compute_f2_all_traj_pairs");
  init_muon_line();

  std::string tpw_ensemble = ensemble;
  if (ensemble == "24D-0.00107-physical-pion" || ensemble == "24D-0.00107-refine-field") {tpw_ensemble = "24D-0.00107";}
  if (ensemble == "24D-0.0174-physical-pion" || ensemble == "24D-0.0174-refine-field") {tpw_ensemble = "24D-0.0174";}
  if (ensemble == "32D-0.00107-physical-pion" || ensemble == "32D-0.00107-refine-field") {tpw_ensemble = "32D-0.00107";}
  if (ensemble == "32Dfine-0.0001-physical-pion" || ensemble == "32Dfine-0.0001-refine-field") {tpw_ensemble = "32Dfine-0.0001";}
  if (ensemble == "48I-0.00078-physical-pion" || ensemble == "48I-0.00078-refine-field") {tpw_ensemble = "48I-0.00078";}

  const TwoPointWallEnsembleInfo tpw_info(tpw_ensemble, accuracy);
  F2EnsembleInfo f2_ensemble_info(ensemble, accuracy, mod, xxp_limit);

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

      if (
          ensemble == "24D-0.00107-refine-field" ||
          ensemble == "24D-0.0174-refine-field" ||
          ensemble == "32D-0.00107-refine-field" ||
          ensemble == "32Dfine-0.0001-refine-field" ||
          ensemble == "48I-0.00078-refine-field"
          ) {
        refine_field(two_point_wall_1);
        refine_field(two_point_wall_2);
      }

      compute_f2_one_traj_pair_and_save(traj_pair, f2_ensemble_info, y_info, two_point_wall_1, two_point_wall_2);
      release_lock();
    }
  }
}

QLAT_END_NAMESPACE

#endif
