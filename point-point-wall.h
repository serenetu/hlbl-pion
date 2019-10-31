#ifndef _POINT-POINT-WALL_H
#define _POINT-POINT-WALL_H

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
#include "ensemble.h"

#include <map>
#include <vector>

QLAT_START_NAMESPACE

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
      NUM_POINT_SLOPPY = 1024;
      NUM_POINT_EXACT_1 = 32;
      NUM_POINT_EXACT_2 = 8;

      NUM_WALL_SLOPPY = 64;
      NUM_WALL_EXACT = 2;
    } else if (ENSEMBLE == "48I-0.00078") {
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
    return "/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/" + ENSEMBLE + ssprintf("/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
#if 0
    if (ENSEMBLE == "32Dfine-0.0001") {
      return ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/32Dfine-0.0001/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    } else if (ENSEMBLE == "24D-0.0174") {
      return ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/24D-0.0174/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    } else if (ENSEMBLE == "48I-0.00078") {
      return ssprintf("/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/48I-0.00078/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    }
    return FIELD_ENSEMBLE_ACCURACY_TMIN_OUT_PATH + ssprintf("/results=%04d", traj);
#endif
  }

  std::string get_field_traj_avg_dir(const int traj) const {
    return get_field_traj_dir(traj);
#if 0
    if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "24D-0.0174" || ENSEMBLE == "48I-0.00078") {
      return get_field_traj_dir(traj);
    }
    std::string traj_dir = get_field_traj_dir(traj);
    return traj_dir + ssprintf("/avg ; type=%d", TYPE);
#endif
  }

  void load_field_traj_avg(PionGGElemField& field, const int traj) const {
    const std::string path = get_field_traj_avg_dir(traj);
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
#if 0
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
#endif
  }
  
  void make_field_traj_dir(const int traj) const {
    const std::string field_traj_path = get_field_traj_dir(traj);
    qmkdir_sync_node(field_traj_path);
  }

  bool is_traj_computed(const int traj) const {
    std::string decay_cheng = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/" + ENSEMBLE + ssprintf("/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
    if (does_file_exist_sync_node(decay_cheng) || does_file_exist_sync_node(decay_cheng + "/checkpoint")) {
      return true;
    } else {
      return false;
    }
#if 0
    if (ENSEMBLE == "32Dfine-0.0001" || ENSEMBLE == "24D-0.0174" || ENSEMBLE == "48I-0.00078") {
      std::string decay_cheng = "/home/ljin/application/Public/Qlat-CPS-cc/jobs/em-corr/results/" + ENSEMBLE + ssprintf("/results=%d/contraction-with-point/pion_gg/decay_cheng", traj);
      if (does_file_exist_sync_node(decay_cheng) || does_file_exist_sync_node(decay_cheng + "/checkpoint")) {
        return true;
      } else {
        return false;
      }
    }
    return does_file_exist_sync_node(get_field_traj_dir(traj) + "/avg_checkpoint");
#endif
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

QLAT_END_NAMESPACE

#endif
