#ifndef _WALLWALL_H
#define _WALLWALL_H

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

Complex pi_pi_contraction(const WilsonMatrix& wm_from_1_to_2, const WilsonMatrix& wm_from_2_to_1)
{
  return -matrix_trace((SpinMatrix)(ii * gamma5) * (WilsonMatrix)(wm_from_2_to_1 * (SpinMatrix)(ii * gamma5) * wm_from_1_to_2));
}

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

QLAT_END_NAMESPACE

#endif
