//#pragma once
#ifndef _QLAT-ANALYSIS_H
#define _QLAT-ANALYSIS_H

#include <qlat/config.h>
#include <qlat/cache.h>
#include <qlat/utils.h>
#include <qlat/mvector.h>
#include <qlat/matrix.h>
#include <qlat/mpi.h>
#include <qlat/utils-io.h>
#include <qlat/field.h>
#include <qlat/field-utils.h>
#include <qlat/field-fft.h>
#include <qlat/field-rng.h>
#include <qlat/field-comm.h>
#include <qlat/field-serial-io.h>
#include <qlat/field-dist-io.h>
#include <qlat/field-expand.h>
#include <qlat/qed.h>
#include <qlat/qcd.h>
#include <qlat/qcd-utils.h>
#include <qlat/qcd-gauge-transformation.h>
#include <qlat/qcd-smear.h>

QLAT_START_NAMESPACE
#if 0 
struct ConfigurationInfo
{
  int traj;
  std::string path;
  std::string conf_format;
};

struct ConfigurationsInfo
{
  Coordinate total_site;
  std::vector<ConfigurationInfo> infos;
};

inline void load_configuration(GaugeField& gf, const ConfigurationInfo& ci)
{
  if (ci.conf_format == "milc") {
    load_gauge_field_milc(gf, ci.path);
  } else {
    load_gauge_field(gf, ci.path);
  }
}

inline std::string& get_result_path()
{
  static std::string path = ".";
  return path;
}
#endif

inline std::string mk_new_log_path()
{
  const std::string path = get_result_path();
  for (long i = 0; i < 9999; ++i) {
    const std::string fn = ssprintf("%s/logs/log.%04d.txt", path.c_str(), i);
    if (!does_file_exist(fn)) {
      return fn;
    }
  }
  return path + "/logs/log.txt";
}

inline double get_time_limit_from_env()
{
  const std::string ss = get_env("COBALT_STARTTIME");
  const std::string se = get_env("COBALT_ENDTIME");
  if (ss != "" and se != "") {
    TIMER_VERBOSE("get_time_limit_from_env");
    double start_time, end_time;
    reads(start_time, ss);
    reads(end_time, se);
    return end_time - start_time;
  } else {
    // SADJUST ME
    return 2.0 * 60.0 * 60.0;
  }
}

inline void setup(const std::string& name)
{
  static FILE* log = NULL;
  get_result_path() = name;
  qmkdir_sync_node(get_result_path());
  qmkdir_sync_node(get_result_path() + "/logs");
  if (0 == get_id_node()) {
    FILE* new_log = qopen(mk_new_log_path(), "a");
    qassert(new_log != NULL);
    qset_line_buf(new_log);
    get_output_file() = new_log;
    get_monitor_file() = stdout;
    qclose(log);
    log = new_log;
  }
  get_global_rng_state() = RngState("e7958f3917469d569c9f5658dd64bcacfc226b2e5b039e4ac970beea1f2108bf").split(name);
  Timer::max_function_name_length_shown() = 50;
  Timer::max_call_times_for_always_show_info() = 3;
  Timer::minimum_duration_for_show_stop_info() = 60;
  Timer::minimum_autodisplay_interval() = 365 * 24 * 3600;
  get_time_limit() = get_time_limit_from_env();
  get_lock_expiration_time_limit() = get_time_limit();
  get_default_budget() = 0;
  dist_write_par_limit() = 128;
  dist_read_par_limit() = 128;
  displayln_info("get_time_limit() = " + show(get_time_limit()));
}

inline ConfigurationsInfo make_configurations_info_milc()
{
  ConfigurationsInfo csi;
  csi.total_site = Coordinate(24, 24, 24, 64);
  for (int traj = 500; traj <= 1900; traj += 100) {
    ConfigurationInfo ci;
    ci.traj = traj;
    ci.path = get_env("HOME") +
      "/qcdarchive-milc/24c64"
      "/2plus1plus1"
      "/l2464f211b600m0102m0509m635a.hyp." + show(traj);
    ci.conf_format = "milc";
    csi.infos.push_back(ci);
  }
  return csi;
}

inline ConfigurationsInfo make_configurations_info_16c32_mu0p01_ms0p04()
{
  ConfigurationsInfo csi;
  csi.total_site = Coordinate(16, 16, 16, 32);
  for (int traj = 1000; traj <= 4000; traj += 100) {
    ConfigurationInfo ci;
    ci.traj = traj;
    ci.path = get_env("HOME") +
      "/qcdarchive/DWF_iwa_nf2p1/16c32"
      "/2plus1_16nt32_IWASAKI_b2p13_ls16_M1p8_ms0p04_mu0p01_rhmc_multi_timescale_ukqcd"
      "/ckpoint_lat.IEEE64BIG." + show(traj);
    ci.conf_format = "cps";
    csi.infos.push_back(ci);
  }
  return csi;
}

inline ConfigurationsInfo make_configurations_info_test()
{
  ConfigurationsInfo csi;
  // csi.total_site = Coordinate(16, 16, 16, 32);
  // csi.total_site = Coordinate(32, 32, 32, 64);
  csi.total_site = Coordinate(48, 48, 48, 96);
  // csi.total_site = Coordinate(64, 64, 64, 128);
  return csi;
}

inline ConfigurationsInfo make_configurations_info()
{
  // ADJUST ME
  // return make_configurations_info_16c32_mu0p01_ms0p04();
  // return make_configurations_info_milc();
  return make_configurations_info_test();
}

inline int qrename_partial(const std::string& path)
{
  TIMER("qrename_partial");
  return qrename(path + ".partial", path);
}

inline int qrename_partial_info(const std::string& path)
{
  TIMER("qrename_partial_info");
  return qrename_info(path + ".partial", path);
}

#if 0
inline std::string find_conf_24c64_dsdr_mu0p0017_ms0p0850(const int traj)
{
  const std::string parent_path = get_env("HOME") + "/qcdarchive-jtu/24nt64/IWASAKI+DSDR/b1.633/ls24/M1.8/ms0.0850/ml0.00107";
  const std::string fname = "/ckpoint_lat.";
  std::string path;
  path = parent_path +
    "/evol0/configurations"
    + fname + show(traj);
  if (does_file_exist_sync_node(path)) {
    return path;
  }
  path = parent_path +
    "/evol1/configurations"
    + fname + show(traj);
  if (does_file_exist_sync_node(path)) {
    return path;
  }
  path = parent_path +
    "/evol2/configurations"
    + fname + show(traj);
  if (does_file_exist_sync_node(path)) {
    return path;
  }
  return "";
}
#endif

QLAT_END_NAMESPACE
#endif
