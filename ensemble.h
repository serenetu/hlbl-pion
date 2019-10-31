#ifndef _ENSEMBLE_H
#define _ENSEMBLE_H

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

#include <map>
#include <vector>

QLAT_START_NAMESPACE

inline int get_tmin(const std::string& job_tag)
{
  if (
      job_tag == "24D-0.00107" || 
      job_tag == "24D-0.00107-physical-pion" || 
      job_tag == "24D-0.00107-refine-field" || 
      job_tag == "32D-0.00107" || 
      job_tag == "32D-0.00107-physical-pion" ||
      job_tag == "32D-0.00107-refine-field"
      ) {
    return 10;
  } else if (
      job_tag == "24D-0.0174" || 
      job_tag == "24D-0.0174-physical-pion" || 
      job_tag == "24D-0.0174-refine-field"
      ) {
    return 10;
  } else if (
      job_tag == "32Dfine-0.0001" || 
      job_tag == "32Dfine-0.0001-physical-pion" || 
      job_tag == "32Dfine-0.0001-refine-field"
      ) {
    return 14;
  } else if (
      job_tag == "48I-0.00078" || 
      job_tag == "48I-0.00078-physical-pion" || 
      job_tag == "48I-0.00078-refine-field"
      ) {
    return 16;
  } else {
    qassert(false);
  }
  return 8;
}

double get_ainv(const std::string& ensemble) {
  double ainv = 0.;
  if (ensemble == "24D-0.00107") {
    ainv = 1.015;
  } else if (ensemble == "24D-0.0174") {
    ainv = 1.015;
  } else if (ensemble == "32D-0.00107") {
    ainv = 1.015;
  } else if (ensemble == "32Dfine-0.0001") {
    ainv = 1.378;
  } else if (ensemble == "48I-0.00078") {
    ainv = 1.73;
  } else {
    qassert(false);
  }
  return ainv;
}

double get_a(const std::string& ensemble) {  // in fm
  double ainv = get_ainv(ensemble);
  double a = 1. / ainv * 0.197;  // 1 GeV-1 = .197 fm
  return a;
}

int get_l(const std::string& ensemble) {
  int l = 0;
  if (ensemble == "24D-0.00107") {
    l = 24;
  } else if (ensemble == "24D-0.0174") {
    l = 24;
  } else if (ensemble == "32D-0.00107") {
    l = 32;
  } else if (ensemble == "32Dfine-0.0001") {
    l = 32;
  } else if (ensemble == "48I-0.00078") {
    l = 48;
  } else {
    qassert(false);
  }
  return l;
}

double get_mpi(const std::string ensemble) {
  double m_pi = 0.;
  if (ensemble == "24D-0.00107") {
    m_pi = 0.13975;
  } else if (ensemble == "24D-0.0174") {
    m_pi = 0.3357;
  } else if (ensemble == "32D-0.00107") {
    m_pi = 0.139474;
  } else if (ensemble == "32Dfine-0.0001") {
    m_pi = 0.10468;
  } else if (ensemble == "48I-0.00078") {
    m_pi = 0.08049;
  } else {
    qassert(false);
  }
  return m_pi;
}

double get_fpi(const std::string& ensemble) {
  double f_pi = 0.;
  if (ensemble == "24D-0.00107") {
    f_pi = 0.13055;
  } else if (ensemble == "32D-0.00107") {
    f_pi = 0.13122;
  } else if (ensemble == "32Dfine-0.0001") {
    f_pi = 0.09490;
  } else if (ensemble == "48I-0.00078") {
    // f_pi = 0.07580 / sqrt(2.);
    f_pi = 0.07580;
  } else {
    qassert(false);
  }
  return f_pi;
}

double get_zw(const std::string& ensemble) {
  double zw = 0.;
  if (ensemble == "24D-0.00107") {
    zw = 131683077.512;
  } else if (ensemble == "24D-0.0174") {
    zw = 58760419.01434206;
  } else if (ensemble == "32D-0.00107") {
    zw = 319649623.111;
  } else if (ensemble == "32Dfine-0.0001") {
    zw = 772327306.431;
  } else if (ensemble == "48I-0.00078") {
    zw = 5082918150.729124;
  } else {
    qassert(false);
  }
  return zw;
}

double get_zv(const std::string& ensemble) {
  double zv = 0.;
  if (ensemble == "24D-0.00107") {
    zv = 0.72672;
  } else if (ensemble == "24D-0.0174") {
    zv = 0.72672;
  } else if (ensemble == "32D-0.00107") {
    zv = 0.7260;
  } else if (ensemble == "32Dfine-0.0001") {
    zv = 0.68339;
  } else if (ensemble == "48I-0.00078") {
    zv = 0.71076;
  } else {
    qassert(false);
  }
  return zv;
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
    double physical_pion_mass = 0.1349770;
    if (ENSEMBLE == "24D-0.00107") {
      PION = get_mpi(ENSEMBLE);
      AINV = get_ainv(ENSEMBLE);

      TRAJ_START = 1000;
      TRAJ_END = 3000;
    } else if (ENSEMBLE == "24D-0.00107-physical-pion" || ENSEMBLE == "24D-0.00107-refine-field") {
      AINV = get_ainv("24D-0.00107");
      PION = physical_pion_mass / AINV;

      TRAJ_START = 1000;
      TRAJ_END = 3000;
    } else if (ENSEMBLE == "24D-0.0174") {
      PION = get_mpi(ENSEMBLE);
      AINV = get_ainv(ENSEMBLE);

      TRAJ_START = 200;
      TRAJ_END = 1000;
    } else if (ENSEMBLE == "24D-0.0174-physical-pion" || ENSEMBLE == "24D-0.0174-refine-field") {
      AINV = get_ainv("24D-0.0174");
      PION = physical_pion_mass / AINV;

      TRAJ_START = 200;
      TRAJ_END = 1000;
    } else if (ENSEMBLE == "32D-0.00107") {
      PION = get_mpi(ENSEMBLE);
      AINV = get_ainv(ENSEMBLE);

      TRAJ_START = 680;
      TRAJ_END = 2000;
    } else if (ENSEMBLE == "32D-0.00107-physical-pion" || ENSEMBLE == "32D-0.00107-refine-field") {
      AINV = get_ainv("32D-0.00107");
      PION = physical_pion_mass / AINV;

      TRAJ_START = 680;
      TRAJ_END = 2000;
    } else if (ENSEMBLE == "32Dfine-0.0001") {
      PION = get_mpi(ENSEMBLE);
      AINV = get_ainv(ENSEMBLE);

      TRAJ_START = 200;
      TRAJ_END = 2000;
    } else if (ENSEMBLE == "32Dfine-0.0001-physical-pion" || ENSEMBLE == "32Dfine-0.0001-refine-field") {
      AINV = get_ainv("32Dfine-0.0001");
      PION = physical_pion_mass / AINV;

      TRAJ_START = 200;
      TRAJ_END = 2000;
    } else if (ENSEMBLE == "48I-0.00078") {
      PION = get_mpi(ENSEMBLE);
      AINV = get_ainv(ENSEMBLE);

      TRAJ_START = 500;
      TRAJ_END = 3000;
    } else if (ENSEMBLE == "48I-0.00078-physical-pion" || ENSEMBLE == "48I-0.00078-refine-field") {
      AINV = get_ainv("48I-0.00078");
      PION = physical_pion_mass / AINV;

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


QLAT_END_NAMESPACE

#endif
