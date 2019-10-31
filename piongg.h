#ifndef _PIONGG_H
#define _PIONGG_H

#include <qlat/qlat.h>
#include <math.h>
#include "ensemble.h"
#include "hlbl-utils.h"
#include "my-utils.h"
#include "point-point-wall.h"

QLAT_START_NAMESPACE

struct FREnsembleInfo : public EnsembleInfo {
  std::string ACCURACY;
	bool IS_INTERPOLATION;
  // double F_PI;

  std::string FR_OUT_PATH = "fr";
  std::string FR_ENSEMBLE_ACCURACY_OUT_PATH;

  void init(const std::string& ensemble_, const std::string& accuracy_, const bool is_interpolation_) {
    ACCURACY = accuracy_;
		IS_INTERPOLATION = is_interpolation_;
    qassert(ACCURACY == "ama" || ACCURACY == "sloppy");
    qassert(
        ENSEMBLE == "24D-0.00107" ||
        ENSEMBLE == "32D-0.00107" ||
        ENSEMBLE == "32Dfine-0.0001" ||
        ENSEMBLE == "48I-0.00078"
        )
		if (IS_INTERPOLATION) {
      FR_ENSEMBLE_ACCURACY_OUT_PATH = FR_OUT_PATH + "/" + ENSEMBLE + "-interpolation" + "/" + ACCURACY;
		} else {
      FR_ENSEMBLE_ACCURACY_OUT_PATH = FR_OUT_PATH + "/" + ENSEMBLE + "/" + ACCURACY;
		}
    return;
  }

  void show_info() const {
    main_displayln_info("FREnsembleInfo:");
    main_displayln_info("ACCURACY: " + ACCURACY);
		main_displayln_info(ssprintf("IS_INTERPOLATION %d", IS_INTERPOLATION));
    main_displayln_info("FR_OUT_PATH: " + FR_OUT_PATH);
    main_displayln_info("FR_ENSEMBLE_ACCURACY_OUT_PATH: " + FR_ENSEMBLE_ACCURACY_OUT_PATH);
  }

  FREnsembleInfo(const std::string& ensemble_, const std::string& accuracy_, const bool is_interpolation_=false) : EnsembleInfo(ensemble_) {
    init(ensemble_, accuracy_, is_interpolation_);
    show_info();
  }

  std::string get_fr_traj_path(const int traj) const {
    return FR_ENSEMBLE_ACCURACY_OUT_PATH + ssprintf("/results=%04d", traj);
  }

  bool is_fr_traj_computed(const int traj) const {
    return does_file_exist_sync_node(get_fr_traj_path(traj));
  }

  void make_fr_ensemble_accuracy_dir() const {
    qassert(FR_OUT_PATH != "" && ENSEMBLE != "" && ACCURACY != "");
    qmkdir_sync_node(FR_OUT_PATH);
		if (IS_INTERPOLATION) {
      qmkdir_sync_node(FR_OUT_PATH + "/" + ENSEMBLE + "-interpolation");
		} else {
      qmkdir_sync_node(FR_OUT_PATH + "/" + ENSEMBLE);
		}
    qmkdir_sync_node(FR_ENSEMBLE_ACCURACY_OUT_PATH);
    return;
  }
};

double compute_zp(int l, double m_pi) {
  return 1. / (2. * m_pi * std::pow((double) l, 3.));
}

Complex compute_factor(const std::string& ensemble) {
  double zw = get_zw(ensemble);
  double zp = compute_zp(get_l(ensemble), get_mpi(ensemble));
  double zv = get_zv(ensemble);
  double f_pi = get_fpi(ensemble);
  double m_pi = get_mpi(ensemble);

  double q_u = 2. / 3.;
  double q_d = -1. / 3.;
  Complex res = -1. / sqrt(zw * zp) * std::pow(zv, 2.) * (std::pow(q_u, 2.) - std::pow(q_d, 2.));
  res *= -3. * std::pow(PI, 2.) / f_pi / (Complex(0., 1.) * m_pi);  // first '-' coming from i * i
  return res;
}

std::vector<Coordinate> get_3d_neighbors(const CoordinateD& x, const Coordinate& total_site) {
  CoordinateD x_ = relative_coordinate(x, (CoordinateD) total_site);
  std::vector<Coordinate> neighbors(8, Coordinate(0, 0, 0, 0));
  neighbors[0] = regular_coordinate(
		  relative_coordinate(Coordinate(int(floor(x_[0])), int(floor(x_[1])), int(floor(x_[2])), 0), total_site),
		  total_site);
  neighbors[1] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(1, 0, 0, 0), total_site), total_site);
  neighbors[2] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(0, 1, 0, 0), total_site), total_site);
  neighbors[3] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(1, 1, 0, 0), total_site), total_site);
  neighbors[4] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(0, 0, 1, 0), total_site), total_site);
  neighbors[5] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(1, 0, 1, 0), total_site), total_site);
  neighbors[6] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(0, 1, 1, 0), total_site), total_site);
  neighbors[7] = regular_coordinate(relative_coordinate(neighbors[0] + Coordinate(1, 1, 1, 0), total_site), total_site);
  return neighbors;
}

template <class T>
T compute_1d_interpolation(
    double right_p, double p_left, 
    const T& left, const T& right) 
{
  T res = right_p * left + p_left * right;
  return res;
}

template <class T>
T get_p_val_3d(const CoordinateD& p, const Coordinate& total_site, const FieldM<T, 1>& field) {
  std::vector<Coordinate> neighbors = get_3d_neighbors(p, total_site);

  double x_right_p = smod((double) neighbors[1][0] - (double) p[0], (double) total_site[0]);
  double x_p_left = smod((double) p[0] - (double) neighbors[0][0], (double) total_site[0]);
	qassert(0. <= x_right_p && x_right_p <= 1.);
	qassert(0. <= x_p_left && x_p_left <= 1.);
  T p_01 = compute_1d_interpolation(x_right_p, x_p_left, field.get_elem(neighbors[0]), field.get_elem(neighbors[1]));
  T p_23 = compute_1d_interpolation(x_right_p, x_p_left, field.get_elem(neighbors[2]), field.get_elem(neighbors[3]));
  T p_45 = compute_1d_interpolation(x_right_p, x_p_left, field.get_elem(neighbors[4]), field.get_elem(neighbors[5]));
  T p_67 = compute_1d_interpolation(x_right_p, x_p_left, field.get_elem(neighbors[6]), field.get_elem(neighbors[7]));

  double y_right_p = smod((double) neighbors[2][1] - (double) p[1], (double) total_site[1]);
  double y_p_left = smod((double) p[1] - (double) neighbors[0][1], (double) total_site[1]);
	qassert(0. <= y_right_p && y_right_p <= 1.);
	qassert(0. <= y_p_left && y_p_left <= 1.);
  T p_0123 = compute_1d_interpolation(y_right_p, y_p_left, p_01, p_23);
  T p_4567 = compute_1d_interpolation(y_right_p, y_p_left, p_45, p_67);

  double z_right_p = smod((double) neighbors[4][2] - (double) p[2], (double) total_site[2]);
  double z_p_left = smod((double) p[2] - (double) neighbors[0][2], (double) total_site[2]);
	qassert(0. <= z_right_p && z_right_p <= 1.);
	qassert(0. <= z_p_left && z_p_left <= 1.);
  T p_01234567 = compute_1d_interpolation(z_right_p, z_p_left, p_0123, p_4567);

  return p_01234567;
}

void compute_fr_field_one_traj(
    const std::string& ensemble,
    FieldM<Complex, 1>& fr_field, 
    PionGGElemField& two_point_wall)
{
  TIMER_VERBOSE("compute_fr_field_one_traj");

  const Geometry& geo = two_point_wall.geo;
  const Coordinate total_site = geo.total_site();

  fr_field.init(geo);

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lx = geo.coordinate_from_index(index);
    Coordinate x = geo.coordinate_g_from_l(lx);
    x = relative_coordinate(x, total_site);

    if (x == Coordinate(0, 0, 0, 0)) {
      fr_field.get_elem(lx) = Complex(1., 0.);
      continue;
    }

    if (x[3] != 0) {
      fr_field.get_elem(lx) = Complex(0., 0.);
      continue; 
    }

    int r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    PionGGElem& pgge = two_point_wall.get_elem(lx);

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
    if (aa == 0) {
      fr_field.get_elem(lx) = Complex(0., 0.);
      continue;
    }

    Complex la = Complex(0., 0.);
    for (int mu = 0; mu < 4; ++mu) {
      for (int nu = 0; nu < 4; ++nu) {
        la += pgge.v[mu][nu] * (double) a[mu][nu];
      }
    }

    Complex f = la / (double) aa * std::pow((double) r2, 2.);
    fr_field.get_elem(lx) = f * compute_factor(ensemble);
  }
}

std::vector<Complex> compute_fr_interpolation_one_traj(
    PionGGElemField& two_point_wall, 
    const FREnsembleInfo& fr_info,
    double r_max = 4., // unit of fm
    int num_r = 40,
    int num_cos_theta = 20,
    int num_phi = 20)
{
  TIMER_VERBOSE("compute_fr_interpolation_one_traj");
	qassert(0 == get_rank());

  FieldM<Complex, 1> fr_field;
  compute_fr_field_one_traj(fr_info.ENSEMBLE, fr_field, two_point_wall);
  const Geometry& geo = fr_field.geo;
  const Coordinate total_site = geo.total_site();

  double r_step = r_max / num_r * fr_info.AINV / 0.197; // unit of lattice spacing
  double cos_theta_step = 2. / num_cos_theta;
  double phi_step = 2. * PI / num_phi;

  std::vector<Complex> fr_list(num_r + 1, Complex(0., 0.));

  fr_list[0] = Complex(1., 0.);
#pragma omp parallel for
  for (int i_r = 1; i_r < num_r + 1; ++i_r) {
		double r = i_r * r_step;
		Complex fr(0., 0.);
    std::vector<Complex> fr_cos_theta_list(num_cos_theta + 1, Complex(0., 0.));
    for (int i_cos_theta = 0; i_cos_theta < num_cos_theta + 1; ++i_cos_theta) {
			double cos_theta = -1. + (double) i_cos_theta * cos_theta_step;
			double theta = acos(cut_1(cos_theta));
      Complex fr_cos_theta(0., 0.);
      for (int i_phi = 0; i_phi < num_phi; ++i_phi) {
				double phi = -PI + (double) i_phi * phi_step;

				double x = r * sin(theta) * cos(phi);
				double y = r * sin(theta) * sin(phi);
				double z = r * cos_theta;
        if (
            x < (double) (-total_site[0] / 2) || (double) (total_site[0] / 2 - 1) < x ||
            y < (double) (-total_site[1] / 2) || (double) (total_site[1] / 2 - 1) < y ||
            z < (double) (-total_site[2] / 2) || (double) (total_site[2] / 2 - 1) < z
            ) {
          continue;
        }
        CoordinateD p(x, y, z, 0.); // relative coor
				p = mod(p, (CoordinateD) total_site); // coor in lattice
				fr_cos_theta += get_p_val_3d(p, total_site, fr_field) * phi_step;
      }
			fr_cos_theta_list[i_cos_theta] = fr_cos_theta;
    }
    for (int i_cos_theta = 0; i_cos_theta < num_cos_theta; ++i_cos_theta) {
		  fr += (fr_cos_theta_list[i_cos_theta] + fr_cos_theta_list[i_cos_theta + 1]) / 2. * cos_theta_step;
		}
		fr_list[i_r] = fr / (4. * PI);
  }

  return fr_list;
}

std::vector<Complex> compute_fr_one_traj(
    const FREnsembleInfo& fr_info, 
    PionGGElemField& two_point_wall, 
    const int r_max = 100)
{
  TIMER_VERBOSE("compute_fr_one_traj");
  std::vector<Complex> fr(r_max, Complex(0., 0.));
  std::vector<long> fr_cnt(r_max, 0);

  const Geometry& geo = two_point_wall.geo;
  const Coordinate total_site = geo.total_site();

  for (long index = 0; index < geo.local_volume(); ++index)
  {
    const Coordinate lx = geo.coordinate_from_index(index);
    Coordinate x = geo.coordinate_g_from_l(lx);
    x = relative_coordinate(x, total_site);

    if (x[3] != 0) {continue; }

    int r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
    if (r2 >= fr.size()) {continue;}
    PionGGElem& pgge = two_point_wall.get_elem(lx);

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

    Complex f(0., 0.);
    f = la / (double) aa * std::pow((double) r2, 2.);

    fr[r2] += f;
    fr_cnt[r2] += 1;
  }

  sync_node();
  for (int i = 0; i < fr.size(); ++i) {
    qlat::glb_sum_double(fr[i]);
    qlat::glb_sum_long(fr_cnt[i]);
  }
  sync_node();
  for (int i = 0; i < fr.size(); ++i) {
    if (fr_cnt[i] == 0) {continue;}
    fr[i] /= (double) fr_cnt[i];
    fr[i] *= compute_factor(fr_info.ENSEMBLE);
  }
  return fr;
}

std::vector<Complex> compute_fr_all_traj(const std::string& ensemble, const std::string accuracy, const int r_max = 1000)
{
  TIMER_VERBOSE("compute_fr_all_traj");
  const TwoPointWallEnsembleInfo tpw_info(ensemble, accuracy);
  const FREnsembleInfo fr_info(ensemble, accuracy);
  fr_info.make_fr_ensemble_accuracy_dir();

  for (int traj = fr_info.TRAJ_START; traj <= fr_info.TRAJ_END; ++traj) {
    if (obtain_lock(fr_info.get_fr_traj_path(traj) + "-lock")) {
      if (!tpw_info.is_traj_computed(traj)) {
        main_displayln_info(fname + ssprintf(": Two Point Wall for Traj=%d Are Not Computed", traj));
        release_lock();
        continue;
      }
      PionGGElemField two_point_wall;
      tpw_info.load_field_traj_avg(two_point_wall, traj);
      std::vector<Complex> fr_one_traj = compute_fr_one_traj(fr_info, two_point_wall, r_max);

      main_displayln_info(ssprintf("fr in traj %d:", traj));
      main_displayln_info(show_vec_complex(fr_one_traj));

      // save
      write_data_from_0_node(&fr_one_traj[0], fr_one_traj.size(), fr_info.get_fr_traj_path(traj));

      release_lock();
    }

  }
}

std::vector<Complex> compute_fr_interpolation_all_traj(
    const std::string& ensemble, 
    const std::string accuracy, 
    double r_max = 4.,
    int num_r = 40,
    int num_cos_theta = 20,
    int num_phi = 20)
{
  TIMER_VERBOSE("compute_fr_field_all_traj");
  const TwoPointWallEnsembleInfo tpw_info(ensemble, accuracy);
  const FREnsembleInfo fr_info(ensemble, accuracy, true);
  fr_info.make_fr_ensemble_accuracy_dir();

  for (int traj = fr_info.TRAJ_START; traj <= fr_info.TRAJ_END; ++traj) {
    if (obtain_lock(fr_info.get_fr_traj_path(traj) + "-lock")) {
      if (!tpw_info.is_traj_computed(traj)) {
        main_displayln_info(fname + ssprintf(": Two Point Wall for Traj=%d Are Not Computed", traj));
        release_lock();
        continue;
      }
      PionGGElemField two_point_wall;
      tpw_info.load_field_traj_avg(two_point_wall, traj);
      std::vector<Complex> fr_interpolation_one_traj = compute_fr_interpolation_one_traj(two_point_wall, fr_info, r_max, num_r, num_cos_theta, num_phi);

      main_displayln_info(ssprintf("fr interpolation in traj %d:", traj));
      main_displayln_info(show_vec_complex(fr_interpolation_one_traj));

      // save
      write_data_from_0_node(&fr_interpolation_one_traj[0], fr_interpolation_one_traj.size(), fr_info.get_fr_traj_path(traj));

      release_lock();
    }
  }
}

QLAT_END_NAMESPACE

#endif
