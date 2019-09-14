/*
  Simple interpolation library

  Modified from Christoph Lehner's code
*/

#pragma once

#include <qlat/qlat.h>

#include <vector>
#include <cmath>
#include <cassert>

struct InterpolationDim {
  int n;
  double xhigh;
  double xlow;
};

template <class C, class I>
struct InterpolationNd {
  //
  std::vector<C> data;
  std::vector<InterpolationDim> dims;
  //
  void init()
  {
    dims.resize(0);
    data.resize(0);
  }
  //
  void add_dimension(const int n, const double xhigh, const double xlow)
    // use n-grid in next dimension with i=0 <> xlow and i=n-1 <> xhigh
  {
    InterpolationDim d = { n, xhigh, xlow };
    dims.push_back(d);
    adjust_data();
  }
  //
  void adjust_data()
  {
    size_t size = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
      size *= dims[i].n;
    }
    data.resize(size);
  }
  //
  size_t size() const
  {
    return data.size();
  }
  //
  std::vector<int> get_icoor(size_t idx) const
  {
    std::vector<int> icoor(dims.size());
    for (size_t j=0; j < dims.size(); j++) {
      icoor[j] = idx % dims[j].n;
      idx /= dims[j].n;
    }
    return icoor;
  }
  //
  size_t get_index(const std::vector<int>& icoor) const
  {
    size_t idx = 0;
    for (int j=(int)dims.size()-1; j >= 0; j--) {
      idx *= dims[j].n;
      idx += icoor[j];
    }
    return idx;
  }
  //
  std::vector<int> index_plus(const std::vector<int>& input, const int i) const
  {
    std::vector<int> ret = input;
    ret[i] += 1;
    if (ret[i] == dims[i].n) {
      // this is intentional for use below
      ret[i] -= 1;
    }
    return ret;
  }
  //
  std::vector<double> coor_from_icoor(const std::vector<int>& icoor) const
  {
    std::vector<double> x(icoor.size());
    for (size_t j = 0; j < dims.size(); j++) {
      const InterpolationDim& d = dims[j];
      if (1 == d.n) {
        x[j] = 0.5 * (d.xhigh + d.xlow);
      } else {
        double lam = (double)icoor[j] / (double)(d.n - 1);
        x[j] = d.xlow + lam * (d.xhigh - d.xlow);
      }
    }
    return x;
  }
  //
  std::vector<double> get_coor(size_t idx) const
    // get coordinate of grid point i
  {
    return coor_from_icoor(get_icoor(idx));
  }
  //
  C& operator[](const size_t idx)
  {
    return data[idx];
  }
  //
  const C& operator[](const size_t idx) const
  {
    return data[idx];
  }
  //
  C operator()(const std::vector<double>& x) const
    // get interpolated value at point x
  {
    qassert(dims.size() == x.size());
    std::vector<int> il(dims.size());
    std::vector<double> vl(dims.size());
    // get coordinate left of x in each dimension
    for (size_t j=0;j<dims.size();j++) {
      const InterpolationDim& d = dims[j];
      if (1 == d.n) {
        il[j] = 0;
        vl[j] = 0.0;
      } else {
        double fj = (x[j] - d.xlow) / (d.xhigh - d.xlow); // 0 <= fj <= 1
        if (false == (0.0 <= fj && fj <= 1.0)) {
          fdisplayln(stdout, ssprintf("WARNING: interpolation fj out of range. j=%d fj=%f", j, fj));
          if (fj > 1.0) {
            fj = 1.0;
          } else if (fj < 0.0) {
            fj = 0.0;
          } else {
            assert(false);
          }
        }
        int ileft = (int)(fj * (d.n - 1)); // fringe case for x[j] == d.xhigh, ileft=d.n-1
        int iright = (ileft == d.n-1) ? ileft : ileft + 1;
        double dj = (d.xhigh - d.xlow) / (double)(d.n - 1);
        double lam = (x[j] - d.xlow - dj*ileft) / dj;
        if (false == (0.0 <= lam && lam <= 1.0)) {
          fdisplayln(stdout, ssprintf("WARNING: interpolation lam out of range. j=%d lam=%f", j, lam));
          if (lam > 1.0) {
            lam = 1.0;
          } else if (lam < 0.0) {
            lam = 0.0;
          } else {
            assert(false);
          }
        }
        il[j] = ileft;
        vl[j] = lam;
      }
    }
    return I()(*this,il,vl);
  }
};

template <class C>
struct SimpleInterpolator
{
  typedef InterpolationNd<C,SimpleInterpolator<C> > IP;
  //
  C operator()(const IP& ip, std::vector<int>& il, std::vector<double>& vl) const
    // do interpolation
  {
    C v0 = ip[ip.get_index(il)];
    for (size_t j=0;j<ip.dims.size();j++) {
      C dj = ip[ip.get_index(ip.index_plus(il,j))] - v0;
      v0 += dj * vl[j];
    }
    return v0;
  }
};

template <class C>
struct BilinearInterpolator
{
  typedef InterpolationNd<C,BilinearInterpolator<C> > IP;
  //
  C operator()(const IP& ip, std::vector<int>& il, std::vector<double>& vl) const
    // n-dimensional box, interpolate one dimension at a time
    //
    // cube_{000...0} is fundamental vertex of box
    // cube_{100...0} is 1 hop in 0-dimension
    // ....
    // cube_{111...1} is 1 hop in all dimensions
    //
    // for n dimensions this means 2^n points are involved in interpolation
    // for n = 5 this means 32 points
  {
    std::vector<C> cube(1<<il.size());
    for (size_t points=0;points<cube.size();points++) {
      std::vector<int> i = il;
      for (size_t d=0;d<il.size();d++) {
        if ((1 << d) & points) {
          i = ip.index_plus(i,d);
        }
      }
      //printf("Point %d: %d %d %d %d %d\n",points,i[0],i[1],i[2],i[3],i[4]);
      cube[points] = ip[ip.get_index(i)];
    }
    // exit(1);
    // do interpolation for each dimension, each time reducing cube.size() by a factor of two
    for (size_t d=0;d<il.size();d++) {
      std::vector<C> cubep;
      double a = 1.0 - vl[d];
      double b = vl[d];
      // all even indices in cube are at xlow, all odd indices are at xhigh ->
      for (size_t points=0;points<cube.size();points+=2) {
        cubep.push_back(cube[points]*a + cube[points+1]*b);
      }
      cube = cubep;
    }
    return cube[0];
  }
};

template <class C>
struct InterpolationBilinearNd : public InterpolationNd<C,BilinearInterpolator<C> >
{
};

inline void test_interpolationBilinear()
{
  TIMER_VERBOSE("test_interpolationBilinear");
  using namespace qlat;
  if (0 != get_id_node()) {
    return;
  }
  RngState rs("test_interpolationBilinear");
  const int test_size = 4;
  InterpolationBilinearNd<double> interpolation;
  // InterpolationNd<double, SimpleInterpolator<double> > interpolation;
  const double limit = 2.0;
  const int dimN = 16;
  interpolation.add_dimension(dimN, limit, -limit);
  interpolation.add_dimension(dimN, limit, -limit);
  interpolation.add_dimension(dimN, limit, -limit);
  interpolation.add_dimension(dimN, limit, -limit);
  interpolation.add_dimension(dimN, limit, -limit);
  std::vector<double> shift(5);
  for (int k = 0; k < test_size; ++k) {
    for (size_t i = 0; i < shift.size(); ++i) {
      shift[i] = uRandGen(rs, PI, 0.0);
    }
    for (size_t idx = 0; idx < interpolation.size(); ++idx) {
      std::vector<double> x = interpolation.get_coor(idx);
      interpolation[idx] = std::sin(x[0] + shift[0]) * std::sin(x[1] + shift[1]) *
        std::sin(x[2] + shift[2]) * std::sin(x[3] + shift[3]) * std::sin(x[4] + shift[4]);
    }
    for (size_t i = 0; i < test_size; ++i) {
      std::vector<double> x(5);
      x[0] = uRandGen(rs, limit, -limit);
      x[1] = uRandGen(rs, limit, -limit);
      x[2] = uRandGen(rs, limit, -limit);
      x[3] = uRandGen(rs, limit, -limit);
      x[4] = uRandGen(rs, limit, -limit);
      const double fi = interpolation(x);
      const double f = std::sin(x[0] + shift[0]) * std::sin(x[1] + shift[1]) *
        std::sin(x[2] + shift[2]) * std::sin(x[3] + shift[3]) * std::sin(x[4] + shift[4]);
      fdisplayln(stdout,
          ssprintf("%4d %4d: %10.4f %10.4f %13.7f %10.3f%%", k, i,
            f, fi, fi-f, 200.0 * (fi - f)/std::abs(fi+f)));
    }
  }
}
