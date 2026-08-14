#pragma once
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include "xtensor-python/pytensor.hpp"
#include "boozermagneticfield.h"

using std::logic_error;
using std::shared_ptr;

#include <vector>       // For std::vector in Phihat
#include <algorithm>    // For std::sort in Phihat
#include <numeric>      // For std::iota in Phihat
#include <stdexcept>    // For std::invalid_argument Phihat
#include <set>          // For std::set in Phihat
#include <xtensor/xview.hpp> // To access parts of the xtensor
                             // (for ShearAlfvenWave and ShearAlfvenHarmonic)

/**
* @brief Transverse Shear Alfvén Wave in Boozer coordinates
*
* See Paul. et al,
* JPP (2023;89(5):905890515. doi:10.1017/S0022377823001095), and refs. therein.
**/
class ShearAlfvenWave {
public:
    using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;

protected:
    virtual void _Phi_impl(Array2& Phi) {
        throw logic_error("_Phi_impl was not implemented");
    }

    virtual void _dPhidpsi_impl(Array2& dPhidpsi) {
        throw logic_error("_dPhidpsi_impl was not implemented");
    }

    virtual void _dPhidtheta_impl(Array2& dPhidtheta) {
        throw logic_error("_dPhidtheta_impl was not implemented");
    }

    virtual void _dPhidzeta_impl(Array2& dPhidzeta) {
        throw logic_error("_dPhidzeta_impl was not implemented");
    }

    virtual void _Phidot_impl(Array2& Phidot) {
        throw logic_error("_Phidot_impl was not implemented");
    }

    virtual void _alpha_impl(Array2& alpha) {
        throw logic_error("_alpha_impl was not implemented");
    }

    virtual void _dalphadpsi_impl(Array2& dalphadpsi) {
        throw logic_error("_dalphadpsi_impl was not implemented");
    }

    virtual void _dalphadtheta_impl(Array2& dalphadtheta) {
        throw logic_error("_dalphadtheta_impl was not implemented");
    }

    virtual void _dalphadzeta_impl(Array2& dalphadzeta) {
        throw logic_error("_dalphadzeta_impl was not implemented");
    }

    virtual void _alphadot_impl(Array2& alphadot) {
        throw logic_error("_alphadot_impl was not implemented");
    }
    shared_ptr<BoozerMagneticField> B0;
    Array2 points;
    Array2 data_Phi;
    Array2 data_dPhidpsi, data_dPhidtheta, data_dPhidzeta, data_Phidot;
    Array2 data_alpha;
    Array2 data_alphadot, data_dalphadpsi, data_dalphadtheta, data_dalphadzeta;
    long npoints;
public:
    ShearAlfvenWave(shared_ptr<BoozerMagneticField> B0field)
        : B0(B0field) {
        Array2 vals({{0., 0., 0., 0.}});
        this->set_points(vals);
    }

    virtual ~ShearAlfvenWave() {}

    virtual void set_points(Array2& p) {
        if (p.shape(1) != 4) {
            throw std::invalid_argument("Input tensor must have 4 columns: Boozer coordinates, and time (s, theta, zeta, time)");
        }
        npoints = p.shape(0);
        points.resize({npoints, 4});
        memcpy(points.data(), p.data(), 4 * npoints * sizeof(double));
        // Set points for B0 using the first three columns of p
        // (s, theta, zeta):
        Array2 p_b0 = xt::view(p, xt::all(), xt::range(0, 3));
        B0->set_points(p_b0);
    }

    Array2 get_points() {
        return points;
    }

    Array2& Phi_ref() {
        data_Phi.resize({npoints, 1});
        _Phi_impl(data_Phi);
        return data_Phi;
    }

    Array2& dPhidpsi_ref() {
        data_dPhidpsi.resize({npoints, 1});
        _dPhidpsi_impl(data_dPhidpsi);
        return data_dPhidpsi;
    }

    Array2& Phidot_ref() {
        data_Phidot.resize({npoints, 1});
        _Phidot_impl(data_Phidot);
        return data_Phidot;
    }

    Array2& dPhidtheta_ref() {
        data_dPhidtheta.resize({npoints, 1});
        _dPhidtheta_impl(data_dPhidtheta);
        return data_dPhidtheta;
    }

    Array2& dPhidzeta_ref() {
        data_dPhidzeta.resize({npoints, 1});
        _dPhidzeta_impl(data_dPhidzeta);
        return data_dPhidzeta;
    }

    Array2& alpha_ref() {
        data_alpha.resize({npoints, 1});
        _alpha_impl(data_alpha);
        return data_alpha;
    }

    Array2& alphadot_ref() {
        data_alphadot.resize({npoints, 1});
        _alphadot_impl(data_alphadot);
        return data_alphadot;
    }

    Array2& dalphadtheta_ref() {
        data_dalphadtheta.resize({npoints, 1});
        _dalphadtheta_impl(data_dalphadtheta);
        return data_dalphadtheta;
    }

    Array2& dalphadpsi_ref() {
        data_dalphadpsi.resize({npoints, 1});
        _dalphadpsi_impl(data_dalphadpsi);
        return data_dalphadpsi;
    }

    Array2& dalphadzeta_ref() {
        data_dalphadzeta.resize({npoints, 1});
        _dalphadzeta_impl(data_dalphadzeta);
        return data_dalphadzeta;
    }

    std::shared_ptr<BoozerMagneticField> get_B0() const {
        return B0;
    }

    void set_B0(shared_ptr<BoozerMagneticField> B0field) {
        if (!B0field) {
            throw std::invalid_argument("B0 field must be provided.");
        }
        B0 = B0field;
        // Ensure B0 has the same points as the wave (s, theta, zeta).
        if (points.size() > 0) {
            Array2 p_b0 = xt::view(points, xt::all(), xt::range(0, 3));
            B0->set_points(p_b0);
        }
    }

    Array2 Phi() { return Phi_ref(); }
    Array2 dPhidpsi() { return dPhidpsi_ref(); }
    Array2 Phidot() { return Phidot_ref(); }
    Array2 dPhidtheta() { return dPhidtheta_ref(); }
    Array2 dPhidzeta() { return dPhidzeta_ref(); }
    Array2 alpha() { return alpha_ref(); }
    Array2 alphadot() { return alphadot_ref(); }
    Array2 dalphadtheta() { return dalphadtheta_ref(); }
    Array2 dalphadpsi() { return dalphadpsi_ref(); }
    Array2 dalphadzeta() { return dalphadzeta_ref(); }
};

/**
* @brief Class representing the profile of scalar potential
* with respect to normalized flux Boozer coordinate `s`.
*
* The `Phihat` class represents scalar potential profile (`Phihat`)
* as a function of the normalized toroidal Boozer coordinate `s`.
* It uses linear interpolation to compute the value of the
* scalar potential and its derivative at any given point within the domain.
*/
class Phihat {
private:
  std::vector<double> s_values;
  std::vector<double> Phihat_values;

  /**
  * @brief Validates the input vectors of normalized flux
  * Boozer coordinate `s` and corresponding scalar potential `Phi` values.
  *
  * Ensures that `s_values` and `Phihat_values` have the
  * same size and that all `s_values` are unique.
  *
  * @throws std::invalid_argument if the input vectors are not of
  * the same size or if `s_values` contains duplicates.
  */
  void validateInput() const {
    if (s_values.size() != Phihat_values.size()) {
      throw std::invalid_argument(
          "s_values and Phihat_values must have the same size.");
    }
    if (std::set<double>(s_values.begin(), s_values.end()).size() !=
        s_values.size()) {
      throw std::invalid_argument(
          "s_values contains duplicate entries; all s must be unique.");
    }
    if (s_values.size() < 2) {
      throw std::invalid_argument(
          "s_values must contain at least two points for interpolation.");
    }
  }


  /**
  * @brief Sorts the input data based on `s_values`.
  *
  * Sorts both `s_values` and `Phihat_values` in ascending order
  * of `s_values` to ensure correct interpolation.
  */
  void sortData() {
    std::vector<size_t> indices(s_values.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {
      return s_values[i1] < s_values[i2];
    });

    auto sorted_s_values = s_values;
    auto sorted_Phihat_values = Phihat_values;
    for (size_t i = 0; i < indices.size(); ++i) {
      s_values[i] = sorted_s_values[indices[i]];
      Phihat_values[i] = sorted_Phihat_values[indices[i]];
    }
  }

public:

  /**
  * @brief Constructs a Phihat object with the given `s` and `Phihat` values.
  *
  * Initializes the `Phihat` object with vectors of
  * `s` coordinates and their corresponding `Phihat` values.
  *
  * @param s_vals Vector of `s` coordinates.
  * @param Phihat_vals Vector of scalar potential values corresponding to `s`.
  * @throws std::invalid_argument if input vectors are not valid.
  */
  Phihat(const std::vector<double> &s_vals,
         const std::vector<double> &Phihat_vals)
      : s_values(s_vals), Phihat_values(Phihat_vals) {
    validateInput();
    sortData();
  }

  /**
  * @brief Interpolates the scalar potential `Phihat`
  * at a given `s` coordinate.
  *
  * Computes the value of the scalar potential `Phihat` using linear
  * interpolation between the nearest data points.
  * If `s` is outside the range of `s_values`,
  * returns the nearest boundary value.
  *
  * @param s The normalized toroidal Boozer coordinate.
  * @return Interpolated scalar potential value `Phihat` at the given `s`.
  */
  double operator()(double s) const {

    if (s < s_values.front()) {
      return Phihat_values.front();
    }
    if (s > s_values.back()) {
      return Phihat_values.back();
    }

    size_t i_left = 0;
    size_t i_right = s_values.size() - 1;
    for (int i = s_values.size() - 1; i >= 0; --i) {
      if (s_values[i] <= s && (i + 1 < s_values.size())) {
        i_left = i;
        i_right = i + 1;
        break;
      }
    }

    double slope = (Phihat_values[i_right] - Phihat_values[i_left]) /
                   (s_values[i_right] - s_values[i_left]);

    double Phi_at_s = Phihat_values[i_left] + slope * (s-s_values[i_left]);
    return Phi_at_s;
  }

  /**
  * @brief Computes the derivative of the scalar potential `Phihat`
  * at a given `s` coordinate.
  *
  * Computes the slope of `Phihat` at a given `s` using
  * linear interpolation between the nearest data points.
  * If `s` is outside the range of `s_values`, returns 0.0.
  *
  * @param s The normalized toroidal Boozer coordinate.
  * @return The derivative of `Phihat` at the given `s`.
  */
  double derivative(double s) const {
    if (s < s_values.front() || s > s_values.back()) {
      return 0.0;
    }

    size_t i_left = 0;
    size_t i_right = s_values.size() - 1;
    for (int i = s_values.size() - 1; i >= 0; --i) {
      if (s_values[i] <= s && (i + 1 < s_values.size())) {
        i_left = i;
        i_right = i + 1;
        break;
      }
    }

    return (Phihat_values[i_right] - Phihat_values[i_left]) /
           (s_values[i_right] - s_values[i_left]);
  }

  /**
  * @brief Returns the sorted s_values used for interpolation.
  *
  * @return A vector of sorted s_values.
  */
  const std::vector<double>& get_s_basis() const {
    return s_values;
  }
};


/**
* @brief Class representing a single harmonic Shear Alfvén Wave.
* See Paul. et al, JPP (2023;89(5):905890515. doi:10.1017/S0022377823001095)
*
* Initializes the Shear Alfvén Wave with the scalar potential of the form
* \f$ \Phi = \hat{\Phi}(s) \sin(m \theta - n \zeta + \omega t + \text{phase}) \f$
* and vector potential alpha determined by the ideal
* Ohm's law (i.e., zero electric field along the field line).
*
*/
class ShearAlfvenHarmonic : public ShearAlfvenWave {
public:
    using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;
    Phihat phihat;
    int Phim; // Poloidal mode number.
    int Phin; // Toroidal mode number.
    double omega; // Frequency of the wave.
    double phase; // Phase offset of the wave.

    void set_points(Array2& p) override {
      ShearAlfvenWave::set_points(p);
      npoints = p.shape(0);
      // Precompute the data for the wave:
      auto data_iota = B0->iota_ref();
      auto data_G = B0->G_ref();
      auto data_diotadpsi = B0->diotads_ref() / B0->psi0;
      Array2 data_d_alpha_fac_dpsi;
      if (B0->field_type == "nok" || B0->field_type == "") {
        auto data_I = B0->I_ref();
        auto data_dGdpsi = B0->dGds_ref() / B0->psi0;
        auto data_dIdpsi = B0->dIds_ref() / B0->psi0;
        data_alpha_fac = (data_iota * Phim - Phin) /
          (omega * (data_G + data_iota * data_I));
        data_d_alpha_fac_dpsi =
          (data_diotadpsi * Phim) / (omega * (data_G + data_iota * data_I)) -
          data_alpha_fac / (data_G + data_iota * data_I) *
          (data_dGdpsi + data_diotadpsi * data_I + data_iota * data_dIdpsi);
      } else {
        data_alpha_fac = (data_iota * Phim - Phin) / (omega * data_G);
        data_d_alpha_fac_dpsi = data_diotadpsi * Phim / (omega * data_G);
      }

      data_Phi.resize({npoints, 1});
      data_dPhidpsi.resize({npoints, 1});
      data_dPhidtheta.resize({npoints, 1});
      data_dPhidzeta.resize({npoints, 1});
      data_Phidot.resize({npoints, 1});
      if (B0->field_type == "nok" || B0->field_type == "") {
        data_alpha.resize({npoints, 1});
        data_dalphadzeta.resize({npoints, 1});
      }
      data_alphadot.resize({npoints, 1});
      data_dalphadpsi.resize({npoints, 1});
      data_dalphadtheta.resize({npoints, 1});
      data_dalphadzeta.resize({npoints, 1});
      for (std::size_t i = 0; i < npoints; ++i) {
        double s = p(i,0);
        double theta = p(i,1);
        double zeta = p(i,2);
        double time = p(i,3);
        double data_cos =
            cos(Phim * theta - Phin * zeta +
            omega * time + phase);
        double data_sin =
            sin(Phim * theta - Phin * zeta +
            omega * time + phase);
        double data_phihat = phihat(s);
        double data_dphihatdpsi = phihat.derivative(s) / (B0->psi0);
        data_Phi(i, 0) = data_phihat * data_sin;
        data_dPhidpsi(i, 0) = data_dphihatdpsi * data_sin;
        data_Phidot(i, 0) = data_phihat * data_cos * omega;
        data_dPhidtheta(i, 0) = data_Phidot(i, 0) * (Phim / omega);
        data_dPhidzeta(i, 0) = -data_Phidot(i, 0) * (Phin / omega);
        if (B0->field_type == "nok" || B0->field_type == "") {
          data_alpha(i, 0) = -data_Phi(i, 0) * data_alpha_fac(i, 0);
          data_dalphadzeta(i, 0) = -data_dPhidzeta(i, 0) * data_alpha_fac(i, 0);
        }
        data_alphadot(i, 0) = -data_Phidot(i, 0) * data_alpha_fac(i, 0);
        data_dalphadpsi(i, 0) = -data_dPhidpsi(i, 0) * data_alpha_fac(i, 0)
            - data_Phi(i, 0) * data_d_alpha_fac_dpsi(i, 0);
        data_dalphadtheta(i, 0) = -data_dPhidtheta(i, 0) * data_alpha_fac(i, 0);
      }
    }

protected:
  Array2 data_Phi, data_dPhidpsi, data_dPhidtheta, data_dPhidzeta, data_Phidot;
  Array2 data_alpha, data_alphadot, data_dalphadpsi, data_dalphadtheta, data_dalphadzeta;
  Array2 data_alpha_fac;

  void _Phi_impl(Array2& Phi) override {
    Phi = data_Phi;
  }

  void _dPhidpsi_impl(Array2& dPhidpsi) override {
    dPhidpsi = data_dPhidpsi;
  }

  void _dPhidtheta_impl(Array2& dPhidtheta) override {
    dPhidtheta = data_dPhidtheta;
  }

  void _dPhidzeta_impl(Array2& dPhidzeta) override {
    dPhidzeta = data_dPhidzeta;
  }

  void _Phidot_impl(Array2& Phidot) override {
    Phidot = data_Phidot;
  }

  void _alpha_impl(Array2& alpha) override {
    // Data only precomputed if non-vacuum
    if (B0->field_type == "vac") {
      alpha = - data_Phi * data_alpha_fac;
    } else {
      alpha = data_alpha;
    }
  }

  void _alphadot_impl(Array2& alphadot) override {
    alphadot = data_alphadot;
  }

  void _dalphadpsi_impl(Array2& dalphadpsi) override {
    dalphadpsi = data_dalphadpsi;
  }

  void _dalphadtheta_impl(Array2& dalphadtheta) override {
    dalphadtheta = data_dalphadtheta;
  }

  void _dalphadzeta_impl(Array2& dalphadzeta) override {
    // Data only precomputed if non-vacuum
    if (B0->field_type == "vac") {
      dalphadzeta = -data_dPhidzeta * data_alpha_fac;
    } else {
      dalphadzeta = data_dalphadzeta;
    }
  }

  public:
      /**
      * @brief Constructor for the ShearAlfvenHarmonic class.
      *
      * Initializes the Shear Alfvén Harmonic with a given profile `phihat`, mode numbers `m` and `n`,
      * wave frequency `omega`, phase `phase`, and equilibrium magnetic field `B0`.
      *
      * @param phihat_in Profile of the scalar potential.
      * @param Phim Poloidal mode number.
      * @param Phin Toroidal mode number.
      * @param omega Frequency of the wave.
      * @param phase Phase offset of the wave.
      * @param B0field Shared pointer to the equilibrium Boozer magnetic field.
      */
      ShearAlfvenHarmonic(
          const Phihat& phihat_in,
          int Phim,
          int Phin,
          double omega,
          double phase,
          shared_ptr<BoozerMagneticField> B0field
      ) :
      ShearAlfvenWave(B0field),
      phihat(phihat_in),
      Phim(Phim),
      Phin(Phin),
      omega(omega),
      phase(phase) {}

      /**
      * @brief Returns radial amplitude Phihat of the ShearAlfvenHarmonic
      */
      const Phihat& get_phihat() const {
        return phihat;
      }
};


/**
* @brief Class representing a superposition of multiple Shear Alfvén waves.
*
* This class models the superposition of multiple Shear Alfvén waves, combining their scalar
* potential `Phi`, vector potential `alpha`, and their respective derivatives.
*/
class ShearAlfvenWavesSuperposition : public ShearAlfvenWave {
public:
  using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;
  //List of waves in superposition:
  std::vector<std::shared_ptr<ShearAlfvenWave>> waves;

  /**
  * @brief Adds a new wave to the superposition.
  *
  *  Adds a new wave to the superposition after verifying
  *  that it has the same equilibrium magnetic field `B0`.
  *
  * @param wave Shared pointer to a ShearAlfvenWave object to be added.
  * @throws std::invalid_argument if the wave's `B0` field does not
  * match the superposition's `B0`.
  */
  void add_wave(const std::shared_ptr<ShearAlfvenWave>& wave) {
    if (wave->get_B0() != this->B0) {
      throw std::invalid_argument(
        "The wave's B0 field does not match the superposition's B0 field."
      );
    }
    waves.push_back(wave);
    refresh_fusion();
  }

  /**
  * @brief Constructor for ShearAlfvenWavesSuperposition.
  *
  * Initializes the superposition with a base wave,
  * setting its `B0` field as the reference field
  * for all subsequent waves added to the superposition.
  *
  * @param base_wave Shared pointer to the initial ShearAlfvenWave object.
  * @throws std::invalid_argument if the base wave is not provided.
  */
  ShearAlfvenWavesSuperposition(std::shared_ptr<ShearAlfvenWave> base_wave)
    : ShearAlfvenWave(base_wave->get_B0()) {
    if (!base_wave) {
      throw std::invalid_argument(
        "Base wave must be provided to initialize the superposition."
      );
    }
    add_wave(base_wave);
  }

  /**
  * @brief Sets the points (s, theta, zeta, time)
  *
  * Sets the points for the superposition and propagates them to all waves.
  *
  * @param p A tensor representing the points in Boozer coordinates
  *          and time (s, theta, zeta, time).
  */
  void set_points(Array2& p) override {
    ShearAlfvenWave::set_points(p);  // stores the points and sets B0 once
    if (fused) {
      evaluate_fused(p);
    } else {
      for (const auto& wave : waves) {
        wave->set_points(p);  // Propagate points to each wave
      }
    }
  }

  std::shared_ptr<ShearAlfvenWave> get_wave(size_t index) const {
    if (index >= waves.size()) {
      throw std::out_of_range("Wave index out of range");
    }
    // The fused path never hands the points to the individual waves, so bring
    // this one up to date before it is handed out to be queried on its own.
    if (fused && points.size() > 0) {
      Array2 p = points;
      waves[index]->set_points(p);
    }
    return waves[index];
  }

  size_t size() const {
    return waves.size();
  }

private:
  /* Fused evaluation of a superposition of ShearAlfvenHarmonics.
   *
   * All waves share one B0 (add_wave enforces it), so when every wave is a
   * ShearAlfvenHarmonic the whole superposition can be evaluated in a single
   * pass. The generic path below instead calls set_points on every wave --
   * each of which re-sets B0 and re-reads its flux functions, allocates ten
   * output arrays and builds xtensor temporaries -- and then sums N per-wave
   * arrays again for each quantity read. A tracing RHS evaluates one point at
   * a time, so that per-wave framework cost, not the arithmetic, dominates.
   *
   * Harmonics are accumulated in the order they appear in `waves`, matching
   * the summation order of the generic path.
   */
  std::vector<ShearAlfvenHarmonic*> harmonics;  // non-owning; valid iff fused
  bool fused = false;
  Array2 fused_Phi, fused_dPhidpsi, fused_dPhidtheta, fused_dPhidzeta,
      fused_Phidot, fused_alpha, fused_alphadot, fused_dalphadpsi,
      fused_dalphadtheta, fused_dalphadzeta;

  void refresh_fusion() {
    harmonics.clear();
    for (const auto& wave : waves) {
      auto* h = dynamic_cast<ShearAlfvenHarmonic*>(wave.get());
      if (!h) {  // e.g. a nested superposition or an interpolated wave
        harmonics.clear();
        fused = false;
        return;
      }
      harmonics.push_back(h);
    }
    fused = !harmonics.empty();
  }

  void evaluate_fused(Array2& p) {
    // retains_K mirrors the branch in ShearAlfvenHarmonic::set_points
    const bool retains_K = (B0->field_type == "nok" || B0->field_type == "");
    const double psi0 = B0->psi0;

    // Flux functions are read once here rather than once per harmonic.
    auto& data_iota = B0->iota_ref();
    auto& data_G = B0->G_ref();
    auto& data_diotads = B0->diotads_ref();
    Array2 empty;
    auto& data_I = retains_K ? B0->I_ref() : empty;
    auto& data_dGds = retains_K ? B0->dGds_ref() : empty;
    auto& data_dIds = retains_K ? B0->dIds_ref() : empty;

    fused_Phi.resize({npoints, 1});
    fused_dPhidpsi.resize({npoints, 1});
    fused_dPhidtheta.resize({npoints, 1});
    fused_dPhidzeta.resize({npoints, 1});
    fused_Phidot.resize({npoints, 1});
    fused_alpha.resize({npoints, 1});
    fused_alphadot.resize({npoints, 1});
    fused_dalphadpsi.resize({npoints, 1});
    fused_dalphadtheta.resize({npoints, 1});
    fused_dalphadzeta.resize({npoints, 1});

    for (long i = 0; i < npoints; ++i) {
      const double s = p(i, 0);
      const double theta = p(i, 1);
      const double zeta = p(i, 2);
      const double time = p(i, 3);

      const double iota = data_iota(i, 0);
      const double G = data_G(i, 0);
      const double diotadpsi = data_diotads(i, 0) / psi0;
      double I = 0., dGdpsi = 0., dIdpsi = 0.;
      if (retains_K) {
        I = data_I(i, 0);
        dGdpsi = data_dGds(i, 0) / psi0;
        dIdpsi = data_dIds(i, 0) / psi0;
      }

      double sum_Phi = 0., sum_dPhidpsi = 0., sum_dPhidtheta = 0.,
             sum_dPhidzeta = 0., sum_Phidot = 0., sum_alpha = 0.,
             sum_alphadot = 0., sum_dalphadpsi = 0., sum_dalphadtheta = 0.,
             sum_dalphadzeta = 0.;

      for (const auto* h : harmonics) {
        double alpha_fac, d_alpha_fac_dpsi;
        if (retains_K) {
          const double denom = G + iota * I;
          alpha_fac = (iota * h->Phim - h->Phin) / (h->omega * denom);
          d_alpha_fac_dpsi = (diotadpsi * h->Phim) / (h->omega * denom) -
                             alpha_fac / denom *
                                 (dGdpsi + diotadpsi * I + iota * dIdpsi);
        } else {
          alpha_fac = (iota * h->Phim - h->Phin) / (h->omega * G);
          d_alpha_fac_dpsi = diotadpsi * h->Phim / (h->omega * G);
        }

        const double arg =
            h->Phim * theta - h->Phin * zeta + h->omega * time + h->phase;
        const double data_cos = cos(arg);
        const double data_sin = sin(arg);
        const double phihat_s = h->phihat(s);
        const double dphihatdpsi = h->phihat.derivative(s) / psi0;

        const double Phi = phihat_s * data_sin;
        const double dPhidpsi = dphihatdpsi * data_sin;
        const double Phidot = phihat_s * data_cos * h->omega;
        const double dPhidtheta = Phidot * (h->Phim / h->omega);
        const double dPhidzeta = -Phidot * (h->Phin / h->omega);

        sum_Phi += Phi;
        sum_dPhidpsi += dPhidpsi;
        sum_Phidot += Phidot;
        sum_dPhidtheta += dPhidtheta;
        sum_dPhidzeta += dPhidzeta;
        if (retains_K) {
          sum_alpha += -Phi * alpha_fac;
          sum_dalphadzeta += -dPhidzeta * alpha_fac;
        }
        sum_alphadot += -Phidot * alpha_fac;
        sum_dalphadpsi += -dPhidpsi * alpha_fac - Phi * d_alpha_fac_dpsi;
        sum_dalphadtheta += -dPhidtheta * alpha_fac;
      }

      fused_Phi(i, 0) = sum_Phi;
      fused_dPhidpsi(i, 0) = sum_dPhidpsi;
      fused_dPhidtheta(i, 0) = sum_dPhidtheta;
      fused_dPhidzeta(i, 0) = sum_dPhidzeta;
      fused_Phidot(i, 0) = sum_Phidot;
      // alpha and dalphadzeta are only defined when the field retains K; the
      // per-wave path leaves them untouched in the vacuum case, so report
      // zero rather than a stale value.
      fused_alpha(i, 0) = sum_alpha;
      fused_dalphadzeta(i, 0) = sum_dalphadzeta;
      fused_alphadot(i, 0) = sum_alphadot;
      fused_dalphadpsi(i, 0) = sum_dalphadpsi;
      fused_dalphadtheta(i, 0) = sum_dalphadtheta;
    }
  }

protected:
  void _Phi_impl(Array2& Phi) override {
    if (fused) {
      Phi = fused_Phi;
      return;
    }
    Phi.fill(0.0);
    for (const auto& wave : waves) {
      Phi += wave->Phi();
    }
  }

  void _dPhidpsi_impl(Array2& dPhidpsi) override {
    if (fused) {
      dPhidpsi = fused_dPhidpsi;
      return;
    }
    dPhidpsi.fill(0.0);
    for (const auto& wave : waves) {
      dPhidpsi += wave->dPhidpsi();
    }
  }

  void _dPhidtheta_impl(Array2& dPhidtheta) override {
    if (fused) {
      dPhidtheta = fused_dPhidtheta;
      return;
    }
    dPhidtheta.fill(0.0);
    for (const auto& wave : waves) {
      dPhidtheta += wave->dPhidtheta();
    }
  }

  void _dPhidzeta_impl(Array2& dPhidzeta) override {
    if (fused) {
      dPhidzeta = fused_dPhidzeta;
      return;
    }
    dPhidzeta.fill(0.0);
    for (const auto& wave : waves) {
      dPhidzeta += wave->dPhidzeta();
    }
  }

  void _Phidot_impl(Array2& Phidot) override {
    if (fused) {
      Phidot = fused_Phidot;
      return;
    }
    Phidot.fill(0.0);
      for (const auto& wave : waves) {
      Phidot += wave->Phidot();
    }
  }

  void _alpha_impl(Array2& alpha) override {
    if (fused) {
      alpha = fused_alpha;
      return;
    }
    alpha.fill(0.0);
    for (const auto& wave : waves) {
      alpha += wave->alpha();
    }
  }

  void _dalphadpsi_impl(Array2& dalphadpsi) override {
    if (fused) {
      dalphadpsi = fused_dalphadpsi;
      return;
    }
    dalphadpsi.fill(0.0);
    for (const auto& wave : waves) {
      dalphadpsi += wave->dalphadpsi();
    }
  }

  void _dalphadtheta_impl(Array2& dalphadtheta) override {
    if (fused) {
      dalphadtheta = fused_dalphadtheta;
      return;
    }
    dalphadtheta.fill(0.0);
    for (const auto& wave : waves) {
      dalphadtheta += wave->dalphadtheta();
    }
  }

  void _dalphadzeta_impl(Array2& dalphadzeta) override {
    if (fused) {
      dalphadzeta = fused_dalphadzeta;
      return;
    }
    dalphadzeta.fill(0.0);
    for (const auto& wave : waves) {
      dalphadzeta += wave->dalphadzeta();
    }
  }

  void _alphadot_impl(Array2& alphadot) override {
    if (fused) {
      alphadot = fused_alphadot;
      return;
    }
    alphadot.fill(0.0);
    for (const auto& wave : waves) {
      alphadot += wave->alphadot();
    }
  }
};
