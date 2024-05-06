// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/Adm.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#include <cmath>

#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

namespace Xcts {

void adm_mass_volume_integrand(
    gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& energy_density,
    const Scalar<DataVector>& christoffel_deriv) {
  tenex::evaluate(
      result,
      (1.0 / (16 * M_PI)) *
          (16.0 * M_PI * pow<5>(conformal_factor()) * energy_density() +
           (1.0 / (pow<3>(conformal_factor()))) *
               extrinsic_curvature(ti::i, ti::j) *
               extrinsic_curvature(ti::k, ti::l) *
               inv_conformal_metric(ti::I, ti::K) *
               inv_conformal_metric(ti::J, ti::L) -
           conformal_factor() * conformal_ricci_scalar() -
           pow<5>(conformal_factor()) * pow<2>(trace_extrinsic_curvature()) +
           christoffel_deriv()));
}

Scalar<DataVector> adm_mass_volume_integrand(
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  const tnsr::I<DataVector, 3> contracted_christoffel = tenex::evaluate<ti::I>(
      conformal_christoffel_second_kind(ti::I, ti::n, ti::j) *
      inv_conformal_metric(ti::N, ti::J));
  // const auto christoffel =
  //     partial_derivative(inv_conformal_metric,mesh,inv_jacobian);
  // const auto christoffel_contracted =
  // tenex::evaluate<ti::J>(-christoffel(ti::i,ti::I,ti::J));
  const auto christoffel_deriv =
      divergence(contracted_christoffel, mesh, inv_jacobian);
  return christoffel_deriv;
}

Scalar<DataVector> adm_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& energy_density, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian) {
  Scalar<DataVector> result{get_size(get(conformal_factor))};
  adm_mass_volume_integrand(
      make_not_null(&result), conformal_factor, conformal_metric,
      inv_conformal_metric, conformal_ricci_scalar, extrinsic_curvature,
      trace_extrinsic_curvature, energy_density,
      adm_mass_volume_integrand(inv_conformal_metric,
                                conformal_christoffel_second_kind, mesh,
                                inv_jacobian));
  return result;
}

void adm_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  const auto contracted_christoffel = tenex::evaluate<ti::I>(
      conformal_christoffel_second_kind(ti::I, ti::n, ti::j) *
      inv_conformal_metric(ti::N, ti::J));
  tenex::evaluate<ti::I>(
      result, (1.0 / (16.0 * M_PI)) * (contracted_christoffel(ti::I) -
                                       8.0 * conformal_factor_deriv(ti::I)));
}

tnsr::I<DataVector, 3> adm_mass_surface_integrand(
    const tnsr::I<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  tnsr::I<DataVector, 3> result{};
  adm_mass_surface_integrand(make_not_null(&result), conformal_factor_deriv,
                             inv_conformal_metric,
                             conformal_christoffel_second_kind);
  return result;
}

// Intermediate terms for ADM linear momentum and angular momentum
void adm_intermediate_P(gsl::not_null<tnsr::II<DataVector, 3>*> result,
                        const Scalar<DataVector>& conformal_factor,
                        const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
                        const tnsr::II<DataVector, 3>& inv_spatial_metric,
                        const Scalar<DataVector>& trace_extrinsic_curvature) {
  tenex::evaluate<ti::I, ti::J>(
      result,
      pow<10>(conformal_factor()) *
          (inv_extrinsic_curvature(ti::I, ti::J) -
           trace_extrinsic_curvature() * inv_spatial_metric(ti::I, ti::J)));
}

tnsr::II<DataVector, 3> adm_intermediate_P(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  tnsr::II<DataVector, 3> result{get_size(get(conformal_factor))};
  adm_intermediate_P(make_not_null(&result), conformal_factor,
                     inv_extrinsic_curvature, inv_spatial_metric,
                     trace_extrinsic_curvature);
  return result;
}

void adm_intermediate_G(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& intermediate_P,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  const auto cmetric_jk_P_JK = tenex::evaluate(conformal_metric(ti::j, ti::k) *
                                               intermediate_P(ti::J, ti::K));
  const auto contracted_con_christoffel_Jjk = tenex::evaluate<ti::k>(
      conformal_christoffel_second_kind(ti::J, ti::j, ti::k));
  const auto cmetric_IL_par_l_lnconfac = tenex::evaluate<ti::I>(
      inv_conformal_metric(ti::I, ti::L) * conformal_factor_deriv(ti::l) /
      conformal_factor());

  tenex::evaluate<ti::I>(
      result,
      conformal_christoffel_second_kind(ti::I, ti::j, ti::k) *
              intermediate_P(ti::J, ti::K) +
          contracted_con_christoffel_Jjk(ti::k) * intermediate_P(ti::I, ti::K) -
          2.0 * cmetric_jk_P_JK() * cmetric_IL_par_l_lnconfac(ti::I));
}

tnsr::I<DataVector, 3> adm_intermediate_G(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& intermediate_P,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind

) {
  tnsr::I<DataVector, 3> result{get_size(get(conformal_factor))};
  adm_intermediate_G(make_not_null(&result), conformal_factor,
                     conformal_factor_deriv, intermediate_P, conformal_metric,
                     inv_conformal_metric, conformal_christoffel_second_kind);
  return result;
}

// ADM linear momentum
void adm_linear_momentum_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  const auto intermediate_P =
      adm_intermediate_P(conformal_factor, inv_extrinsic_curvature,
                         inv_spatial_metric, trace_extrinsic_curvature);
  tenex::evaluate<ti::I, ti::J>(result,
                                1 / (8 * M_PI) * intermediate_P(ti::I, ti::J));
}
tnsr::II<DataVector, 3> adm_linear_momentum_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  tnsr::II<DataVector, 3> result{get_size(get(conformal_factor))};
  adm_linear_momentum_surface_integrand(
      make_not_null(&result), conformal_factor, inv_extrinsic_curvature,
      inv_spatial_metric, trace_extrinsic_curvature);
  return result;
}

// FIXME::Decay the value of G eqn 22 of 1506.01689
// Note that we have already included the negative sign
void adm_linear_momentum_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  const tnsr::II<DataVector, 3> intermediate_P =
      adm_intermediate_P(conformal_factor, inv_extrinsic_curvature,
                         inv_spatial_metric, trace_extrinsic_curvature);
  const tnsr::I<DataVector, 3> intermediate_G =
      adm_intermediate_G(conformal_factor, conformal_factor_deriv,
                         intermediate_P, conformal_metric, inv_conformal_metric,
                         conformal_christoffel_second_kind);
  tenex::evaluate<ti::I>(result, -1 / (8 * M_PI) * intermediate_G(ti::I));
}

tnsr::I<DataVector, 3> adm_linear_momentum_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  tnsr::I<DataVector, 3> result;
  adm_linear_momentum_volume_integrand(
      make_not_null(&result), conformal_factor, trace_extrinsic_curvature,
      conformal_factor_deriv, conformal_metric, inv_extrinsic_curvature,
      inv_spatial_metric, inv_conformal_metric,
      conformal_christoffel_second_kind);
  return result;
}

// ADM angular momentum
void adm_angular_momentum_surface_integrand_full(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  const auto intermediate_P =
      adm_intermediate_P(conformal_factor, inv_extrinsic_curvature,
                         inv_spatial_metric, trace_extrinsic_curvature);

  for (size_t i = 0; i < 3; i++) {
    // Angular momentum is defined in a cyclic manner
    size_t indx1 = i % 3;
    size_t indx2 = (i + 1) % 3;
    size_t indx3 = (i + 2) % 3;
    for (size_t j = 0; j < 3; j++) {
      result->get(indx3, j) =
          (1.0 / (8 * M_PI)) *
          (coordinates.get(indx1) * intermediate_P.get(indx2, j) -
           coordinates.get(indx2) * intermediate_P.get(indx1, j));
    }
  }
}

tnsr::II<DataVector, 3> adm_angular_momentum_surface_integrand_full(
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  tnsr::II<DataVector, 3> result;
  adm_angular_momentum_surface_integrand_full(
      make_not_null(&result), coordinates, conformal_factor,
      inv_extrinsic_curvature, inv_spatial_metric, trace_extrinsic_curvature);
  return result;
}

// Negative sign is already included
void adm_angular_momentum_volume_integrand_full(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  const tnsr::II<DataVector, 3> intermediate_P =
      adm_intermediate_P(conformal_factor, inv_extrinsic_curvature,
                         inv_spatial_metric, trace_extrinsic_curvature);
  const tnsr::I<DataVector, 3> intermediate_G =
      adm_intermediate_G(conformal_factor, conformal_factor_deriv,
                         intermediate_P, conformal_metric, inv_conformal_metric,
                         conformal_christoffel_second_kind);
  for (size_t i = 0; i < 3; i++) {
    // Angular momentum is defined in a cyclic manner
    size_t indx1 = i % 3;
    size_t indx2 = (i + 1) % 3;
    size_t indx3 = (i + 2) % 3;
    result->get(indx3) = -(1.0 / (8 * M_PI)) *
                         (coordinates.get(indx1) * intermediate_G.get(indx2) -
                          coordinates.get(indx2) * intermediate_G.get(indx1));
  }
}

tnsr::I<DataVector, 3> adm_angular_momentum_volume_integrand_full(
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  tnsr::I<DataVector, 3> result;
  adm_angular_momentum_volume_integrand_full(
      make_not_null(&result), coordinates, conformal_factor,
      trace_extrinsic_curvature, conformal_factor_deriv, conformal_metric,
      inv_extrinsic_curvature, inv_spatial_metric, inv_conformal_metric,
      conformal_christoffel_second_kind);
  return result;
}

// In practice the the full form involves cancellation of large volume terms
// which introduces errors. We use only the surface term and that too in the
// region where K~0 and g~\eta (maximal slicing and conformal flatness)
void adm_angular_momentum_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature) {
  for (size_t i = 0; i < 3; i++) {
    // Angular momentum is defined in a cyclic manner
    size_t indx1 = i % 3;
    size_t indx2 = (i + 1) % 3;
    size_t indx3 = (i + 2) % 3;
    for (size_t j = 0; j < 3; j++) {
      result->get(indx3, j) =
          (1.0 / (8 * M_PI)) * pow<10>(conformal_factor.get()) *
          (coordinates.get(indx1) * inv_extrinsic_curvature.get(indx2, j) -
           coordinates.get(indx2) * inv_extrinsic_curvature.get(indx1, j));
    }
  }
}

tnsr::II<DataVector, 3> adm_angular_momentum_surface_integrand(
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature) {
  tnsr::II<DataVector, 3> result;
  adm_angular_momentum_surface_integrand(make_not_null(&result), coordinates,
                                         conformal_factor,
                                         inv_extrinsic_curvature);
  return result;
}
}  // namespace Xcts
