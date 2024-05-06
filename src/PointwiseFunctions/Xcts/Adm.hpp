// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
    const Scalar<DataVector>& christoffel_deriv);

Scalar<DataVector> adm_mass_volume_integrand(
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian);

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
                          Frame::Inertial>& inv_jacobian);

void adm_mass_surface_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

tnsr::I<DataVector, 3> adm_mass_surface_integrand(
    const tnsr::I<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

// Intermediate terms for ADM linear momentum and angular momentum
void adm_intermediate_P(gsl::not_null<tnsr::II<DataVector, 3>*> result,
                        const Scalar<DataVector>& conformal_factor,
                        const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
                        const tnsr::II<DataVector, 3>& inv_spatial_metric,
                        const Scalar<DataVector>& trace_extrinsic_curvature);

tnsr::II<DataVector, 3> adm_intermediate_P(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature);

void adm_intermediate_G(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& intermediate_P,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

tnsr::I<DataVector, 3> adm_intermediate_G(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::II<DataVector, 3>& intermediate_P,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind

);

// ADM linear momentum
void adm_linear_momentum_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature);

tnsr::II<DataVector, 3> adm_linear_momentum_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature);

void adm_linear_momentum_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

tnsr::I<DataVector, 3> adm_linear_momentum_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

// ADM angular momentum full (Not used because the volume term gives errors)
void adm_angular_momentum_surface_integrand_full(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature);

tnsr::II<DataVector, 3> adm_angular_momentum_surface_integrand_full(
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const Scalar<DataVector>& trace_extrinsic_curvature);

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
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

tnsr::I<DataVector, 3> adm_angular_momentum_volume_integrand_full(
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const tnsr::i<DataVector, 3>& conformal_factor_deriv,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind);

// The following formula is derived under the assumption that K~0 and g~\eta
// (maximal slicing and conformal flatness). Under these assumptions the volume
// term becomes zero and the surface term reduces to the
void adm_angular_momentum_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature);

tnsr::II<DataVector, 3> adm_angular_momentum_surface_integrand(
    const tnsr::I<DataVector, 3>& coordinates,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::II<DataVector, 3>& inv_extrinsic_curvature);
}  // namespace Xcts
