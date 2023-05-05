// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/DirichletMinkowski.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
template <size_t Dim>
struct ConvertMinkowski {
  using unpacked_container = int;
  using packed_container =
      gh::Solutions::WrappedGr<gr::Solutions::Minkowski<Dim>>;
  using packed_type = double;

  static packed_container create_container() { return {}; }

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    // No way of getting the args from the boundary condition.
    return Dim;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container /*unpacked*/,
                          const size_t /*grid_point_index*/) {
    *packed = create_container();
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  const auto box_analytic_soln = db::create<db::AddSimpleTags<
      Tags::Time, Tags::AnalyticSolution<gh::Solutions::WrappedGr<
                      gr::Solutions::Minkowski<Dim>>>>>(
      0.5, ConvertMinkowski<Dim>::create_container());

  helpers::test_boundary_condition_with_python<
      gh::BoundaryConditions::DirichletMinkowski<Dim>,
      gh::BoundaryConditions::BoundaryCondition<Dim>, gh::System<Dim>,
      tmpl::list<gh::BoundaryCorrections::UpwindPenalty<Dim>>,
      tmpl::list<ConvertMinkowski<Dim>>>(
      make_not_null(&gen),
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions."
      "DirichletMinkowski",
      tuples::TaggedTuple<
          helpers::Tags::PythonFunctionForErrorMessage<>,
          helpers::Tags::PythonFunctionName<
              gr::Tags::SpacetimeMetric<DataVector, Dim>>,
          helpers::Tags::PythonFunctionName<gh::Tags::Pi<Dim>>,
          helpers::Tags::PythonFunctionName<gh::Tags::Phi<Dim>>,
          helpers::Tags::PythonFunctionName<
              gh::ConstraintDamping::Tags::ConstraintGamma1>,
          helpers::Tags::PythonFunctionName<
              gh::ConstraintDamping::Tags::ConstraintGamma2>,
          helpers::Tags::PythonFunctionName<gr::Tags::Lapse<DataVector>>,
          helpers::Tags::PythonFunctionName<gr::Tags::Shift<DataVector, Dim>>>{
          "error", "spacetime_metric", "pi", "phi", "constraint_gamma1",
          "constraint_gamma2", "lapse", "shift"},
      "DirichletMinkowski:\n", Index<Dim - 1>{Dim == 1 ? 1 : 5},
      box_analytic_soln,
      tuples::TaggedTuple<
          helpers::Tags::Range<gh::ConstraintDamping::Tags::ConstraintGamma1>,
          helpers::Tags::Range<gh::ConstraintDamping::Tags::ConstraintGamma2>>{
          std::array{0.0, 1.0}, std::array{0.0, 1.0}});
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BoundaryConditions.DirichletMinkowski",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  test<1>();
  test<2>();
  test<3>();
}
