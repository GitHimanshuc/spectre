// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Translation",
                  "[Domain][Unit]") {
  // define vars for FunctionOfTime::PiecewisePolynomial f(t) = t**2.
  double t = -1.0;
  const double dt = 0.6;
  const double final_time = 4.0;
  constexpr size_t deriv_order = 3;

  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0}, {-2.0}, {2.0}, {0.0}}};

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list["translation"] =
      std::make_unique<Polynomial>(t, init_func, final_time + dt);

  const FoftPtr& f_of_t = f_of_t_list.at("translation");

  const CoordinateMaps::TimeDependent::Translation trans_map{"translation"};
  // test serialized/deserialized map
  const auto trans_map_deserialized = serialize_and_deserialize(trans_map);

  const std::array<double, 1> point_xi{{3.2}};

  while (t < final_time) {
    const std::array<double, 1> trans_x{{square(t)}};
    const std::array<double, 1> frame_vel{{f_of_t->func_and_deriv(t)[1][0]}};

    CHECK_ITERABLE_APPROX(trans_map(point_xi, t, f_of_t_list),
                          point_xi + trans_x);
    CHECK_ITERABLE_APPROX(
        trans_map.inverse(point_xi + trans_x, t, f_of_t_list).get(), point_xi);
    CHECK_ITERABLE_APPROX(trans_map.frame_velocity(point_xi, t, f_of_t_list),
                          frame_vel);

    CHECK_ITERABLE_APPROX(trans_map_deserialized(point_xi, t, f_of_t_list),
                          point_xi + trans_x);
    CHECK_ITERABLE_APPROX(
        trans_map_deserialized.inverse(point_xi + trans_x, t, f_of_t_list)
            .get(),
        point_xi);
    CHECK_ITERABLE_APPROX(trans_map_deserialized.frame_velocity(
                              point_xi + trans_x, t, f_of_t_list),
                          frame_vel);

    t += dt;
  }

  // time-independent checks
  CHECK(trans_map.inv_jacobian(point_xi).get(0, 0) == 1.0);
  CHECK(trans_map_deserialized.inv_jacobian(point_xi).get(0, 0) == 1.0);
  CHECK(trans_map.jacobian(point_xi).get(0, 0) == 1.0);
  CHECK(trans_map_deserialized.jacobian(point_xi).get(0, 0) == 1.0);

  // Check inequivalence operator
  CHECK_FALSE(trans_map != trans_map);
  CHECK_FALSE(trans_map_deserialized != trans_map_deserialized);

  // Check serialization
  CHECK(trans_map == trans_map_deserialized);
  CHECK_FALSE(trans_map != trans_map_deserialized);

  test_coordinate_map_argument_types(trans_map, point_xi, t, f_of_t_list);
  CHECK(not CoordinateMaps::TimeDependent::Translation{}.is_identity());
}
}  // namespace domain
