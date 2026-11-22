#include <cmath>
#include <limits>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/core/math_utils.hpp"

using Catch::Matchers::WithinAbs;
using namespace vertirta::core;

namespace {

template <typename A, typename B, typename = void>
struct CanAdd : std::false_type {};

template <typename A, typename B>
struct CanAdd<A, B, std::void_t<decltype(std::declval<A>() + std::declval<B>())>> : std::true_type {};

}  // namespace

TEST_CASE("Strong frames block cross-frame addition and satisfy involutive axis conversions") {
  static_assert(CanAdd<Vec3Ned, Vec3Ned>::value, "Same-frame addition must compile");
  static_assert(!CanAdd<Vec3Ned, Vec3Enu>::value, "Cross-frame NED+ENU addition must be deleted");
  static_assert(!CanAdd<Vec3Frd, Vec3Flu>::value, "Cross-frame FRD+FLU addition must be deleted");
  static_assert(!CanAdd<Vec3Ned, Vec3Frd>::value, "Cross-frame NED+FRD addition must be deleted");

  const Vec3Enu e{10.0, 25.0, 40.0};
  const Vec3Ned n = enu_to_ned(e);
  REQUIRE_THAT(n.x, WithinAbs(25.0, 1e-12));
  REQUIRE_THAT(n.y, WithinAbs(10.0, 1e-12));
  REQUIRE_THAT(n.z, WithinAbs(-40.0, 1e-12));
  const Vec3Enu e_rt = ned_to_enu(n);
  REQUIRE_THAT(e_rt.x, WithinAbs(e.x, 1e-12));
  REQUIRE_THAT(e_rt.y, WithinAbs(e.y, 1e-12));
  REQUIRE_THAT(e_rt.z, WithinAbs(e.z, 1e-12));

  const Vec3Frd f{1.0, -2.0, -2353.6};
  const Vec3Flu fl = frd_to_flu(f);
  REQUIRE_THAT(fl.x, WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(fl.y, WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(fl.z, WithinAbs(2353.6, 1e-12));
  const Vec3Frd f_rt = flu_to_frd(fl);
  REQUIRE_THAT(f_rt.z, WithinAbs(f.z, 1e-12));
}

TEST_CASE("PX4 FRD/NED scalar-first quaternion converts accurately to ROS 2 FLU/ENU scalar-last") {
  // DR-1 Section 1.1 numerical benchmark: heading 035 deg NE, pitch +10 deg, roll +15 deg
  const QuatPx4FrdNed q_px4{0.945380, 0.098027, 0.121511, 0.286149};
  const QuatRosFluEnu q_ros = px4_to_ros_quat(q_px4);
  REQUIRE_THAT(q_ros.x, WithinAbs(0.155237, 1e-5));
  REQUIRE_THAT(q_ros.y, WithinAbs(-0.016606, 1e-5));
  REQUIRE_THAT(q_ros.z, WithinAbs(0.466147, 1e-5));
  REQUIRE_THAT(q_ros.w, WithinAbs(0.870823, 1e-5));

  const QuatPx4FrdNed rt = ros_to_px4_quat(q_ros);
  REQUIRE_THAT(rt.w, WithinAbs(q_px4.w, 1e-12));
  REQUIRE_THAT(rt.x, WithinAbs(q_px4.x, 1e-12));
  REQUIRE_THAT(rt.y, WithinAbs(q_px4.y, 1e-12));
  REQUIRE_THAT(rt.z, WithinAbs(q_px4.z, 1e-12));

  // Unit-normalized quaternion round-trip inner product > 1 - 1e-12
  const double inv_n = 1.0 / std::sqrt(q_px4.w * q_px4.w + q_px4.x * q_px4.x +
                                       q_px4.y * q_px4.y + q_px4.z * q_px4.z);
  const QuatPx4FrdNed q_unit{q_px4.w * inv_n, q_px4.x * inv_n, q_px4.y * inv_n, q_px4.z * inv_n};
  const QuatPx4FrdNed rt_unit = ros_to_px4_quat(px4_to_ros_quat(q_unit));
  const double dot = std::abs(rt_unit.w * q_unit.w + rt_unit.x * q_unit.x +
                              rt_unit.y * q_unit.y + rt_unit.z * q_unit.z);
  REQUIRE(dot > 1.0 - 1e-12);
}

TEST_CASE("Constexpr EGM2008 patch and Kartal-to-LTFJ geodesy avoid +36.7 m double-add") {
  constexpr double N_kartal = egm2008_undulation_m(40.8860, 29.1880);
  static_assert(N_kartal > 20.0 && N_kartal < 50.0, "EGM2008 patch must be constexpr and sane");
  REQUIRE_THAT(N_kartal, WithinAbs(36.7, 0.15));

  const double h_hae = amsl_to_hae(12.0, 40.8860, 29.1880);
  REQUIRE_THAT(h_hae, WithinAbs(48.7, 0.15));
  REQUIRE_THAT(hae_to_amsl(h_hae, 40.8860, 29.1880), WithinAbs(12.0, 1e-12));

  // Kartal datum maps to local ENU origin (0, 0, 0)
  const Vec3Enu origin = geodetic_to_local_enu({40.8860, 29.1880, 48.7});
  REQUIRE_THAT(origin.x, WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(origin.y, WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(origin.z, WithinAbs(0.0, 1e-6));

  // Exact LTFJ ARP (40d53m54sN, 29d18m33sE, 95.1 m AMSL + 36.58 m N = 131.68 m HAE)
  const Vec3Enu ltfj_enu = geodetic_to_local_enu({40.898333, 29.309167, 131.68});
  const double horiz_dist_m = std::hypot(ltfj_enu.x, ltfj_enu.y);
  REQUIRE_THAT(horiz_dist_m, WithinAbs(10309.0, 40.0));
  REQUIRE(horiz_dist_m > 9260.0 + 1000.0);  // > 1 km outside 5 NM LTFJ CTR

  // Round-trip ENU -> Geodetic -> ENU (< 1 mm accuracy over 15 km)
  const GeodeticWgs84 ltfj_rt = local_enu_to_geodetic(ltfj_enu);
  REQUIRE_THAT(ltfj_rt.lat_deg, WithinAbs(40.898333, 1e-7));
  REQUIRE_THAT(ltfj_rt.lon_deg, WithinAbs(29.309167, 1e-7));
  REQUIRE_THAT(ltfj_rt.h_hae_m, WithinAbs(131.68, 1e-3));
}

TEST_CASE("Freestanding clamp_f and clamp_d handle bounds and NaN safely") {
  REQUIRE_THAT(clamp_f(5.0f, 0.0f, 3.0f), WithinAbs(3.0f, 1e-6f));
  REQUIRE_THAT(clamp_f(-2.0f, 0.0f, 3.0f), WithinAbs(0.0f, 1e-6f));
  REQUIRE_THAT(clamp_d(1.5, 0.0, 3.0), WithinAbs(1.5, 1e-12));
  REQUIRE(is_finite_f(1.0f));
  REQUIRE_FALSE(is_finite_f(std::numeric_limits<float>::quiet_NaN()));
  REQUIRE_FALSE(is_finite_d(std::numeric_limits<double>::infinity()));
  // NaN input clamps safely to lo (deterministic, never propagates NaN silently)
  REQUIRE_THAT(clamp_f(std::numeric_limits<float>::quiet_NaN(), 0.0f, 3.0f), WithinAbs(0.0f, 1e-6f));
}
