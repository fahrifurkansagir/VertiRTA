"""Contract tests pinning VertiRTA-X8-240 vehicle and Istanbul LTFJ/LTBX airspace configs."""
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]
VEHICLE_YAML = ROOT / "config" / "vehicle_x8_240.yaml"
AIRSPACE_YAML = ROOT / "config" / "istanbul_ltfj_airspace.yaml"
FORBIDDEN_PORTS = {4173, 4174, 8765, 9222, 9223}


def test_vehicle_x8_240_contracts():
    v = yaml.safe_load(VEHICLE_YAML.read_text(encoding="utf-8"))
    assert v["model_name"] == "VertiRTA-X8-240"
    assert v["mass_kg"] == 240.0
    assert v["tw_nominal"] == 2.5
    assert v["inertia_kg_m2"] == [85.0, 95.0, 140.0]

    geom = v["geometry"]
    assert geom["rotor_count"] == 8
    assert geom["arm_radius_m"] == 1.35
    assert geom["rotor_radius_m"] == 0.75
    assert geom["coax_spacing_m"] == 0.35

    prop = v["propulsion"]
    assert prop["eta_coax_hover"] == 0.85
    assert prop["tau_up_s"] == 0.085
    assert prop["tau_coast_s"] == 0.424
    assert prop["omega_max_rad_s"] == 240.0
    assert prop["ke_bus_vs_rad"] <= 1.36
    assert abs(prop["k_t_upper"] * 240.0**2 - 795.4) < 1.5
    assert abs(prop["k_t_lower"] * 240.0**2 - 676.1) < 1.5

    batt = v["battery"]
    assert batt["pack_config"] == "112S12P"
    assert batt["pack_mass_kg"] == 68.0
    assert batt["specific_energy_wh_kg"] >= 315.0
    assert batt["v_nom_v"] >= 400.0
    assert abs(batt["r0_ohm"] - 0.1363) < 1e-4
    assert abs(batt["r1_ohm"] - 0.0840) < 1e-4
    assert batt["c1_f"] == 300.0

    lim = v["limits"]
    assert lim["v_max_mps"] == 22.0
    assert lim["r_slz_min_m"] >= 5.0
    assert lim["slz_max_slope_deg"] <= 8.0
    assert lim["slz_max_roughness_m"] <= 0.15


def test_istanbul_ltfj_airspace_and_port_contracts():
    a = yaml.safe_load(AIRSPACE_YAML.read_text(encoding="utf-8"))
    datum = a["datum"]
    assert datum["lat_deg"] == 40.8860
    assert datum["lon_deg"] == 29.1880
    assert datum["alt_amsl_m"] == 12.0
    assert datum["egm2008_n_m"] == 36.7
    assert abs(datum["alt_hae_m"] - (datum["alt_amsl_m"] + datum["egm2008_n_m"])) < 1e-6

    assert a["odd_bounds"]["max_lon_near_j2_deg"] < 29.190

    ltfj = a["airspace"]["ltfj_ctr"]
    assert abs(ltfj["arp_lat_deg"] - 40.898333) < 1e-4
    assert abs(ltfj["arp_lon_deg"] - 29.309167) < 1e-4
    assert ltfj["radius_m"] == 9260.0
    assert abs(ltfj["nw_j2_lat_deg"] - 40.91279) < 1e-4
    assert abs(ltfj["nw_j2_lon_deg"] - 29.19212) < 1e-4
    assert abs(ltfj["ceiling_hae_m"] - 798.58) < 0.2

    ltbx = a["airspace"]["ltbx_ctr"]
    assert ltbx["radius_m"] == 9260.0
    assert abs(ltbx["ceiling_hae_m"] - 646.35) < 0.2

    ports = a["ports"]
    assert ports["foxglove_ws"] == 8891
    assert ports["cesium_http"] == 5180
    assert ports["micro_xrce_agent_udp"] == 8888
    assert set(ports.values()).isdisjoint(FORBIDDEN_PORTS)
