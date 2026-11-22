"""Strict TDD test suite for Plan 1B Task 8 (viz/blender/ Colab G4 5.2 LTS Cycles OptiX).

Pins all DR-5 / Plan 1B Task 8 / Review Focus #2 locks:
- Driver-gated Blender 5.2 LTS (driver >= 575) / 4.5 LTS (535..574) selector,
  explicitly rejecting Blender 4.2/4.3 (no NVIDIA Blackwell sm_120 OptiX).
- Blender >= 5.0 mat.use_nodes deprecation guard and ShaderNodeMix RGBA socket
  indexing strictly via [6]/[7]/[2] (NEVER string "A"/"B"/"Result"!).
- argparse.parse_known_args() splitting cleanly after "--".
- 3-regime coaxial rotor VFX (>1200 BLUR_DISC, 0<rpm<=1200 ARC_BLADES_STEP5
  with motion_steps=5, ==0 STOPPED_BLADE + GeoNodes smoke trigger on fault tick),
  volumetric 4D CBVF funnel, and non-shadow-casting LTFJ/LTBX curtains.
- Resumable 150-frame slice planner (12 slices for 1,800 frames), 3-frame <18 GB
  VRAM preflight, hevc_nvenc -preset p5 (NEVER -preset slow!), and valid Colab
  G4 notebook JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from viz.blender.apply_vrta_and_render_setup import (
    CAMERA_CUTS_1800F,
    apply_vrta_to_scene,
    build_cbvf_funnel_mesh_data,
    build_failed_rotor_smoke_spec,
    classify_rotor_vfx_regime,
    compute_rotor_vfx_state,
    configure_cycles_blackwell_optix,
    parse_blender_script_args,
    select_camera_for_frame,
)
from viz.blender.blender_env_check import (
    build_colab_install_commands,
    parse_nvidia_driver_major,
    select_blender_release,
    validate_blender_version_for_blackwell,
)
from viz.blender.build_istanbul_scene import (
    MIX_RGBA_A_IDX,
    MIX_RGBA_B_IDX,
    MIX_RGBA_OUT_IDX,
    build_ctr_curtain_object,
    build_marmara_ocean_shader,
    build_terracotta_roof_shader,
    build_turkish_rooftop_concrete_slz_shader,
    create_material_with_nodes,
    validate_offline_scene_tos_clean,
)
from viz.blender.mux_hud_ffmpeg import (
    build_ffmpeg_nvenc_cmd,
    format_hud_telemetry_payload,
)
from viz.blender.render_slices import (
    assert_vram_within_budget,
    build_blender_slice_cmd,
    build_preflight_3frame_cmd,
    compute_frame_slices,
    parse_cycles_peak_vram_gb,
)
from viz.cesium.vrta_reader import VrtaFileHeader, VrtaFrame100Hz, write_vrta_file


class _FakeSocket:
    def __init__(self, name: str, idx: int) -> None:
        self.name = name
        self.idx = idx
        self.default_value: Any = None


class _FakeSocketCollection:
    """Mimics Blender 4.x/5.x ShaderNodeMix where string 'A'/'B'/'Result' bind hidden FLOAT sockets."""

    def __init__(self, node_type: str, is_output: bool) -> None:
        self._node_type = node_type
        self._is_output = is_output
        self._by_idx: dict[int, _FakeSocket] = {}
        self._by_name: dict[str, _FakeSocket] = {}

    def __getitem__(self, key: int | str) -> _FakeSocket:
        if isinstance(key, str):
            if self._node_type == "ShaderNodeMix" and key in {"A", "B", "Result"}:
                raise AssertionError(
                    f"DR-5 Erratum #3 violation: string lookup {key!r} on ShaderNodeMix binds hidden FLOAT socket!"
                )
            if key not in self._by_name:
                self._by_name[key] = _FakeSocket(key, -1)
            return self._by_name[key]
        if key not in self._by_idx:
            self._by_idx[key] = _FakeSocket(f"idx_{key}", key)
        return self._by_idx[key]


class _FakeNode:
    def __init__(self, node_type: str) -> None:
        self.type = node_type
        self.data_type = "FLOAT"
        self.blend_type = "MIX"
        self.operation = "ADD"
        self.wave_type = "BANDS"
        self.inputs = _FakeSocketCollection(node_type, is_output=False)
        self.outputs = _FakeSocketCollection(node_type, is_output=True)
        self.color_ramp = type(
            "ColorRamp",
            (),
            {
                "elements": [
                    type("Elem", (), {"position": 0.0, "color": (0, 0, 0, 1)})(),
                    type("Elem", (), {"position": 1.0, "color": (1, 1, 1, 1)})(),
                ]
            },
        )()


class _FakeNodeTree:
    def __init__(self) -> None:
        self.created: list[_FakeNode] = []
        self.links_created: list[tuple[_FakeSocket, _FakeSocket]] = []
        self._bsdf = _FakeNode("ShaderNodeBsdfPrincipled")

        class _Nodes:
            def __init__(self, outer: _FakeNodeTree) -> None:
                self._outer = outer

            def new(self, node_type: str) -> _FakeNode:
                node = _FakeNode(node_type)
                self._outer.created.append(node)
                return node

            def get(self, name: str) -> _FakeNode:
                return self._outer._bsdf

        class _Links:
            def __init__(self, outer: _FakeNodeTree) -> None:
                self._outer = outer

            def new(self, src: _FakeSocket, dst: _FakeSocket) -> None:
                self._outer.links_created.append((src, dst))

        self.nodes = _Nodes(self)
        self.links = _Links(self)


class _FakeMaterial:
    def __init__(self, name: str, forbid_use_nodes: bool) -> None:
        self.name = name
        self.node_tree = _FakeNodeTree()
        self._forbid_use_nodes = forbid_use_nodes
        self._use_nodes_called = False

    @property
    def use_nodes(self) -> bool:
        return True

    @use_nodes.setter
    def use_nodes(self, value: bool) -> None:
        if self._forbid_use_nodes:
            raise AssertionError("mat.use_nodes must NOT be set on Blender >= 5.0!")
        self._use_nodes_called = bool(value)


def _make_fake_bpy(version: tuple[int, int, int] = (5, 2, 4)) -> Any:
    forbid = version >= (5, 0, 0)
    materials: list[_FakeMaterial] = []
    objects_list: list[Any] = []

    class _MatStore:
        def new(self, name: str) -> _FakeMaterial:
            m = _FakeMaterial(name, forbid_use_nodes=forbid)
            materials.append(m)
            return m

    class _Mesh:
        def __init__(self, name: str) -> None:
            self.name = name
            self.verts: Any = None
            self.faces: Any = None
            self.materials: list[Any] = []

        def from_pydata(self, verts: Any, edges: Any, faces: Any) -> None:
            self.verts = verts
            self.faces = faces

        def update(self) -> None:
            pass

    class _MeshStore:
        def new(self, name: str) -> _Mesh:
            return _Mesh(name)

    class _Obj:
        def __init__(self, name: str, data: Any) -> None:
            self.name = name
            self.data = data
            self.visible_shadow = True
            self.visible_diffuse = True

    class _ObjStore:
        def new(self, name: str, data: Any) -> _Obj:
            obj = _Obj(name, data)
            objects_list.append(obj)
            return obj

    class _SceneColl:
        def __init__(self) -> None:
            self.linked: list[Any] = []

        def link(self, obj: Any) -> None:
            self.linked.append(obj)

    dev_optix = type("Dev", (), {"name": "RTX PRO 6000 Blackwell", "type": "OPTIX", "use": False})()
    dev_cpu = type("Dev", (), {"name": "Xeon CPU", "type": "CPU", "use": True})()
    cycles_prefs = type(
        "CyclesPrefs",
        (),
        {
            "compute_device_type": "NONE",
            "devices": [dev_optix, dev_cpu],
            "get_devices": lambda self: None,
        },
    )()

    scene_coll = _SceneColl()
    scene = type(
        "Scene",
        (),
        {
            "render": type("Render", (), {})(),
            "cycles": type("Cycles", (), {})(),
            "view_settings": type("ViewSettings", (), {})(),
            "collection": type("Coll", (), {"objects": scene_coll})(),
        },
    )()

    return type(
        "FakeBpy",
        (),
        {
            "app": type("App", (), {"version": version})(),
            "data": type("Data", (), {"materials": _MatStore(), "meshes": _MeshStore(), "objects": _ObjStore()})(),
            "context": type(
                "Context",
                (),
                {
                    "scene": scene,
                    "preferences": type(
                        "Prefs", (), {"addons": {"cycles": type("Addon", (), {"preferences": cycles_prefs})()}}
                    )(),
                },
            )(),
        },
    )()


def test_pbr_mix_nodes_use_rgba_indices_6_7_2_and_guard_blender5_use_nodes() -> None:
    """Review Focus #2 & DR-5 Erratum #3: ShaderNodeMix RGBA [6]/[7]/[2] + Blender >=5 use_nodes guard."""
    assert (MIX_RGBA_A_IDX, MIX_RGBA_B_IDX, MIX_RGBA_OUT_IDX) == (6, 7, 2)
    src = Path("viz/blender/build_istanbul_scene.py").read_text(encoding="utf-8")
    assert 'inputs["A"]' not in src and 'inputs["B"]' not in src and 'outputs["Result"]' not in src
    assert "inputs[MIX_RGBA_A_IDX]" in src and "inputs[MIX_RGBA_B_IDX]" in src
    assert "outputs[MIX_RGBA_OUT_IDX]" in src

    # Blender 5.2 LTS: MUST NOT touch mat.use_nodes; ShaderNodeMix links strictly via [6],[7],[2]
    bpy5 = _make_fake_bpy((5, 2, 4))
    mat_slz = build_turkish_rooftop_concrete_slz_shader(bpy5)
    mat_roof = build_terracotta_roof_shader(bpy5)
    mat_sea = build_marmara_ocean_shader(bpy5)
    assert mat_slz.name == "Mat_Istanbul_Concrete_Rooftop_SLZ"
    assert mat_roof.name == "Mat_Istanbul_Terracotta_Roof"
    assert mat_sea.name == "Mat_Marmara_Sea"
    assert any((src_s.idx == 2 or dst_s.idx in (6, 7)) for src_s, dst_s in mat_slz.node_tree.links_created)

    # Blender 4.5 LTS: MUST enable mat.use_nodes = True
    bpy45 = _make_fake_bpy((4, 5, 16))
    mat45 = create_material_with_nodes(bpy45, "Test45")
    assert mat45._use_nodes_called is True

    # Non-shadow-casting holographic LTFJ/LTBX curtains
    ltfj_obj = build_ctr_curtain_object(bpy5, "ltfj")
    ltbx_obj = build_ctr_curtain_object(bpy5, "ltbx")
    assert ltfj_obj.visible_shadow is False and ltfj_obj.visible_diffuse is False
    assert ltbx_obj.visible_shadow is False and ltbx_obj.visible_diffuse is False

    # Offline scene ToS audit rejects Bing/Esri/EOX/FABDEM
    assert validate_offline_scene_tos_clean(["viz/geo/data/kartal_15km_egm2008.tif"]) is True
    with pytest.raises(ValueError, match="ToS|CC BY-NC-SA"):
        validate_offline_scene_tos_clean(["https://tiles.maps.eox.at/s2cloudless-2024"])


def test_blender_driver_gate_blackwell_and_cli_rotor_vfx_and_smoke() -> None:
    """DR-5 Blackwell lock (5.2 >=575 / 4.5 535..574, reject 4.2/4.3), parse_known_args, 3-regime VFX."""
    assert parse_nvidia_driver_major("NVIDIA RTX PRO 6000 Blackwell, 575.51.03, 97887 MiB") == 575
    rel52 = select_blender_release(575)
    rel45 = select_blender_release(560)
    assert rel52["series"] == "5.2" and rel45["series"] == "4.5"
    with pytest.raises(RuntimeError, match="driver|535"):
        select_blender_release(525)

    assert validate_blender_version_for_blackwell((5, 2, 4)) is True
    assert validate_blender_version_for_blackwell((4, 5, 16)) is True
    for legacy in ((4, 2, 9), (4, 3, 2)):
        with pytest.raises(ValueError, match="Blackwell|sm_120"):
            validate_blender_version_for_blackwell(legacy)

    install_cmds = "\n".join(build_colab_install_commands(575))
    assert "5.2" in install_cmds and "OPTIX" in install_cmds

    # Blender CLI arg parsing after '--' via parse_known_args
    args = parse_blender_script_args(
        ["blender", "-b", "scene.blend", "--", "--cycles-device", "OPTIX", "--vrta", "a.vrta", "--samples", "128"]
    )
    assert args.vrta == "a.vrta" and args.samples == 128

    # 3-regime rotor VFX & smoke trigger
    assert (
        classify_rotor_vfx_regime(2400.0),
        classify_rotor_vfx_regime(450.0),
        classify_rotor_vfx_regime(0.0),
    ) == ("BLUR_DISC", "ARC_BLADES_STEP5", "STOPPED_BLADE")

    rotor_states = compute_rotor_vfx_state([2400.0, 2400.0, 0.0, 900.0, 2400.0, 2400.0, 2400.0, 2400.0], 0b11111011)
    assert rotor_states[2]["regime"] == "STOPPED_BLADE" and rotor_states[2]["smoke_emitter_active"] is True
    assert rotor_states[3]["regime"] == "ARC_BLADES_STEP5" and rotor_states[3]["motion_steps"] == 5
    assert rotor_states[0]["show_blur_disc"] is True

    # Cycles OptiX Blackwell setup configures GPU-only OPTIX + persistent data + AgX
    bpy5 = _make_fake_bpy((5, 2, 4))
    configure_cycles_blackwell_optix(bpy5)
    assert bpy5.context.scene.render.use_persistent_data is True
    assert bpy5.context.scene.view_settings.view_transform == "AgX"
    devs = bpy5.context.preferences.addons["cycles"].preferences.devices
    assert devs[0].use is True and devs[1].use is False

    # 4-camera 1,800-frame cuts, 4D CBVF funnel mesh, and smoke fault detection
    assert len(CAMERA_CUTS_1800F) == 4
    assert select_camera_for_frame(1)["focal_mm"] == 35
    assert select_camera_for_frame(500)["focal_mm"] == 50
    assert select_camera_for_frame(1000)["focal_mm"] == 24
    assert select_camera_for_frame(1600)["focal_mm"] == 85

    funnel = build_cbvf_funnel_mesh_data((0.0, 0.0, 60.0), agl_m=60.0, active_motor_mask=0b11111011)
    assert funnel["status"] == "DEGRADED_7_ROTOR" and len(funnel["vertices_enu_m"]) == 25

    f0 = VrtaFrame100Hz(seq=0, t_sim_us=0, active_motor_mask=0xFF)
    f1 = VrtaFrame100Hz(seq=1, t_sim_us=33_333, active_motor_mask=0b11111011)
    smoke = build_failed_rotor_smoke_spec([f0, f1], fps=30)
    assert smoke["fault_frame_1based"] == 2 and smoke["failed_rotor_indices"] == [2]


def test_slices_preflight_nvenc_p5_and_colab_notebook_json(tmp_path: Path) -> None:
    """150-frame resumable slices, <18 GB VRAM preflight, hevc_nvenc -preset p5, valid Colab ipynb."""
    slices = compute_frame_slices(total_frames=1800, slice_size=150)
    assert len(slices) == 12 and slices[0] == (1, 150) and slices[-1] == (1651, 1800)

    # Resuming after slice 1 (frames 1..150) skips slice 1 cleanly
    resumed = compute_frame_slices(total_frames=1800, slice_size=150, completed_frames=set(range(1, 151)))
    assert len(resumed) == 11 and resumed[0] == (151, 300)

    preflight_cmd = " ".join(build_preflight_3frame_cmd("m.blend", "s.py", "out/pf_####", "a.vrta"))
    assert "--cycles-device OPTIX" in preflight_cmd and "--cycles-print-stats" in preflight_cmd

    slice_cmd = " ".join(build_blender_slice_cmd("m.blend", "s.py", "out/frame_####", "a.vrta", 1, 150))
    assert "-s 1 -e 150 -a -- --cycles-device OPTIX" in slice_cmd

    peak_gb = parse_cycles_peak_vram_gb("I1122 Cycles: Peak memory usage: 14512.40 MB")
    assert peak_gb == pytest.approx(14.5124, abs=0.05)
    assert assert_vram_within_budget(peak_gb, max_gb=18.0) is True
    with pytest.raises(RuntimeError, match="VRAM"):
        assert_vram_within_budget(21.5, max_gb=18.0)

    # ffmpeg NVENC MUST use -preset p5 (NEVER -preset slow!)
    ff_cmd = " ".join(build_ffmpeg_nvenc_cmd("frames/frame_%04d.png", "hud/hud_%04d.png", "out.mp4"))
    assert "-c:v hevc_nvenc" in ff_cmd and "-preset p5" in ff_cmd and "-preset slow" not in ff_cmd
    assert "-tag:v hvc1" in ff_cmd and "-movflags +faststart" in ff_cmd
    with pytest.raises(ValueError, match="p5|slow"):
        build_ffmpeg_nvenc_cmd("frames/frame_%04d.png", "hud/hud_%04d.png", "out.mp4", preset="slow")

    hud = format_hud_telemetry_payload(
        VrtaFrame100Hz(seq=50, t_sim_us=500_000, v_term_v=384.0, soc_pct=71.5, agl_m=32.0, active_motor_mask=251),
        frame_idx=15,
        total_frames=1800,
    )
    assert hud["failed_motors"] == [2] and hud["soc_pct"] == pytest.approx(71.5)

    # Colab G4 notebook must be valid nbformat 4 JSON referencing 5.2 LTS / 4.5 LTS / 150f / p5
    nb_path = Path("viz/blender/colab_render_cinema.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 6
    nb_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    for token in ("5.2", "4.5", "575", "OPTIX", "150", "hevc_nvenc", "blender_env_check"):
        assert token in nb_src

    # End-to-end .vrta -> apply_vrta_to_scene deterministic resampling & fault smoke check
    vrta_log = tmp_path / "cinema_test.vrta"
    frames = [
        VrtaFrame100Hz(seq=i, t_sim_us=i * 10_000, pos_enu=(float(i), 0.0, 50.0), active_motor_mask=0xFF if i < 5 else 251)
        for i in range(10)
    ]
    write_vrta_file(vrta_log, VrtaFileHeader(), frames)
    bpy5 = _make_fake_bpy((5, 2, 4))
    summary = apply_vrta_to_scene(bpy5, vrta_log, fps=30)
    assert summary["raw_frames"] == 10 and summary["cinema_frames"] >= 3
    assert summary["smoke_spec"]["has_fault"] is True and summary["smoke_spec"]["failed_rotor_indices"] == [2]
