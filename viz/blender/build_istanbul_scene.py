"""Procedural Istanbul Coastal PBR Materials & Non-Shadow-Casting CTR Curtain Builder.

Enforces DR-5 Erratum #3 / Review Focus #2 / Plan 1B Task 8 locks:
- ``ShaderNodeMix`` with ``data_type = 'RGBA'`` MUST bind color sockets strictly
  via numeric indices ``MIX_RGBA_A_IDX = 6``, ``MIX_RGBA_B_IDX = 7``, and
  ``MIX_RGBA_OUT_IDX = 2``. String lookups (``"A"``, ``"B"``, ``"Result"``)
  silently bind Blender's hidden ``FLOAT`` sockets and turn roofs flat gray!
- Blender >= 5.0 deprecates ``mat.use_nodes = True``; guarded via
  ``if bpy_mod.app.version < (5, 0, 0): mat.use_nodes = True``.
- Translucent holographic ``LTFJ``/``LTBX`` CTR walls consume exact DHMI ENU
  curtain meshes from ``viz.geo.airspace_polygons`` and enforce
  ``visible_shadow = False`` and ``visible_diffuse = False``.
- Layer-2 ToS firewall rejects any Bing/Esri/Google/EOX/FABDEM/Cesium ion asset.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from viz.geo.airspace_polygons import build_enu_curtain_mesh_data
from viz.geo.build_istanbul_corridor import validate_asset_url_tos
from viz.geo.vertirta_geodesy import ANCHOR_LAT_DEG, ANCHOR_LON_DEG

# CRITICAL (DR-5 Erratum #3 & Review Focus #2):
# Blender 4.x / 5.x ShaderNodeMix (data_type='RGBA') numeric socket indices.
# Index 0=Factor(Float), 1=Factor(Vector), 2=A(Float), 3=B(Float),
# 4=A(Vector), 5=B(Vector), 6=A(RGBA), 7=B(RGBA); Outputs: 0=Float, 1=Vector, 2=RGBA.
MIX_RGBA_A_IDX: int = 6
MIX_RGBA_B_IDX: int = 7
MIX_RGBA_OUT_IDX: int = 2

_FORBIDDEN_LOCAL_TOKENS: tuple[str, ...] = (
    "eox",
    "s2cloudless",
    "fabdem",
    "bing",
    "virtualearth",
    "arcgisonline",
    "esri_world_imagery",
    "cesium_ion",
    "google_3d_tiles",
)


def create_material_with_nodes(bpy_mod: Any, name: str) -> Any:
    """Create a Blender material with Blender 5.0+ ``use_nodes`` deprecation guard."""
    mat = bpy_mod.data.materials.new(name=name)
    if tuple(bpy_mod.app.version) < (5, 0, 0):
        mat.use_nodes = True
    return mat


def validate_offline_scene_tos_clean(asset_paths_or_urls: Sequence[str]) -> bool:
    """Audit every texture/DEM/vector path or URL entering the offline .blend scene.

    Raises:
        ValueError: If any URL fails ``validate_asset_url_tos`` or any local path
            references forbidden CC BY-NC-SA or commercial scraped assets.
    """
    for item in asset_paths_or_urls:
        raw = str(item).strip()
        if "://" in raw:
            validate_asset_url_tos(raw)
        else:
            lowered = raw.lower()
            for tok in _FORBIDDEN_LOCAL_TOKENS:
                if tok in lowered:
                    raise ValueError(
                        f"ToS / CC BY-NC-SA violation in offline Blender scene asset {raw!r} "
                        f"(matched forbidden token {tok!r})"
                    )
    return True


def build_turkish_rooftop_concrete_slz_shader(bpy_mod: Any) -> Any:
    """1 m AGL procedural Kartal/Dragos reinforced-concrete rooftop SLZ PBR shader."""
    mat = create_material_with_nodes(bpy_mod, "Mat_Istanbul_Concrete_Rooftop_SLZ")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")

    tex_coord = nodes.new("ShaderNodeTexCoord")

    noise_macro = nodes.new("ShaderNodeTexNoise")
    noise_macro.inputs["Scale"].default_value = 0.35
    noise_macro.inputs["Detail"].default_value = 8.0
    noise_macro.inputs["Roughness"].default_value = 0.65

    noise_micro = nodes.new("ShaderNodeTexNoise")
    noise_micro.inputs["Scale"].default_value = 18.0
    noise_micro.inputs["Detail"].default_value = 10.0

    joints = nodes.new("ShaderNodeTexBrick")
    joints.inputs["Scale"].default_value = 0.45
    joints.inputs["Mortar Size"].default_value = 0.012
    joints.inputs["Mortar Smooth"].default_value = 0.2
    joints.inputs["Color1"].default_value = (0.34, 0.33, 0.31, 1.0)
    joints.inputs["Color2"].default_value = (0.28, 0.28, 0.27, 1.0)
    joints.inputs["Mortar"].default_value = (0.09, 0.09, 0.09, 1.0)

    mix_col = nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.blend_type = "MULTIPLY"
    mix_col.inputs[0].default_value = 0.32

    ramp_rough = nodes.new("ShaderNodeValToRGB")
    ramp_rough.color_ramp.elements[0].position = 0.25
    ramp_rough.color_ramp.elements[0].color = (0.68, 0.68, 0.68, 1.0)
    ramp_rough.color_ramp.elements[1].position = 0.80
    ramp_rough.color_ramp.elements[1].color = (0.94, 0.94, 0.94, 1.0)

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.015

    links.new(tex_coord.outputs["Object"], noise_macro.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], noise_micro.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], joints.inputs["Vector"])
    # STRICT numeric RGBA sockets [6], [7], [2] (NEVER string keys!)
    links.new(joints.outputs["Color"], mix_col.inputs[MIX_RGBA_A_IDX])
    links.new(noise_macro.outputs["Fac"], mix_col.inputs[MIX_RGBA_B_IDX])
    links.new(mix_col.outputs[MIX_RGBA_OUT_IDX], bsdf.inputs["Base Color"])
    links.new(noise_macro.outputs["Fac"], ramp_rough.inputs["Fac"])
    links.new(ramp_rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(noise_micro.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Specular IOR Level"].default_value = 0.42
    bsdf.inputs["Coat Weight"].default_value = 0.0
    return mat


def build_terracotta_roof_shader(bpy_mod: Any) -> Any:
    """Procedural Anatolian coastal terracotta tile roof PBR shader."""
    mat = create_material_with_nodes(bpy_mod, "Mat_Istanbul_Terracotta_Roof")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")

    tex_coord = nodes.new("ShaderNodeTexCoord")
    tiles = nodes.new("ShaderNodeTexBrick")
    tiles.inputs["Scale"].default_value = 2.4
    tiles.inputs["Color1"].default_value = (0.46, 0.16, 0.08, 1.0)
    tiles.inputs["Color2"].default_value = (0.36, 0.12, 0.06, 1.0)
    tiles.inputs["Mortar"].default_value = (0.14, 0.09, 0.06, 1.0)

    weather = nodes.new("ShaderNodeTexNoise")
    weather.inputs["Scale"].default_value = 1.6
    weather.inputs["Detail"].default_value = 6.0

    mix_col = nodes.new("ShaderNodeMix")
    mix_col.data_type = "RGBA"
    mix_col.blend_type = "MULTIPLY"
    mix_col.inputs[0].default_value = 0.25

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.30
    bump.inputs["Distance"].default_value = 0.025

    links.new(tex_coord.outputs["Object"], tiles.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], weather.inputs["Vector"])
    links.new(tiles.outputs["Color"], mix_col.inputs[MIX_RGBA_A_IDX])
    links.new(weather.outputs["Fac"], mix_col.inputs[MIX_RGBA_B_IDX])
    links.new(mix_col.outputs[MIX_RGBA_OUT_IDX], bsdf.inputs["Base Color"])
    links.new(tiles.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    bsdf.inputs["Roughness"].default_value = 0.76
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    return mat


def build_marmara_ocean_shader(bpy_mod: Any) -> Any:
    """Sea of Marmara specular water with dual-scale procedural swell and ripples."""
    mat = create_material_with_nodes(bpy_mod, "Mat_Marmara_Sea")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes.get("Principled BSDF")

    tex_coord = nodes.new("ShaderNodeTexCoord")
    swell = nodes.new("ShaderNodeTexWave")
    swell.wave_type = "BANDS"
    swell.inputs["Scale"].default_value = 0.08
    swell.inputs["Distortion"].default_value = 4.5
    swell.inputs["Detail"].default_value = 5.0

    ripples = nodes.new("ShaderNodeTexNoise")
    ripples.inputs["Scale"].default_value = 1.8
    ripples.inputs["Detail"].default_value = 9.0
    ripples.inputs["Roughness"].default_value = 0.55

    mix_h = nodes.new("ShaderNodeMath")
    mix_h.operation = "ADD"
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.08

    bsdf.inputs["Base Color"].default_value = (0.012, 0.044, 0.068, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.045
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Specular IOR Level"].default_value = 0.55
    bsdf.inputs["Coat Weight"].default_value = 0.15

    links.new(tex_coord.outputs["Object"], swell.inputs["Vector"])
    links.new(tex_coord.outputs["Object"], ripples.inputs["Vector"])
    links.new(swell.outputs["Fac"], mix_h.inputs[0])
    links.new(ripples.outputs["Fac"], mix_h.inputs[1])
    links.new(mix_h.outputs["Value"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def build_nishita_istanbul_sky(
    bpy_mod: Any,
    *,
    lat_deg: float = ANCHOR_LAT_DEG,
    lon_deg: float = ANCHOR_LON_DEG,
    sun_elevation_deg: float = 24.0,
    sun_rotation_deg: float = 235.0,
) -> dict[str, Any]:
    """Configure physical Nishita sky locked to Istanbul late-afternoon cross-light."""
    world = getattr(bpy_mod.context.scene, "world", None)
    spec = {
        "sky_type": "NISHITA",
        "lat_deg": float(lat_deg),
        "lon_deg": float(lon_deg),
        "sun_elevation_rad": math.radians(sun_elevation_deg),
        "sun_rotation_rad": math.radians(sun_rotation_deg),
        "air_density": 1.15,
        "dust_density": 1.4,
        "ozone_density": 1.0,
    }
    if world is not None and getattr(world, "node_tree", None) is not None:
        nt = world.node_tree
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = "NISHITA"
        sky.sun_elevation = spec["sun_elevation_rad"]
        sky.sun_rotation = spec["sun_rotation_rad"]
        sky.air_density = spec["air_density"]
        sky.dust_density = spec["dust_density"]
        bg = nt.nodes.get("Background")
        if bg is not None:
            nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    return spec


def build_ctr_curtain_material(bpy_mod: Any, zone: str) -> Any:
    """Holographic translucent LayerWeight -> MixShader (Transparent + Emission) CTR wall."""
    key = zone.strip().lower()
    if key not in {"ltfj", "ltbx"}:
        raise ValueError(f"Unsupported CTR zone {zone!r}; expected 'ltfj' or 'ltbx'")
    mat = create_material_with_nodes(bpy_mod, f"Mat_CTR_{key.upper()}_Curtain")
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.inputs["Blend"].default_value = 0.25
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (
        (0.95, 0.14, 0.12, 1.0) if key == "ltfj" else (1.0, 0.68, 0.08, 1.0)
    )
    emission.inputs["Strength"].default_value = 2.2
    mix_sh = nodes.new("ShaderNodeMixShader")
    out_mat = nodes.get("Material Output")

    links.new(layer_weight.outputs["Facing"], mix_sh.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix_sh.inputs[1])
    links.new(emission.outputs["Emission"], mix_sh.inputs[2])
    links.new(mix_sh.outputs["Shader"], out_mat.inputs["Surface"])
    return mat


def build_ctr_curtain_object(
    bpy_mod: Any, zone: str, *, arc_segments: int = 360
) -> Any:
    """Build non-shadow-casting 3D ENU CTR wall mesh for LTFJ or LTBX."""
    mesh_data = build_enu_curtain_mesh_data(zone, arc_segments=arc_segments)
    icao = mesh_data["icao"]
    mesh = bpy_mod.data.meshes.new(name=f"Mesh_CTR_{icao}")
    mesh.from_pydata(mesh_data["vertices_enu_m"], [], mesh_data["quad_faces"])
    mesh.update()
    mat = build_ctr_curtain_material(bpy_mod, zone)
    if hasattr(mesh.materials, "append"):
        mesh.materials.append(mat)
    obj = bpy_mod.data.objects.new(name=f"CTR_{icao}_Wall", data=mesh)
    # CRITICAL (DR-5): holographic airspace walls must NEVER cast fake shadows on Istanbul!
    obj.visible_shadow = False
    obj.visible_diffuse = False
    bpy_mod.context.scene.collection.objects.link(obj)
    return obj
