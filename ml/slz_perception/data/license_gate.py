# SPDX-License-Identifier: Apache-2.0
"""Zero-ShareAlike, Zero-NC, Defense-Safe License Gate for VertiRTA SLZ Perception.

Enforces DR-7 primary-source license locks:
  - Approved 5-source permissive training set:
      1) FLAIR-1 (OPEN LICENCE 2.0 / etalab-2.0)
      2) OpenEarthMap 41/100 programmatic whitelist (CC BY 4.0 + DL-DE-BY-2.0 ONLY)
      3) FloodNet-Supervised v1.0 (CDLA-Permissive-1.0)
      4) SkyScenes (MIT)
      5) Gazebo_Istanbul_8K (Apache-2.0 + Copernicus GLO-30; uses_osm_odbl=False)
  - Hard-banned from all splits:
      RescueNet (CC BY-NC-ND), FloodNet-VQA (CC BY-NC-ND), LoveDA (CC BY-NC-SA/GE-ToS),
      DroneDeploy (no license), FlyAwareV2 (GPL-3.0 + UAVid), SynDrone (undocumented),
      DINOv3 (ITAR/defense ban), SegFormer / MiT-B0 (NVIDIA NC), Mid-Air (CC BY-NC-SA),
      UAVid (CC BY-NC-SA), Depth-Anything-V2-Base/Large/Giant (CC-BY-NC),
      EOX Sentinel-2 Cloudless (CC BY-NC-SA), FABDEM (CC BY-NC-SA).
  - Strictly quarantined to EVAL-ONLY (never training):
      AeroScapes (CC BY-SA 4.0), ICG_Semantic_Drone (NC), UAVScenes (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


class LicenseViolationError(RuntimeError):
    """Raised whenever a dataset, region, teacher, or license violates Apache-2.0 purity."""


@dataclass(frozen=True)
class DatasetEntry:
    """Single dataset or model asset entry in a VertiRTA training/eval manifest."""

    name: str
    split: str
    license: str
    uses_osm_odbl: bool = False
    regions: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class ManifestReport:
    """Verified audit output produced by `verify_training_manifest`."""

    approved_train_entries: tuple[str, ...]
    quarantined_eval_entries: tuple[str, ...]
    oem_whitelisted_regions: frozenset[str]
    checked_count: int
    zero_sharealike_verified: bool


_EVAL_ONLY_SPLITS: frozenset[str] = frozenset({"eval", "test", "eval_only", "zero_shot_eval"})
_TRAIN_SPLITS: frozenset[str] = frozenset({"train", "val", "trainval", "finetune", "distill", "oe"})
_MODEL_SPLITS: frozenset[str] = frozenset({"teacher", "backbone", "student"})

# Normalized tokens -> canonical dataset key
_HARD_BANNED_KEYS: dict[str, str] = {
    "rescuenet": "RescueNet (CC BY-NC-ND 4.0)",
    "floodnetvqa": "FloodNet-VQA (CC BY-NC-ND)",
    "loveda": "LoveDA (CC BY-NC-SA / Academic + Google Earth ToS)",
    "dronedeploy": "DroneDeploy (no open license)",
    "flyawarev2": "FlyAwareV2 (GPL-3.0 + UAVid)",
    "flyaware": "FlyAwareV2 (GPL-3.0 + UAVid)",
    "syndrone": "SynDrone (undocumented data license)",
    "dinov3": "DINOv3 (ITAR / defense prohibition)",
    "segformer": "SegFormer / MiT (NVIDIA Non-Commercial)",
    "mitb0": "SegFormer MiT-B0 (NVIDIA Non-Commercial)",
    "midair": "Mid-Air (CC BY-NC-SA 4.0)",
    "uavid": "UAVid (CC BY-NC-SA 4.0)",
    "depthanythingv2base": "Depth-Anything-V2-Base (CC-BY-NC-4.0)",
    "depthanythingv2large": "Depth-Anything-V2-Large (CC-BY-NC-4.0)",
    "depthanythingv2giant": "Depth-Anything-V2-Giant (CC-BY-NC-4.0)",
    "dav2base": "Depth-Anything-V2-Base (CC-BY-NC-4.0)",
    "dav2large": "Depth-Anything-V2-Large (CC-BY-NC-4.0)",
    "dav2giant": "Depth-Anything-V2-Giant (CC-BY-NC-4.0)",
    "dav2b": "Depth-Anything-V2-Base (CC-BY-NC-4.0)",
    "dav2l": "Depth-Anything-V2-Large (CC-BY-NC-4.0)",
    "dav2g": "Depth-Anything-V2-Giant (CC-BY-NC-4.0)",
    "eox": "EOX Sentinel-2 Cloudless (CC BY-NC-SA 4.0)",
    "fabdem": "FABDEM (CC BY-NC-SA 4.0)",
}

_EVAL_ONLY_QUARANTINE: dict[str, str] = {
    "aeroscape": "AeroScapes",
    "aeroscapes": "AeroScapes",
    "aeroscapesccbysa4": "AeroScapes",
    "icg": "ICG_Semantic_Drone",
    "icgsemanticdrone": "ICG_Semantic_Drone",
    "icgsemanticdronenc": "ICG_Semantic_Drone",
    "uavscenes": "UAVScenes",
    "uavscenesccbyncsa4": "UAVScenes",
}

_APPROVED_TRAIN_DATASETS: dict[str, str] = {
    "flair1": "FLAIR-1",
    "flair12": "FLAIR-1",
    "flair1etalab2": "FLAIR-1",
    "openearthmap": "OpenEarthMap",
    "openearthmap41": "OpenEarthMap",
    "openearthmapfiltered41entries": "OpenEarthMap",
    "oem41": "OpenEarthMap",
    "floodnetsupervisedv10": "FloodNet-Supervised-v1.0",
    "floodnetsupervised": "FloodNet-Supervised-v1.0",
    "skyscenes": "SkyScenes",
    "skyscenesmit": "SkyScenes",
    "gazeboistanbul8k": "Gazebo_Istanbul_8K",
    "gazeboistanbul8kauthor": "Gazebo_Istanbul_8K",
}

_APPROVED_MODEL_ASSETS: dict[str, str] = {
    "dinov2vitl14": "DINOv2-ViT-L/14",
    "dinov2vitb14": "DINOv2-ViT-B/14",
    "depthanythingv2small": "Depth-Anything-V2-Small",
    "depthanythingv2vits": "Depth-Anything-V2-Small",
    "dav2small": "Depth-Anything-V2-Small",
    "pidnets": "PIDNet-S",
}

_OEM_ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "cc by 4.0",
        "cc-by-4.0",
        "cc-by 4.0",
        "cc by 4.0 international",
        "creative commons attribution 4.0",
        "creative commons attribution 4.0 international",
        "dl-de-by-2.0",
        "dl-de/by-2-0",
        "dl de by 2.0",
        "data licence germany - attribution - version 2.0",
        "data licence germany – attribution – version 2.0",
    }
)

_FORBIDDEN_LICENSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bby[-_\s]*sa\b", re.IGNORECASE),
    re.compile(r"\bby[-_\s]*nc\b", re.IGNORECASE),
    re.compile(r"\bby[-_\s]*nd\b", re.IGNORECASE),
    re.compile(r"\bnc[-_\s]*sa\b", re.IGNORECASE),
    re.compile(r"\bnc[-_\s]*nd\b", re.IGNORECASE),
    re.compile(r"share\s*alike", re.IGNORECASE),
    re.compile(r"non[-\s]*commercial", re.IGNORECASE),
    re.compile(r"no[-\s]*derivatives", re.IGNORECASE),
    re.compile(r"\b(?:a|l)?gpl\b", re.IGNORECASE),
    re.compile(r"\bodbl\b", re.IGNORECASE),
    re.compile(r"\bpublic\s+domain\b", re.IGNORECASE),
    re.compile(r"\bproprietary\b", re.IGNORECASE),
    re.compile(r"\bunknown\b", re.IGNORECASE),
    re.compile(r"\bitar\b", re.IGNORECASE),
)


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _normalize_license_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class _HtmlTableRowParser(HTMLParser):
    """Minimal dependency-free HTML table row extractor for OpenEarthMap attribution.html."""

    def __init__(self) -> None:
        super().__init__()
        self.header_rows: list[list[str]] = []
        self.data_rows: list[list[str]] = []
        self._in_tr = False
        self._in_cell = False
        self._cell_is_th = False
        self._row_has_td = False
        self._current_cells: list[str] = []
        self._cell_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t == "tr":
            self._in_tr = True
            self._row_has_td = False
            self._current_cells = []
        elif self._in_tr and t in ("td", "th"):
            self._in_cell = True
            self._cell_is_th = t == "th"
            if t == "td":
                self._row_has_td = True
            self._cell_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("td", "th") and self._in_cell:
            self._current_cells.append("".join(self._cell_chunks).strip())
            self._in_cell = False
            self._cell_chunks = []
        elif t == "tr" and self._in_tr:
            if self._current_cells:
                if self._row_has_td:
                    self.data_rows.append(self._current_cells)
                else:
                    self.header_rows.append(self._current_cells)
            self._in_tr = False


def parse_oem_permissive_whitelist(
    attribution_html: Path | str,
    expect_total: int | None = 100,
) -> set[str]:
    """Parse OpenEarthMap attribution.html and return ONLY CC BY 4.0 + DL-DE-BY-2.0 regions.

    Public Domain, blank/unspecified, CC BY-SA 4.0, and CC BY-NC-SA 4.0 entries are strictly
    excluded because OpenEarthMap Zenodo terms place Public Domain / blank regions under
    CC BY-NC-SA 4.0.
    """
    html_path = Path(attribution_html)
    if not html_path.is_file():
        raise LicenseViolationError(
            f"OpenEarthMap attribution HTML not found at '{html_path}'; cannot verify 41/100 whitelist."
        )

    parser = _HtmlTableRowParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    parser.close()

    rows = parser.data_rows
    if expect_total is not None and len(rows) != expect_total:
        raise LicenseViolationError(
            f"OpenEarthMap attribution row count mismatch: expected {expect_total}, found {len(rows)}."
        )

    license_col: int | None = None
    region_col = 0
    for hdr in parser.header_rows:
        lower_hdr = [c.strip().lower() for c in hdr]
        for idx, col_name in enumerate(lower_hdr):
            if "licen" in col_name:
                license_col = idx
            if "region" in col_name or "city" in col_name or "dataset" in col_name:
                region_col = idx

    whitelisted: set[str] = set()
    for row in rows:
        if len(row) < 2:
            continue
        region = row[region_col].strip().lower() if region_col < len(row) else ""
        if not region:
            continue
        lic_idx = license_col if (license_col is not None and license_col < len(row)) else len(row) - 1
        lic_norm = _normalize_license_text(row[lic_idx])
        if lic_norm in _OEM_ALLOWED_LICENSES:
            whitelisted.add(region)

    if expect_total == 100 and len(whitelisted) != 41:
        raise LicenseViolationError(
            f"Official OpenEarthMap 100-entry whitelist must yield exactly 41 permissive regions "
            f"(35 CC BY 4.0 + 6 DL-DE-BY-2.0), got {len(whitelisted)}."
        )

    return whitelisted


def _check_hard_banned(name: str) -> None:
    key = _normalize_key(name)
    for banned_key, reason in _HARD_BANNED_KEYS.items():
        if key == banned_key or key.startswith(banned_key):
            raise LicenseViolationError(f"Hard-banned asset '{name}': {reason}.")


def _check_license_clean(entry: DatasetEntry) -> None:
    lic = entry.license.strip()
    if not lic:
        raise LicenseViolationError(
            f"Training entry '{entry.name}' has an empty license string (defaults to all rights reserved)."
        )
    for pat in _FORBIDDEN_LICENSE_PATTERNS:
        if pat.search(lic):
            raise LicenseViolationError(
                f"Training entry '{entry.name}' carries forbidden non-permissive license '{entry.license}'."
            )


def verify_training_manifest(
    entries: Iterable[DatasetEntry],
    oem_attribution_html: Path | str | None = None,
    *,
    expect_oem_total: int | None = 100,
) -> ManifestReport:
    """Verify every asset in a training/evaluation manifest against VertiRTA zero-ShareAlike locks.

    Raises:
        LicenseViolationError: On any banned asset, eval-only asset in a training split,
            OSM ODbL contamination in training, non-permissive license taint, unknown
            training dataset, or unlisted OpenEarthMap region.
    """
    entry_list = list(entries)
    if not entry_list:
        raise LicenseViolationError("Manifest must contain at least one dataset entry.")

    approved_train: list[str] = []
    quarantined_eval: list[str] = []
    oem_whitelist: frozenset[str] = frozenset()

    for entry in entry_list:
        _check_hard_banned(entry.name)

        split = entry.split.strip().lower()
        key = _normalize_key(entry.name)

        if key in _EVAL_ONLY_QUARANTINE:
            canonical_eval = _EVAL_ONLY_QUARANTINE[key]
            if split not in _EVAL_ONLY_SPLITS:
                raise LicenseViolationError(
                    f"Eval-only quarantined dataset '{entry.name}' ({canonical_eval}) "
                    f"cannot be used with split='{entry.split}'."
                )
            quarantined_eval.append(canonical_eval)
            continue

        if split in _EVAL_ONLY_SPLITS:
            quarantined_eval.append(entry.name)
            continue

        if entry.uses_osm_odbl:
            raise LicenseViolationError(
                f"Dataset '{entry.name}' sets uses_osm_odbl=True; OSM ODbL footprints are strictly "
                f"isolated from training weights."
            )

        _check_license_clean(entry)

        if split in _MODEL_SPLITS:
            if key not in _APPROVED_MODEL_ASSETS:
                raise LicenseViolationError(
                    f"Unapproved model asset '{entry.name}' for split='{entry.split}'."
                )
            continue

        if split not in _TRAIN_SPLITS:
            raise LicenseViolationError(
                f"Unrecognized split '{entry.split}' for entry '{entry.name}'."
            )

        if key not in _APPROVED_TRAIN_DATASETS:
            raise LicenseViolationError(
                f"Dataset '{entry.name}' is not in the locked 5-source permissive training allowlist."
            )

        canonical_train = _APPROVED_TRAIN_DATASETS[key]
        if canonical_train == "OpenEarthMap":
            if oem_attribution_html is None:
                raise LicenseViolationError(
                    "OpenEarthMap training entry requires oem_attribution_html to verify 41/100 whitelist."
                )
            if not oem_whitelist:
                oem_whitelist = frozenset(
                    parse_oem_permissive_whitelist(
                        oem_attribution_html, expect_total=expect_oem_total
                    )
                )
            if not entry.regions:
                raise LicenseViolationError(
                    "OpenEarthMap training entry must specify explicit whitelisted region names."
                )
            bad_regions = sorted(
                r for r in entry.regions if r.strip().lower() not in oem_whitelist
            )
            if bad_regions:
                raise LicenseViolationError(
                    f"OpenEarthMap regions not in permissive CC BY 4.0 / DL-DE-BY-2.0 whitelist: {bad_regions}"
                )

        approved_train.append(canonical_train)

    return ManifestReport(
        approved_train_entries=tuple(approved_train),
        quarantined_eval_entries=tuple(quarantined_eval),
        oem_whitelisted_regions=oem_whitelist,
        checked_count=len(entry_list),
        zero_sharealike_verified=True,
    )
