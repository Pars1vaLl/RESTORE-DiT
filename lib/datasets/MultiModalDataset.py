"""
MultiModalDataset — flexible multi-modal satellite dataset for cloud removal
and related downstream tasks (segmentation, change detection).

Supported modalities
--------------------
  • Sentinel-2  (optical, 10–13 bands, 10 m)      — required
  • Sentinel-1  (SAR, 2 bands VV+VH, 10 m)         — optional (use_s1)
  • Gaofen-1    (optical, 4 bands BGRN, 8–16 m)    — optional (use_gaofen1)

Initialisation modes
--------------------
  1. Manifest CSV (recommended):
       Pass `manifest_path` pointing to a CSV file.
       Required columns : tile_id, s2_cloudy_path, s2_cloudfree_path, split
       Optional columns : s1_path, gaofen1_path, region, date, bounds, crs,
                          valid_ratio, cloud_ratio

  2. Folder structure:
       Pass `root` containing these sub-directories:
         root/s2_cloudy/      → <tile_id>.npy  (C, H, W) or (T, C, H, W)
         root/s2_cloudfree/   → <tile_id>.npy
         root/s1/             → <tile_id>.npy  (optional)
         root/gaofen1/        → <tile_id>.npy  (optional)
       A manifest CSV inside root/ is auto-detected if it exists.

Normalisation assumptions
-------------------------
  Sentinel-2:
    • Input DN assumed to be surface reflectance × 10 000 (ESA convention).
    • Clamped to [0, s2_norm_max] then divided → [0, 1].
    • Default s2_norm_max = 10 000 (full reflectance range).
      Use 8 000 to match the PASTIS convention.
    • If rescale=True: further mapped to [-1, 1] via Normalize(0.5, 0.5).

  Sentinel-1:
    • Two normalisation modes:
        a) stats-file (default when s1_norm_stats_path is provided):
             JSON with {"mean": [...], "std": [...]} per channel.
             z-score → clamp [-2, 2] → divide by 2  →  [-1, 1].
        b) linear (fallback):
             Assumes linear amplitude/power values already in [0, 1].
             Or enable s1_input_db=True for dB input → clip [-25, 0] → [0, 1].
    • Optionally rescaled to [-1, 1] when rescale=True.

  Gaofen-1:
    • Same philosophy as S2 (optical sensor).
    • Clamped to [0, gf1_norm_max] then divided → [0, 1].
    • Default gf1_norm_max = 3 000 (surface reflectance × 10 000, land < 0.30).
    • If your pipeline stores raw 12-bit DN: set gf1_norm_max = 4 095.
    • If rescale=True: further mapped to [-1, 1] via Normalize(0.5, 0.5).

Output sample format
--------------------
  {
    "tile_id"  : str,
    "input"    : {
        "s2"      : Tensor (C, H, W) or (T, C, H, W),
        "s1"      : Tensor or None,
        "gaofen1" : Tensor or None,
    },
    "target"   : {
        "s2_cloudfree" : Tensor (same shape as input s2),
    },
    "meta"     : {
        "region"      : str,
        "date"        : str,
        "paths"       : dict,
        "bounds"      : str,
        "crs"         : str,
        "valid_ratio" : float,
        "cloud_ratio" : float,
        "gf1_valid"   : bool,   # False when GF-1 file was missing/invalid
        "s1_valid"    : bool,
    },
  }

Downstream changes required when switching from PASTISDataset
--------------------------------------------------------------
  • batch['x']               → batch['input']['s2']
  • batch['cond']            → batch['input']['s1']
  • batch['y']               → batch['target']['s2_cloudfree']
  • batch['position_days']   → NOT included (MultiModalDataset is tile-based,
                               not temporal-sequence-based by default)
  See README or the "Minimal downstream changes" section in the PR for details.

Extension points
----------------
  For rice segmentation  : add a 'seg_path' column to the manifest and return
                           batch['target']['seg_mask'].
  For change detection   : add 'date_a_path' / 'date_b_path' columns and return
                           both in batch['input'].
"""

import csv
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as tdata
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Sensor constants ───────────────────────────────────────────────────────────

# Sentinel-2
S2_DEFAULT_NORM_MAX: float = 10_000.0   # reflectance × 10000 ceiling
S2_NUM_BANDS: int = 10                  # full PASTIS band count (B2–B8A, B11, B12)

# Sentinel-1 (SAR, linear amplitude)
S1_NUM_BANDS: int = 2                   # VV, VH
S1_DB_MIN: float = -25.0               # dB lower bound for land surfaces
S1_DB_MAX: float = 0.0                 # dB upper bound

# Gaofen-1 multispectral (BGRN)
GF1_NUM_BANDS: int = 4                 # Blue, Green, Red, NIR
GF1_DEFAULT_NORM_MAX: float = 3_000.0  # typical land surface reflectance × 10000


# ── Utilities ──────────────────────────────────────────────────────────────────

def _safe_float(val: Union[str, float, None], default: float = 0.0) -> float:
    """Convert a manifest CSV string value to float, falling back to default."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Main dataset class ─────────────────────────────────────────────────────────

class MultiModalDataset(tdata.Dataset):
    """
    Flexible multi-modal satellite dataset.

    See module docstring for full documentation.
    """

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(
        self,
        # ── Data source ───────────────────────────────────────────────────
        root: Optional[str] = None,
        manifest_path: Optional[str] = None,
        split: str = "train",

        # ── Modality toggles ──────────────────────────────────────────────
        use_s1: bool = True,
        use_gaofen1: bool = False,

        # ── Normalisation ─────────────────────────────────────────────────
        s2_norm_max: float = S2_DEFAULT_NORM_MAX,
        gf1_norm_max: float = GF1_DEFAULT_NORM_MAX,
        # Path to a JSON file with {"mean": [...], "std": [...]} for S1.
        # If None, falls back to linear normalisation.
        s1_norm_stats_path: Optional[str] = None,
        # Set True if Sentinel-1 values are stored in dB (default: linear).
        s1_input_db: bool = False,
        # Rescale all tensors from [0, 1] → [-1, 1] after normalisation.
        rescale: bool = True,

        # ── Shape constraints ─────────────────────────────────────────────
        # Expected spatial tile size (H, W).  Used for cross-modal shape checks.
        tile_size: Tuple[int, int] = (256, 256),
        # Expected channel counts per modality (used for sanity checks).
        s2_expected_channels: Optional[int] = None,   # None → no check
        s1_expected_channels: Optional[int] = S1_NUM_BANDS,
        gf1_expected_channels: int = GF1_NUM_BANDS,

        # ── Misc ──────────────────────────────────────────────────────────
        # If True, include file paths in the 'meta' dict (useful for debugging).
        return_paths: bool = True,
        # Absorb unknown keyword arguments from OmegaConf config merges.
        **kwargs,
    ):
        super().__init__()

        # ── Validate inputs ───────────────────────────────────────────────
        if root is None and manifest_path is None:
            raise ValueError(
                "Provide at least one of `root` (folder-structure mode) "
                "or `manifest_path` (CSV mode)."
            )

        # ── Store configuration ───────────────────────────────────────────
        self.root = root
        self.split = split
        self.use_s1 = use_s1
        self.use_gaofen1 = use_gaofen1

        self.s2_norm_max = s2_norm_max
        self.gf1_norm_max = gf1_norm_max
        self.s1_input_db = s1_input_db
        self.rescale = rescale

        self.tile_size = tile_size            # (H, W)
        self.s2_expected_channels = s2_expected_channels
        self.s1_expected_channels = s1_expected_channels
        self.gf1_expected_channels = gf1_expected_channels

        self.return_paths = return_paths

        # ── S1 normalisation stats (optional) ────────────────────────────
        self.s1_norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if s1_norm_stats_path is not None:
            self.s1_norm_stats = self._load_s1_norm_stats(s1_norm_stats_path)

        # ── Build sample list ─────────────────────────────────────────────
        if manifest_path is not None:
            self.samples = self._build_samples_from_manifest(manifest_path, split)
            logger.info(
                f"[MultiModalDataset] Loaded {len(self.samples)} samples "
                f"(split='{split}') from manifest: {manifest_path}"
            )
        else:
            # Auto-detect manifest inside root/
            auto_manifest = os.path.join(root, "manifest.csv")
            if os.path.isfile(auto_manifest):
                self.samples = self._build_samples_from_manifest(auto_manifest, split)
                logger.info(
                    f"[MultiModalDataset] Auto-detected manifest at {auto_manifest}. "
                    f"Loaded {len(self.samples)} samples."
                )
            else:
                self.samples = self._build_samples_from_folder(root, split)
                logger.info(
                    f"[MultiModalDataset] Scanned folder structure under {root}. "
                    f"Loaded {len(self.samples)} samples."
                )

        if len(self.samples) == 0:
            warnings.warn(
                f"[MultiModalDataset] No samples found for split='{split}'. "
                "Check your manifest or folder structure.",
                UserWarning,
            )

        # Log which modalities are active
        active = ["S2"]
        if use_s1:
            active.append("S1")
        if use_gaofen1:
            active.append("Gaofen-1")
        logger.info(f"[MultiModalDataset] Active modalities: {', '.join(active)}")
        if use_gaofen1:
            logger.info(
                f"[MultiModalDataset] GF-1 normalization max = {gf1_norm_max}. "
                f"Expected channels = {gf1_expected_channels}."
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.samples[idx]
        tile_id = str(meta.get("tile_id", idx))

        # ── Load Sentinel-2 cloudy (model input) ──────────────────────────
        s2_cloudy = self._load_raster(
            meta["s2_cloudy_path"],
            name="S2 cloudy",
            tile_id=tile_id,
            expected_channels=self.s2_expected_channels,
        )
        s2_cloudy = self._normalize_s2(s2_cloudy)

        # ── Load Sentinel-2 cloud-free (model target) ─────────────────────
        s2_cloudfree = self._load_raster(
            meta["s2_cloudfree_path"],
            name="S2 cloud-free",
            tile_id=tile_id,
            expected_channels=self.s2_expected_channels,
        )
        s2_cloudfree = self._normalize_s2(s2_cloudfree)

        # ── Sanity check: S2 cloudy and cloud-free must have the same shape ─
        if s2_cloudy.shape != s2_cloudfree.shape:
            raise ValueError(
                f"[tile {tile_id}] Shape mismatch between S2 cloudy "
                f"{tuple(s2_cloudy.shape)} and cloud-free {tuple(s2_cloudfree.shape)}."
            )

        # ── Load Sentinel-1 (optional) ────────────────────────────────────
        s1 = None
        s1_valid = False
        if self.use_s1:
            s1_path = meta.get("s1_path", "")
            if s1_path:
                s1, s1_valid = self._load_and_normalize_s1(
                    s1_path, tile_id=tile_id
                )
            else:
                warnings.warn(
                    f"[tile {tile_id}] use_s1=True but 's1_path' is empty. "
                    "S1 will be None for this sample.",
                    UserWarning,
                )

        # ── Load Gaofen-1 (optional) ──────────────────────────────────────
        # GF-1 is an optical sensor like S2 but at different resolution/bands.
        # It is handled separately because:
        #   • Different channel count (4 vs 10)
        #   • Different normalisation ceiling
        #   • Typically a single acquisition (not a time series)
        gaofen1 = None
        gf1_valid = False
        if self.use_gaofen1:
            gf1_path = meta.get("gaofen1_path", "")
            if gf1_path:
                gaofen1, gf1_valid = self._load_and_normalize_gf1(
                    gf1_path, tile_id=tile_id, reference_shape=s2_cloudy.shape
                )
            else:
                warnings.warn(
                    f"[tile {tile_id}] use_gaofen1=True but 'gaofen1_path' is empty. "
                    "GF-1 will be None for this sample.",
                    UserWarning,
                )

        # ── Cross-modal pairing sanity check ──────────────────────────────
        # Verifies that all loaded rasters describe the same geographic tile.
        self._validate_pairing(
            tile_id=tile_id,
            s2_cloudy=s2_cloudy,
            s2_cloudfree=s2_cloudfree,
            s1=s1,
            gaofen1=gaofen1,
        )

        # ── Assemble output ───────────────────────────────────────────────
        out = {
            "tile_id": tile_id,
            "input": {
                "s2": s2_cloudy,
                "s1": s1,                 # None when not loaded
                "gaofen1": gaofen1,       # None when not loaded
            },
            "target": {
                "s2_cloudfree": s2_cloudfree,
            },
            "meta": {
                "region":      meta.get("region", ""),
                "date":        meta.get("date", ""),
                "bounds":      meta.get("bounds", ""),
                "crs":         meta.get("crs", ""),
                "valid_ratio": _safe_float(meta.get("valid_ratio"), 1.0),
                "cloud_ratio": _safe_float(meta.get("cloud_ratio"), 0.0),
                "gf1_valid":   gf1_valid,
                "s1_valid":    s1_valid,
                # Include file paths only when requested (helps debugging)
                "paths": {
                    "s2_cloudy":    meta.get("s2_cloudy_path", ""),
                    "s2_cloudfree": meta.get("s2_cloudfree_path", ""),
                    "s1":           meta.get("s1_path", ""),
                    "gaofen1":      meta.get("gaofen1_path", ""),
                } if self.return_paths else {},
            },
        }

        return out

    # ── Sample builders ───────────────────────────────────────────────────────

    def _build_samples_from_manifest(
        self, manifest_path: str, split: str
    ) -> List[Dict]:
        """
        Parse a manifest CSV file and return a list of sample dicts for `split`.

        Required columns : tile_id, s2_cloudy_path, s2_cloudfree_path, split
        Optional columns : s1_path, gaofen1_path, region, date, bounds, crs,
                           valid_ratio, cloud_ratio
        """
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"[MultiModalDataset] Manifest not found: {manifest_path}"
            )

        samples: List[Dict] = []
        missing_required: List[str] = []

        with open(manifest_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []

            # Validate required columns on first read
            required_cols = {"tile_id", "s2_cloudy_path", "s2_cloudfree_path"}
            missing_cols = required_cols - set(fieldnames)
            if missing_cols:
                raise ValueError(
                    f"[MultiModalDataset] Manifest {manifest_path} is missing "
                    f"required columns: {sorted(missing_cols)}"
                )

            for i, row in enumerate(reader, start=2):  # row 1 = header
                # Filter by split (if the column exists)
                if "split" in row and row["split"].strip() != split:
                    continue

                # Basic completeness check for required paths
                for col in ("s2_cloudy_path", "s2_cloudfree_path"):
                    if not row.get(col, "").strip():
                        missing_required.append(
                            f"row {i}: '{col}' is empty"
                        )

                samples.append(dict(row))

        if missing_required:
            warnings.warn(
                f"[MultiModalDataset] {len(missing_required)} rows have empty "
                f"required path fields:\n" + "\n".join(missing_required[:10]),
                UserWarning,
            )

        return samples

    def _build_samples_from_folder(self, root: str, split: str) -> List[Dict]:
        """
        Build sample descriptors by scanning a folder structure.

        Expects:
            root/s2_cloudy/<tile_id>.npy
            root/s2_cloudfree/<tile_id>.npy
            root/s1/<tile_id>.npy         (optional)
            root/gaofen1/<tile_id>.npy    (optional)
            root/splits/<split>.txt       (optional – one tile_id per line)
        """
        s2_cloudy_dir = os.path.join(root, "s2_cloudy")
        s2_cf_dir = os.path.join(root, "s2_cloudfree")

        if not os.path.isdir(s2_cloudy_dir):
            raise FileNotFoundError(
                f"[MultiModalDataset] Expected folder not found: {s2_cloudy_dir}"
            )
        if not os.path.isdir(s2_cf_dir):
            raise FileNotFoundError(
                f"[MultiModalDataset] Expected folder not found: {s2_cf_dir}"
            )

        # Discover tile IDs from the S2 cloudy folder
        all_tile_ids = sorted(
            p.stem for p in Path(s2_cloudy_dir).glob("*.npy")
        )

        # Apply split filter via optional split txt file
        split_file = os.path.join(root, "splits", f"{split}.txt")
        if os.path.isfile(split_file):
            with open(split_file) as fh:
                allowed = {line.strip() for line in fh if line.strip()}
            all_tile_ids = [t for t in all_tile_ids if t in allowed]
            logger.info(
                f"[MultiModalDataset] Applied split filter from {split_file}. "
                f"{len(all_tile_ids)} tiles remain."
            )

        s1_dir = os.path.join(root, "s1")
        gf1_dir = os.path.join(root, "gaofen1")

        samples = []
        for tid in all_tile_ids:
            entry: Dict[str, str] = {
                "tile_id": tid,
                "s2_cloudy_path":    os.path.join(s2_cloudy_dir, f"{tid}.npy"),
                "s2_cloudfree_path": os.path.join(s2_cf_dir,     f"{tid}.npy"),
            }
            # Optional modalities – paths are included even if files are absent;
            # _load_and_normalize_* handles missing files gracefully.
            if os.path.isdir(s1_dir):
                entry["s1_path"] = os.path.join(s1_dir, f"{tid}.npy")
            if os.path.isdir(gf1_dir):
                entry["gaofen1_path"] = os.path.join(gf1_dir, f"{tid}.npy")

            samples.append(entry)

        return samples

    # ── Raster loaders ────────────────────────────────────────────────────────

    def _load_raster(
        self,
        path: str,
        name: str,
        tile_id: str,
        expected_channels: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load a raster from a .npy file and return a float32 tensor.

        Accepted shapes: (C, H, W) or (T, C, H, W).

        Sanity checks performed:
            • File existence
            • Array dimensionality
            • Channel count (when expected_channels is not None)
            • Spatial tile size matches self.tile_size
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[{name} / tile {tile_id}] File not found: {path}"
            )

        try:
            arr = np.load(path).astype(np.float32)
        except Exception as exc:
            raise IOError(
                f"[{name} / tile {tile_id}] Failed to load {path}: {exc}"
            ) from exc

        t = torch.from_numpy(arr)

        # ── Dimensionality check ──────────────────────────────────────────
        if t.ndim == 3:
            C, H, W = t.shape
        elif t.ndim == 4:
            _T, C, H, W = t.shape
        else:
            raise ValueError(
                f"[{name} / tile {tile_id}] Unexpected array shape {tuple(t.shape)}. "
                f"Expected (C, H, W) or (T, C, H, W). File: {path}"
            )

        # ── Channel count check ───────────────────────────────────────────
        if expected_channels is not None and C != expected_channels:
            raise ValueError(
                f"[{name} / tile {tile_id}] Channel mismatch: "
                f"expected {expected_channels}, got {C}. File: {path}"
            )

        # ── Spatial size check ───────────────────────────────────────────
        H_exp, W_exp = self.tile_size
        if H != H_exp or W != W_exp:
            raise ValueError(
                f"[{name} / tile {tile_id}] Spatial size mismatch: "
                f"expected {self.tile_size}, got ({H}, {W}). File: {path}"
            )

        return t

    def _load_and_normalize_s1(
        self, path: str, tile_id: str
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        Load Sentinel-1 data, apply normalisation, return (tensor, valid).

        Returns (None, False) on any file-level error; raises on shape errors.

        Normalisation logic:
            If s1_norm_stats is set (JSON with mean/std):
                z-score  →  clamp [-2, 2]  →  /2  →  [-1, 1]
            Else if s1_input_db=True:
                clip to [S1_DB_MIN, S1_DB_MAX]  →  rescale to [0, 1]
            Else (linear amplitude/power assumed):
                values passed through as-is (expected to be in [0, 1])

            If rescale=True: all paths finish with Normalize(0.5, 0.5) → [-1, 1]
        """
        if not os.path.isfile(path):
            warnings.warn(
                f"[S1 / tile {tile_id}] File not found: {path}. "
                "Returning None (s1_valid=False).",
                UserWarning,
            )
            return None, False

        try:
            s1 = self._load_raster(
                path,
                name="S1",
                tile_id=tile_id,
                expected_channels=self.s1_expected_channels,
            )
        except (FileNotFoundError, IOError, ValueError) as exc:
            warnings.warn(str(exc) + " Returning None (s1_valid=False).", UserWarning)
            return None, False

        # Replace NaN with 0
        s1 = torch.nan_to_num(s1, nan=0.0)

        if self.s1_norm_stats is not None:
            # Per-channel z-score using pre-computed statistics
            mean, std = self.s1_norm_stats  # both shape (C,)
            # s1 shape is (C, H, W) or (T, C, H, W)
            if s1.ndim == 3:
                s1 = (s1 - mean[:, None, None]) / std[:, None, None]
            else:  # (T, C, H, W)
                s1 = (s1 - mean[None, :, None, None]) / std[None, :, None, None]
            s1 = torch.clamp(s1, -2.0, 2.0) / 2.0  # → [-1, 1]
        elif self.s1_input_db:
            # Linear map from [S1_DB_MIN, S1_DB_MAX] to [0, 1]
            s1 = (s1 - S1_DB_MIN) / (S1_DB_MAX - S1_DB_MIN)
            s1 = torch.clamp(s1, 0.0, 1.0)
            if self.rescale:
                s1 = transforms.Normalize([0.5], [0.5])(s1)
        else:
            # Assume values already in [0, 1] (linear amplitude or power)
            if self.rescale:
                s1 = transforms.Normalize([0.5], [0.5])(s1)

        return s1, True

    def _load_and_normalize_gf1(
        self,
        path: str,
        tile_id: str,
        reference_shape: torch.Size,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        """
        Load Gaofen-1 data, apply normalisation, return (tensor, valid).

        This is the core GF-1 loading function.  All sanity checks specific
        to the GF-1 modality are performed here.

        Sanity checks:
            1. File existence
            2. Dimensionality (3-D or 4-D)
            3. Channel count == gf1_expected_channels
            4. Spatial size == tile_size
            5. Inconsistent pairing: spatial size matches S2 reference shape
            6. All-NaN detection

        Normalisation:
            clamp to [0, gf1_norm_max]  →  /gf1_norm_max  →  [0, 1]
            If rescale=True: Normalize(0.5, 0.5) → [-1, 1]

        Args:
            path:            Path to the GF-1 .npy file.
            tile_id:         Tile identifier for error messages.
            reference_shape: Shape of the already-loaded S2 tensor, used to
                             detect spatial pairing inconsistencies.

        Returns:
            (tensor, valid):
                tensor – float32 Tensor, zeros of expected shape when invalid.
                valid  – True iff the file was loaded successfully.
        """
        H_exp, W_exp = self.tile_size

        def _zeros():
            return torch.zeros(
                self.gf1_expected_channels, H_exp, W_exp, dtype=torch.float32
            )

        # ── Sanity check 1: file existence ───────────────────────────────
        if not os.path.isfile(path):
            warnings.warn(
                f"[GF1 / tile {tile_id}] File not found: {path}. "
                "Returning zeros (gf1_valid=False).",
                UserWarning,
            )
            return _zeros(), False

        # ── Load raw array ────────────────────────────────────────────────
        try:
            arr = np.load(path).astype(np.float32)
        except Exception as exc:
            warnings.warn(
                f"[GF1 / tile {tile_id}] Failed to load {path}: {exc}. "
                "Returning zeros (gf1_valid=False).",
                UserWarning,
            )
            return _zeros(), False

        gf1 = torch.from_numpy(arr)

        # ── Sanity check 2: dimensionality ───────────────────────────────
        if gf1.ndim == 3:
            C, H, W = gf1.shape                 # single acquisition
        elif gf1.ndim == 4:
            # Multi-temporal: use only the first frame.
            # TODO: for full time-series support, extend this branch.
            logger.debug(
                f"[GF1 / tile {tile_id}] Got 4-D array {tuple(gf1.shape)}. "
                "Using first temporal frame."
            )
            gf1 = gf1[0]
            C, H, W = gf1.shape
        else:
            raise ValueError(
                f"[GF1 / tile {tile_id}] Unexpected array shape {tuple(gf1.shape)}. "
                f"Expected (C, H, W) or (T, C, H, W). File: {path}"
            )

        # ── Sanity check 3: channel count ────────────────────────────────
        if C != self.gf1_expected_channels:
            raise ValueError(
                f"[GF1 / tile {tile_id}] Channel mismatch: "
                f"expected {self.gf1_expected_channels} channels (BGRN), "
                f"got {C}. File: {path}"
            )

        # ── Sanity check 4: spatial size ──────────────────────────────────
        if H != H_exp or W != W_exp:
            raise ValueError(
                f"[GF1 / tile {tile_id}] Spatial size mismatch: "
                f"expected {self.tile_size}, got ({H}, {W}). "
                f"Ensure GF-1 tiles are resampled to the same grid as S2. "
                f"File: {path}"
            )

        # ── Sanity check 5: cross-modal pairing consistency ───────────────
        # Compare the H, W of GF-1 against the already-loaded S2 tile.
        if reference_shape[-2] != H or reference_shape[-1] != W:
            raise ValueError(
                f"[GF1 / tile {tile_id}] Pairing inconsistency: "
                f"S2 spatial size {(reference_shape[-2], reference_shape[-1])} "
                f"does not match GF-1 size ({H}, {W}). "
                f"Files may not correspond to the same geographic tile."
            )

        # ── Sanity check 6: all-NaN tile ──────────────────────────────────
        if torch.isnan(gf1).all():
            warnings.warn(
                f"[GF1 / tile {tile_id}] Tile is entirely NaN. "
                "Returning zeros (gf1_valid=False).",
                UserWarning,
            )
            return _zeros(), False

        # Warn (but keep) all-zero tiles — may be valid no-data region
        if (gf1 == 0).all():
            warnings.warn(
                f"[GF1 / tile {tile_id}] Tile is entirely zero. "
                "This may indicate a no-data region. Proceeding with caution.",
                UserWarning,
            )

        # ── Replace residual NaNs with 0 before arithmetic ────────────────
        gf1 = torch.nan_to_num(gf1, nan=0.0)

        # ── Normalisation ─────────────────────────────────────────────────
        # Optical sensor: same philosophy as Sentinel-2.
        # Assumption: GF-1 DN = surface reflectance × 10000.
        #   Typical non-snow land: refl < 0.30  →  DN < 3000  (gf1_norm_max default).
        #   Snow/cloud may exceed 0.30; those are clamped to 1.0 after normalisation.
        gf1 = torch.clamp(gf1, 0.0, self.gf1_norm_max) / self.gf1_norm_max  # → [0, 1]

        # Optional rescale to [-1, 1] (consistent with S2 treatment)
        if self.rescale:
            gf1 = transforms.Normalize([0.5], [0.5])(gf1)   # → [-1, 1]

        return gf1, True

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _normalize_s2(self, t: torch.Tensor) -> torch.Tensor:
        """
        Normalise Sentinel-2 tensor.

        clamp to [0, s2_norm_max]  →  divide  →  [0, 1]
        If rescale=True: Normalize(0.5, 0.5) → [-1, 1]
        """
        t = torch.nan_to_num(t, nan=0.0)
        t = torch.clamp(t, 0.0, self.s2_norm_max) / self.s2_norm_max
        if self.rescale:
            t = transforms.Normalize([0.5], [0.5])(t)
        return t

    # ── Validation helpers ────────────────────────────────────────────────────

    def _validate_pairing(
        self,
        tile_id: str,
        s2_cloudy: torch.Tensor,
        s2_cloudfree: torch.Tensor,
        s1: Optional[torch.Tensor],
        gaofen1: Optional[torch.Tensor],
    ) -> None:
        """
        Cross-modal pairing checks.

        Verifies that all loaded rasters have consistent spatial dimensions.
        Raises ValueError on hard inconsistencies; warns on soft ones.
        """
        ref_hw = s2_cloudy.shape[-2:]  # (H, W)

        if s2_cloudfree.shape[-2:] != ref_hw:
            raise ValueError(
                f"[tile {tile_id}] Spatial size mismatch between "
                f"S2 cloudy {tuple(s2_cloudy.shape)} and "
                f"S2 cloud-free {tuple(s2_cloudfree.shape)}."
            )

        if s1 is not None and s1.shape[-2:] != ref_hw:
            raise ValueError(
                f"[tile {tile_id}] Spatial size mismatch between "
                f"S2 {tuple(ref_hw)} and S1 {tuple(s1.shape[-2:])}."
            )

        # GF-1 spatial check already performed in _load_and_normalize_gf1,
        # but we add a final consistency assertion here as a belt-and-suspenders.
        if gaofen1 is not None and gaofen1.shape[-2:] != ref_hw:
            raise ValueError(
                f"[tile {tile_id}] Spatial size mismatch between "
                f"S2 {tuple(ref_hw)} and GF-1 {tuple(gaofen1.shape[-2:])}."
            )

    # ── S1 stats loader ───────────────────────────────────────────────────────

    @staticmethod
    def _load_s1_norm_stats(
        path: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load per-channel mean and std from a JSON file.

        Expected JSON format:
            {"mean": [float, ...], "std": [float, ...]}

        Returns:
            (mean, std) as float32 tensors of shape (C,).
        """
        with open(path, "r") as fh:
            stats = json.load(fh)
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std  = torch.tensor(stats["std"],  dtype=torch.float32)
        # Guard against zero std to avoid division by zero
        std = torch.clamp(std, min=1e-6)
        logger.info(f"[MultiModalDataset] Loaded S1 norm stats from {path}.")
        return mean, std
