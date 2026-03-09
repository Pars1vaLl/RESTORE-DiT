"""
Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
License MIT

Extended by: RESTORE-DiT team
Extension: Added optional Gaofen-1 (GF-1) satellite support as a third modality.
           All existing Sentinel-1 / Sentinel-2 behavior is fully preserved.
           GF-1 is loaded only when `use_gaofen1=True` is passed to the constructor.
"""

import json
import os
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from torchvision import transforms
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf

# ── Gaofen-1 sensor constants ──────────────────────────────────────────────────
# GF-1 PMS and WFV sensors carry 4 multispectral bands in order:
#   Band 1 (Blue)  : 0.45–0.52 μm
#   Band 2 (Green) : 0.52–0.59 μm
#   Band 3 (Red)   : 0.63–0.69 μm
#   Band 4 (NIR)   : 0.77–0.89 μm
GF1_DEFAULT_CHANNELS = 4

# Normalization ceiling for GF-1 surface reflectance (×10000 convention).
# Typical non-snow land reflectance: < 0.30  →  DN < 3000.
# If your pipeline stores raw 12-bit DN (0–4095), set gaofen1_norm_max=4095.
# If ToA reflectance can exceed 0.30, increase accordingly (e.g. 10000).
GF1_DEFAULT_NORM_MAX = 3000.0


class PASTISDataset(tdata.Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        channels: str = 'bgr-nir',
        filter_settings: Optional[Dict | DictConfig] = None,
        crop_settings: Optional[Dict | DictConfig] = None,
        pe_strategy: str = 'day-of-year',
        mask_kwargs: Optional[Dict | DictConfig] = None,
        augment: bool = False,
        max_seq_length: Optional[int] = None,
        rescale: bool = True,
        ifTestClip: bool = False,
        ifCTHW: bool = False,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        date_rescale=False,
        # ── Gaofen-1 extension (new parameters) ────────────────────────────
        # All parameters below are additive; they do not affect existing behavior
        # when left at their default values (use_gaofen1=False).
        use_gaofen1: bool = False,
        gaofen1_root: Optional[str] = None,
        gaofen1_channels: int = GF1_DEFAULT_CHANNELS,
        gaofen1_norm_max: float = GF1_DEFAULT_NORM_MAX,
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.

        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target

        Args:
            root (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use. If you are using PASTIS-R, you have access to
                Sentinel-2 imagery and Sentinel-1 observations in Ascending and Descending orbits,
                respectively S2, S1A, and S1D.
                For example use sats=['S2', 'S1A'] for Sentinel-2 + Sentinel-1 ascending time series,
                or sats=['S2', 'S1A','S1D'] to retrieve all time series.
                If you are using PASTIS, only  S2 observations are available.
        """
        super(PASTISDataset, self).__init__()
        if filter_settings is None:
            filter_settings = {'type': 'cloud-free', 'min_length': 10, 'return_valid_obs_only': True, 'max_t_sampling': None}
        self.filter_settings = filter_settings
        self.max_seq_length = max_seq_length
        self.rescale = rescale
        self.channels = channels
        self.root = root
        self.split = split
        self.mask_path = os.path.join(root, "REAL_MASKS")
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats
        self.date_rescale = date_rescale

        # Image size
        self.crop_settings = crop_settings
        self.image_size = crop_settings.shape if self.crop_settings.enabled else (128, 128)

        # Fixed sequence length? If yes, the `collate_fn` function of the data loader pads samples to the same temporal
        # length before collating them to a batch
        if filter_settings and filter_settings.get('type', None) is not None:
            self.variable_seq_length = filter_settings.return_valid_obs_only
        else:
            self.variable_seq_length = False

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(root, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        if split == "train":
            folds = [1, 2, 3, 4]
        elif split == "test":
            folds = [5]

        # Select Fold samples
        self.meta_patch = pd.concat(
            [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
        )

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norms = {}
            for s in self.sats:
                if s != "S2":
                    with open(
                        os.path.join(root, "NORM_{}_patch.json".format(s)), "r"
                    ) as file:
                        normvals = json.loads(file.read())
                    selected_folds = folds if folds is not None else range(1, 6)
                    means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                    stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                    self.norms[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                    self.norms[s] = (
                        torch.from_numpy(self.norms[s][0]).float(),
                        torch.from_numpy(self.norms[s][1]).float(),
                    )
        else:
            self.norms = None

        # Get bad frames
        with open(os.path.join(root, "bad_frames.json"), "r") as file:
            bad_frames = json.load(file)
        self.bad_frames = {int(k): v for k, v in bad_frames.items()}

        # Save the number of channels, the indices of the RGB channels, and the index of the NIR channel
        if 'bgr' == self.channels[:3]:
            # self.channels in ['bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = [0, 1, 2]                         # B2, B3, B4
        else:
            # self.channels in ['all', 'all-mask']
            self.num_channels = 10
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = list(np.arange(10))               # all 10 bands

        if '-nir' in self.channels:
            # self.channels in ['bgr-nir', 'bgr-nir-mask']
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [6]                              # B8
        elif 'all' in self.channels:
            self.c_index_nir = torch.Tensor([6]).long()
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        # ── Gaofen-1 (GF-1) extension ─────────────────────────────────────────
        # GF-1 is handled entirely outside the `sats` loop because:
        #   1. It is an optical sensor (not SAR), with different normalization.
        #   2. It may have no temporal metadata aligned with PASTIS date tables.
        #   3. It is treated as a single-acquisition context image per patch.
        # The existing S1/S2 code path is completely untouched.
        self.use_gaofen1 = use_gaofen1
        # Default path follows the PASTIS naming convention: <root>/DATA_GF1/GF1_<patch_id>.npy
        self.gaofen1_root = gaofen1_root or os.path.join(root, "DATA_GF1")
        self.gaofen1_channels = gaofen1_channels
        self.gaofen1_norm_max = gaofen1_norm_max
        if use_gaofen1:
            print(f"[GF1] Gaofen-1 support enabled. Looking for data in: {self.gaofen1_root}")
            if not os.path.isdir(self.gaofen1_root):
                warnings.warn(
                    f"[GF1] Data directory not found: {self.gaofen1_root}. "
                    "GF-1 tiles will return zeros for missing patches.",
                    UserWarning,
                )

        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.root,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays

            data = {s: torch.from_numpy(a) for s, a in data.items()}


            if self.norm:
                for s, d in data.items():
                    if s == "S2":
                        data[s] = torch.clamp(d, 0, 8000) / 8000
                    else:
                        mid = (d - self.norms[s][0][None, :, None, None]) / self.norms[s][1][None, :, None, None]
                        data[s] = torch.clamp(mid, -2, 2) / 2 # SAR data is rescale to [-1, 1]


            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.root, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                heatmap = np.load(
                    os.path.join(
                        self.root,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.root,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.root,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.root, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()

            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        sample_from_S1 = True
        if sample_from_S1:
            Num = 35
            t_sampled_ablation = torch.linspace(0, len(data['S1A']) - 1, steps=Num, dtype=torch.long)

            data['S1A'] = data['S1A'][t_sampled_ablation, :, :, :]
            dates['S1A'] = dates['S1A'][t_sampled_ablation]

        # Temporally subsample/trim the sequence

        t_sampled = self._subsample_sequence(data["S2"], self.bad_frames[id_patch])

        # Extract the subsampled time steps
        if 'S2' in self.sats:
            data["S2"] = data["S2"][t_sampled, :, :, :]
            data["S2"] = data["S2"][:, self.s2_channels, :, :]
            dates["S2"] = dates["S2"][t_sampled]

            if self.rescale:
                trans_scale = transforms.Normalize([0.5], [0.5])
                data['S2'] = trans_scale(data['S2']) # [-1, 1]

        # Extract the matched SAR frames
        if 'S1D' in self.sats:
            dates["S1D"], t_sampled_SAR = match_sequences_tensor(dates['S2'], dates['S1D'])
            data["S1D"] = data["S1D"][t_sampled_SAR, :, :, :]
        else:
            dates["S1D"] = None
            data["S1D"] = None

        if 'S1A' in self.sats:
            dates["S1A"], t_sampled_SAR = match_sequences_tensor(dates['S2'], dates['S1A'])
            data["S1A"] = data["S1A"][t_sampled_SAR, :, :, :]
        else:
            dates["S1A"] = None
            data["S1A"] = None

        cond = data["S1D"] if data["S1D"] is not None else data["S1A"]
        position_days_cond = dates["S1D"] if dates["S1D"] is not None else dates["S1A"]


        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        # masks for training
        masks_pool = np.load(self.mask_path + "/{}.npy".format(id_patch)) # [T, 1, H, W]
        masks_pool = torch.from_numpy(masks_pool).float()
        # randomly select "len(t_sampled)" frames from masks (T frames)
        if self.split == 'train':
            masks = self.select_random_frames(masks_pool, len(t_sampled)) # [len(t_sampled), 1, H, W]
        else:
            masks = self.select_fixed_frames(masks_pool, len(t_sampled))
        # masks: (0: clear; 1: Thick cloud; 2: Thin cloud; 3: Cloud shadow)
        masks = torch.where(masks > 0, 1.0, masks)

        # Real cloud_mask for evaluation (used to maskout the cloud pixels in the target)
        cloud_mask = masks_pool[t_sampled].clone()
        cloud_mask = torch.where(cloud_mask > 0, 1.0, cloud_mask)

        # frames_input
        frames_input = data["S2"].clone()
        flag = (masks == 1).expand_as(frames_input)
        frames_input[flag] = 1    # fill masked pixels with 1

        # days since the first observation in the sequence
        days = dates['S2'] - dates['S2'][0]

        # if date_rescale: rescale each date to the closest multiple of 10
        if self.date_rescale:
            dates['S2'] = ((dates['S2'] / 10).round() * 10).int()
            # print(dates['S2'])

        # ── Gaofen-1 (optional, loaded outside the sats loop) ─────────────────
        # We always include the key so pad_collate sees a consistent schema.
        # When use_gaofen1=False: key is absent (callers that don't know about GF-1
        # are unaffected).
        # When use_gaofen1=True but the file is missing: zeros tensor is returned
        # and 'gf1_valid' is set to False so downstream code can skip the sample.
        if self.use_gaofen1:
            gaofen1, gf1_valid = self._load_gaofen1(id_patch)

        # Assemble output
        out = {
            'x': frames_input,
            'cond': cond,
            'y': data["S2"],
            'masks': masks,      # (T x 1 x H x W)
            'position_days': dates['S2'],
            'position_days_cond': position_days_cond,
            'days': days,  # number of days since the first observation in the sequence, (T, )
            'sample_index': id_patch,
            'label': target,
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir,
            'cloud_mask': cloud_mask,
        }

        # Add GF-1 keys only when the modality is requested.
        # Keeping them absent when not requested preserves full backward compatibility
        # with all existing training/inference scripts.
        if self.use_gaofen1:
            out['gaofen1'] = gaofen1          # (C, H, W) float32 tensor
            out['gf1_valid'] = gf1_valid       # bool: False → file was missing/invalid

        return out


    def _subsample_sequence(self, sample: torch.Tensor, bad_index: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
           images is at most `self.filter_settings.max_t_sampling` days,
        3) trims the sequence to a maximum temporal length.

        Args:
            sample:           torch.Tensor, (T, C, H, W), image time series.
            bad_index:        list, indices of the invalid frames.

        Returns:
            t_sampled:        torch.Tensor, length T.
        """

        # Generate a mask to exclude invalid frames:
        # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
        seq_length = sample.shape[0]

        if self.filter_settings.type == 'cloud-free':
            # Indices of available and cloud-free images
            masks_valid_obs = torch.ones(seq_length, )
            masks_valid_obs[bad_index] = 0
            # masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])

        # elif self.filter_settings.type == 'cloud-free_consecutive':
        #     subseq = self._longest_consecutive_seq(sample)
        #     masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])
        #     masks_valid_obs[:subseq['start']] = 0
        #     masks_valid_obs[subseq['end'] + 1:] = 0
        else:
            masks_valid_obs = torch.ones(seq_length, )

        if self.filter_settings.get('return_valid_obs_only', True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled

    # ── Gaofen-1 helper ───────────────────────────────────────────────────────

    def _load_gaofen1(self, id_patch: int) -> Tuple[torch.Tensor, bool]:
        """
        Load and normalise Gaofen-1 data for one PASTIS patch.

        Expected file layout (mirrors the PASTIS DATA_* convention):
            <gaofen1_root>/GF1_<id_patch>.npy
            Shape: (C, H, W)  – single acquisition, or
                   (T, C, H, W) – short time series

        Normalisation (optical sensor, same philosophy as S2):
            1. Replace NaN → 0
            2. Clamp to [0, gaofen1_norm_max]
            3. Divide by gaofen1_norm_max  →  [0, 1]
            4. If self.rescale=True: apply Normalize([0.5],[0.5])  →  [-1, 1]

        Sanity checks performed:
            • File existence
            • Array dimensionality (must be 3-D or 4-D)
            • Channel count must match self.gaofen1_channels
            • Spatial size must match self.image_size
            • All-NaN tile detection

        Returns:
            (tensor, valid):
                tensor – float32 Tensor ready for model consumption;
                         zeros of shape (gaofen1_channels, H, W) when invalid.
                valid  – True iff the file was loaded successfully.
        """
        gf1_path = os.path.join(self.gaofen1_root, f"GF1_{id_patch}.npy")
        H_exp, W_exp = self.image_size

        def _zeros():
            """Return a zero tensor with the expected GF-1 shape."""
            return torch.zeros(self.gaofen1_channels, H_exp, W_exp, dtype=torch.float32)

        # ── Sanity check 1: file existence ───────────────────────────────────
        if not os.path.isfile(gf1_path):
            warnings.warn(
                f"[GF1] Missing file for patch {id_patch}: {gf1_path}. "
                "Returning zero tensor (gf1_valid=False).",
                UserWarning,
            )
            return _zeros(), False

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            gf1_np = np.load(gf1_path).astype(np.float32)
        except Exception as exc:
            warnings.warn(
                f"[GF1] Failed to load {gf1_path}: {exc}. "
                "Returning zero tensor (gf1_valid=False).",
                UserWarning,
            )
            return _zeros(), False

        gf1 = torch.from_numpy(gf1_np)

        # ── Sanity check 2: dimensionality ───────────────────────────────────
        if gf1.ndim == 3:
            C, H, W = gf1.shape            # single acquisition (C, H, W)
        elif gf1.ndim == 4:
            _T, C, H, W = gf1.shape        # short time series  (T, C, H, W)
            # Use only the first frame for now; extend here for full series support
            gf1 = gf1[0]
        else:
            raise ValueError(
                f"[GF1] Patch {id_patch}: unexpected array shape {tuple(gf1.shape)}. "
                f"Expected (C, H, W) or (T, C, H, W). File: {gf1_path}"
            )

        # ── Sanity check 3: channel count ─────────────────────────────────────
        if C != self.gaofen1_channels:
            raise ValueError(
                f"[GF1] Patch {id_patch}: channel mismatch. "
                f"Expected {self.gaofen1_channels} channels, got {C}. "
                f"File: {gf1_path}"
            )

        # ── Sanity check 4: spatial size matches S2 tiles ─────────────────────
        if H != H_exp or W != W_exp:
            raise ValueError(
                f"[GF1] Patch {id_patch}: spatial size mismatch with S2. "
                f"Expected ({H_exp}, {W_exp}), got ({H}, {W}). "
                f"Ensure GF-1 tiles are co-registered and resampled to the same grid. "
                f"File: {gf1_path}"
            )

        # ── Sanity check 5: all-NaN tile ──────────────────────────────────────
        if torch.isnan(gf1).all():
            warnings.warn(
                f"[GF1] Patch {id_patch}: tile is entirely NaN. "
                "Returning zero tensor (gf1_valid=False).",
                UserWarning,
            )
            return _zeros(), False

        # Warn (but do not discard) for all-zero tiles – may be valid no-data
        if (gf1 == 0).all():
            warnings.warn(
                f"[GF1] Patch {id_patch}: tile is entirely zero. "
                "This may indicate a no-data region. Proceeding with caution.",
                UserWarning,
            )

        # ── Replace residual NaNs with 0 before arithmetic ────────────────────
        gf1 = torch.nan_to_num(gf1, nan=0.0)

        # ── Normalisation ─────────────────────────────────────────────────────
        # Mirrors the S2 normalisation style:
        #   clamp to valid range → divide to get [0, 1]
        # Assumption: GF-1 DN represents surface reflectance × 10000.
        #   Typical land: refl < 0.30  →  DN < 3000  (default gaofen1_norm_max).
        #   Adjust gaofen1_norm_max via config if your preprocessing differs.
        gf1 = torch.clamp(gf1, 0.0, self.gaofen1_norm_max) / self.gaofen1_norm_max

        # Optional rescale to [-1, 1] – same treatment as S2 when self.rescale=True
        if self.rescale:
            gf1 = transforms.Normalize([0.5], [0.5])(gf1)

        return gf1, True

    # ── End Gaofen-1 helper ───────────────────────────────────────────────────

    def select_random_frames(self, mask, num_frames):
        T = mask.shape[0]
        indices = torch.randperm(T)[:num_frames]
        selected_frames = torch.index_select(mask, 0, indices)
        return selected_frames
    def select_fixed_frames(self, mask, num_frames):
        indices = torch.arange(0, num_frames)
        selected_frames = torch.index_select(mask, 0, indices)
        return selected_frames

def prepare_dates(date_dict, reference_date):
    """Date formating."""
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


def match_sequences_tensor(A, B):
    M = A.size(0)
    N = B.size(0)

    # Initialize C with a large negative integer as a placeholder
    placeholder = -10 ** 6
    C = torch.full_like(A, placeholder)
    indices = torch.full_like(A, -1, dtype=torch.long)  # Indices of elements in B used for C

    # Step 1: Fix positions for exact matches
    for i in range(M):
        a = A[i].item()
        if a in B:
            pos_in_b = (B == a).nonzero(as_tuple=True)[0].item()
            C[i] = a
            indices[i] = pos_in_b

    # Step 2: Fill the gaps for non-exact matches
    for i in range(M):
        if C[i].item() == placeholder:
            a = A[i].item()
            # Find the best match for a in B
            pos = torch.searchsorted(B, torch.tensor([a]), right=False).item()

            # Check to the left and right to find the closest element in B
            left_pos = pos - 1
            right_pos = pos if pos < N else N - 1

            left_diff = float('inf') if left_pos < 0 else abs(B[left_pos].item() - a)
            right_diff = float('inf') if right_pos >= N else abs(B[right_pos].item() - a)

            # Choose the position with the smallest difference
            if left_diff <= right_diff and left_pos >= 0:
                best_pos = left_pos
            else:
                best_pos = right_pos

            # Assign the best match to C
            C[i] = B[best_pos]
            indices[i] = best_pos

    return C, indices