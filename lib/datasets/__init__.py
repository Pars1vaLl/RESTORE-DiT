from typing import Dict, Type
from .PASTISDataset import PASTISDataset
# MultiModalDataset: manifest-CSV-based loader that supports S2 + S1 + Gaofen-1.
# Use dataset="multimodal" in your config to activate it.
from .MultiModalDataset import MultiModalDataset

DATASETS = {
    "pastis":      PASTISDataset,
    "multimodal":  MultiModalDataset,
}
