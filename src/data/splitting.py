"""Reproducible stratified splitting for the chest X-ray dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.preprocessing import build_dataset_dataframe


DEFAULT_SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def _file_hash(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_group_counts(group_sizes: Iterable[int], targets: list[int], rng: np.random.Generator) -> list[int]:
    """Assign duplicate groups to splits while respecting class-level targets."""
    sizes = list(group_sizes)
    order = rng.permutation(len(sizes))
    assigned = [0] * len(sizes)
    current = [0] * len(targets)
    for group_index in order:
        size = sizes[group_index]
        candidates = [index for index, target in enumerate(targets) if current[index] + size <= target]
        if not candidates:
            candidates = list(range(len(targets)))
        split_index = min(candidates, key=lambda index: (current[index] / max(targets[index], 1), current[index]))
        assigned[group_index] = split_index
        current[split_index] += size
    return assigned


def _split_targets(total: int, ratios: tuple[float, float, float]) -> list[int]:
    """Calculate integer split targets whose sum equals the class total."""
    raw = np.asarray(ratios) * total
    targets = np.floor(raw).astype(int)
    remainder = total - int(targets.sum())
    for index in np.argsort(-(raw - targets))[:remainder]:
        targets[index] += 1
    return targets.tolist()


def create_stratified_split_manifest(
    dataset_dir: str | Path,
    output_path: str | Path,
    random_state: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> pd.DataFrame:
    """Create a reproducible 70/15/15 manifest without hash duplicates across splits."""
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain three values summing to 1.0")

    dataframe = build_dataset_dataframe(dataset_dir).copy()
    if dataframe.empty:
        raise ValueError("The dataset does not contain any images.")

    dataframe["content_hash"] = dataframe["path"].map(_file_hash)
    rng = np.random.default_rng(random_state)
    assignments: dict[str, str] = {}
    split_names = list(DEFAULT_SPLIT_NAMES)

    for label, label_records in dataframe.groupby("label", sort=True):
        groups = label_records.groupby("content_hash", sort=True).size()
        targets = _split_targets(len(label_records), ratios)
        group_assignments = _allocate_group_counts(groups.tolist(), targets, rng)
        for content_hash, split_index in zip(groups.index, group_assignments):
            assignments[str(content_hash)] = split_names[split_index]

    dataframe["split"] = dataframe["content_hash"].map(assignments)
    dataframe = dataframe.drop(columns=["content_hash"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output, index=False)
    return dataframe


def load_split_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """Load a previously generated split manifest."""
    manifest = pd.read_csv(manifest_path)
    required = {"split", "label", "path", "target"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Split manifest is missing columns: {sorted(missing)}")
    return manifest
