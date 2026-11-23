"Feature extraction disk cache and dataset config resolver."

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.contrib.concurrent import thread_map

from diquark.config.config_manager import ConfigManager
from diquark.data.loader import DataLoader
from diquark.data.preprocessor import Preprocessor
from diquark.features import (
    BaseFeatureExtractor,
    NewFeatureExtractor,
    OldFeatureExtractor,
)


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "features"


def compute_feature_cache_key(
    path_dict: Mapping[str, Path | str],
    index_start: int,
    index_stop: int | None,
    feature_extraction_cfg: Mapping[str, Any],
    feature_names: Sequence[str],
) -> str:
    """Compute a deterministic SHA-256 key for extracted features.

    Worker counts and batch size are intentionally excluded so re-runs with
    different parallelism hit the same cache entry.
    """
    sources = []
    for key in sorted(path_dict.keys()):
        resolved = Path(path_dict[key]).resolve()

        if not resolved.exists():
            raise FileNotFoundError(f"Input file not found for '{key}': '{resolved}'")

        stat = resolved.stat()
        sources.append(
            {
                "key": str(key),
                "path": str(resolved),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    payload = {
        "sources": sources,
        "index_start": int(index_start),
        "index_stop": int(index_stop) if index_stop is not None else None,
        "feature_extraction": dict(sorted(feature_extraction_cfg.items())),
        "feature_names": list(feature_names),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class FeatureCache:
    "Atomic Parquet + metadata cache for extracted feature DataFrames."

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def cache_path(self, label: str, key: str) -> Path:
        safe_label = _sanitize_label(label)
        return self.cache_dir / f"{safe_label}_{key}.parquet"

    def meta_path(self, label: str, key: str) -> Path:
        safe_label = _sanitize_label(label)
        return self.cache_dir / f"{safe_label}_{key}.meta.json"

    def load(self, label: str, key: str) -> pd.DataFrame | None:
        path = self.cache_path(label, key)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def save(
        self,
        df: pd.DataFrame,
        label: str,
        key: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_path(label, key)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)

        if metadata is not None:
            meta_p = self.meta_path(label, key)
            tmp_meta = meta_p.with_name(f".{meta_p.name}.{os.getpid()}.tmp")
            with open(tmp_meta, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=str)
            os.replace(tmp_meta, meta_p)

        return path


def resolve_dataset_spec(config_path: str | Path) -> dict[str, Any]:
    "Parse dataset paths, cross-sections, and feature extractor spec from a run config."
    config_file = Path(config_path)
    config = ConfigManager(config_file)
    config_dir = config_file.parent

    backgrounds_cfg_rel = Path(config.get_required("data.backgrounds_file"))
    backgrounds_cfg_path = (
        backgrounds_cfg_rel
        if backgrounds_cfg_rel.is_absolute()
        else config_dir / backgrounds_cfg_rel
    )
    backgrounds_config = ConfigManager(backgrounds_cfg_path)
    backgrounds: dict[str, object] = backgrounds_config.get("backgrounds", {})

    background_path_dict: dict[str, Path] = {}
    background_cross_sections: dict[str, float] = {}
    for key, data in backgrounds.items():
        assert isinstance(data, dict), "'backgrounds' should be an object/mapping"
        file_path = Path(data["file"])
        if not file_path.exists():
            raise FileNotFoundError(
                f"Could not find background file at path '{file_path}'"
            )
        dict_key = f"BKG:{key}"
        background_path_dict[dict_key] = file_path
        background_cross_sections[dict_key] = float(data["cross_section"])

    signal_cfg_rel = Path(config.get_required("data.signal_file"))
    signal_cfg_path = (
        signal_cfg_rel if signal_cfg_rel.is_absolute() else config_dir / signal_cfg_rel
    )
    signal_config = ConfigManager(signal_cfg_path)

    signal_file = Path(signal_config.get_required("signal.file"))
    if not signal_file.exists():
        raise FileNotFoundError(f"Could not find signal file at path '{signal_file}'")
    signal_cross_section = float(signal_config.get_required("signal.cross_section"))

    cross_sections = background_cross_sections | {"SIG:Suu": signal_cross_section}
    path_dict = background_path_dict | {"SIG:Suu": signal_file}
    assert set(cross_sections.keys()) == set(path_dict.keys())

    index_start = int(config.get("data.index_start", 0))
    raw_stop = config.get("data.index_stop", None)
    index_stop = int(raw_stop) if raw_stop is not None else None

    feature_set = str(config.get("feature_extraction.feature_set", "new"))
    n_jets = int(config.get_required("feature_extraction.n_jets"))

    if feature_set == "new":
        batch_parallel_workers = int(
            config.get(
                "feature_extraction.batch_parallel_workers",
                config.get("feature_extraction.num_workers", 8),
            )
        )
        batch_size = int(config.get("feature_extraction.batch_size", 2048))
        raw_max_comb = config.get(
            "feature_extraction.max_combination_jets",
            config.get("feature_extraction.n_combination_jets", 12),
        )
        max_combination_jets = int(raw_max_comb) if raw_max_comb is not None else None
        raw_max_eta = config.get("feature_extraction.max_jet_eta", None)
        max_jet_eta = float(raw_max_eta) if raw_max_eta is not None else None
        raw_min_pt = config.get("feature_extraction.min_jet_pt", None)
        min_jet_pt = float(raw_min_pt) if raw_min_pt is not None else None
        chi_mass = float(signal_config.get_required("signal.chi_mass"))
        suu_mass = float(signal_config.get_required("signal.suu_mass"))

        feature_extractor: BaseFeatureExtractor = NewFeatureExtractor(
            max_jets=n_jets,
            chi_mass=chi_mass,
            suu_mass=suu_mass,
            batch_size=batch_size,
            batch_parallel_workers=batch_parallel_workers,
            max_combination_jets=max_combination_jets,
            max_jet_eta=max_jet_eta,
            min_jet_pt=min_jet_pt,
        )
        cache_fe_cfg: dict[str, Any] = {
            "feature_set": feature_set,
            "n_jets": n_jets,
            "max_combination_jets": feature_extractor.max_combination_jets,
            "max_jet_eta": max_jet_eta,
            "min_jet_pt": min_jet_pt,
            "chi_mass": chi_mass,
            "suu_mass": suu_mass,
        }
    elif feature_set == "old":
        feature_extractor = OldFeatureExtractor(max_jets=n_jets)
        cache_fe_cfg = {
            "feature_set": feature_set,
            "n_jets": n_jets,
        }
    else:
        raise ValueError(f"Unknown/unsupported feature set: '{feature_set}'")

    dataset_parallel_workers = int(
        config.get(
            "feature_extraction.dataset_parallel_workers",
            config.get("feature_extraction.max_workers", 4),
        )
    )
    cache_dir_val = config.get("feature_extraction.cache_dir", None)
    label = f"{signal_cfg_path.stem}_{feature_set}_nj{n_jets}"

    return {
        "config": config,
        "path_dict": path_dict,
        "cross_sections": cross_sections,
        "index_start": index_start,
        "index_stop": index_stop,
        "feature_extractor": feature_extractor,
        "cache_fe_cfg": cache_fe_cfg,
        "dataset_parallel_workers": dataset_parallel_workers,
        "cache_dir": Path(cache_dir_val) if cache_dir_val else None,
        "label": label,
    }


def load_or_extract_dataset(
    config_path: str | Path,
    cache_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    "Load extracted feature DataFrame from cache, or extract from ROOT and cache it."
    spec = resolve_dataset_spec(config_path)
    effective_cache_dir = (
        Path(cache_dir) if cache_dir is not None else spec["cache_dir"]
    )
    feature_extractor: BaseFeatureExtractor = spec["feature_extractor"]
    cache_key = compute_feature_cache_key(
        spec["path_dict"],
        spec["index_start"],
        spec["index_stop"],
        spec["cache_fe_cfg"],
        feature_extractor.feature_names,
    )

    cache = (
        FeatureCache(effective_cache_dir) if effective_cache_dir is not None else None
    )
    cache_hit = False
    cache_path: Path | None = None
    df: pd.DataFrame | None = None

    if cache is not None:
        df = cache.load(spec["label"], cache_key)
        if df is not None:
            cache_hit = True
            cache_path = cache.cache_path(spec["label"], cache_key)

    if df is None:
        loader = DataLoader(
            spec["path_dict"],
            index_start=spec["index_start"],
            index_stop=spec["index_stop"],
        )
        data = loader.load_data()
        extracted = thread_map(
            feature_extractor.compute_all,
            data.values(),
            max_workers=spec["dataset_parallel_workers"],
            desc="Extracting features",
        )
        if len(extracted[0].keys()) != len(feature_extractor.feature_names):
            raise AssertionError(
                f"Number of extracted features ({len(extracted[0].keys())}) doesn't match "
                f"number of feature names ({len(feature_extractor.feature_names)})"
            )
        features = dict(zip(data.keys(), extracted))
        preprocessor = Preprocessor(spec["config"].get("preprocessing", {}))
        df = preprocessor.create_dataframe(features)

        if cache is not None:
            cache_meta = {
                "config_path": str(Path(config_path).resolve()),
                "cache_key": cache_key,
                "label": spec["label"],
                "n_rows": len(df),
                "n_features": len(feature_extractor.feature_names),
                "truth_counts": {
                    str(k): int(v) for k, v in df["Truth"].value_counts().items()
                },
                "feature_extraction": spec["cache_fe_cfg"],
            }
            cache_path = cache.save(df, spec["label"], cache_key, metadata=cache_meta)

    metadata = {
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "cache_path": str(cache_path) if cache_path is not None else None,
        "label": spec["label"],
        "feature_names": list(feature_extractor.feature_names),
        "total_luminosity": float(spec["config"].get("data.total_luminosity", 3000)),
        "preprocessing": dict(spec["config"].get("preprocessing", {})),
    }
    return df, spec["cross_sections"], metadata
