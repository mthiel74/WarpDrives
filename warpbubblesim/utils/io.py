"""
Input/output utilities for WarpBubbleSim.

Handles loading configurations, saving data, and exporting figures.
"""

import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def save_array(
    data: np.ndarray,
    filepath: str | Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a numpy array with optional metadata.

    Parameters
    ----------
    data : np.ndarray
        Array to save.
    filepath : str or Path
        Output file path. Extension determines format (.npy or .npz).
    metadata : dict, optional
        Metadata to save alongside the array.
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npy':
        np.save(filepath, data)
        if metadata:
            meta_path = filepath.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
    elif filepath.suffix == '.npz':
        if metadata:
            np.savez(filepath, data=data, **{f"meta_{k}": v for k, v in metadata.items()})
        else:
            np.savez(filepath, data=data)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


def load_array(filepath: str | Path) -> tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Load a numpy array and optional metadata.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    tuple
        (data, metadata) where metadata may be None.
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npy':
        data = np.load(filepath)
        metadata = None
        meta_path = filepath.with_suffix('.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        return data, metadata
    elif filepath.suffix == '.npz':
        loaded = np.load(filepath, allow_pickle=True)
        data = loaded['data']
        metadata = {k[5:]: loaded[k] for k in loaded.files if k.startswith('meta_')}
        return data, metadata if metadata else None
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


def save_figure(
    fig,
    filepath: str | Path,
    dpi: int = 150,
    transparent: bool = False,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    filepath : str or Path
        Output file path.
    dpi : int
        Resolution in dots per inch.
    transparent : bool
        Whether to use transparent background.
    bbox_inches : str
        Bounding box setting.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)


def load_yaml_config(filepath: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    filepath : str or Path
        Path to YAML file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], filepath: str | Path) -> None:
    """
    Save a configuration to YAML file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    filepath : str or Path
        Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_output_directory(base_dir: str | Path, name: str) -> Path:
    """
    Create a uniquely named output directory.

    Parameters
    ----------
    base_dir : str or Path
        Base directory for outputs.
    name : str
        Name prefix for the directory.

    Returns
    -------
    Path
        Path to created directory.
    """
    from datetime import datetime

    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_simulation_state(
    state: Dict[str, Any],
    filepath: str | Path
) -> None:
    """
    Export full simulation state to a directory.

    Parameters
    ----------
    state : dict
        Dictionary containing arrays and metadata.
    filepath : str or Path
        Output directory path.
    """
    filepath = Path(filepath)
    filepath.mkdir(parents=True, exist_ok=True)

    # Save arrays
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            np.save(filepath / f"{key}.npy", value)
        elif isinstance(value, dict):
            save_yaml_config(value, filepath / f"{key}.yaml")

    # Save manifest
    manifest = {
        'keys': list(state.keys()),
        'shapes': {k: v.shape if isinstance(v, np.ndarray) else 'dict'
                   for k, v in state.items()}
    }
    save_yaml_config(manifest, filepath / 'manifest.yaml')


def import_simulation_state(filepath: str | Path) -> Dict[str, Any]:
    """
    Import a full simulation state from a directory.

    Parameters
    ----------
    filepath : str or Path
        Directory containing saved state.

    Returns
    -------
    dict
        Loaded state dictionary.
    """
    filepath = Path(filepath)

    manifest = load_yaml_config(filepath / 'manifest.yaml')
    state = {}

    for key in manifest['keys']:
        npy_path = filepath / f"{key}.npy"
        yaml_path = filepath / f"{key}.yaml"

        if npy_path.exists():
            state[key] = np.load(npy_path)
        elif yaml_path.exists():
            state[key] = load_yaml_config(yaml_path)

    return state
