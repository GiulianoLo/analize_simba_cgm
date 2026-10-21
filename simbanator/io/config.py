"""Simulation paths for simbanator, kept in one visible, hand-editable file.

Source code never contains hardcoded data paths: every simulation is an entry in
``config/simulations.json`` at the repository root. Edit it by hand or with
:func:`add_simulation`. Example::

    {
      "simulations": {
        "simba_m100n1024": {
          "data_dir": "/mnt/data/SIMBA/simba100_snaps",
          "catalog_dir": "/mnt/data/SIMBA/simba100_snaps/Groups",
          "file_format": "m100n1024_{snap:03d}.hdf5"
        }
      }
    }

Which file is used, in order:

1. the path in the ``SIMBANATOR_CONFIG`` environment variable;
2. ``config/simulations.json`` in the repository;
3. the old hidden ``~/.simbanator/config.json``, read with a one-time notice to move it.

Transition: while the old hidden file exists, every write is mirrored into it, because
some notebooks still read it directly. Once they read through :func:`load_config`, delete it.
"""

import json
import os
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG = REPO_ROOT / "config" / "simulations.json"
CONFIG_ENV = "SIMBANATOR_CONFIG"
LEGACY_CONFIG = Path.home() / ".simbanator" / "config.json"

# Kept for backward compatibility (tests and older code patch these two names).
# When CONFIG_FILE is set to anything other than its default, it wins over the lookup.
CONFIG_DIR = PROJECT_CONFIG.parent
CONFIG_FILE = PROJECT_CONFIG
_DEFAULT_CONFIG_FILE = CONFIG_FILE
_legacy_notice_shown = False


def config_path():
    """The config file in use (see the module docstring for the lookup order)."""
    global _legacy_notice_shown
    if Path(CONFIG_FILE) != _DEFAULT_CONFIG_FILE:
        return Path(CONFIG_FILE)
    env = os.environ.get(CONFIG_ENV)
    if env:
        return Path(env).expanduser()
    if PROJECT_CONFIG.exists():
        return PROJECT_CONFIG
    if LEGACY_CONFIG.exists():
        if not _legacy_notice_shown:
            warnings.warn(
                f"Reading simulation paths from the old hidden file {LEGACY_CONFIG}. "
                f"Copy it to {PROJECT_CONFIG} to use the visible config.", stacklevel=3)
            _legacy_notice_shown = True
        return LEGACY_CONFIG
    return PROJECT_CONFIG


# ── read / write ──────────────────────────────────────────────────────

def load_config():
    """Load the configuration dict, or return defaults if no file exists."""
    path = config_path()
    if not path.exists():
        return {"simulations": {}}
    with open(path) as f:
        return json.load(f)


def save_config(config):
    """Write *config* to the config file in use (and mirror it into the old hidden file, if present)."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    if path != LEGACY_CONFIG and Path(CONFIG_FILE) == _DEFAULT_CONFIG_FILE and LEGACY_CONFIG.exists():
        with open(LEGACY_CONFIG, "w") as f:
            json.dump(config, f, indent=2)
    return path


# ── query helpers ─────────────────────────────────────────────────────

def get_simulation_config(name):
    """Return the config dict for simulation *name*.

    Raises
    ------
    KeyError
        If *name* is not in the config file (with a helpful message).
    """
    config = load_config()
    sims = config.get("simulations", {})
    if name not in sims:
        available = list(sims.keys()) or ["(none configured)"]
        raise KeyError(
            f"Simulation '{name}' not found: it is not configured in {config_path()}.\n"
            f"Available: {available}\n"
            f"Add it with:\n"
            f"  simbanator.add_simulation('{name}', data_dir='/path/to/data')\n"
            f"Or pass data_dir= directly to Simulation()."
        )
    return sims[name]


def list_simulations():
    """Return a dict of all configured simulations."""
    return load_config().get("simulations", {})


# ── mutation helpers ──────────────────────────────────────────────────

def add_simulation(name, data_dir, catalog_dir=None, file_format=None):
    """Add or update a simulation entry in the config file.

    Parameters
    ----------
    name : str
        Identifier for this simulation (e.g. ``'simba_m100n1024'``).
    data_dir : str
        Directory containing snapshot / catalog files.
    catalog_dir : str, optional
        Directory containing Caesar catalogs.  Defaults to
        ``<data_dir>/Groups/``.
    file_format : str, optional
        Python format-string for catalog filenames, using ``{snap}``
        as placeholder.  E.g. ``'m100n1024_{snap:03d}.hdf5'``.
    """
    config = load_config()
    config.setdefault("simulations", {})

    entry = {"data_dir": str(data_dir)}
    if catalog_dir is not None:
        entry["catalog_dir"] = str(catalog_dir)
    if file_format is not None:
        entry["file_format"] = file_format

    config["simulations"][name] = entry
    path = save_config(config)
    print(f"Saved simulation '{name}' → {path}")


def remove_simulation(name):
    """Remove a simulation entry from the config file."""
    config = load_config()
    sims = config.get("simulations", {})
    if name in sims:
        del sims[name]
        path = save_config(config)
        print(f"Removed '{name}' from {path}")
    else:
        print(f"'{name}' not found in config.")
