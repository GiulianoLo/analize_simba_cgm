"""Per-machine configuration for simbanator.

Stores simulation data paths in ``~/.simbanator/config.json`` so that
source code never contains hardcoded absolute paths.  The config is a
simple JSON file you set up once per machine.

Example ``~/.simbanator/config.json``::

    {
      "simulations": {
        "simba_m100n1024": {
          "data_dir": "/mnt/data/SIMBA/simba100_snaps",
          "catalog_dir": "/mnt/data/SIMBA/simba100_snaps/Groups",
          "file_format": "m100n1024_{snap:03d}.hdf5"
        }
      }
    }
"""

import json
import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".simbanator"
CONFIG_FILE = CONFIG_DIR / "config.json"


# ── read / write ──────────────────────────────────────────────────────

def load_config():
    """Load the configuration dict, or return defaults if no file exists."""
    if not CONFIG_FILE.exists():
        return {"simulations": {}}
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(config):
    """Write *config* to ``~/.simbanator/config.json``."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


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
            f"Simulation '{name}' not found in {CONFIG_FILE}.\n"
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
    save_config(config)
    print(f"Saved simulation '{name}' → {CONFIG_FILE}")


def remove_simulation(name):
    """Remove a simulation entry from the config file."""
    config = load_config()
    sims = config.get("simulations", {})
    if name in sims:
        del sims[name]
        save_config(config)
        print(f"Removed '{name}' from {CONFIG_FILE}")
    else:
        print(f"'{name}' not found in config.")
