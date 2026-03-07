"""Tests for simbanator I/O layer.

Run with:  python -m pytest tests/test_io.py -v
Or simply:  python tests/test_io.py
"""

import json
import os
import shutil
import tempfile

import pytest
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Helpers to isolate tests from the real ~/.simbanator config
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect the config dir to a temp dir so tests never touch
    your real ~/.simbanator/config.json."""
    fake_config_dir = tmp_path / ".simbanator"
    fake_config_file = fake_config_dir / "config.json"

    import simbanator.io.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", fake_config_dir)
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", fake_config_file)

    return fake_config_file


@pytest.fixture
def fake_sim_dir(tmp_path):
    """Create a fake simulation directory with empty snapshot and catalog files."""
    data_dir = tmp_path / "sim_data"
    catalog_dir = data_dir / "Groups"
    data_dir.mkdir()
    catalog_dir.mkdir()

    # Create a few empty placeholder files so path checks pass
    for snap in (100, 120, 151):
        (data_dir / f"snap_m100n1024_{snap:03d}.hdf5").touch()
        (catalog_dir / f"m100n1024_{snap:03d}.hdf5").touch()

    return data_dir, catalog_dir


# ======================================================================
#  1. Config system
# ======================================================================

class TestConfig:
    """Test the config read/write/query cycle."""

    def test_empty_config_on_fresh_start(self):
        from simbanator.io.config import load_config
        cfg = load_config()
        assert cfg == {"simulations": {}}

    def test_add_and_list_simulation(self):
        from simbanator.io.config import add_simulation, list_simulations

        add_simulation("test_sim", data_dir="/fake/path",
                       catalog_dir="/fake/path/Groups",
                       file_format="snap_{snap:03d}.hdf5")

        sims = list_simulations()
        assert "test_sim" in sims
        assert sims["test_sim"]["data_dir"] == "/fake/path"
        assert sims["test_sim"]["catalog_dir"] == "/fake/path/Groups"

    def test_get_simulation_config_found(self):
        from simbanator.io.config import add_simulation, get_simulation_config

        add_simulation("found_sim", data_dir="/some/dir")
        cfg = get_simulation_config("found_sim")
        assert cfg["data_dir"] == "/some/dir"

    def test_get_simulation_config_not_found(self):
        from simbanator.io.config import get_simulation_config

        with pytest.raises(KeyError, match="not found"):
            get_simulation_config("nonexistent_sim")

    def test_remove_simulation(self):
        from simbanator.io.config import (add_simulation, remove_simulation,
                                           list_simulations)
        add_simulation("to_remove", data_dir="/tmp")
        assert "to_remove" in list_simulations()

        remove_simulation("to_remove")
        assert "to_remove" not in list_simulations()

    def test_config_persists_to_disk(self, isolated_config):
        from simbanator.io.config import add_simulation

        add_simulation("disk_sim", data_dir="/data/run1")

        # Read the raw JSON to verify
        with open(isolated_config) as f:
            raw = json.load(f)
        assert raw["simulations"]["disk_sim"]["data_dir"] == "/data/run1"


# ======================================================================
#  2. Simulation / Simba classes
# ======================================================================

class TestSimulation:
    """Test the generic Simulation class."""

    def test_explicit_paths(self, fake_sim_dir):
        from simbanator.io.simba import Simulation

        data_dir, catalog_dir = fake_sim_dir
        sim = Simulation("test", data_dir=data_dir, catalog_dir=catalog_dir,
                         file_format="m100n1024_{snap:03d}.hdf5")

        assert sim.name == "test"
        assert sim.data_dir == str(data_dir)
        assert sim.catalog_dir == str(catalog_dir)

        # Path generation
        cat_path = sim.get_catalog_file(151)
        assert cat_path.endswith("m100n1024_151.hdf5")
        assert os.path.isfile(cat_path)

        snap_path = sim.get_snapshot_file(151)
        assert "snap_m100n1024_151.hdf5" in snap_path
        assert os.path.isfile(snap_path)

    def test_from_config(self, fake_sim_dir):
        from simbanator.io.config import add_simulation
        from simbanator.io.simba import Simulation

        data_dir, catalog_dir = fake_sim_dir
        add_simulation("configured_sim",
                       data_dir=str(data_dir),
                       catalog_dir=str(catalog_dir),
                       file_format="m100n1024_{snap:03d}.hdf5")

        sim = Simulation("configured_sim")
        assert sim.data_dir == str(data_dir)
        assert os.path.isfile(sim.get_catalog_file(151))

    def test_no_args_raises(self):
        from simbanator.io.simba import Simulation
        with pytest.raises(TypeError, match="Provide either"):
            Simulation()

    def test_missing_config_raises(self):
        from simbanator.io.simba import Simulation
        with pytest.raises(KeyError, match="not found"):
            Simulation("totally_missing_sim")


class TestSimba:
    """Test the SIMBA-specific subclass."""

    def test_explicit_data_dir(self, fake_sim_dir):
        from simbanator.io.simba import Simba

        data_dir, catalog_dir = fake_sim_dir
        sb = Simba(box=100, data_dir=data_dir, catalog_dir=catalog_dir)

        assert sb.box == 100
        assert sb.name == "simba_m100n1024"
        assert sb.data_dir == str(data_dir)

    def test_from_config(self, fake_sim_dir):
        from simbanator.io.config import add_simulation
        from simbanator.io.simba import Simba

        data_dir, catalog_dir = fake_sim_dir
        add_simulation("simba_m100n1024",
                       data_dir=str(data_dir),
                       catalog_dir=str(catalog_dir))

        sb = Simba(box=100)
        assert sb.data_dir == str(data_dir)

    def test_missing_config_gives_helpful_error(self):
        from simbanator.io.simba import Simba
        with pytest.raises(KeyError, match="not configured"):
            Simba(box=100)

    def test_invalid_box_raises(self, fake_sim_dir):
        from simbanator.io.simba import Simba
        data_dir, _ = fake_sim_dir
        with pytest.raises(ValueError, match="Unknown SIMBA box"):
            Simba(box=999, data_dir=data_dir)

    def test_snapshots_and_redshifts(self, fake_sim_dir):
        from simbanator.io.simba import Simba

        data_dir, catalog_dir = fake_sim_dir
        sb = Simba(box=100, data_dir=data_dir, catalog_dir=catalog_dir)

        # Snapshot range
        assert sb.SNAP_MIN == 6
        assert sb.SNAP_MAX == 151
        assert len(sb.snaps) == 146  # 151 - 6 + 1
        assert sb.snaps[0] == 151    # reversed (latest first)
        assert sb.snaps[-1] == 6

        # Redshifts loaded from bundled package data
        assert sb.zeds is not None
        assert len(sb.zeds) == len(sb.snaps)

        # z(151) should be ~0 (last snapshot is z≈0)
        z_151 = sb.get_z_from_snap(151)
        assert z_151 == pytest.approx(0.0, abs=0.02)

        # z(6) should be high redshift
        z_6 = sb.get_z_from_snap(6)
        assert z_6 > 5.0

    def test_catalog_path_format(self, fake_sim_dir):
        from simbanator.io.simba import Simba

        data_dir, catalog_dir = fake_sim_dir
        sb = Simba(box=100, data_dir=data_dir, catalog_dir=catalog_dir)

        path = sb.get_catalog_file(151)
        assert path.endswith("m100n1024_151.hdf5")
        assert str(catalog_dir) in path

    def test_filters_populated(self, fake_sim_dir):
        from simbanator.io.simba import Simba

        data_dir, catalog_dir = fake_sim_dir
        sb = Simba(box=100, data_dir=data_dir, catalog_dir=catalog_dir)

        assert len(sb.filters) == 28
        assert len(sb.filters_pretty) == 28
        assert "GALEX_FUV" in sb.filters


# ======================================================================
#  3. Output tree (OutputPaths)
# ======================================================================

class TestOutputPaths:
    """Test that OutputPaths creates the full directory tree correctly."""

    def test_creates_root(self, tmp_path):
        from simbanator.io.paths import OutputPaths

        out = OutputPaths("my_sim", base_dir=str(tmp_path / "output"))
        # root should not exist yet (lazy)
        assert not os.path.exists(out.root)

        # Accessing any property triggers creation
        _ = out.progenitors
        assert os.path.isdir(out.root)

    def test_task_directories_created(self, tmp_path):
        from simbanator.io.paths import OutputPaths

        out = OutputPaths("my_sim", base_dir=str(tmp_path / "output"))

        expected_tasks = {
            "progenitors": out.progenitors,
            "filtered_particles": out.filtered_particles,
            "histories": out.histories,
            "radial_profiles": out.radial_profiles,
            "plots": out.plots,
            "sed": out.sed,
        }

        for task_name, task_path in expected_tasks.items():
            assert os.path.isdir(task_path), f"{task_name}/ not created"
            assert task_path.endswith(task_name)

    def test_filtered_snap(self, tmp_path):
        from simbanator.io.paths import OutputPaths

        out = OutputPaths("my_sim", base_dir=str(tmp_path / "output"))
        snap_dir = out.filtered_snap(151)

        assert os.path.isdir(snap_dir)
        assert snap_dir.endswith("snap_151")

    def test_custom_subdir(self, tmp_path):
        from simbanator.io.paths import OutputPaths

        out = OutputPaths("my_sim", base_dir=str(tmp_path / "output"))
        custom = out.subdir("convergence_tests", "run3")

        assert os.path.isdir(custom)
        assert custom.endswith(os.path.join("convergence_tests", "run3"))

    def test_full_tree_structure(self, tmp_path):
        """Verify the complete output tree matches the documented layout."""
        from simbanator.io.paths import OutputPaths

        out = OutputPaths("simba_m100n1024", base_dir=str(tmp_path / "output"))

        # Touch every property to build the full tree
        dirs = [
            out.progenitors,
            out.filtered_particles,
            out.filtered_snap(100),
            out.filtered_snap(151),
            out.histories,
            out.radial_profiles,
            out.plots,
            out.sed,
            out.subdir("custom_analysis"),
        ]

        # Print the tree for visual inspection
        print("\n── Output tree ──")
        for root, subdirs, files in os.walk(out.root):
            level = root.replace(out.root, "").count(os.sep)
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")

        # All directories should exist
        for d in dirs:
            assert os.path.isdir(d)

    def test_repr(self, tmp_path):
        from simbanator.io.paths import OutputPaths

        out = OutputPaths("my_sim", base_dir=str(tmp_path / "output"))
        r = repr(out)
        assert "my_sim" in r
        assert "OutputPaths" in r


# ======================================================================
#  4. End-to-end: config → Simba → OutputPaths
# ======================================================================

class TestEndToEnd:
    """Full round-trip: configure a simulation, create Simba, produce output tree."""

    def test_full_workflow(self, fake_sim_dir, tmp_path):
        from simbanator.io.config import add_simulation
        from simbanator.io.simba import Simba
        from simbanator.io.paths import OutputPaths

        data_dir, catalog_dir = fake_sim_dir

        # Step 1: register the simulation
        add_simulation("simba_m100n1024",
                       data_dir=str(data_dir),
                       catalog_dir=str(catalog_dir))

        # Step 2: create Simba from config
        sb = Simba(box=100)
        assert sb.data_dir == str(data_dir)

        # Step 3: verify it can find data files
        cat = sb.get_catalog_file(151)
        snap = sb.get_snapshot_file(151)
        assert os.path.isfile(cat), f"Catalog file not found: {cat}"
        assert os.path.isfile(snap), f"Snapshot file not found: {snap}"

        # Step 4: create the output tree
        out = OutputPaths(sb.name, base_dir=str(tmp_path / "output"))

        tree = [
            out.progenitors,
            out.filtered_snap(151),
            out.histories,
            out.radial_profiles,
            out.plots,
            out.sed,
        ]
        for d in tree:
            assert os.path.isdir(d)

        # Step 5: verify redshift lookup works
        z = sb.get_z_from_snap(151)
        assert z == pytest.approx(0.0, abs=0.02)

        print("\n✓ Full workflow passed:")
        print(f"  Config:       registered simba_m100n1024")
        print(f"  Simba:        box=100, {len(sb.snaps)} snapshots, z(151)={z:.4f}")
        print(f"  Catalog file: {cat}")
        print(f"  Snap file:    {snap}")
        print(f"  Output root:  {out.root}")
        print(f"  Directories:  {len(tree)} task folders created")


# ──────────────────────────────────────────────────────────────────────
#  Run directly:  python tests/test_io.py
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
