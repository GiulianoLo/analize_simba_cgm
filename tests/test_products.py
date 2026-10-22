"""io.paths (repo_root / output_dir / data_file) and io.products (registry, provenance)."""
import json

import h5py
import numpy as np
import pytest
from astropy.table import Table

from simbanator.io import config, paths, products, registry


@pytest.fixture
def out_root(tmp_path, monkeypatch):
    cfg = tmp_path / "simulations.json"
    cfg.write_text(json.dumps({"simulations": {}, "paths": {"output_root": str(tmp_path / "out")}}))
    monkeypatch.setattr(config, "CONFIG_FILE", cfg)
    return tmp_path / "out"


def test_repo_root_and_data_file():
    root = paths.repo_root()
    assert (root / "simbanator" / "io" / "paths.py").is_file()
    assert paths.data_file("almac11/ks_table.csv") == (root / "obs_data/almac11/ks_table.csv").resolve()
    assert paths.data_file("parameters_master.py").parent.name == "masters"
    with pytest.raises(FileNotFoundError, match="looked in"):
        paths.data_file("no_such_file.res")


def test_output_dir_from_config(out_root):
    assert paths.output_dir("cis25", "tables") == out_root / "cis25" / "tables"
    assert paths.OutputPaths("cis25").root == str(out_root / "cis25")


def test_registry_is_consistent():
    for name, e in registry.PRODUCTS.items():
        assert {"path", "producer"} <= set(e), name
        for dep in e.get("depends_on", ()):
            assert dep in registry.PRODUCTS, (name, dep)
        for k in e.get("params", ()):
            assert k in registry.CONVENTIONS, (name, k)
    assert "ks_tracks" in products.dependents("powderday_selection")
    assert len(products.table()) == len(registry.PRODUCTS)


def test_path_require_read(out_root):
    p = products.path("powderday_selection", sim="cis25")
    assert p == out_root / "cis25/tables/powderday_quenched_selection.fits"
    assert products.path("powderday_selection", sim="cis25", variant="_pt").name.endswith("_pt.fits")
    with pytest.raises(KeyError, match="needs"):
        products.path("history_anchor", sim="cis25")
    with pytest.raises(FileNotFoundError, match="powderday_flux_quenched_m25"):
        products.require("powderday_selection", sim="cis25")
    p.parent.mkdir(parents=True)
    Table({"snap": [105, 134], "gal_id": [3, 7], "pop": [b"Q", b"SF"]}).write(p)
    df = products.read("powderday_selection", sim="cis25")
    assert list(df["pop"]) == ["Q", "SF"]
    with pytest.raises(TypeError, match="container"):
        h = products.path("history_anchor", sim="cis25", tag="z1")
        h.parent.mkdir(parents=True)
        h5py.File(h, "w").close()
        products.read("history_anchor", sim="cis25", tag="z1")


@pytest.mark.parametrize("name,tags,suffix", [
    ("powderday_selection", {"sim": "cis25"}, "fits"),
    ("history_anchor", {"sim": "cis25", "tag": "z1"}, "hdf5"),
    ("ism_prediction_agn_classes", {}, "csv"),
])
def test_stamp_and_check(out_root, name, tags, suffix):
    p = products.path(name, **tags)
    assert products.check(name, **tags)["status"] == "missing"
    p.parent.mkdir(parents=True, exist_ok=True)
    if suffix == "fits":
        Table({"x": np.arange(3)}).write(p)
    elif suffix == "hdf5":
        h5py.File(p, "w").close()
    else:
        p.write_text("x\n1\n")
    st = products.check(name, **tags)
    assert st["status"] == "unstamped" and st["why"] == registry.PRODUCTS[name].get("stale", "")
    products.stamp(p, name)
    st = products.check(name, **tags)
    assert st["status"] == "ok" and st["stamp"]["product"] == name
    if "keep_zero" in registry.PRODUCTS[name].get("params", ()):
        products.stamp(p, name, keep_zero=True)
        st = products.check(name, **tags)
        assert st["status"] == "stale" and "keep_zero=True" in st["why"]
        assert st["dependents"] == products.dependents(name)


def test_conventions_match_the_library():
    from simbanator.particles.grains import H2_RECIPE
    assert registry.CONVENTIONS["h2_recipe"] == H2_RECIPE
