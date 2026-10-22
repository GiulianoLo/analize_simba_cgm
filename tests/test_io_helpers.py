"""Tests for the I/O helpers lifted out of the notebooks (simbanator.io.tables / catalogs / products / qc)."""
import os

import h5py
import numpy as np
import pytest
from astropy.table import Table, MaskedColumn

from simbanator.io import tables, catalogs, products, qc


# ── tables ────────────────────────────────────────────────────────────────────

def test_as_str_decodes_bytes_and_strips():
    col = np.array([b"AGN ", b"none"], dtype="S5")
    out = tables.as_str(col)
    assert out.tolist() == ["AGN", "none"]
    assert (out == "AGN").sum() == 1                      # the comparison that silently failed on bytes


def test_as_str_object_column_and_invalid_bytes():
    assert tables.as_str(np.array([b"a", "b", 3], dtype=object)).tolist() == ["a", "b", "3"]
    with pytest.raises(UnicodeDecodeError):
        tables.as_str(np.array([b"\xff\xfe"], dtype="S2"))


def test_read_table_decodes_string_columns(tmp_path):
    p = tmp_path / "t.fits"
    Table({"cls": np.array([b"AGN", b"SF"], dtype="S3"), "x": [1.0, 2.0]}).write(p)
    t = tables.read_table(p)
    assert t["cls"].dtype.kind == "U" and list(t["cls"]) == ["AGN", "SF"]
    assert tables.read_table(p, decode=False)["cls"].dtype.kind in "SU"


def test_write_table_atomic_and_no_leftovers(tmp_path):
    p = tmp_path / "sub" / "t.fits"
    tables.write_table(Table({"a": [1, 2]}), str(p))
    assert Table.read(p)["a"].tolist() == [1, 2]
    tables.write_table(Table({"a": [3]}), str(p))           # overwrite
    assert Table.read(p)["a"].tolist() == [3]
    assert [f for f in os.listdir(p.parent) if f.startswith(".tmp_")] == []
    with pytest.raises(FileExistsError):
        tables.write_table(Table({"a": [4]}), str(p), overwrite=False)
    assert [f for f in os.listdir(p.parent) if f.startswith(".tmp_")] == []


def test_float_column_masked_and_absent():
    t = Table({"m": MaskedColumn([1.0, 2.0, 3.0], mask=[False, True, False])})
    np.testing.assert_array_equal(tables.float_column(t, "m"), [1.0, np.nan, 3.0])
    assert np.isnan(tables.float_column(t, "nope")).all() and len(tables.float_column(t, "nope")) == 3


def test_to_frame_native_byte_order_and_strings(tmp_path):
    p = tmp_path / "t.fits"
    Table({"v": np.array([1.5, 2.5], dtype=">f8"), "incl": np.array([b"0", b"90"])}).write(p)
    df = tables.to_frame(Table.read(p), ["v", "incl"], string_cols=("incl",))
    assert df["v"].dtype.byteorder in "=|" and df["v"].tolist() == [1.5, 2.5]
    assert df["incl"].tolist() == ["0", "90"]


def test_align_to_rows_plain_and_keyed():
    m = {(100, 1): 5.0, (100, 2): 6.0}
    np.testing.assert_array_equal(tables.align_to_rows(m, [100, 100, 101], [1, 2, 1]), [5.0, 6.0, np.nan])
    m2 = {(100, 1): {"a": 1.0}}
    np.testing.assert_array_equal(tables.align_to_rows(m2, [100, 100], [1, 9], key="a"), [1.0, np.nan])


def test_galaxy_keys_from_columns_or_key_strings():
    assert tables.galaxy_keys(Table({"anchor_snap": [100, 101], "gal_id": [3, 4]})) == [(100, 3), (101, 4)]
    assert tables.galaxy_keys(Table({"gkey": np.array([b"100_3", b"101_4"])})) == [(100, 3), (101, 4)]


def test_tagged_path():
    assert tables.tagged_path("out/t.fits", "agn") == "out/t_agn.fits"


# ── catalogs ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_caesar(tmp_path):
    p = tmp_path / "m25n512_100.hdf5"
    with h5py.File(p, "w") as f:
        sa = f.create_group("simulation_attributes")
        sa.attrs.update(redshift=1.0, scale_factor=0.5, hubble_constant=0.68, omega_matter=0.3, boxsize=25000.0)
        g = f.create_group("galaxy_data")
        g.create_dataset("dicts/masses.stellar", data=np.array([1e10, 2e10, 3e10]))
        g.create_dataset("sfr", data=np.array([1.0, 0.0, 0.5]))
        g.create_dataset("parent_halo_index", data=np.array([0, -1, 1]))
        g.create_dataset("pos", data=np.zeros((3, 3)))
        f.create_dataset("halo_data/dicts/virial_quantities.m200c", data=np.array([1e12, 1e13]))
    return str(p)


def test_dataset_tries_dicts_then_flat(fake_caesar):
    with h5py.File(fake_caesar, "r") as f:
        np.testing.assert_array_equal(catalogs.dataset(f, "masses.stellar"), [1e10, 2e10, 3e10])
        np.testing.assert_array_equal(catalogs.dataset(f, "sfr"), [1.0, 0.0, 0.5])
        assert catalogs.dataset(f, "masses.dust") is None


def test_read_galaxy_columns_index_and_missing(fake_caesar):
    out = catalogs.read_galaxy_columns(fake_caesar, {"ms": "masses.stellar", "sfr": ["sfr_100", "sfr"],
                                                     "md": "masses.dust"}, index=[2, 0, 7])
    np.testing.assert_array_equal(out["ms"], [3e10, 1e10, np.nan])          # 7 is out of range -> NaN
    np.testing.assert_array_equal(out["sfr"], [0.5, 1.0, np.nan])           # second candidate used
    assert np.isnan(out["md"]).all() and out["_missing"] == ["md"]
    with pytest.raises(KeyError):
        catalogs.read_galaxy_columns(fake_caesar, {"md": "masses.dust"}, missing="raise")
    assert "md" not in catalogs.read_galaxy_columns(fake_caesar, {"md": "masses.dust"}, missing="skip")


def test_read_galaxy_columns_missing_file_warns(tmp_path):
    with pytest.warns(UserWarning):
        out = catalogs.read_galaxy_columns(str(tmp_path / "nope.hdf5"), {"ms": "masses.stellar"}, index=[1, 2])
    assert np.isnan(out["ms"]).all() and len(out["ms"]) == 2


def test_parent_halo_columns(fake_caesar):
    out = catalogs.parent_halo_columns(fake_caesar, {"m200c": "virial_quantities.m200c"})
    np.testing.assert_array_equal(out["m200c"], [1e12, np.nan, 1e13])        # galaxy 1 has no halo
    out = catalogs.parent_halo_columns(fake_caesar, {"m200c": "virial_quantities.m200c"}, index=[2, 9])
    np.testing.assert_array_equal(out["m200c"], [1e13, np.nan])


def test_sim_attributes(fake_caesar):
    a = catalogs.sim_attributes(fake_caesar)
    assert a["z"] == 1.0 and a["a"] == 0.5 and a["h"] == 0.68 and a["boxsize"] == 25000.0


def test_snapshot_and_time_helpers():
    assert catalogs.nearest_snapshot(0.98, [100, 105, 110], [1.2, 1.0, 0.8]) == (105, 1.0)
    s, v = catalogs.nearest_snapshot(np.nan, [100], [1.0])
    assert s is None and np.isnan(v)
    assert catalogs.z_tag(0.3) == "z0p3" and catalogs.z_tag(1) == "z1"
    from astropy.cosmology import Planck15
    z_to_t, t_to_z = catalogs.time_redshift_maps()
    assert abs(z_to_t(1.0) - Planck15.age(1.0).value) < 1e-3
    assert abs(t_to_z(z_to_t(2.0)) - 2.0) < 1e-3
    assert catalogs.softening_pkpc(0.5, 0.68, 0.5) == pytest.approx(0.5 / 0.68 * 0.5)


# ── products / qc ─────────────────────────────────────────────────────────────

def test_need_names_the_producer(tmp_path):
    p = tmp_path / "x.fits"
    with pytest.raises(FileNotFoundError, match="run powderday_flux Part 7e first"):
        products.need(str(p), "powderday_flux Part 7e")
    p.write_text("")
    assert products.need(str(p)) == str(p)


def test_read_selection(tmp_path):
    p = tmp_path / "sel.fits"
    Table({"snap": [100, 101], "gal_id": [5, 6]}).write(p)
    t, s, g = products.read_selection(str(p))
    assert s.tolist() == [100, 101] and g.tolist() == [5, 6]


def test_hdf5_status(tmp_path, fake_caesar):
    assert qc.hdf5_status(str(tmp_path / "nope.h5")) == "missing"
    (tmp_path / "empty.h5").write_bytes(b"")
    assert qc.hdf5_status(str(tmp_path / "empty.h5")) == "empty"
    (tmp_path / "bad.h5").write_bytes(b"not an hdf5 file at all")
    assert qc.hdf5_status(str(tmp_path / "bad.h5")).startswith("corrupt:")
    assert qc.hdf5_status(fake_caesar) == "ok"
    assert qc.hdf5_status(fake_caesar, required=("galaxy_data", "missing_group")) == "no:missing_group"
