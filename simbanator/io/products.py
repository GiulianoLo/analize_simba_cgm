"""Shared analysis products: where they live, who makes them, and whether they are current.

The products themselves are listed in :mod:`simbanator.io.registry`. Typical use::

    from simbanator.io import products
    sel = products.read("powderday_selection", sim="cis25")         # a DataFrame
    h5  = products.require("history_anchor", sim="cis25", tag="z1")  # a checked path
    products.stamp(h5, "history_anchor", keep_zero=False)            # writers: record provenance
    products.check("powderday_selection", sim="cis25")               # ok / stale / unstamped / missing

A missing file raises ``FileNotFoundError`` naming the notebook part or job that writes it.

Provenance: :func:`stamp` records the git commit, the library version, the time and the
convention parameters a product depends on (``registry.CONVENTIONS``) inside the file
(HDF5 attributes, FITS header of the first table HDU) or, for CSV and other formats, in a
``<file>.prov.json`` next to it. :func:`check` compares the stamp with the conventions in force.
"""

import datetime
import json
import os
import string
import subprocess
from pathlib import Path

import numpy as np

from .paths import output_dir, repo_root
from .registry import CONVENTIONS, PRODUCTS, TAG_DEFAULTS
from .tables import read_table, to_frame

PROV_PREFIX = "prov_"          # HDF5 attribute / FITS keyword prefix of the provenance stamp


# ── checked paths (kept from before the registry; the notebooks use them) ─────────────────────

def need(path, made_by=None):
    """``path`` if it exists, otherwise ``FileNotFoundError`` naming what produces it."""
    if not os.path.exists(path):
        hint = f": run {made_by} first" if made_by else ""
        raise FileNotFoundError(f"{path} is missing{hint}")
    return path


def read_frame(path, cols=None, made_by=None, string_cols=("incl", "aperture")):
    """A product table as a pandas DataFrame (checked with :func:`need`; see :func:`.tables.to_frame`)."""
    return to_frame(read_table(need(path, made_by)), cols, string_cols=string_cols)


def read_selection(path, made_by=None):
    """A selection catalogue -> ``(table, snap array, gal_id array)``."""
    sel = read_table(need(path, made_by))
    return sel, np.asarray(sel["snap"], int), np.asarray(sel["gal_id"], int)


# ── the registry ──────────────────────────────────────────────────────────────────────────────

def entry(name):
    """The registry entry of product ``name`` (``KeyError`` lists the known names)."""
    try:
        return PRODUCTS[name]
    except KeyError:
        raise KeyError(f"unknown product {name!r}; known: {sorted(PRODUCTS)}") from None


def fields(name):
    """The template fields of product ``name``'s path."""
    return [f for _, f, _, _ in string.Formatter().parse(entry(name)["path"]) if f]


def path(name, **tags):
    """Absolute path of product ``name`` with its template fields filled from ``tags``."""
    tpl = entry(name)["path"]
    vals = {**TAG_DEFAULTS, **tags}
    missing = [f for f in fields(name) if f not in vals]
    if missing:
        raise KeyError(f"product {name!r} ({tpl}) needs {missing}")
    return output_dir() / tpl.format(**vals)


def require(name, **tags):
    """:func:`path`, checked to exist; a missing file names the producer."""
    p = path(name, **tags)
    if not p.exists():
        raise FileNotFoundError(f"{name}: {p} is missing; it is written by {entry(name)['producer']}")
    return p


def read(name, cols=None, frame=True, **tags):
    """Read a table product (FITS / CSV / ECSV): a pandas DataFrame, or the astropy Table with ``frame=False``.

    HDF5 and other containers have no single table: open :func:`require`'s path yourself.
    """
    p = require(name, **tags)
    if p.suffix.lower() in (".h5", ".hdf5", ".npz", ".npy"):
        raise TypeError(f"{name} is a {p.suffix} container, not a table: open products.require(...) directly")
    tab = read_table(str(p), format="ascii.csv") if p.suffix.lower() == ".csv" else read_table(str(p))
    return to_frame(tab, cols) if frame else tab


def dependents(name):
    """Every product built (directly or not) from ``name``, nearest first."""
    out, todo = [], [name]
    while todo:
        cur = todo.pop(0)
        for k, e in PRODUCTS.items():
            if cur in e.get("depends_on", ()) and k not in out:
                out.append(k)
                todo.append(k)
    return out


# ── provenance ────────────────────────────────────────────────────────────────────────────────

def _git_commit():
    try:
        root = str(repo_root())
        rev = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (rev + ("-dirty" if dirty else "")) if rev else "unknown"
    except Exception:
        return "unknown"


def _version():
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "unknown"


def _sidecar(p):
    return Path(str(p) + ".prov.json")


def stamp(file, name, **params):
    """Record the provenance of product ``name`` in ``file`` and return the stamp.

    ``params`` are the convention values the file was built with; the product's registry
    ``params`` not given here are recorded at their current ``CONVENTIONS`` value (so pass a
    value whenever the build used anything else).
    """
    e = entry(name)
    rec = {"product": name, "git": _git_commit(), "simbanator": _version(),
           "written": datetime.datetime.now().isoformat(timespec="seconds")}
    for k in e.get("params", ()):
        rec[k] = params.pop(k, CONVENTIONS[k])
    rec.update(params)
    p = Path(file)
    suf = p.suffix.lower()
    if suf in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "a") as f:
            for k, v in rec.items():
                f.attrs[PROV_PREFIX + k] = json.dumps(v)
    elif suf in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(p, mode="update") as hdul:
            hdr = hdul[1 if len(hdul) > 1 else 0].header
            for k, v in rec.items():
                hdr["HIERARCH " + PROV_PREFIX + k] = json.dumps(v)
    else:
        with open(_sidecar(p), "w") as f:
            json.dump(rec, f, indent=1)
    return rec


def provenance(file):
    """The stamp recorded in ``file`` (``{}`` when it has none)."""
    p = Path(file)
    suf = p.suffix.lower()
    raw = {}
    if suf in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            raw = {k[len(PROV_PREFIX):]: v for k, v in f.attrs.items() if k.startswith(PROV_PREFIX)}
    elif suf in (".fits", ".fit"):
        from astropy.io import fits
        with fits.open(p) as hdul:
            hdr = hdul[1 if len(hdul) > 1 else 0].header
            raw = {k[len(PROV_PREFIX):].lower(): v for k, v in hdr.items()
                   if k.lower().startswith(PROV_PREFIX)}
    elif _sidecar(p).exists():
        with open(_sidecar(p)) as f:
            return json.load(f)
    out = {}
    for k, v in raw.items():
        try:
            out[k] = json.loads(v if isinstance(v, str) else v.decode())
        except Exception:
            out[k] = v
    return out


def check(name, **tags):
    """Status of one product file: a dict with ``status`` and ``why``.

    ``missing`` (no file) / ``unstamped`` (written before the registry; ``why`` gives the known
    reason it is out of date, if any) / ``stale`` (stamped with conventions other than the
    current ones) / ``ok``. ``dependents`` lists what has to be rebuilt after it when it is not ok.
    """
    e = entry(name)
    p = path(name, **tags)
    res = {"product": name, "path": str(p), "producer": e["producer"], "dependents": []}
    if not p.exists():
        res.update(status="missing", why="")
        return res
    prov = provenance(p)
    if not prov:
        res.update(status="unstamped", why=e.get("stale", ""), dependents=dependents(name))
        return res
    bad = [f"{k}={prov.get(k)!r} (now {CONVENTIONS[k]!r})" for k in e.get("params", ())
           if prov.get(k, CONVENTIONS[k]) != CONVENTIONS[k]]
    if bad:
        res.update(status="stale", why="; ".join(bad), dependents=dependents(name))
    else:
        res.update(status="ok", why="")
    res["stamp"] = prov
    return res


def table():
    """The registry as a pandas DataFrame (one row per product; the source of docs/products.md)."""
    import pandas as pd
    rows = [{"product": k, "path": e["path"], "producer": e["producer"],
             "depends_on": ", ".join(e.get("depends_on", ())), "params": ", ".join(e.get("params", ())),
             "stale": e.get("stale", ""), "note": e.get("note", "")} for k, e in PRODUCTS.items()]
    return pd.DataFrame(rows)
