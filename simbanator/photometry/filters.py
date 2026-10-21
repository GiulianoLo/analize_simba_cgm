"""Helpers to download photometric filters from SVO using ``svo_filters``."""

import os
from collections import OrderedDict

import numpy as np


DEFAULT_SVO_FILTERS = OrderedDict([
    ('SLOAN/SDSS.u', 'SLOAN_SDSS.u.dat'),
    ('SLOAN/SDSS.g', 'SLOAN_SDSS.g.dat'),
    ('SLOAN/SDSS.r', 'SLOAN_SDSS.r.dat'),
    ('SLOAN/SDSS.i', 'SLOAN_SDSS.i.dat'),
    ('SLOAN/SDSS.z', 'SLOAN_SDSS.z.dat'),
    ('GALEX/GALEX.FUV', 'GALEX_GALEX.FUV.dat'),
    ('GALEX/GALEX.NUV', 'GALEX_GALEX.NUV.dat'),
])


def _load_curve_from_svo_filters(filter_id):
    try:
        import svo_filters
    except ImportError as exc:
        raise ImportError(
            "svo_filters is required to download filters. Install with: pip install svo_filters"
        ) from exc

    filter_obj = None
    errors = []

    if hasattr(svo_filters, 'Filter'):
        try:
            filter_obj = svo_filters.Filter(filter_id)
        except Exception as exc:
            errors.append(f"svo_filters.Filter failed: {exc}")

    if filter_obj is None and hasattr(svo_filters, 'load_filter'):
        try:
            filter_obj = svo_filters.load_filter(filter_id)
        except Exception as exc:
            errors.append(f"svo_filters.load_filter failed: {exc}")

    if filter_obj is None:
        raise RuntimeError(
            "Could not load filter via svo_filters for "
            f"'{filter_id}'. Tried known APIs. Details: {' | '.join(errors)}"
        )

    wav_candidates = ['wavelength', 'wave', 'lambda_eff', 'lambda_']
    trans_candidates = ['transmission', 'throughput', 'response', 'trans']

    wav = None
    trans = None

    for key in wav_candidates:
        if hasattr(filter_obj, key):
            wav = np.asarray(getattr(filter_obj, key))
            if wav.size:
                break

    for key in trans_candidates:
        if hasattr(filter_obj, key):
            trans = np.asarray(getattr(filter_obj, key))
            if trans.size:
                break

    if wav is None or trans is None or wav.size == 0 or trans.size == 0:
        raise RuntimeError(
            "Loaded filter object does not expose wavelength/transmission arrays. "
            "Expected attributes like wavelength/wave and transmission/throughput."
        )

    wav = np.ravel(wav).astype(float)
    trans = np.ravel(trans).astype(float)
    if wav.shape != trans.shape:
        raise RuntimeError("Wavelength and transmission arrays have different sizes")

    mask = np.isfinite(wav) & np.isfinite(trans)
    wav = wav[mask]
    trans = trans[mask]

    if wav.size < 2:
        raise RuntimeError(f"Filter '{filter_id}' returned too few valid points")

    order = np.argsort(wav)
    return wav[order], trans[order]


def download_svo_filter(filter_id, out_dir, filename=None, overwrite=False):
    """Download one filter from SVO and save as 2-column ``.dat`` file.

    Parameters
    ----------
    filter_id : str
        SVO filter identifier (for example ``'SLOAN/SDSS.u'``).
    out_dir : str
        Output directory where the ``.dat`` file is written.
    filename : str, optional
        Output filename. If None, one is generated from ``filter_id``.
    overwrite : bool, optional
        If False (default), keep existing files and skip download.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        filename = filter_id.replace('/', '_') + '.dat'

    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path) and not overwrite:
        return out_path

    wav, trans = _load_curve_from_svo_filters(filter_id)
    arr = np.column_stack([wav, trans])
    np.savetxt(out_path, arr, fmt='%.8e')
    return out_path


def download_svo_filters(filters, out_dir, overwrite=False):
    """Download multiple SVO filters.

    Parameters
    ----------
    filters : mapping
        Mapping ``{svo_filter_id: output_filename}``.
    out_dir : str
        Directory where files are written.
    overwrite : bool, optional
        Overwrite existing files.

    Returns
    -------
    dict
        ``{svo_filter_id: saved_path}``.
    """
    saved = {}
    for filter_id, filename in filters.items():
        saved[filter_id] = download_svo_filter(
            filter_id,
            out_dir,
            filename=filename,
            overwrite=overwrite,
        )
    return saved


def download_default_svo_filters(out_dir, overwrite=False):
    """Download the default SDSS + GALEX set used by SED photometry tools."""
    return download_svo_filters(DEFAULT_SVO_FILTERS, out_dir, overwrite=overwrite)
