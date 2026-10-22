"""Black-hole accretion and AGN feedback in SIMBA, from the catalogue histories.

SIMBA's feedback modes (Davé+19), as used throughout the notebooks:

- **jet** mode when ``M_BH > 10**JET_LOGMBH`` and ``f_Edd < JET_FEDD``; the jet velocity boost is
  ``w_jet = clip(log10(JET_FEDD / f_Edd), 0, 1)``, reaching full speed (``w_jet >= W_FULL``) at
  ``f_Edd <= 0.02``;
- **X-ray** heating on top of a full-speed jet when the galaxy is gas-poor, ``M_gas/M_* < gate``.
  The gate (``XRAY_FGAS_MAX``, 0.2 in SIMBA) is a knob of several notebooks (environment variable
  ``XRAY_FGAS_MAX``; 0.6 in noxray_analogues), so every function takes it as an argument;
- ``x_coup = w_jet x [f_gas < gate]``: the ISM coupling of the jet.

Two conventions differ between the notebooks and are explicit arguments here (see :func:`jet_weight`):
whether a black hole with ``f_Edd = 0`` (not accreting) gets ``w_jet = 0`` (``require_accretion=True``,
the agn_states / dusty_qgs / starburst convention, the default) or ``w_jet = 1`` (the x_coup
formula of paper_ism_prediction_boxes Part 6, powderday_flux Part 2 and quench_mode), and whether
a missing black hole is NaN or 0.
"""
import numpy as np

__all__ = [
    "JET_LOGMBH", "JET_FEDD", "XRAY_FGAS_MAX", "XRAY_FEDD_MAX", "W_FULL", "ETA_RAD", "MDOT_PER_MBH", "C2_ERG_PER_MSUN",
    "eddington_ratio", "mdot_from_fedd", "bolometric_luminosity",
    "jet_weight", "xray_coupling", "xray_mode", "agn_states", "catalogue_agn_state",
    "ignition_time", "threshold_onset", "relative_onset_time", "onset_category", "pre_event_mean", "post_event_mean",
    "integrate_window", "exposure_integrals", "bh_window_metrics",
    "bh_dataset_path", "read_bh_history", "build_bh_history", "attach_bh_stages",
]

JET_LOGMBH = 7.5            # jet mode needs M_BH above 10^7.5 Msun ...
JET_FEDD = 0.2              # ... and f_Edd below 0.2
XRAY_FGAS_MAX = 0.2         # X-ray heating needs M_gas/M_* below this
XRAY_FEDD_MAX = 0.02        # full-speed jet: f_Edd at or below 0.02 (w_jet = 1)
W_FULL = 0.999              # the same full-speed condition on w_jet
ETA_RAD = 0.1               # radiative efficiency
MDOT_PER_MBH = 1.26e38 / (ETA_RAD * 8.98755e20) / 1.98892e33 * 3.15576e7   # Msun/yr per Msun of BH at f_Edd = 1
C2_ERG_PER_MSUN = 1.98892e33 * 8.98755e20                                   # c^2 per Msun [erg]


# ═════════════════════════════════════════════════════════════════════════════
# Accretion
# ═════════════════════════════════════════════════════════════════════════════

def eddington_ratio(mbh_msun, mdot_msun_yr, eta=ETA_RAD):
    """``f_Edd = Mdot / Mdot_Edd`` with ``L_Edd = 1.26e38 erg/s x M_BH``; NaN without a black hole or a finite Mdot."""
    mbh, mdot = np.asarray(mbh_msun, float), np.asarray(mdot_msun_yr, float)
    mdot_edd = (1.26e38 * mbh / (eta * 8.98755e20)) / 1.98892e33 * 3.15576e7
    with np.errstate(all="ignore"):
        out = np.where((mbh > 0) & np.isfinite(mdot), mdot / mdot_edd, np.nan)
    return out if out.ndim else float(out)


def mdot_from_fedd(fedd, mbh_msun, eta=ETA_RAD):
    """Accretion rate [Msun/yr] from the Eddington ratio and the BH mass."""
    return 1.26e38 / (eta * 8.98755e20) / 1.98892e33 * 3.15576e7 * np.asarray(fedd, float) * np.asarray(mbh_msun, float)


def bolometric_luminosity(mdot_msun_yr, eta=ETA_RAD):
    """``L_bol = eta Mdot c^2`` [erg/s]."""
    return eta * np.asarray(mdot_msun_yr, float) * C2_ERG_PER_MSUN / 3.15576e7


# ═════════════════════════════════════════════════════════════════════════════
# Feedback mode per (snapshot, galaxy)
# ═════════════════════════════════════════════════════════════════════════════

def jet_weight(mbh, fedd, jet_logmbh=JET_LOGMBH, jet_fedd=JET_FEDD, require_accretion=True, missing_bh="zero"):
    """SIMBA's jet velocity weight ``w_jet = clip(log10(jet_fedd / f_Edd), 0, 1)`` for black holes above the jet mass.

    ``require_accretion=True``: a black hole with ``f_Edd <= 0`` (not accreting) has w_jet = 0.
    ``False``: it has w_jet = 1 (log10 of 0.2 / 1e-12, clipped), the x_coup formula of the older
    notebooks. ``missing_bh`` is what a cell without BH data gets: ``"zero"`` or ``"nan"``.
    """
    mbh, fedd = np.asarray(mbh, float), np.asarray(fedd, float)
    ok = np.isfinite(mbh) & np.isfinite(fedd)
    big = ok & (mbh > 10 ** jet_logmbh)
    if require_accretion:
        big &= fedd > 0
    with np.errstate(all="ignore"):
        w = np.where(big, np.clip(np.log10(jet_fedd / np.clip(fedd, 1e-12, None)), 0.0, 1.0), 0.0)
    if missing_bh == "nan":
        w = np.where(ok, w, np.nan)
    return w


def xray_coupling(mbh, fedd, mgas, mstar, gate=XRAY_FGAS_MAX, jet_logmbh=JET_LOGMBH, jet_fedd=JET_FEDD,
                  require_accretion=True, missing_bh="nan"):
    """``x_coup = w_jet x [M_gas/M_* < gate]`` -> ``(x_coup, f_gas, w_jet)``.

    NaN where the gas fraction or (with ``missing_bh="nan"``) the BH data are missing. Pass
    ``require_accretion=False`` to reproduce the products of paper_ism_prediction_boxes Part 6
    and powderday_flux Part 2 (see the module notes).
    """
    mgas, mstar = np.asarray(mgas, float), np.asarray(mstar, float)
    with np.errstate(all="ignore"):
        fgas = np.where(mstar > 0, mgas / mstar, np.nan)
    w = jet_weight(mbh, fedd, jet_logmbh, jet_fedd, require_accretion=require_accretion, missing_bh=missing_bh)
    x = np.where(np.isfinite(w) & np.isfinite(fgas), w * (fgas < gate).astype(float), np.nan)
    return x, fgas, w


def xray_mode(mbh, fedd, mgas, mstar, gate=XRAY_FGAS_MAX, jet_logmbh=JET_LOGMBH, xray_fedd_max=XRAY_FEDD_MAX):
    """1 where the X-ray channel fires (massive BH, ``f_Edd < xray_fedd_max``, gas-poor), 0 elsewhere, NaN without BH or gas data."""
    mbh, fedd, mgas, mstar = (np.asarray(v, float) for v in (mbh, fedd, mgas, mstar))
    with np.errstate(all="ignore"):
        fgas = np.where(mstar > 0, mgas / mstar, np.nan)
    ok = np.isfinite(mbh) & np.isfinite(fedd) & np.isfinite(fgas)
    return np.where(ok, ((mbh > 10 ** jet_logmbh) & (fedd < xray_fedd_max) & (fgas < gate)).astype(float), np.nan)


def agn_states(mstar, mgas, mbh, fedd, gate=XRAY_FGAS_MAX, jet_logmbh=JET_LOGMBH, jet_fedd=JET_FEDD, w_full=W_FULL,
               eta=ETA_RAD):
    """The four exclusive AGN states of an ``(n_t, n_gal)`` history block, plus diagnostics.

    States (each 0/1, NaN where the galaxy does not exist): ``xray`` (full-speed jet and gas-poor),
    ``choked`` (jet but not X-ray), ``radiative`` (accreting, not in jet mode), ``dormant``
    (not accreting; also a missing BH). Diagnostics: ``nobh``, ``pjet`` / ``fjrich`` (partial /
    full-speed jets that do not fire X-rays), ``quasar`` (radiative at ``f_Edd >= jet_fedd``),
    ``lowmbh`` (radiative below the jet mass), ``big``, ``gate``, ``jet``. Extras: ``exist``,
    ``fgas``, ``wjet``, ``mdot`` [Msun/yr].

    ``f_gas = M_gas/M_*`` is SIMBA's own X-ray gate and uses the total gas: a feedback criterion,
    not an ISM diagnostic.
    """
    mstar, mgas, mbh, fedd = (np.asarray(v, float) for v in (mstar, mgas, mbh, fedd))
    with np.errstate(all="ignore"):
        exist = np.isfinite(mstar) & (mstar > 0)
        fgas = np.where(exist, mgas / np.where(exist, mstar, np.nan), np.nan)
        bh = np.isfinite(mbh) & (mbh > 0)
        acc = exist & bh & np.isfinite(fedd) & (fedd > 0)
        big = bh & (mbh > 10 ** jet_logmbh)
        wjet = np.where(acc & big, np.clip(np.log10(jet_fedd / np.clip(fedd, 1e-12, None)), 0.0, 1.0), 0.0)
        mdot = np.where(acc, mdot_from_fedd(np.nan_to_num(fedd), np.nan_to_num(mbh), eta), 0.0)
        jet = acc & big & (fedd < jet_fedd)
        xray = jet & (wjet >= w_full) & (fgas < gate)
        S = {"xray": xray, "choked": jet & ~xray, "radiative": acc & ~jet, "dormant": exist & ~acc}
        D = {"nobh": exist & ~bh, "pjet": jet & ~xray & (wjet < w_full), "fjrich": jet & ~xray & (wjet >= w_full),
             "quasar": (acc & ~jet) & np.isfinite(fedd) & (fedd >= jet_fedd), "lowmbh": (acc & ~jet) & ~big,
             "big": big, "gate": exist & (fgas < gate), "jet": jet}
    f = lambda m: np.where(exist, m.astype(float), np.nan)
    return ({k: f(v) for k, v in S.items()}, {k: f(v) for k, v in D.items()},
            dict(exist=exist, fgas=fgas, wjet=np.where(exist, wjet, np.nan), mdot=np.where(exist, mdot, np.nan)))


def catalogue_agn_state(mbh, jet_flag, jet_logmbh=JET_LOGMBH):
    """The AGN state a single catalogue snapshot carries: ``'jet'``, ``'high_fedd'`` (massive BH, not in jet mode) or ``'low_bh'``."""
    mbh = np.asarray(mbh, float)
    with np.errstate(all="ignore"):
        big = (mbh > 0) & (np.log10(np.where(mbh > 0, mbh, 1.0)) > jet_logmbh)
    return np.where(np.asarray(jet_flag, bool), "jet", np.where(big, "high_fedd", "low_bh"))


# ═════════════════════════════════════════════════════════════════════════════
# Event times along one galaxy's history
# ═════════════════════════════════════════════════════════════════════════════

def ignition_time(t_yr, mdot, t_qt_yr, frac=0.5, floor=1e-3, tolerance_yr=0.05e9):
    """AGN ignition: the first time the BH accretion rate reaches ``frac`` x its pre-quench peak (at least ``floor``).

    Only snapshots up to ``t_QT + tolerance`` are searched. NaN with fewer than 2 snapshots above ``floor``.
    """
    t, md = np.asarray(t_yr, float), np.asarray(mdot, float)
    o = np.argsort(t); t, md = t[o], md[o]
    before = t <= t_qt_yr + tolerance_yr
    m = before & np.isfinite(md) & (md > floor)
    if m.sum() < 2:
        return np.nan
    thr = max(frac * np.nanmax(md[m]), floor)
    cr = np.where(before & np.isfinite(md) & (md >= thr))[0]
    return float(t[cr[0]]) if len(cr) else np.nan


def threshold_onset(t_yr, x, threshold):
    """First time ``x`` reaches ``threshold`` (NaN if never)."""
    t, x = np.asarray(t_yr, float), np.asarray(x, float)
    o = np.argsort(t); t, x = t[o], x[o]
    jj = np.where(np.isfinite(x) & (x >= threshold))[0]
    return float(t[jj[0]]) if len(jj) else np.nan


def relative_onset_time(t_yr, x, t_start_yr, frac=0.5, floor=0.05):
    """First time at or after ``t_start`` that ``x`` reaches ``frac`` x its later peak (at least ``floor``).

    E.g. the coupling onset after ignition: x = x_coup, t_start = t_AGN.
    """
    if not np.isfinite(t_start_yr):
        return np.nan
    t, x = np.asarray(t_yr, float), np.asarray(x, float)
    o = np.argsort(t); t, x = t[o], x[o]
    post = t >= t_start_yr
    m = post & np.isfinite(x) & (x > 0)
    if not m.any():
        return np.nan
    thr = max(frac * np.nanmax(x[m]), floor)
    cr = np.where(post & np.isfinite(x) & (x >= thr))[0]
    return float(t[cr[0]]) if len(cr) else np.nan


def onset_category(t_onset, t_sft, t_qt, has_data=True):
    """Where an onset falls relative to the transition: ``'lead'`` (before SFT), ``'concurrent'`` (by QT), ``'late'``,
    ``'never'`` (data but no onset) or ``'no_AGN'`` (no data)."""
    if np.isfinite(t_onset):
        return "lead" if t_onset < t_sft else ("concurrent" if t_onset <= t_qt else "late")
    return "never" if has_data else "no_AGN"


def pre_event_mean(t_yr, w, t_event_yr, window_yr=1e9):
    """Mean of ``w`` over ``[t_event - window, t_event]`` -> ``(mean, n snapshots)``.

    When no snapshot falls in the window, the finite snapshot nearest to the event is used.
    ``(nan, 0)`` without any finite value.
    """
    t, w = np.asarray(t_yr, float), np.asarray(w, float)
    o = np.argsort(t); t, w = t[o], w[o]
    fin = np.isfinite(w)
    win = (t >= t_event_yr - window_yr) & (t <= t_event_yr) & fin
    if not win.any() and fin.any():
        jj = np.where(fin)[0]
        win = np.zeros_like(fin); win[jj[np.argmin(np.abs(t[jj] - t_event_yr))]] = True
    if not win.any():
        return np.nan, 0
    return float(np.nanmean(w[win])), int(win.sum())


def post_event_mean(t_yr, x, t_event_yr, nmin=2):
    """Mean of ``x`` over the snapshots at or after the event (NaN below ``nmin`` finite values)."""
    if not np.isfinite(t_event_yr):
        return np.nan
    t, x = np.asarray(t_yr, float), np.asarray(x, float)
    ok = (t >= t_event_yr) & np.isfinite(x)
    return float(np.nanmean(x[ok])) if ok.sum() >= nmin else np.nan


def integrate_window(t, y, t0, t1):
    """∫ y dt over ``[t0, t1]`` on the snapshot grid, interpolating at the ends; NaN counts as 0 (no BH / no galaxy = no feedback)."""
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return 0.0
    t, y = np.asarray(t, float), np.nan_to_num(np.asarray(y, float), nan=0.0)
    o = np.argsort(t); t, y = t[o], y[o]
    tt = np.concatenate(([t0], t[(t > t0) & (t < t1)], [t1]))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(tt, t, y), tt))


def exposure_integrals(t_gyr, x_coup, w_jet, fgas, t_sft_gyr, t_qt_gyr, t_anchor_gyr=None, pre_window_gyr=1.0,
                       gate=XRAY_FGAS_MAX):
    """The X-ray and jet doses of one galaxy, all in Gyr.

    Returns a dict: ``dt_sft`` / ``dt_qt`` (anchor minus SFT / QT), ``e_x`` / ``e_jet`` (∫ x_coup / w_jet dt
    from SFT to the anchor), ``e_x_qt`` (from QT), ``t_gaspoor`` (time below the gate since SFT),
    ``f_x`` / ``f_jet`` (doses per unit time since SFT), ``r_x`` (e_x per gas-poor time),
    ``e_x_pre`` / ``e_jet_pre`` (the ``pre_window_gyr`` before SFT), ``fgas_sft``, ``fgas_anchor``,
    ``gate_at_sft`` (1 if gas-poor at SFT) and ``lag_gate`` (first gas-poor time from 1 window
    before SFT, minus SFT). The anchor is the last snapshot unless given.
    """
    t = np.asarray(t_gyr, float); o = np.argsort(t); t = t[o]
    x, w, fg = (np.asarray(v, float)[o] for v in (x_coup, w_jet, fgas))
    gt = np.where(np.isfinite(fg), (fg < gate).astype(float), np.nan)
    ta = float(t[-1]) if t_anchor_gyr is None else float(t_anchor_gyr)
    keys = ("dt_sft", "dt_qt", "e_x", "e_jet", "e_x_qt", "t_gaspoor", "f_x", "f_jet", "r_x", "e_x_pre", "e_jet_pre",
            "lag_gate", "gate_at_sft", "fgas_sft", "fgas_anchor")
    r = {k: np.nan for k in keys}
    if not (np.isfinite(t_sft_gyr) and np.isfinite(t_qt_gyr)):
        return r
    r["dt_sft"], r["dt_qt"] = ta - t_sft_gyr, ta - t_qt_gyr
    r["e_x"], r["e_jet"] = integrate_window(t, x, t_sft_gyr, ta), integrate_window(t, w, t_sft_gyr, ta)
    r["e_x_qt"] = integrate_window(t, x, t_qt_gyr, ta)
    r["t_gaspoor"] = integrate_window(t, gt, t_sft_gyr, ta)
    r["e_x_pre"] = integrate_window(t, x, t_sft_gyr - pre_window_gyr, t_sft_gyr)
    r["e_jet_pre"] = integrate_window(t, w, t_sft_gyr - pre_window_gyr, t_sft_gyr)
    if r["dt_sft"] > 0:
        r["f_x"], r["f_jet"] = r["e_x"] / r["dt_sft"], r["e_jet"] / r["dt_sft"]
    if r["t_gaspoor"] > 0:
        r["r_x"] = r["e_x"] / r["t_gaspoor"]
    fin = np.isfinite(fg)
    if fin.any():
        r["fgas_sft"] = float(np.interp(t_sft_gyr, t[fin], fg[fin])); r["fgas_anchor"] = float(fg[fin][-1])
        r["gate_at_sft"] = float(r["fgas_sft"] < gate)
        jj = np.where(fin & (fg < gate) & (t >= t_sft_gyr - pre_window_gyr))[0]
        if len(jj):
            r["lag_gate"] = float(t[jj[0]] - t_sft_gyr)
    return r


def bh_window_metrics(t_yr, mdot, mbh, fedd, t0_yr, t1_yr, eta=ETA_RAD, jet_logmbh=JET_LOGMBH, jet_fedd=JET_FEDD):
    """Over the snapshots in ``[t0, t1]`` (e.g. SFT to QT): ``E_agn = eta ∫ Mdot dt c^2`` [erg], the BH mass gained,
    and the fraction of snapshots in jet mode -> dict. NaNs with fewer than 2 snapshots."""
    t = np.asarray(t_yr, float); o = np.argsort(t)
    t, md, mb, fe = t[o], *(np.asarray(v, float)[o] for v in (mdot, mbh, fedd))
    out = dict(E_agn=np.nan, dMbh=np.nan, jet_frac=np.nan)
    lo, hi = sorted((t0_yr, t1_yr)) if np.isfinite(t0_yr) and np.isfinite(t1_yr) else (np.nan, np.nan)
    sel = (t >= lo) & (t <= hi)
    if sel.sum() < 2:
        return out
    t, md, mb, fe = t[sel], md[sel], mb[sel], fe[sel]
    good = np.isfinite(md) & np.isfinite(t)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    if good.sum() >= 2:
        out["E_agn"] = eta * trapz(md[good], t[good]) * C2_ERG_PER_MSUN
    if np.isfinite(mb).sum() >= 2:
        mv = mb[np.isfinite(mb)]; out["dMbh"] = mv[-1] - mv[0]
    vs = np.isfinite(mb) & np.isfinite(fe)
    jet = vs & (mb > 10 ** jet_logmbh) & (fe < jet_fedd)
    out["jet_frac"] = jet.sum() / vs.sum() if vs.sum() > 0 else np.nan
    return out


# ═════════════════════════════════════════════════════════════════════════════
# BH histories from the catalogues
# ═════════════════════════════════════════════════════════════════════════════

def bh_dataset_path(f, candidates):
    """First of ``galaxy_data/dicts/<c>`` / ``galaxy_data/<c>`` present in an open catalogue, over ``candidates``; None if none."""
    for c in candidates:
        for p in (f"galaxy_data/dicts/{c}", f"galaxy_data/{c}"):
            if p in f:
                return p
    return None


def read_bh_history(path):
    """A BH history file -> ``{dataset: (n_snap, n_gal) array}``."""
    import h5py
    with h5py.File(path, "r") as f:
        return {k: f[k][:] for k in f.keys()}


def build_bh_history(sb, progenitor_index, snaps, candidates, out_path=None, skip_snaps=None, verbose=True):
    """BH columns along the main branches: one ``(n_snap, n_gal)`` array per entry of ``candidates``.

    ``progenitor_index`` is the ``(n_snap, n_gal)`` matrix of catalogue indices
    (:func:`~simbanator.histories.trees.progenitor_index_matrix`, NaN = no progenitor);
    ``candidates`` maps output names to dataset names tried in order (``{"bh_mass": ["bhmass", ...]}``).
    Snapshots in ``skip_snaps`` or unreadable ones stay NaN (unreadable ones are added to ``skip_snaps``).
    Written to ``out_path`` when given.
    """
    import h5py
    skip_snaps = skip_snaps if skip_snaps is not None else set()
    pidx = np.asarray(progenitor_index, float)
    BH = {k: np.full(pidx.shape, np.nan) for k in candidates}
    for ri, snap in enumerate(snaps):
        snap = int(snap)
        if snap in skip_snaps:
            continue
        try:
            with h5py.File(sb.get_caesar_file(snap), "r") as f:
                valid = np.isfinite(pidx[ri]); cv = np.where(valid)[0]; vi = pidx[ri][valid].astype(int)
                for k, cands in candidates.items():
                    p = bh_dataset_path(f, cands)
                    if p is not None:
                        BH[k][ri, cv] = f[p][:][vi]
        except (OSError, KeyError) as e:
            if verbose:
                print(f"  [skip] snap {snap}: {type(e).__name__}")
            skip_snaps.add(snap)
    if out_path is not None:
        with h5py.File(out_path, "w") as f:
            for k, arr in BH.items():
                f.create_dataset(k, data=arr)
        if verbose:
            print(f"BH history -> {out_path}")
    return BH


# ═════════════════════════════════════════════════════════════════════════════
# Moved from ks_tracks_lib.py (the K-S track measurements), unchanged
# ═════════════════════════════════════════════════════════════════════════════
from ..histories.events import STAGES_BH, nearest_row

def attach_bh_stages(recs, t_cosmic_yr, dt_sft_gyr, stages=STAGES_BH):
    """Add the AGN-feedback epochs to the stage records of `build_stage_records` (in place; returns `recs`).

    dt_sft_gyr : {gid: {stage: dt}} — the epoch of each stage in Gyr RELATIVE TO SFT, the convention of the m25
    notebook's Part 2d windows table (`agn_classifier_windows.fits`: t_agn = ignition, first crossing of half the
    pre-QT peak BHAR; t_jet = first snapshot with the ungated w_jet >= 0.5; both (t - t_SFT)/Gyr).
    A stage gets t_<stage> = t_sft + dt [yr] and row_<stage> = the nearest history row (-1 beyond the anchor,
    row 0 = anchor); it stays undefined (NaN / -1) when the galaxy has no SFT, no entry, or a NaN dt.
    """
    t_cosmic_yr = np.asarray(t_cosmic_yr, float)
    t_anchor = float(t_cosmic_yr[0])
    for rec in recs:
        d = dt_sft_gyr.get(int(rec["gid"]), {}) or {}
        t_sft = float(rec.get("t_sft", np.nan))
        for st in stages:
            dt = d.get(st, np.nan)
            dt = float(dt) if dt is not None else np.nan
            t = t_sft + dt * 1e9 if (np.isfinite(t_sft) and np.isfinite(dt)) else np.nan
            rec["t_%s" % st] = float(t) if np.isfinite(t) else np.nan
            rec["row_%s" % st] = nearest_row(t_cosmic_yr, t, t_anchor=t_anchor)
    return recs
