"""Smoke and behaviour tests for simbanator.plotting.figures (Agg backend, no display)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.table import Table

from simbanator.plotting import figures as F


def test_save_figure_writes_every_format(tmp_path):
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
    paths = F.save_figure(fig, "demo", outdir=str(tmp_path / "out"), prefix="paper_", verbose=False)
    assert [os.path.basename(p) for p in paths] == ["paper_demo.png", "paper_demo.pdf"]
    assert all(os.path.getsize(p) > 0 for p in paths)
    plt.close(fig)


def test_figure_path_router(tmp_path):
    r = [("ism_", "10_ism")]
    assert F.figure_path(str(tmp_path), "ism_x.png", router=r).endswith("10_ism/ism_x.png")
    assert F.figure_path(str(tmp_path), "other.png", router=r).endswith("99_misc/other.png")
    assert F.figure_path(str(tmp_path), "x.png") == str(tmp_path / "x.png")


def test_paper_style_is_scoped():
    before = matplotlib.rcParams["axes.labelsize"]
    with F.paper_style():
        assert matplotlib.rcParams["axes.labelsize"] == 21
    assert matplotlib.rcParams["axes.labelsize"] == before


def test_grid_equalise_and_frame():
    fig, axs = F.panel_grid(2, ["m100", "m25"])
    axs[0, 0].plot([0, 1], [0, 1]); axs[0, 1].plot([0, 5], [0, 3])
    axs[1, 0].plot([0, 1], [0, 1]); axs[1, 0]._auto_y = True
    axs[1, 1].plot([0, 2], [1, 9]); axs[1, 1]._auto_y = True
    F.equalise_rows(fig, axs)
    assert axs[0, 0].get_ylim() == axs[0, 1].get_ylim() and axs[0, 0].get_xlim() == axs[0, 1].get_xlim()
    lo, hi = axs[1, 0].get_ylim()
    assert lo < 0 and hi > 9                                   # 15 % beyond the drawn lines
    assert len(fig.artists) == 1                               # the frame around the first column
    plt.close(fig)


def test_labels_scalebar_captions():
    fig, ax = plt.subplots()
    F.panel_letter(ax, 1, 2, "dust"); assert ax.get_title(loc="right") == "(b3) dust"
    F.scalebar(ax, (0, 100, 0, 100), 20); F.scalebar(ax, (0, 1e4, 0, 1e4), 2000); F.scalebar(ax, (0, 2, 0, 2), 0.5)
    assert [t.get_text() for t in ax.texts] == ["20 kpc", "2 Mpc", "500 pc"]
    F.panel_label(ax, "G9"); F.zone_frame(ax, "tab:red", tag="core")
    plt.close(fig)
    assert F.caption_lines(["a" * 10, "b" * 10], width=15) == "a" * 10 + "\n" + "b" * 10
    wrapped = F.wrap_caption("x " * 50 + "$a b c d e$", width=20)
    assert all(len(l) <= 20 for l in wrapped.split("\n")[:-1]) and "$a b c d e$" in wrapped


def test_spaces_and_limits():
    spec = {"log": True, "floor": 1e-3}
    v = np.array([1e-5, 1.0, 100.0])
    np.testing.assert_allclose(F.to_plot_space(spec, v), [-3, 0, 2])
    np.testing.assert_allclose(F.from_plot_space(spec, [0, 2]), [1, 100])
    assert np.isnan(F.to_plot_space({"log": True}, [-1.0])[0])
    lo, hi = F.robust_limits(np.r_[np.arange(20.0), 1e6])
    assert hi < 100                                             # the outlier does not stretch it
    assert F.robust_limits(np.ones(10), zero_iqr_fallback=False) is None
    assert F.clipped_limits(np.r_[np.ones(20), 1e9], log=True)[1] < 10


def test_tracks_and_strips():
    rng = np.random.default_rng(0)
    T = Table(dict(stage=np.array(["sft", "qt", "end"] * 10), is_ul=np.zeros(30, bool),
                   logSigmaH2=rng.normal(size=30), logSigmaSFR=rng.normal(size=30)))
    fig, ax = plt.subplots()
    meds = F.draw_median_track(ax, T, "C0", ["sft", "qt", "end"], {"sft": "o", "qt": "s", "end": "*"}, end_stages=("end",))
    assert [m["stage"] for m in meds] == ["sft", "qt", "end"] and all("x" in m for m in meds)
    tr = {"x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]), "lo": np.array([-1.0, 0.0]), "hi": np.array([1.0, 2.0])}
    F.draw_track(ax, tr, {"log": False}, {"log": True}, "C1")
    ps = F.significance_strip(ax, [0, 1, 2], [rng.normal(size=30), rng.normal(size=3)],
                              [rng.normal(3, 1, 30), rng.normal(size=30)], 0.0, 0.05)
    assert ps[0] < 1e-3 and np.isnan(ps[1])
    axb = F.count_histogram_behind(ax, np.arange(3.0), {"a": [1, 2, 3]}, {"a": "C2"})
    assert axb.get_ylim()[1] == pytest.approx(3 * 2.8)
    assert F.kde_contour_or_scatter(ax, rng.normal(size=50), rng.normal(size=50), "C3") == "contour"
    assert F.kde_contour_or_scatter(ax, [1.0, 2.0], [1.0, 2.0], "C3") == "scatter"
    plt.close(fig)


def test_font_floor_patches_once():
    from matplotlib.axes import Axes
    F.apply_font_floor(9)
    F.apply_font_floor(9)                                       # idempotent
    fig, ax = plt.subplots()
    t = ax.text(0, 0, "x", fontsize=4)
    assert t.get_fontsize() == 9
    assert getattr(Axes.text, "_fontfloor", False)
    plt.close(fig)
