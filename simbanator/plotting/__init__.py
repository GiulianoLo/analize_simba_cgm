"""Figures: history plots, paper style, and particle rendering.

- ``figures``    paper style + saving, panel grids, labels, captions, axis spaces, robust limits, tracks, strips
- ``plots``      history and merger-track plots
- ``animation``  simple animations
- ``rendering``  particle images (needs sphviewer, yt for some classes; loaded on first use)
"""
from .plots import HistoryPlots, plot_merger_rate_by_phase, plot_main_galaxy_track, plot_neighborhood_track
from .animation import create_animation
from .figures import *  # noqa: F401,F403  (the names listed in figures.__all__)
from .figures import __all__ as _figures_all

_LAZY = {n: "rendering" for n in ("ParticleProjectionRender", "plot_particle_projections", "RenderRGB",
                                    "SingleRender", "ReducedRender", "find_rot_ax", "rotation_matrices_from_angles",
                                    "get_normalized_image", "log_stretch", "apply_dust_screen", "gamma_correction")}

def __getattr__(name):
    # Heavy modules load on first use, so importing this package stays light.
    sub = _LAZY.get(name)
    if sub is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{sub}"), name)

