"""Figures: history plots, paper style, and particle rendering.

- ``plots``      history and merger-track plots
- ``animation``  simple animations
- ``rendering``  particle images (needs sphviewer, yt for some classes; loaded on first use)
"""
from .plots import HistoryPlots, plot_merger_rate_by_phase, plot_main_galaxy_track, plot_neighborhood_track
from .animation import create_animation

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

