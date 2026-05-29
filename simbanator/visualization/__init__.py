"""Visualization tools for plotting, rendering, and animation."""

from .plots import HistoryPlots, plot_merger_rate_by_phase
from .animation import create_animation

# Heavy-dependency modules (yt, sphviewer) available via explicit import:
#   from simbanator.visualization.rendering import RenderRGB, SingleRender

__all__ = ["HistoryPlots", "create_animation", "plot_merger_rate_by_phase"]
