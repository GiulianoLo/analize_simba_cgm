"""Radial and resolved analysis of particle files.

- ``cutouts``   per-galaxy / halo / aperture particle extraction
- ``frames``    centring, principal axes, face-on and sightline projection
- ``profiles``  radial profiles
- ``grains``    dust and metal conversions (zone budgets and grain clocks to follow)
"""
from .cutouts import extract_particles
from .frames import shrink_center, principal_axes, rotate_to_frame, sightline_unit_vectors, projected_radius
from .profiles import radial_profile
from .grains import Z_to_OH12, Dust_to_Metal

__all__ = [
    "extract_particles",
    "shrink_center", "principal_axes", "rotate_to_frame", "sightline_unit_vectors", "projected_radius",
    "radial_profile", "Z_to_OH12", "Dust_to_Metal",
]
