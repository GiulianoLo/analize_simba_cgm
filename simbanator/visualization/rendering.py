"""SPH particle rendering for RGB composites and videos.

Requires optional heavy dependencies: ``yt``, ``py-sphviewer``, ``caesar``.
Install with ``pip install simbanator[full]``.

Classes
-------
RenderRGB
    Multi-component (gas + stars + optional dust) RGB blended images.
SingleRender
    Single-component maps with optional velocity stream plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

import yt
import caesar
import sphviewer as sph
from sphviewer.tools import camera_tools, QuickView, Blend

from ..io.paths import SavePaths


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def find_rot_ax(L, t=None, p=None, spos='faceon'):
    """Compute camera angles from an angular-momentum vector *L*."""
    x_vec = np.array([1, 0, 0])
    y_vec = np.array([0, 1, 0])
    cos_theta = np.dot(L, x_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(L))
    cos_phi = np.dot(L, y_vec) / (np.linalg.norm(y_vec) * np.linalg.norm(L))

    offset = 90 if spos == 'faceon' else 0
    if t is None:
        t = np.rad2deg(np.arccos(cos_theta)) + offset
    if p is None:
        p = np.rad2deg(np.arccos(cos_phi)) + offset
    return t, p


def rotation_matrices_from_angles(theta, phi):
    """Return the combined R_z(theta) @ R_y(phi) rotation matrix."""
    theta, phi = np.radians(theta), np.radians(phi)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    R_y = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)],
    ])
    return R_z @ R_y


def get_normalized_image(image, vmin=1, vmax=99, mode='linear', zscale=False):
    """Percentile-clip and optionally log/sqrt-scale an image array."""
    lo = np.percentile(image, vmin)
    hi = np.percentile(image, vmax)
    image = np.clip(image, lo, hi)
    if mode == 'log':
        mk = image > 0
        image[mk] = np.log10(image[mk])
    elif mode == 'sqrt':
        image = np.sqrt(np.clip(image, 0, None))
    if zscale:
        mean, std = np.mean(image), np.std(image)
        if std > 0:
            image = (image - mean) / std
    lo2, hi2 = image.min(), image.max()
    if hi2 > lo2:
        return (image - lo2) / (hi2 - lo2)
    return image - lo2


def apply_dust_screen(image, dust):
    """Attenuate *image* using a normalised *dust* map."""
    dust = (dust - dust.min()) / (dust.max() - dust.min() + 1e-30)
    return image * (1 - dust)


def gamma_correction(image, gamma=2.2):
    """Apply gamma correction to *image*."""
    return image ** gamma


# -----------------------------------------------------------------------
# Data loader
# -----------------------------------------------------------------------

def _load_particle_data(snapfile, catfile, gal_id, propr,
                        region=False, ifdust=False,
                        dim=('Msun', 'Msun', 'Msun')):
    """Load snapshot data and return particle arrays."""
    ds = yt.load(snapfile)
    obj = caesar.load(catfile)
    ad = ds.all_data()
    gal = obj.galaxies[gal_id]

    def get_data(particle_type, prop, dim_unit, indices=None):
        pos = ad[particle_type, 'Coordinates']
        mass = ad[particle_type, prop[:-2] if '_s' in prop else prop]
        if indices is not None:
            pos, mass = pos[indices], mass[indices]
        pos = pos.in_units('kpc').value
        if '_s' in prop:
            mass = ds.arr(mass, 'code_mass').in_units(dim_unit).value
        else:
            mass = mass.in_units(dim_unit).value
        return pos, mass

    if region:
        gas_pos, gas_mass = get_data('PartType0', propr[0], dim[0])
        star_pos, star_mass = get_data('PartType4', propr[1], dim[1])
        dust = get_data('PartType0', propr[2], dim[2]) if ifdust else (None, None)
    else:
        gas_pos, gas_mass = get_data('PartType0', propr[0], dim[0], gal.glist)
        star_pos, star_mass = get_data('PartType4', propr[1], dim[1], gal.slist)
        dust = get_data('PartType0', propr[2], dim[2], gal.glist) if ifdust else (None, None)

    return ds, obj, gal, gas_pos, gas_mass, star_pos, star_mass, dust[0], dust[1]


# -----------------------------------------------------------------------
# RenderRGB
# -----------------------------------------------------------------------

class RenderRGB:
    """Multi-component SPH rendering with gas/star/dust blending.

    Parameters
    ----------
    snapfile, catfile : str
        Paths to the snapshot and Caesar catalog.
    id : int
        Galaxy index.
    propr : tuple of str
        Property names for (gas, stars, dust).
    region : bool
        Use all particles instead of galaxy members.
    ifdust : bool
        Include a dust-screen component.
    dim : list of str
        Unit strings for each component.
    """

    def __init__(self, snapfile, catfile, id, propr,
                 region=False, ifdust=True, dim=('Msun', 'Msun', 'Msun')):
        (self.ds, self.obj, self.gal,
         self.gas_pos, self.gas_mass,
         self.star_pos, self.star_mass,
         self.dust_pos, self.dust_mass) = _load_particle_data(
            snapfile, catfile, id, propr, region, ifdust, dim
        )
        self.a = self.obj.simulation.scale_factor
        self.ifdust = ifdust
        self.phys_ext = None
        self.region = region

    def set_camera(self, center=None, extent=5, t=None, p=None,
                   r='infinity', roll=0, xsize=400, ysize=400,
                   zoom=None, spos='faceon'):
        """Configure an sphviewer Camera."""
        if center is None:
            center = self.gal.minpotpos.in_units('kpc').value
        L = self.gal.rotation['gas_L']
        tn, pn = find_rot_ax(L, t, p, spos)
        return sph.Camera(
            x=center[0], y=center[1], z=center[2],
            r=r, t=tn, p=pn, roll=roll,
            extent=[-extent, extent, -extent, extent],
            xsize=xsize, ysize=ysize, zoom=zoom,
        )

    def set_particles(self):
        """Create sphviewer Particles objects."""
        particles = [
            sph.Particles(self.gas_pos, self.gas_mass),
            sph.Particles(self.star_pos, self.star_mass),
        ]
        if self.ifdust:
            particles.append(sph.Particles(self.dust_pos, self.dust_mass))
        return particles

    def set_rgb(self, particle, camera, update):
        """Render a single component and return the raw image."""
        if update is not None:
            S = sph.Scene(particle)
            S.update_camera(**update)
        else:
            S = sph.Scene(particle, Camera=camera)
        R = sph.Render(S)
        self.phys_ext = R.get_extent()
        return R.get_image()

    def generate_images(self, camera, vmin=None, vmax=None):
        """Generate a blended gas + star image.

        Returns
        -------
        np.ndarray
            RGBA image array suitable for ``imshow``.
        """
        particles = self.set_particles()
        rgbs = [self.set_rgb(p, camera=camera, update=None) for p in particles[:2]]

        gas = get_normalized_image(rgbs[0], vmin, vmax, mode='sqrt', zscale=True)
        stars = get_normalized_image(rgbs[1], vmin, vmax, mode='sqrt', zscale=True)

        if self.ifdust:
            dust_screen = self.set_rgb(particles[2], camera=camera, update=None)
            dust_screen = get_normalized_image(dust_screen, vmin, vmax, mode='linear', zscale=True)
            gas = apply_dust_screen(gas, dust_screen)
            stars = apply_dust_screen(stars, dust_screen)

        blend = Blend.Blend(cm.Greys_r(stars), cm.afmhot(gas))
        return blend.Screen()

    def plot(self, image, xl, yl, name, correct=False):
        """Display and save a rendered image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        if isinstance(correct, float):
            image = gamma_correction(image, correct)
        ax.imshow(image, extent=self.phys_ext)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

        paths = SavePaths()
        base_dir = paths.create_subdir(paths.get_filetype_path('plot'), 'renders')
        prefix = 'region_' if self.region else ''
        fig.savefig(os.path.join(base_dir, f'{prefix}{name}.png'))

    def set_video(self, num_frames, p=None, t=None, r='infinity',
                  extent=5, del_p=360, del_t=0, xsize=500, ysize=500,
                  vmin=None, vmax=None, spos='faceon', zoom=1.):
        """Render individual frames for a rotation video."""
        targets = [self.gal.minpotpos.in_units('kpc').value]
        L = self.gal.rotation['gas_L']
        tn, pn = find_rot_ax(L, t, p, spos)
        anchors = {
            'sim_times': np.linspace(0, 1, num_frames),
            'id_frames': np.arange(num_frames),
            'r': [r] * num_frames,
            'id_targets': [0] * num_frames,
            't': np.linspace(tn, tn + del_t, num_frames),
            'p': np.linspace(pn, pn + del_p, num_frames),
            'zoom': [zoom] * num_frames,
            'extent': [extent] * num_frames,
        }
        data = camera_tools.get_camera_trajectory(targets, anchors)

        paths = SavePaths()
        frame_dir = paths.create_subdir(
            paths.create_subdir(paths.get_filetype_path('plot'), 'videos'),
            'frames',
        )
        particles = self.set_particles()
        for h, cam_update in enumerate(data):
            cam_update.update({'xsize': xsize, 'ysize': ysize, 'roll': 0})
            rgbs = [self.set_rgb(p, camera=None, update=cam_update)
                    for p in particles[:2]]
            gas = get_normalized_image(rgbs[0], vmin, vmax, mode='sqrt', zscale=True)
            stars = get_normalized_image(rgbs[1], vmin, vmax, mode='sqrt', zscale=False)

            if self.ifdust:
                dust_img = self.set_rgb(particles[2], camera=None, update=cam_update)
                dust_img = get_normalized_image(dust_img, vmin, vmax, mode='linear', zscale=True)
                gas = apply_dust_screen(gas, dust_img)
                stars = apply_dust_screen(stars, dust_img)

            blend = Blend.Blend(cm.Greys_r(stars), cm.afmhot(gas))
            plt.imsave(f'{frame_dir}/image_{h:04d}.png', blend.Screen())

    def create_video(self, name, interval=100):
        """Stitch rendered frames into a GIF."""
        paths = SavePaths()
        frame_dir = paths.create_subdir(
            paths.create_subdir(paths.get_filetype_path('plot'), 'videos'),
            'frames',
        )
        prefix = 'region_' if self.region else ''
        save_path = os.path.join(
            paths.create_subdir(paths.get_filetype_path('plot'), 'videos'),
            f'{prefix}{name}.gif',
        )
        image_files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
        images = [Image.open(os.path.join(frame_dir, f)) for f in image_files]
        images[0].save(save_path, save_all=True, append_images=images[1:],
                       duration=interval, loop=0)
        print(f"GIF saved as {save_path}")

    def flush(self):
        """Remove all frame PNGs from the frames directory."""
        paths = SavePaths()
        frame_dir = paths.create_subdir(
            paths.create_subdir(paths.get_filetype_path('plot'), 'videos'),
            'frames',
        )
        for fn in os.listdir(frame_dir):
            if fn.lower().endswith('.png'):
                os.remove(os.path.join(frame_dir, fn))


# -----------------------------------------------------------------------
# SingleRender
# -----------------------------------------------------------------------

class SingleRender:
    """Single-component SPH map with optional velocity stream plot.

    Parameters
    ----------
    snapfile, catfile : str
        Paths to the snapshot and Caesar catalog.
    id : int
        Galaxy index.
    propr : tuple
        ``(particle_type, property_name)``.
    region : bool
        Use all particles.
    dim : str
        Unit for the property (e.g. ``'Msun'``).
    """

    def __init__(self, snapfile, catfile, id, propr, region=False, dim='Msun'):
        self.ds = yt.load(snapfile)
        self.obj = caesar.load(catfile)
        self.ad = self.ds.all_data()
        self.gal = self.obj.galaxies[id]
        self.propr = propr
        self.region = region
        self.dim = dim
        self.phys_ext = None
        self._initialize_data()

    def _initialize_data(self):
        def get_data(ptype, prop, dim_unit, indices=None):
            pos = self.ad[ptype, 'Coordinates']
            mass = self.ad[ptype, prop[:-2] if '_s' in prop else prop]
            if indices is not None:
                pos, mass = pos[indices], mass[indices]
            pos = pos.in_units('kpc').value
            if '_s' in prop:
                mass = self.ds.arr(mass, 'code_mass').in_units(dim_unit).value
            else:
                mass = mass.in_units(dim_unit).value
            return pos, mass

        if self.region:
            self.pos, self.mass = get_data(self.propr[0], self.propr[1], self.dim)
        else:
            idx = self.gal.glist if self.propr[0] == 'PartType0' else self.gal.slist
            self.pos, self.mass = get_data(self.propr[0], self.propr[1], self.dim, idx)

        self.a = self.obj.simulation.scale_factor
        self.pos *= self.a
        self.mass *= self.a

    def single_map(self, center=None, ex=5, t=None, p=None,
                   r='infinity', roll=0, xsize=400, ysize=400,
                   zoom=None, spos='faceon', cmap='viridis',
                   vmin=1, vmax=99, mode='log', zscale=False):
        """Generate a single projected map."""
        if center is None:
            center = self.gal.minpotpos.in_units('kpc').value * self.a
        L = self.gal.rotation['gas_L']
        t, p = find_rot_ax(L, t, p, spos)
        P = sph.Particles(self.pos, self.mass)
        C = sph.Camera(
            x=center[0], y=center[1], z=center[2],
            r=r, t=t, p=p, roll=roll,
            extent=[-ex, ex, -ex, ex],
            xsize=xsize, ysize=ysize, zoom=zoom,
        )
        S = sph.Scene(P, Camera=C)
        R = sph.Render(S)
        self.phys_ext = R.get_extent()
        return get_normalized_image(R.get_image(), vmin, vmax, mode, zscale)

    def stream_plot(self, center=None, ex=5, r='infinity',
                    t=None, p=None, xl='x', yl='y', spos='faceon'):
        """Overlay velocity streamlines on a density map."""
        L = self.gal.rotation['gas_L']
        t, p = find_rot_ax(L, t, p, spos=spos)
        R_mat = rotation_matrices_from_angles(t, p)

        if self.propr[0] != 'PartType0':
            raise ValueError('Stream plots require gas particles (PartType0)')

        if self.region:
            vel = self.ad['PartType0', 'Velocities'].in_units('m/s').value * self.a
            hsml = self.ad['PartType0', 'SmoothingLength'].in_units('kpc').value * self.a
        else:
            vel = self.ad['PartType0', 'Velocities'][self.gal.glist].in_units('m/s').value * self.a
            hsml = self.ad['PartType0', 'SmoothingLength'][self.gal.glist].in_units('kpc').value * self.a

        if center is None:
            center = self.gal.minpotpos.in_units('kpc').value * self.a

        pos = np.dot(self.pos - center, R_mat.T) + center
        vel = np.dot(vel, R_mat.T)

        qp = QuickView(pos, hsml=hsml, r='infinity',
                        x=center[0], y=center[1], z=center[2],
                        plot=False, extent=[-ex, ex, -ex, ex], logscale=False)
        density_field = qp.get_image()
        extent = qp.get_extent()

        eps = 1e-10
        vfield = []
        for i in range(2):
            qv = QuickView(pos, vel[:, i], hsml=hsml, r='infinity',
                            x=center[0], y=center[1], z=center[2],
                            plot=False, extent=[-ex, ex, -ex, ex], logscale=False)
            frac = qv.get_image() / (density_field + eps)
            frac[~np.isfinite(frac)] = 0.
            vfield.append(frac)

        fig, ax = plt.subplots(figsize=(12, 12))
        X = np.linspace(extent[0], extent[1], 500)
        Y = np.linspace(extent[2], extent[3], 500)
        ax.imshow(np.log1p(density_field), origin='lower', extent=extent, cmap='bone')

        vmag = np.sqrt(vfield[0] ** 2 + vfield[1] ** 2)
        color = np.log1p(vmag)
        max_c = np.max(color)
        color = color / max_c if max_c > 0 else np.zeros_like(color)
        lw = 2 * color

        ax.streamplot(X, Y, vfield[0], vfield[1],
                      color=color, density=1.5, cmap='jet', linewidth=lw, arrowsize=1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.minorticks_on()
        ax.set_xlabel(xl, size=25)
        ax.set_ylabel(yl, size=25)
        plt.show()
        return fig, ax

    def plot(self, image, xl, yl, name, vmin=None, vmax=None):
        """Display and save a single-component map."""
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(image, extent=self.phys_ext, cmap='viridis')
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, label='Mass [Msun]')

        paths = SavePaths()
        base_dir = paths.create_subdir(paths.get_filetype_path('plot'), 'renders')
        prefix = 'map_region_' if self.region else 'map_'
        fig.savefig(os.path.join(base_dir, f'{prefix}{self.propr[1]}_{name}.png'))
