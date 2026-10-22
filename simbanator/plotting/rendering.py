"""SPH particle rendering for RGB composites and videos.

Requires optional heavy dependencies: ``py-sphviewer`` always, plus ``yt`` and ``caesar``
for the classes that read a snapshot.  Install with ``pip install simbanator[full]``.

Classes
-------
RenderRGB
    Multi-component (gas + stars + optional dust) RGB blended images, from a snapshot.
SingleRender
    Single-component maps with optional velocity stream plots, from a snapshot.
ReducedRender
    The same maps, composites and velocity field from a *reduced particle cutout*
    (``build_reduced_particles_job.py``) — no snapshot and no catalogue, so a grid of many
    galaxies × epochs is cheap.  Needs only py-sphviewer and h5py.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

import sphviewer as sph
from sphviewer.tools import camera_tools, QuickView, Blend

try:                       # only the snapshot-reading classes need these; ReducedRender does not
    import yt
    import caesar
except ImportError:        # pragma: no cover - environment-dependent
    yt = caesar = None


def _require_yt():
    """Raise a useful error instead of `NoneType has no attribute load`."""
    if yt is None or caesar is None:
        raise ImportError(
            "This renderer reads a snapshot and needs yt + caesar "
            "(pip install simbanator[full]).  ReducedRender works without them."
        )


def _infer_sim_name(simulation_obj, fallback='default'):
    """Best-effort simulation-name extraction from a Caesar simulation object."""
    if simulation_obj is None:
        return fallback

    for attr in ('name', 'sim_name', 'simulation_name', 'label'):
        value = getattr(simulation_obj, attr, None)
        if isinstance(value, str) and value:
            return value

    return fallback




# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------
class ParticleProjectionRender:
    """
    Simple particle projection plotter for visual selection checks.
    Parameters
    ----------
    snapfile, catfile : str
        Paths to the snapshot and Caesar catalog.
    id : int
        Galaxy index.
    particle_type : str
        Particle type (e.g. 'PartType0' for gas, 'PartType4' for stars).
    region : bool
        Use all particles instead of galaxy members.
    """
    def __init__(self, snapfile, catfile, id, particle_type='PartType0', region=False):
        import yt
        import caesar
        ds = yt.load(snapfile)
        obj = caesar.load(catfile)
        ad = ds.all_data()
        gal = obj.galaxies[id]
        self.a = obj.simulation.scale_factor
        self.region = region
        self.particle_type = particle_type
        self.gal = gal
        self.ds = ds
        self.obj = obj
        self.ad = ad
        if region:
            self.positions = ad[particle_type, 'Coordinates'].in_units('kpc').value * self.a
        else:
            idx = gal.glist if particle_type == 'PartType0' else gal.slist
            self.positions = ad[particle_type, 'Coordinates'][idx].in_units('kpc').value * self.a

    def plot(self, figsize=(12, 4), marker='o', s=2, alpha=0.6, title=None):
        """
        Plot particle positions projected onto the three axes (xy, xz, yz).
        """
        positions = self.positions
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].scatter(positions[:,0], positions[:,1], marker=marker, s=s, alpha=alpha)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('XY projection')
        axes[1].scatter(positions[:,0], positions[:,2], marker=marker, s=s, alpha=alpha)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Z')
        axes[1].set_title('XZ projection')
        axes[2].scatter(positions[:,1], positions[:,2], marker=marker, s=s, alpha=alpha)
        axes[2].set_xlabel('Y')
        axes[2].set_ylabel('Z')
        axes[2].set_title('YZ projection')
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

def plot_particle_projections(positions, figsize=(12, 4), marker='o', s=2, alpha=0.6, title=None):
    """
    Plot particle positions projected onto the three axes (xy, xz, yz).
    Parameters:
        positions (ndarray): Array of shape (N, 3) with particle positions.
        figsize (tuple): Figure size.
        marker (str): Marker style for scatter plot.
        s (int): Marker size.
        alpha (float): Marker transparency.
        title (str): Optional figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].scatter(positions[:,0], positions[:,1], marker=marker, s=s, alpha=alpha)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY projection')
    axes[1].scatter(positions[:,0], positions[:,2], marker=marker, s=s, alpha=alpha)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ projection')
    axes[2].scatter(positions[:,1], positions[:,2], marker=marker, s=s, alpha=alpha)
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ projection')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

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
    """Percentile-clip and optionally log/sqrt/asinh-scale an image array.

    Parameters
    ----------
    image : ndarray
    vmin, vmax : float or None
        Percentile bounds for clipping (default 1/99).  None uses those defaults.
    mode : {'linear', 'log', 'sqrt', 'asinh'}
        Stretch applied after clipping.  ``'asinh'`` is the Lupton et al. (2004)
        stretch recommended for SPH composites.
    zscale : bool
        Apply Z-score normalisation after stretching.  Avoid combining with
        ``mode='log'`` or ``mode='sqrt'`` — they already compress dynamic range.
    """
    if vmin is None:
        vmin = 1
    if vmax is None:
        vmax = 99
    image = image.copy().astype(float)
    lo = np.percentile(image, vmin)
    hi = np.percentile(image, vmax)
    image = np.clip(image, lo, hi)
    if mode == 'log':
        floor = lo if lo > 0 else image[image > 0].min() if np.any(image > 0) else 1e-30
        image = np.log10(np.clip(image, floor, None))
    elif mode == 'sqrt':
        image = np.sqrt(np.clip(image, 0, None))
    elif mode == 'asinh':
        # Lupton et al. (2004): asinh stretch that preserves colour ratios
        stretch = np.percentile(image, 50) + 1e-30
        image = np.arcsinh(image / stretch)
    if zscale:
        mean, std = np.mean(image), np.std(image)
        if std > 0:
            image = (image - mean) / std
    lo2, hi2 = image.min(), image.max()
    if hi2 > lo2:
        return (image - lo2) / (hi2 - lo2)
    return image - lo2


def log_stretch(image, dyn=3.5, pmax=99.9):
    """Log stretch over a FIXED dynamic range below the bright end -> [0, 1].

    The percentile clip of :func:`get_normalized_image` saturates whenever the source fills
    only a small part of the frame: if the galaxy covers less than ``100 - vmax`` per cent of
    the pixels, *every* one of its pixels sits above the clip and comes out at the same value —
    a uniform blob on a black field, which is exactly what a 5 kpc galaxy in a 30 kpc frame
    looks like at ``vmax=99.5``.  This anchors the top of the scale at the ``pmax`` percentile
    of the POSITIVE pixels and shows ``dyn`` decades below it, so the stretch depends on the
    source's own dynamic range and not on how much empty sky surrounds it.

    Parameters
    ----------
    image : ndarray
        A surface-density map; non-positive pixels map to 0.
    dyn : float
        Decades displayed below the bright end.  2-3 for a high-contrast core, 4-5 to bring
        out faint outskirts.
    pmax : float
        Percentile of the positive pixels that saturates to 1 (99.9 clips only the brightest
        few pixels, which are usually one smoothing length across).
    """
    img = np.asarray(image, float)
    pos = img[np.isfinite(img) & (img > 0)]
    if pos.size == 0:
        return np.zeros_like(img)
    hi = float(np.percentile(pos, pmax))
    dyn = float(dyn)
    if not (hi > 0) or dyn <= 0:
        return np.zeros_like(img)
    lo = hi / (10.0 ** dyn)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log10(np.clip(np.nan_to_num(img, nan=0.0), lo, hi) / lo) / dyn
    return np.clip(out, 0.0, 1.0)


def apply_dust_screen(image, dust, tau_max=2.0):
    """Attenuate *image* via Beer-Lambert exponential extinction.

    Parameters
    ----------
    image : ndarray
        Normalised [0, 1] channel image.
    dust : ndarray
        Dust surface density map (any units; will be normalised to [0, 1]).
    tau_max : float
        Peak optical depth.  ``tau_max=2`` gives ~86 % attenuation at max dust
        surface density; ``tau_max=0`` disables attenuation.
    """
    dust = (dust - dust.min()) / (dust.max() - dust.min() + 1e-30)
    return image * np.exp(-tau_max * dust)


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
    _require_yt()
    ds = yt.load(snapfile)
    obj = caesar.load(catfile)
    ad = ds.all_data()
    gal = obj.galaxies[gal_id]

    def get_data(particle_type, prop, dim_unit, indices=None):
        pos = ad[particle_type, 'Coordinates']
        mass = ad[particle_type, prop[:-2] if '_s' in prop else prop]
        if indices is not None:
            pos, mass = pos[indices], mass[indices]
        pos = pos.in_units('kpc').value # * obj.simulation.scale_factor
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
        print('center=======================>', center)
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
            
        print(self.gas_pos)
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

    def generate_images(self, camera, vmin=1, vmax=99.5,
                        mode='log', tau_max=1.5,
                        star_cmap=cm.Greys_r, gas_cmap=cm.afmhot):
        """Generate a blended gas + star image.

        Parameters
        ----------
        vmin, vmax : float
            Percentile bounds for channel normalisation (default 1/99.5).
        mode : str
            Stretch for gas and star channels: ``'log'`` (default), ``'sqrt'``,
            or ``'asinh'``.
        tau_max : float
            Optical depth for dust Beer-Lambert attenuation (default 1.5).
        star_cmap, gas_cmap : matplotlib colormap
            Colormaps for the two channels fed into the Screen blend.

        Returns
        -------
        np.ndarray
            RGBA image array suitable for ``imshow``.
        """
        particles = self.set_particles()
        rgbs = [self.set_rgb(p, camera=camera, update=None) for p in particles[:2]]

        gas   = get_normalized_image(rgbs[0], vmin, vmax, mode=mode,     zscale=False)
        stars = get_normalized_image(rgbs[1], vmin, vmax, mode=mode,     zscale=False)

        if self.ifdust:
            dust_screen = self.set_rgb(particles[2], camera=camera, update=None)
            dust_screen = get_normalized_image(dust_screen, vmin, vmax,
                                               mode='linear', zscale=False)
            gas   = apply_dust_screen(gas,   dust_screen, tau_max)
            stars = apply_dust_screen(stars, dust_screen, tau_max)

        blend = Blend.Blend(star_cmap(stars), gas_cmap(gas))
        return blend.Screen()

    def plot(self, image, xl, yl, name, correct=False, output_dir=None):
        """Display and save a rendered image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        if isinstance(correct, float):
            image = gamma_correction(image, correct)
        ax.imshow(image, extent=self.phys_ext)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)

        if output_dir is None:
            sim_name = _infer_sim_name(getattr(self.obj, 'simulation', None))
            output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'renders')
        os.makedirs(output_dir, exist_ok=True)
        prefix = 'region_' if self.region else ''
        fig.savefig(os.path.join(output_dir, f'{prefix}{name}.png'))

    def set_video(self, num_frames, p=None, t=None, r='infinity',
                  extent=5, del_p=360, del_t=0, xsize=500, ysize=500,
                  vmin=1, vmax=99.5, mode='log', tau_max=1.5,
                  spos='faceon', zoom=1., output_dir=None):
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

        if output_dir is None:
            sim_name = _infer_sim_name(getattr(self.obj, 'simulation', None))
            output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'videos')
        frame_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        particles = self.set_particles()
        for h, cam_update in enumerate(data):
            cam_update.update({'xsize': xsize, 'ysize': ysize, 'roll': 0})
            rgbs = [self.set_rgb(p, camera=None, update=cam_update)
                    for p in particles[:2]]
            gas   = get_normalized_image(rgbs[0], vmin, vmax, mode=mode, zscale=False)
            stars = get_normalized_image(rgbs[1], vmin, vmax, mode=mode, zscale=False)

            if self.ifdust:
                dust_img = self.set_rgb(particles[2], camera=None, update=cam_update)
                dust_img = get_normalized_image(dust_img, vmin, vmax,
                                                mode='linear', zscale=False)
                gas   = apply_dust_screen(gas,   dust_img, tau_max)
                stars = apply_dust_screen(stars, dust_img, tau_max)

            blend = Blend.Blend(cm.Greys_r(stars), cm.afmhot(gas))
            plt.imsave(f'{frame_dir}/image_{h:04d}.png', blend.Screen())

    def create_video(self, name, interval=100, output_dir=None):
        """Stitch rendered frames into a GIF."""
        if output_dir is None:
            sim_name = _infer_sim_name(getattr(self.obj, 'simulation', None))
            output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'videos')
        frame_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        prefix = 'region_' if self.region else ''
        save_path = os.path.join(output_dir, f'{prefix}{name}.gif')
        image_files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
        images = [Image.open(os.path.join(frame_dir, f)) for f in image_files]
        images[0].save(save_path, save_all=True, append_images=images[1:],
                       duration=interval, loop=0)
        print(f"GIF saved as {save_path}")

    def flush(self, output_dir=None):
        """Remove all frame PNGs from the frames directory."""
        if output_dir is None:
            sim_name = _infer_sim_name(getattr(self.obj, 'simulation', None))
            output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'videos')
        frame_dir = os.path.join(output_dir, 'frames')
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
        ``(particle_type, field_name)`` — e.g. ``('PartType0', 'Temperature')``,
        ``('PartType0', 'StarFormationRate')``, ``('PartType0', 'H2I')``,
        ``('PartType4', 'Masses')``.
    region : bool
        Use all particles instead of galaxy members.
    dim : str or None
        yt unit string for the field (e.g. ``'Msun'``, ``'K'``,
        ``'Msun/yr'``, ``'cm**-3'``).  Pass ``None`` to use the raw value
        without any unit conversion (useful for dimensionless quantities).
    as_code_mass : bool
        If ``True``, wrap the raw array in ``ds.arr(..., 'code_mass')`` before
        unit conversion.  Required for SIMBA fields like ``Dust_Masses`` that
        yt reads without units.  Equivalent to the legacy ``'FieldName_s'``
        suffix convention.
    """

    def __init__(self, snapfile, catfile, id, propr, region=False,
                 dim='Msun', as_code_mass=False):
        _require_yt()
        self.ds = yt.load(snapfile)
        self.obj = caesar.load(catfile)
        self.ad = self.ds.all_data()
        self.gal = self.obj.galaxies[id]
        self.propr = propr
        self.region = region
        self.dim = dim
        self.as_code_mass = as_code_mass
        self.phys_ext = None
        self._initialize_data()

    def _initialize_data(self):
        self.a = self.obj.simulation.scale_factor

        # Legacy: field names ending in '_s' signal a SIMBA code_mass field.
        # Preferred: pass as_code_mass=True with the real field name instead.
        prop = self.propr[1]
        use_code_mass = self.as_code_mass
        if not use_code_mass and prop.endswith('_s'):
            use_code_mass = True
            prop = prop[:-2]

        def get_data(ptype, actual_prop, dim_unit, indices=None):
            pos = self.ad[ptype, 'Coordinates']
            raw = self.ad[ptype, actual_prop]
            if indices is not None:
                pos, raw = pos[indices], raw[indices]
            pos = pos.in_units('kpc').value * self.a
            if use_code_mass:
                field = self.ds.arr(raw, 'code_mass')
            else:
                field = raw
            if dim_unit is None:
                val = np.asarray(field)
            else:
                val = field.in_units(dim_unit).value
            return pos, val

        idx = self.gal.glist if self.propr[0] == 'PartType0' else self.gal.slist
        indices = None if self.region else idx
        self.pos, self.mass = get_data(self.propr[0], prop, self.dim, indices)


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

    def plot(self, image, xl, yl, name, vmin=None, vmax=None,
             cmap='viridis', output_dir=None):
        """Display and save a single-component map."""
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(image, extent=self.phys_ext, cmap=cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        cbar_label = (f'{self.propr[1]} [{self.dim}]'
                      if self.dim else self.propr[1])
        fig.colorbar(im, ax=ax, label=cbar_label)

        if output_dir is None:
            sim_name = _infer_sim_name(getattr(self.obj, 'simulation', None))
            output_dir = os.path.join(os.getcwd(), 'output', sim_name, 'renders')
        os.makedirs(output_dir, exist_ok=True)
        prefix = 'map_region_' if self.region else 'map_'
        fig.savefig(os.path.join(output_dir, f'{prefix}{self.propr[1]}_{name}.png'))


# -----------------------------------------------------------------------
# ReducedRender — straight from a reduced particle cutout, no snapshot
# -----------------------------------------------------------------------

class ReducedRender:
    """Face-on / edge-on SPH renders straight from a reduced particle cutout.

    The files written by ``build_reduced_particles_job.py`` hold the 100 kpc neighbourhood of
    one galaxy at one snapshot — the *same* particles the Σ-profile and K-S products are
    measured from — together with the centre and the stellar principal frame.  Rendering from
    them needs only ``py-sphviewer`` and ``h5py``: no snapshot, no Caesar catalogue, so a grid
    of galaxies × epochs renders in seconds, where :class:`RenderRGB` and :class:`SingleRender`
    would each load a full box per panel.  The trade is that only the fields the reduced job
    stored are available (see :attr:`FIELDS`), and only out to ``rmax_kpc``.

    Positions in the file are already relative to the centre, in physical kpc, and the stored
    ``evecs`` follow the ``pos @ evecs`` convention of
    :func:`simbanator.utils.geometry.principal_axes` — columns 0-1 span the disc plane, column
    2 is the minor axis.  ``spos='faceon'`` images columns (0, 1) with 2 along the line of
    sight; ``spos='edgeon'`` images (0, 2) with 1 down the line of sight.  Every map is
    returned as ``image[y, x]`` with its extent, i.e. for
    ``imshow(image, extent=extent, origin='lower')``.

    Parameters
    ----------
    path : str
        The reduced file to read.
    members_only : bool
        Keep only the particles flagged as galaxy members (default ``True``).  ``False``
        renders the whole cutout — the CGM, and any neighbour that falls inside it.
    frame : str
        What ``faceon`` / ``edgeon`` refer to.  ``'stellar'`` (default) is the stored stellar
        principal frame — the inertia tensor of the member stars, which for a spheroidal
        quenched galaxy is nearly degenerate and says nothing about where its gas disc lies.
        ``'H2_L'``, ``'gas_L'``, ``'dust_L'`` or ``'star_L'`` build the frame from the
        **angular momentum** of that component instead (:meth:`set_frame`): column 2 along
        the spin, so ``faceon`` looks down the spin axis and ``edgeon`` along the disc.
    frame_rmax : float
        The radius [kpc] inside which that angular momentum is summed (default 10).
    frame_members : bool
        Sum it over the galaxy's members only (default ``True``), whatever ``members_only``
        says — the CGM and the neighbours must not set the ISM's orientation.

    Examples
    --------
    >>> r = ReducedRender.from_epoch(reduced_dir, snap=61, gx=3)
    >>> img, ext = r.sigma_map('dust', ex=20)             # Msun kpc^-2, face-on
    >>> ax.imshow(np.log10(img), extent=ext, origin='lower')
    >>> rgb, ext = r.rgb(ex=20)                           # stars + gas, dust-attenuated
    >>> X, Y, VX, VY, den = r.velocity_field(ex=20)       # -> ax.streamplot(X, Y, VX, VY)
    """

    #: ``name -> (component, dataset, label)`` for the fields the reduced job stores.
    FIELDS = {
        'gas':  ('gas',  'm_gas',  r'$\Sigma_{\rm gas}$'),
        'HI':   ('gas',  'm_HI',   r'$\Sigma_{\rm HI}$'),
        'H2':   ('gas',  'm_H2',   r'$\Sigma_{\rm H_2}$'),
        'dust': ('gas',  'm_dust', r'$\Sigma_{\rm dust}$'),
        'sfr':  ('gas',  'sfr',    r'$\Sigma_{\rm SFR}$'),
        'star': ('star', 'm_star', r'$\Sigma_\star$'),
    }

    _VIEWS = {'faceon': (0, 1, 2), 'edgeon': (0, 2, 1)}

    #: ``frame -> (component, weight)`` for the angular-momentum frames of :meth:`set_frame`.
    FRAMES = {'H2_L': ('gas', 'm_H2'), 'gas_L': ('gas', 'm_gas'), 'dust_L': ('gas', 'm_dust'), 'star_L': ('star', 'm_star')}

    def __init__(self, path, members_only=True, frame='stellar', frame_rmax=10.0, frame_members=True):
        import h5py

        self.path = path
        self.members_only = bool(members_only)
        self.attrs, self.parts = {}, {}
        with h5py.File(path, 'r') as f:
            for k in f.attrs:
                self.attrs[k] = f.attrs[k]
            for grp in ('gas', 'star'):
                if grp not in f:
                    continue
                keep = slice(None)
                if self.members_only and 'member' in f[grp]:
                    keep = np.asarray(f[grp]['member'][:], bool)
                self.parts[grp] = {k: np.asarray(f[grp][k][keep]) for k in f[grp].keys()}

        evecs = np.asarray(self.attrs.get('evecs', np.eye(3)), float)
        if np.linalg.det(evecs) < 0:      # keep the frame right-handed, so the LOS sign is physical
            evecs = evecs.copy()
            evecs[:, 2] *= -1.0
        self.evecs_stellar = evecs
        self.evecs = evecs
        self.frame = 'stellar'
        self.frame_info = dict(frame='stellar', reason='the stored stellar principal frame')
        self.snap = int(self.attrs.get('snap', -1))
        self.gx = int(self.attrs.get('gx', -1))
        self.redshift = float(self.attrs.get('redshift', np.nan))
        self.rmax_kpc = float(self.attrs.get('rmax_kpc', np.nan))
        if frame != 'stellar':
            self.set_frame(frame, rmax=frame_rmax, members=frame_members)

    def __repr__(self):
        n = {k: len(v.get('m_gas', v.get('m_star', ()))) for k, v in self.parts.items()}
        return (f'<ReducedRender snap={self.snap} gx={self.gx} z={self.redshift:.3f} '
                f'{"members" if self.members_only else "cutout"} {n}>')

    @classmethod
    def from_epoch(cls, reduced_dir, snap, gx, prefix='m25n512', **kwargs):
        """Open the reduced job's own layout: ``<dir>/snap_NNN/<prefix>_snapNNN_galGGGGGG.h5``."""
        path = os.path.join(reduced_dir, 'snap_%03d' % int(snap),
                            '%s_snap%03d_gal%06d.h5' % (prefix, int(snap), int(gx)))
        return cls(path, **kwargs)

    @staticmethod
    def epoch_path(reduced_dir, snap, gx, prefix='m25n512'):
        """The path :meth:`from_epoch` would open — for existence checks before a grid is drawn."""
        return os.path.join(reduced_dir, 'snap_%03d' % int(snap),
                            '%s_snap%03d_gal%06d.h5' % (prefix, int(snap), int(gx)))

    # ── geometry ────────────────────────────────────────────────────────
    def angular_momentum(self, comp='gas', weight='m_H2', rmax=10.0, members=True):
        """The weighted angular momentum of one component inside ``rmax`` kpc, about the centre and
        about the selection's own mass-weighted mean velocity -> ``(L, n_used, coherence)``; ``L`` is
        ``None`` when the file holds no velocities or the selection is empty.  ``coherence`` =
        |sum w r x v| / sum w |r x v|: 1 for a disc, ~1/sqrt(n) for random motions — the test of
        whether there is a disc to orient at all."""
        grp = self.parts.get(comp, {})
        if 'vel' not in grp or 'pos' not in grp or weight not in grp:
            return None, 0, np.nan
        pos = np.asarray(grp['pos'], float)
        vel = np.asarray(grp['vel'], float)
        w = np.asarray(grp[weight], float)
        r = np.sqrt(np.sum(pos ** 2, axis=1))
        sel = np.isfinite(w) & (w > 0) & (r <= float(rmax)) & np.all(np.isfinite(vel), axis=1)
        if members and 'member' in grp:
            sel &= np.asarray(grp['member'], bool)
        n = int(sel.sum())
        if n == 0:
            return None, 0, np.nan
        v0 = np.average(vel[sel], axis=0, weights=w[sel])
        l_i = w[sel, None] * np.cross(pos[sel], vel[sel] - v0)
        L = np.sum(l_i, axis=0)
        denom = float(np.sum(np.linalg.norm(l_i, axis=1)))
        coh = float(np.linalg.norm(L) / denom) if denom > 0 else np.nan
        return L, n, coh

    def set_frame(self, frame='H2_L', rmax=10.0, members=True, n_min=20):
        """Point column 2 of the projection frame along the angular momentum of ``frame``'s component
        (:attr:`FRAMES`), so that ``faceon`` looks down the spin axis of the gas disc rather than
        down the stellar minor axis.  Falls back to the stored stellar frame — and says so in
        :attr:`frame_info` — when the file has no velocities, fewer than ``n_min`` weighted
        particles inside ``rmax``, or a vanishing spin.  Returns :attr:`frame_info`."""
        if frame == 'stellar':
            self.evecs, self.frame = self.evecs_stellar, 'stellar'
            self.frame_info = dict(frame='stellar', reason='the stored stellar principal frame')
            return self.frame_info
        if frame not in self.FRAMES:
            raise ValueError(f"frame must be 'stellar' or one of {sorted(self.FRAMES)}, got {frame!r}")
        comp, weight = self.FRAMES[frame]
        L, n, coh = self.angular_momentum(comp, weight, rmax=rmax, members=members)
        info = dict(frame=frame, comp=comp, weight=weight, rmax_kpc=float(rmax), members=bool(members), n_used=int(n),
                    coherence=float(coh) if np.isfinite(coh) else np.nan)
        if L is None or n < int(n_min) or not np.all(np.isfinite(L)) or np.linalg.norm(L) <= 0:
            info.update(frame='stellar', reason=(f'no velocities / {weight} in {comp}' if L is None and n == 0 else
                                                 f'only {n} weighted particles inside {rmax:g} kpc (n_min {n_min})'))
            self.evecs, self.frame = self.evecs_stellar, 'stellar'
            self.frame_info = info
            return info
        z = L / np.linalg.norm(L)
        x0 = self.evecs_stellar[:, 0]                         # keep the image x near the stellar major axis when possible
        x = x0 - np.dot(x0, z) * z
        if np.linalg.norm(x) < 1e-3:
            x = np.array([1.0, 0.0, 0.0]) - z[0] * z
            if np.linalg.norm(x) < 1e-3:
                x = np.array([0.0, 1.0, 0.0]) - z[1] * z
        x /= np.linalg.norm(x)
        y = np.cross(z, x)                                    # right-handed: det = +1
        self.evecs = np.column_stack([x, y, z])
        self.frame = frame
        tilt = float(np.degrees(np.arccos(np.clip(abs(np.dot(z, self.evecs_stellar[:, 2])), 0.0, 1.0))))
        info.update(L_norm=float(np.linalg.norm(L)), tilt_vs_stellar_deg=tilt, reason='ok')
        self.frame_info = info
        return info

    def set_evecs(self, evecs, label='imposed', info=None):
        """Impose a projection frame from outside — e.g. the spin measured at another epoch of the same galaxy,
        so a row of panels shares one orientation. ``evecs`` must be orthonormal (columns = axes)."""
        e = np.asarray(evecs, float)
        if e.shape != (3, 3) or not np.allclose(e.T @ e, np.eye(3), atol=1e-6):
            raise ValueError('evecs must be a 3x3 orthonormal matrix (columns = axes)')
        if np.linalg.det(e) < 0:
            e = e.copy(); e[:, 2] *= -1.0
        self.evecs, self.frame = e, str(label)
        self.frame_info = dict(info or {}, frame=str(label), reason='imposed')
        return self.frame_info

    def project(self, comp='gas', spos='faceon', what='pos'):
        """A stored (N, 3) vector rotated into the image frame: columns ``(x, y, LOS)``."""
        if spos not in self._VIEWS:
            raise ValueError(f"spos must be one of {sorted(self._VIEWS)}, got {spos!r}")
        v = np.asarray(self.parts[comp][what], float)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"{comp}/{what} is not an (N, 3) array")
        return (v @ self.evecs)[:, list(self._VIEWS[spos])]

    def _field(self, name):
        """``name`` -> (component, dataset).  Unknown names are read from the gas group."""
        if name in self.FIELDS:
            comp, key, _ = self.FIELDS[name]
            return comp, key
        return 'gas', name

    # ── the renders ─────────────────────────────────────────────────────
    def _render(self, xyz, weight, ex, xsize=400, ysize=None):
        """One sphviewer projection -> ``(image[y, x], extent)``; zeros when nothing is left."""
        ysize = xsize if ysize is None else ysize
        extent = [-float(ex), float(ex), -float(ex), float(ex)]
        xyz = np.ascontiguousarray(np.asarray(xyz, float))
        weight = np.ascontiguousarray(np.asarray(weight, float))
        if len(xyz) == 0 or not np.any(np.isfinite(weight)):
            return np.zeros((ysize, xsize)), extent
        P = sph.Particles(xyz, weight)
        C = sph.Camera(x=0, y=0, z=0, r='infinity', t=0, p=0, roll=0,
                       extent=extent, xsize=xsize, ysize=ysize)
        R = sph.Render(sph.Scene(P, Camera=C))
        return np.asarray(R.get_image(), float), list(R.get_extent())

    def sigma_map(self, field='dust', ex=20.0, spos='faceon', xsize=400, ysize=None,
                  normalize=True):
        """Surface density of *field* in the image plane: ``(image, extent)``.

        With ``normalize=True`` (default) the map is rescaled so that
        ``image.sum() * pixel_area`` equals the mass of the particles whose projected
        position falls inside the frame, i.e. the map is in ``[field units] kpc^-2``
        whatever py-sphviewer's internal normalisation is.  Particles outside the frame
        still smear in at the edges, so that equality is a calibration, not a budget.
        """
        comp, key = self._field(field)
        w = np.asarray(self.parts[comp][key], float)
        xyz = self.project(comp, spos)
        img, extent = self._render(xyz, w, ex, xsize, ysize)
        if normalize and img.sum() > 0:
            inside = (np.abs(xyz[:, 0]) <= ex) & (np.abs(xyz[:, 1]) <= ex)
            tot = float(np.nansum(w[inside]))
            pix = ((extent[1] - extent[0]) / img.shape[1]) * ((extent[3] - extent[2]) / img.shape[0])
            if tot > 0:
                img = img * (tot / (float(img.sum()) * pix))
        return img, extent

    def weighted_map(self, quantity, comp='gas', weight='m_gas', ex=20.0, spos='faceon',
                     xsize=400, ysize=None, fill=np.nan):
        """Mass-weighted line-of-sight mean of a per-particle scalar: ``(image, extent)``.

        ``quantity`` is a dataset name (``'temp'``, ``'zgas'``, ...) or an array of length N;
        the string ``'vlos'`` gives the line-of-sight velocity of the rotated frame.  This is
        the ``project(q*m) / project(m)`` trick :meth:`SingleRender.stream_plot` uses.
        """
        w = np.asarray(self.parts[comp][weight], float)
        if isinstance(quantity, str):
            q = (self.project(comp, spos, what='vel')[:, 2] if quantity == 'vlos'
                 else np.asarray(self.parts[comp][quantity], float))
        else:
            q = np.asarray(quantity, float)
        xyz = self.project(comp, spos)
        num, extent = self._render(xyz, w * q, ex, xsize, ysize)
        den, _ = self._render(xyz, w, ex, xsize, ysize)
        out = np.divide(num, den, out=np.full_like(num, fill), where=den > 0)
        return out, extent

    def velocity_field(self, ex=20.0, spos='faceon', comp='gas', weight='m_gas', ngrid=128,
                       subtract_bulk=True, fill=0.0):
        """The mass-weighted velocity field in the image plane, ready for ``streamplot``.

        Returns ``(X, Y, VX, VY, density)`` on an ``ngrid × ngrid`` grid, velocities in the
        units the file stores (km/s, peculiar).  Face-on, a rotating disc closes its
        streamlines into circles and a dispersion-supported galaxy does not — which is the
        split :attr:`kappa_rot` measures, drawn instead of scored.
        """
        xyz = self.project(comp, spos)
        vel = self.project(comp, spos, what='vel')
        w = np.asarray(self.parts[comp][weight], float)
        if subtract_bulk and np.nansum(w) > 0:
            vel = vel - np.average(vel, axis=0, weights=w)
        den, extent = self._render(xyz, w, ex, ngrid, ngrid)
        comps = []
        for i in (0, 1):
            num, _ = self._render(xyz, w * vel[:, i], ex, ngrid, ngrid)
            f = np.divide(num, den, out=np.full_like(num, fill), where=den > 0)
            comps.append(np.nan_to_num(f, nan=fill, posinf=fill, neginf=fill))
        X, Y = np.meshgrid(np.linspace(extent[0], extent[1], ngrid),
                           np.linspace(extent[2], extent[3], ngrid))
        return X, Y, comps[0], comps[1], den

    #: black -> saturated-hue colormaps, one per component, so a Screen blend stays legible
    CHANNEL_CMAPS = {
        'star': mcolors.LinearSegmentedColormap.from_list(
            'sb_star', ['#000000', '#2a2a33', '#9a9aa6', '#ffffff']),
        'gas':  mcolors.LinearSegmentedColormap.from_list(
            'sb_gas',  ['#000000', '#0a2e63', '#1f7fc4', '#a6e4ff']),
        'H2':   mcolors.LinearSegmentedColormap.from_list(
            'sb_h2',   ['#000000', '#083d2e', '#1f9d6b', '#b9f5cf']),
        'HI':   mcolors.LinearSegmentedColormap.from_list(
            'sb_hi',   ['#000000', '#0a2e63', '#1f7fc4', '#a6e4ff']),
        'dust': mcolors.LinearSegmentedColormap.from_list(
            'sb_dust', ['#000000', '#4a1200', '#c1440e', '#ffc27a']),
        'sfr':  mcolors.LinearSegmentedColormap.from_list(
            'sb_sfr',  ['#000000', '#4a0d3d', '#c33ba5', '#ffc8ee']),
    }

    def rgb(self, ex=20.0, spos='faceon', channels=('star', 'gas', 'dust'), cmaps=None,
            dyn=(4.0, 3.5, 3.0), pmax=99.9, stretch='dyn', tau_max=0.0, xsize=400,
            vmin=1, vmax=99.5, mode='log'):
        """Screen-blend one coloured channel per component into a composite: ``(rgba, extent)``.

        Each name in ``channels`` (any key of :attr:`FIELDS`) is rendered, stretched to [0, 1]
        and painted with its own black-to-hue colormap from :attr:`CHANNEL_CMAPS` (override with
        ``cmaps``), then the channels are Screen-blended — so stars, gas and dust stay
        *distinguishable* instead of merging into one warm blob.

        ``stretch='dyn'`` (the default) uses :func:`log_stretch`: ``dyn`` decades below the
        ``pmax`` percentile, per channel (a scalar applies to all).  ``stretch='percentile'``
        reproduces :meth:`RenderRGB.generate_images` exactly — :func:`get_normalized_image` with
        ``vmin`` / ``vmax`` / ``mode`` — which saturates a small source into a uniform blob and
        is kept only for comparison.  ``tau_max > 0`` additionally attenuates every channel by
        the dust column, as :class:`RenderRGB` does; with dust drawn as its own channel that
        attenuation is usually double-counting, so it is off by default.
        """
        cmaps = dict(self.CHANNEL_CMAPS, **(cmaps or {}))
        dyns = ([float(dyn)] * len(channels) if np.isscalar(dyn)
                else [float(d) for d in dyn] + [3.5] * max(0, len(channels) - len(dyn)))
        out, extent = None, [-float(ex), float(ex), -float(ex), float(ex)]
        screen = None
        if tau_max > 0 and 'm_dust' in self.parts.get('gas', {}):
            screen, _ = self._render(self.project('gas', spos), self.parts['gas']['m_dust'],
                                     ex, xsize)
            screen = get_normalized_image(screen, vmin, vmax, mode='linear', zscale=False)
        for _i, name in enumerate(channels):
            comp, key = self._field(name)
            if comp not in self.parts or key not in self.parts[comp]:
                continue
            img, extent = self._render(self.project(comp, spos), self.parts[comp][key], ex, xsize)
            img = (log_stretch(img, dyns[_i], pmax) if stretch == 'dyn'
                   else get_normalized_image(img, vmin, vmax, mode=mode, zscale=False))
            if screen is not None:
                img = apply_dust_screen(img, screen, tau_max)
            rgba = cmaps.get(name, cm.Greys_r)(img)
            out = rgba if out is None else Blend.Blend(out, rgba).Screen()
        if out is None:
            out = np.zeros((xsize, xsize, 4))
            out[..., 3] = 1.0
        return out, extent

    # ── numbers to write on the picture ────────────────────────────────
    def totals(self, ex=None, spos='faceon', members=False):
        """Summed masses (and SFR) of the loaded particles.

        ``ex`` restricts to the drawn frame, ``members=True`` to the particles the catalogue
        calls galaxy members — so the catalogue numbers are still available when the whole
        region is loaded.  ``n_gas`` / ``n_star`` count what went into the sums.
        """
        out = {}
        for comp in ('gas', 'star'):
            if comp not in self.parts:
                continue
            keep = np.ones(len(next(iter(self.parts[comp].values()))), bool)
            if members and 'member' in self.parts[comp]:
                keep &= np.asarray(self.parts[comp]['member'], bool)
            if ex is not None:
                xy = self.project(comp, spos)
                keep &= (np.abs(xy[:, 0]) <= ex) & (np.abs(xy[:, 1]) <= ex)
            out['n_' + comp] = int(keep.sum())
            for name, (c_, key, _) in self.FIELDS.items():
                if c_ == comp and key in self.parts[comp]:
                    out[key] = float(np.nansum(np.asarray(self.parts[comp][key], float)[keep]))
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Helpers lifted from the notebooks (multiscale_figure, agn_feedback_view)
# ═════════════════════════════════════════════════════════════════════════════

def make_camera(center, extent, xsize, ysize, t, p):
    """An orthographic sphviewer camera centred on ``center`` with a square field of half-width ``extent``."""
    return sph.Camera(x=float(center[0]), y=float(center[1]), z=float(center[2]), r="infinity", t=float(t), p=float(p),
                      roll=0, extent=[-extent, extent, -extent, extent], xsize=xsize, ysize=ysize)


def render_image(pos, mass, camera):
    """SPH surface-density image of particles through ``camera`` -> ``(image, extent)``."""
    R = sph.Render(sph.Scene(sph.Particles(pos, mass), Camera=camera))
    return R.get_image(), R.get_extent()


def blend_gas_stars(gas_img, star_img, dust_img=None, tau_max=1.5, gas_cmap=cm.magma, star_cmap=cm.Greys_r,
                    gas_vmin=50, gas_vmax=99.5, gas_mode="log", star_vmin=50, star_vmax=99.7, star_mode="log", gamma=1.0):
    """Gas and star SPH maps blended into one RGBA composite (``Screen``), optionally behind a dust screen.

    ``*_vmin`` / ``*_vmax`` are percentiles: raise ``vmin`` to clip the diffuse background to black (the
    full-box and CGM panels), push ``vmax`` toward 100 so a peaked core saturates less; ``gamma`` > 1
    darkens the mid-tones (``Screen`` only brightens).
    """
    gas_n = get_normalized_image(gas_img, vmin=gas_vmin, vmax=gas_vmax, mode=gas_mode)
    star_n = get_normalized_image(star_img, vmin=star_vmin, vmax=star_vmax, mode=star_mode)
    if dust_img is not None:
        dust_n = get_normalized_image(dust_img, vmin=5, vmax=99, mode="log")
        gas_n = apply_dust_screen(gas_n, dust_n, tau_max)
        star_n = apply_dust_screen(star_n, dust_n, tau_max)
    if gamma != 1.0:
        gas_n = np.clip(gas_n, 0, 1) ** gamma
        star_n = np.clip(star_n, 0, 1) ** gamma
    return Blend.Blend(star_cmap(star_n), gas_cmap(gas_n)).Screen()


def project_mass_weighted(particles, quantity, extent, spos="edgeon", t=None, p=None):
    """Mass-weighted line-of-sight projection of a per-particle scalar (or Σ with ``quantity=None``).

    ``particles``: dict with ``pos``, ``mass``, ``hsml``, ``center``, ``L`` (the spin used to orient
    the view, :func:`find_rot_ax`). Returns ``(map, surface density, extent)``.
    """
    t, p = find_rot_ax(particles["L"], t, p, spos)
    R = rotation_matrices_from_angles(t, p)
    cen = particles["center"]
    pos_rot = (particles["pos"] - cen) @ R.T + cen
    common = dict(hsml=particles["hsml"], r="infinity", x=cen[0], y=cen[1], z=cen[2], plot=False,
                  extent=[-extent, extent, -extent, extent], logscale=False)
    qd = QuickView(pos_rot, particles["mass"], **common)
    den = qd.get_image()
    ext = qd.get_extent()
    if quantity is None:
        return den, den, ext
    num = QuickView(pos_rot, quantity * particles["mass"], **common).get_image()
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0), den, ext
