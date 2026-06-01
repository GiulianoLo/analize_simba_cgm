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

