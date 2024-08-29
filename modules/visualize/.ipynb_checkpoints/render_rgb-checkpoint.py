
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import yt, caesar
import sphviewer as sph
from sphviewer.tools import camera_tools
from sphviewer.tools import QuickView
from sphviewer.tools import Blend

import modules as anal
import modules.anal_func as anal_func

from PIL import Image
from modules.io_paths.savepaths import SavePaths


def find_rot_ax(L, t=None, p=None, spos='faceon'):
    """Find the rotation angles for the given angular momentum vector."""
    x_vec = np.array([1, 0, 0])
    y_vec = np.array([0, 1, 0])
    cos_theta = np.dot(L, x_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(L))
    cos_phi = np.dot(L, y_vec) / (np.linalg.norm(y_vec) * np.linalg.norm(L))
    
    if t is None and spos=='faceon':
        t = np.rad2deg(np.arccos(cos_theta))+90
    if p is None and spos=='faceon':
        p = np.rad2deg(np.arccos(cos_phi))+90

    if t is None and spos=='edgeon':
        t = np.rad2deg(np.arccos(cos_theta))
    if p is None and spos=='edgeon':
        p = np.rad2deg(np.arccos(cos_phi))
    
    return t, p

def rotation_matrices_from_angles(theta, phi):
    """Generate rotation matrices for angles theta (rotation around Z) and phi (rotation around Y)."""
    theta = np.radians(theta)
    phi = np.radians(phi)
    
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    R_y = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    
    return R_z @ R_y


def get_normalized_image(image, vmin=1, vmax=99, mode='linear', zscale=False):
    """Normalize the image array using percentiles and optionally apply a logarithmic scale."""
    vmin = np.percentile(image, vmin)
    vmax = np.percentile(image, vmax)
    image = np.clip(image, vmin, vmax)
    if mode=='log':
        mk = image > 0
        image[mk] = np.log10(image[mk])
    if mode=='sqrt':
        image = np.sqrt(image)
    if zscale:
        mean = np.mean(image)
        std = np.std(image)
        image = (image-mean)/std
    normalized_image = (image - image.min()) / (image.max() - image.min())
    return normalized_image




def apply_dust_screen(image, dust):
    dust = (dust-dust.min())/(dust.max()-dust.min())
    inverted_dust = 1-dust
    # Apply the dust screen to the image
    screened = image * inverted_dust
    
    return screened


def gamma_correction(image, gamma=2.2):
    image = (image)**gamma
    return image


def decide_load(snapfile, catfile, id, propr, region=False, ifdust=False, dim=['Msun', 'Msun', 'Msun']):
    """Load snapshot data and return particle information."""
    ds = yt.load(snapfile)
    obj = caesar.load(catfile)
    ad = ds.all_data()
    gal = obj.galaxies[id]

    def get_data(particle_type, prop, dim, indices=None):
        pos = ad[particle_type, 'Coordinates']
        mass = ad[particle_type, prop[:-2] if '_s' in prop else prop]
        if indices is not None:
            pos = pos[indices]
            mass = mass[indices]
        
        pos = pos.in_units('kpc').value
        mass = ds.arr(mass, 'code_mass').in_units(dim).value if '_s' in prop else mass.in_units(dim).value
        return pos, mass

    if region:
        print('Doing region...')
        gas_pos, gas_mass = get_data('PartType0', propr[0], dim=dim[0])
        star_pos, star_mass = get_data('PartType4', propr[1], dim=dim[1])
        dust_pos, dust_mass = get_data('PartType0', propr[2], dim=dim[2]) if ifdust else (None, None)
    else:
        gas_pos, gas_mass = get_data('PartType0', propr[0], dim=dim[0], indices=gal.glist)
        star_pos, star_mass = get_data('PartType4', propr[1], dim=dim[1], indices=gal.slist)
        dust_pos, dust_mass = get_data('PartType0', propr[2], dim=dim[2], indices=gal.glist) if ifdust else (None, None)

    return ds, obj, gal, gas_pos, gas_mass, star_pos, star_mass, dust_pos, dust_mass


class RenderRGB:
    """Class for rendering and blending particle data images."""
    
    def __init__(self, snapfile, catfile, id, propr, region=False, ifdust=True, dim=['Msun', 'Msun', 'Msun']):
        self.ds, self.obj, self.gal, self.gas_pos, self.gas_mass, self.star_pos, self.star_mass, *dust_data = decide_load(snapfile, catfile, id, propr, region, ifdust, dim)
        self.a = self.obj.simulation.scale_factor
        self.ifdust = ifdust
        self.dust_pos, self.dust_mass = dust_data if ifdust else (None, None)
        self.phys_ext = None
        self.region = region

    def set_camera(self, center=None, extent=5, t=None, p=None, r='infinity', roll=0, xsize=400, ysize=400, zoom=None, spos='faceon'):
        """Configure the camera for rendering."""
        if center==None:
            center = self.gal.minpotpos.in_units('kpc').value
        L = self.gal.rotation['gas_L']
        tn, pn = find_rot_ax(L, t, p, spos)
        print(f"Camera settings: Center={center}, Extent={extent}, Theta={tn}, Phi={pn}, Roll={roll}, Radius={r}, XSize={xsize}, YSize={ysize}")
        return sph.Camera(x=center[0], y=center[1], z=center[2], r=r, t=tn, p=pn, roll=roll,
                          extent=[-extent, extent, -extent, extent], xsize=xsize, ysize=ysize, zoom=zoom)

    def set_particles(self):
        """Create particle objects for SPH rendering."""
        P_gas = sph.Particles(self.gas_pos, self.gas_mass)
        P_star = sph.Particles(self.star_pos, self.star_mass)
        if self.ifdust:
            P_dust = sph.Particles(self.dust_pos, self.dust_mass)
            return P_gas, P_star, P_dust
        return P_gas, P_star

    def set_rgb(self, particle, camera, update):
        """Render and normalize the RGB image for given particles."""
        if update!=None:
            S = sph.Scene(particle)
            S.update_camera(**update)
        else:
            S = sph.Scene(particle, Camera=camera)
        R = sph.Render(S)
        img = R.get_image()
        self.phys_ext = R.get_extent()
        return img


    def generate_images(self, camera, vmin=None, vmax=None):
        """Generate and blend images of particles."""
        particles = self.set_particles()
        color_maps = [cm.afmhot, cm.Greys_r]
        rgbs = [self.set_rgb(part, camera=camera, update=None) for part in particles[:2]]
        if self.ifdust:
            dust_screen = self.set_rgb(particles[2], camera=camera, update=None)
            dust_screen = get_normalized_image(dust_screen, vmin=vmin, vmax=vmax, mode='linear', zscale=True)
            gas = get_normalized_image(rgbs[0], vmin, vmax, mode='sqrt', zscale=True)
            stars = get_normalized_image(rgbs[1], vmin, vmax, mode='sqrt', zscale=True)
            gas = apply_dust_screen(gas, dust_screen)
            stars = apply_dust_screen(stars, dust_screen)
        else:
            gas = get_normalized_image(rgbs[0], vmin, vmax, mode='sqrt', zscale=True)
            stars = get_normalized_image(rgbs[1], vmin, vmax, mode='sqrt', zscale=True)
            
        blend = Blend.Blend(color_maps[1](stars), color_maps[0](gas))
        return blend.Screen()

    def plot(self, image, xl, yl, name, correct=False):
        """Plot and save the rendered image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        if isinstance(correct, float):
            image = gamma_correction(image, correct)
        ax.imshow(image, extent=self.phys_ext)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        path = SavePaths()
        base_dir = path.create_subdir(path.get_filetype_path('plot'), 'renders')
        if self.region:
            fig.savefig(os.path.join(base_dir, f'region_{name}.png'))
        else:
            fig.savefig(os.path.join(base_dir, f'{name}.png'))

    def set_video(self, num_frames, p=None, t=None, r='infinity',
                  extent=5, del_p=360, del_t=0, xsize=500, ysize=500, vmin=None, vmax=None, spos='faceon', zoom=1.):
        """Render frames and create a video."""
        targets = [self.gal.minpotpos.in_units('kpc').value]
        L = self.gal.rotation['gas_L']
        tn, pn = find_rot_ax(L, t, p, spos)
        anchors = {
            'sim_times': np.linspace(0, 1, num_frames),
            'id_frames': np.arange(num_frames),
            'r': [r] * num_frames,
            'id_targets': [0] * num_frames,
            't': np.linspace(t, t+del_t, num_frames),
            'p': np.linspace(p, p+del_p, num_frames),
            'zoom': [zoom] * num_frames,
            'extent': [extent] * num_frames
        }
        print(f'Setting camera with {num_frames} frames, p={pn}, t={tn}, r={r}, extent={extent}')
        data = camera_tools.get_camera_trajectory(targets, anchors)
        path = SavePaths()
        frame_dir = path.create_subdir(path.create_subdir(path.get_filetype_path('plot'), 'videos'), 'frames')
        particles = self.set_particles()
        color_maps = [cm.afmhot, cm.Greys_r]
        for h, i in enumerate(data):
            i.update({'xsize': xsize, 'ysize': ysize, 'roll': 0})
            rgbs = [self.set_rgb(part, camera=None, update=i) for part in particles[:2]]
            if self.ifdust:
                dust_screen = self.set_rgb(particles[2], camera=None, update=i)
                dust_screen = get_normalized_image(dust_screen, vmin=vmin, vmax=vmax, mode='linear', zscale=True)
                gas = get_normalized_image(rgbs[0], vmin, vmax, mode='sqrt', zscale=True)
                stars = get_normalized_image(rgbs[1], vmin, vmax, mode='sqrt', zscale=False)
                gas = apply_dust_screen(gas, dust_screen)
                stars = apply_dust_screen(stars, dust_screen)
            else:
                gas = get_normalized_image(rgbs[0], vmin, vmax, mode='sqrt', zscale=True)
                stars = get_normalized_image(rgbs[1], vmin, vmax, mode='sqrt', zscale=False)
                
            blend = Blend.Blend(color_maps[1](stars), color_maps[0](gas))
            plt.imsave(f'{frame_dir}/image_{h:04d}.png', blend.Screen())

    def create_video(self, name, interval=100):
        """Create a GIF from the rendered frames."""
        path = SavePaths()
        frame_dir = path.create_subdir(path.create_subdir(path.get_filetype_path('plot'), 'videos'), 'frames')
        if self.region:
            save_path = os.path.join(path.create_subdir(path.get_filetype_path('plot'), 'videos'), f'region_{name}.gif')
        else:
            save_path = os.path.join(path.create_subdir(path.get_filetype_path('plot'), 'videos'), f'{name}.gif')
            
        image_files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
        images = [Image.open(os.path.join(frame_dir, file)) for file in image_files]
        images[0].save(
            save_path,
            save_all=True,
            append_images=images[1:],
            duration=interval,
            loop=0
        )
        print(f"GIF saved as {save_path}")

    def flush(self):
        path = SavePaths()
        frame_dir = path.create_subdir(path.create_subdir(path.get_filetype_path('plot'), 'videos'), 'frames')
        for filename in os.listdir(frame_dir):
            if filename.lower().endswith('.png'):
                file_path = os.path.join(frame_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)





class SingleRender:
    def __init__(self, snapfile, catfile, id, propr, region=False, dim='Msun'):
        """
        Initializes the SingleRender object.
        
        Parameters:
        - snapfile: Path to the snapshot file.
        - catfile: Path to the catalog file.
        - id: Galaxy ID.
        - propr: Property to retrieve (tuple of particle type and property).
        - region: If True, use region data; otherwise, use galaxy data.
        - dim: Dimension for mass (e.g., 'Msun').
        """
        self.ds = yt.load(snapfile)
        self.obj = caesar.load(catfile)
        self.ad = self.ds.all_data()
        self.gal = self.obj.galaxies[id]
        self.propr = propr
        self.region = region
        self.dim = dim
        
        self._initialize_data()

    def _initialize_data(self):
        def get_data(particle_type, prop, dim, indices=None):
            pos = self.ad[particle_type, 'Coordinates']
            mass = self.ad[particle_type, prop[:-2] if '_s' in prop else prop]
            if indices is not None:
                pos = pos[indices]
                mass = mass[indices]

            pos = pos.in_units('kpc').value
            mass = self.ds.arr(mass, 'code_mass').in_units(dim).value if '_s' in prop else mass.in_units(dim).value
            return pos, mass

        if self.region:
                self.pos, self.mass = get_data(self.propr[0], self.propr[1], dim=self.dim)
        else:
            if self.propr[0] == 'PartType0':
                self.pos, self.mass = get_data(self.propr[0], self.propr[1], dim=self.dim, indices=self.gal.glist)
            else:
                self.pos, self.mass = get_data(self.propr[0], self.propr[1], dim=self.dim, indices=self.gal.slist)
        
        self.a = self.obj.simulation.scale_factor
        self.pos = self.pos*self.a
        self.mass = self.mass*self.a
        self.phys_ext = None

    def single_map(self, center=None, ex=5, t=None, p=None, r='infinity', roll=0,
                   xsize=400, ysize=400, zoom=None, spos='faceon', cmap='viridis', vmin=1, vmax=99, mode='log', zscale=False):
        """
        Generates a single map of the simulation data.
        
        Parameters:
        - center: Coordinates of the center for the map.
        - extent: Extent of the map in kpc.
        - t, p: Angles for camera orientation.
        - r: Radius for the camera.
        - roll: Camera roll angle.
        - xsize, ysize: Size of the output image.
        - zoom: Zoom factor for the camera.
        - spos: Orientation of the plot ('faceon' or 'edgeon').
        - cmap: Colormap for the image.
        - vmin, vmax: Color scaling for the image.
        """
        if center is None:
            center = self.gal.minpotpos.in_units('kpc').value*self.a
        
        L = self.gal.rotation['gas_L']
        t, p = find_rot_ax(L, t, p, spos)
        P = sph.Particles(self.pos, self.mass)
        C = sph.Camera(x=center[0], y=center[1], z=center[2],
                       r=r, t=t, p=p, roll=roll,
                       extent=[-ex, ex, -ex, ex],
                       xsize=xsize, ysize=ysize, zoom=zoom)
        S = sph.Scene(P, Camera=C)
        R = sph.Render(S)
        self.phys_ext = R.get_extent()
        img = R.get_image()

        return get_normalized_image(img, vmin, vmax, mode, zscale)

    def stream_plot(self, center=None, ex=5, r='infinity', t=None, p=None, xl='x', yl='y', spos='faceon'):
        """
        Generates a stream plot of the velocity field over the density field.
        
        Parameters:
        - center: Coordinates of the center for the plot.
        - extent: Extent of the plot in kpc.
        - r: Radius for the plot.
        - t, p: Angles for rotation.
        - xl, yl: Labels for the x and y axes.
        - spos: Orientation of the plot ('faceon' or 'edgeon').
        """
        L = self.gal.rotation['gas_L']
        t, p = find_rot_ax(L, t, p, spos=spos)
        R = rotation_matrices_from_angles(t, p)
        
        if self.region:
            if self.propr[0] == 'PartType0':
                vel = self.ad['PartType0', 'Velocities'].in_units('m/s').value * self.a
                hsml = self.ad['PartType0', 'SmoothingLength'].in_units('kpc').value * self.a
            else:
                raise ValueError('Particles must be gas')
        else:
            if self.propr[0] == 'PartType0':
                vel = self.ad['PartType0', 'Velocities'][self.gal.glist].in_units('m/s').value * self.a
                hsml = self.ad['PartType0', 'SmoothingLength'][self.gal.glist].in_units('kpc').value * self.a
            else:
                raise ValueError('Particles must be gas')
        
        if center is None:
            center = self.gal.minpotpos.in_units('kpc').value * self.a
    
        pos = np.dot(self.pos - center, R.T)
        vel = np.dot(vel, R.T)
        pos += center
    
        qp = QuickView(pos, hsml=hsml, r='infinity', x=center[0], y=center[1], z=center[2],
                       plot=False, extent=[-ex, ex, -ex, ex], logscale=False)
        density_field = qp.get_image()
        extent = qp.get_extent()
        
        epsilon = 1e-10  # Small value to prevent division by zero
        vfield = []
        for i in range(2):
            qv = QuickView(pos, vel[:, i], hsml=hsml, r='infinity', x=center[0], y=center[1], z=center[2],
                           plot=False, extent=[-ex, ex, -ex, ex], logscale=False)
            frac = qv.get_image() / (density_field + epsilon)
            frac[~np.isfinite(frac)] = 0.  # Set non-finite values to zero
            vfield.append(frac)
        
        fig = plt.figure(1, figsize=(12, 12))
        ax = fig.add_subplot(111)
        X = np.linspace(extent[0], extent[1], 500)
        Y = np.linspace(extent[2], extent[3], 500)
        ax.imshow(np.log1p(density_field), origin='lower', extent=extent, cmap='bone')
        
        # Calculate stream plot colors based on velocity magnitude
        velocity_magnitude = np.sqrt(vfield[0]**2 + vfield[1]**2)
        color = np.log1p(velocity_magnitude)
        
        max_color = np.max(color)
        if max_color > 0:
            color /= max_color  # Normalize only if max_color is greater than 0
        else:
            color = np.zeros_like(color)  # Set to zeros if max_color is 0
    
        lw = 2 * color
        ax.streamplot(X, Y, vfield[0], vfield[1], color=color, density=1.5, cmap='jet', linewidth=lw, arrowsize=1)
        
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.minorticks_on()
        ax.set_xlabel(xl, size=25)
        ax.set_ylabel(yl, size=25)
        plt.show()
        return fig, ax




    def plot(self, image, xl, yl, name, vmin=None, vmax=None):
        """Plot and save the rendered image."""
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(image, extent=self.phys_ext, cmap='viridis')  
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mass [Msun]')
        
        path = SavePaths()
        base_dir = path.create_subdir(path.get_filetype_path('plot'), 'renders')
        if self.region:
            fig.savefig(os.path.join(base_dir, f'map_region_{self.propr[1]}_{name}.png'))
        else:
            fig.savefig(os.path.join(base_dir, f'map_{self.propr[1]}_{name}.png'))
