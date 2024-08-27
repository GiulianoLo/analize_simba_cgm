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




def get_normalized_image(image, lower_percentile=2, upper_percentile=98, log_scale=True):
    """Normalize the image array using percentiles and optionally apply a logarithmic scale."""
    # Compute the vmin and vmax based on the percentiles
    vmin = np.percentile(image, lower_percentile)
    vmax = np.percentile(image, upper_percentile)
    # Clip the image to the computed vmin and vmax
    image = np.clip(image, vmin, vmax)
    # Normalize the image to the range [0, 1]
    normalized_image = (image - vmin) / (vmax - vmin)
    if log_scale:
        # Apply logarithmic scaling
        normalized_image = np.log1p(normalized_image)  # log1p is log(1 + x) to avoid log(0)
        # Normalize again to [0, 1] after log transformation
        normalized_image = normalized_image / np.max(normalized_image)
    
    return normalized_image


# def get_normalized_image(image, lower_percentile=2, upper_percentile=98, log_scale=True):
#     """Normalize the image array using percentiles and optionally apply a logarithmic scale."""
#     # Make a copy of the image to avoid modifying the original
#     image = np.array(image, copy=True)
    
#     # Apply logarithmic transformation if needed
#     if log_scale:
#         # Avoid log of zero by adding a small constant
#         image = np.where(image > 0, np.log10(image), 0)
    
#     # Compute vmin and vmax based on the percentiles
#     vmin = np.percentile(image, lower_percentile)
#     vmax = np.percentile(image, upper_percentile)
    
#     # Clip the image to the range [vmin, vmax]
#     image = np.clip(image, vmin, vmax)
    
#     # Normalize the image to the range [0, 1]
#     normalized_image = (image - vmin) / (vmax - vmin)
    
#     return normalized_image








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

    def set_rgb(self, particle, color_map, camera, update, vmin=None, vmax=None):
        """Render and normalize the RGB image for given particles."""
        if update!=None:
            S = sph.Scene(particle)
            S.update_camera(**update)
        else:
            S = sph.Scene(particle, Camera=camera)
        R = sph.Render(S)
        img = R.get_image()
        self.phys_ext = R.get_extent()
        return color_map(get_normalized_image(img, vmin, vmax))



    def generate_images(self, camera, vmin=None, vmax=None):
        """Generate and blend images of particles."""
        particles = self.set_particles()
        color_maps = [cm.magma, cm.Greys_r] + ([cm.gist_heat_r] if len(particles) == 3 else [])
        rgbs = [self.set_rgb(part, cmap, camera=camera, update=None, vmin=vmin, vmax=vmax) for part, cmap in zip(particles, color_maps)]
        blend = Blend.Blend(rgbs[1], rgbs[0])
        if self.ifdust:
            blend = blend.Overlay()
            blend = Blend.Blend(blend, rgbs[2])
        return blend.Overlay()

    def plot(self, image, xl, yl, name):
        """Plot and save the rendered image."""
        fig, ax = plt.subplots(figsize=(12, 12))
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
        color_maps = [cm.magma, cm.Greys_r, cm.inferno_r]
        for h, i in enumerate(data):
            i.update({'xsize': xsize, 'ysize': ysize, 'roll': 0})
            rgbs = [self.set_rgb(particle, color_map, camera=None, update=i, vmin=vmin, vmax=vmax) for particle, color_map in zip(particles, color_maps)]
            blend = Blend.Blend(rgbs[1], rgbs[0])
            if len(rgbs) > 2:
                blend = Blend.Blend(rgbs[1], rgbs[0]).Overlay()
                blend = Blend.Blend(blend, rgbs[2])
                
            output = blend.Overlay()
            plt.imsave(f'{frame_dir}/image_{h:04d}.png', output)

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




class SingleRender():
    def __init__(self, snapfile, catfile, id, propr, region=False, dim='Msun'):
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
            pos, mass = get_data(propr[0], propr[1], dim=dim)
        else:
            pos, mass = get_data(propr[0], propr[1], dim=dim, indices=gal.glist)
    
        self.a = obj.simulation.scale_factor
        self.phys_ext = None
        self.region = region
        self.pos = pos
        self.mass = mass
        self.gal = gal
        self.propr = propr

    def single_map(self, center=None, extent=5, t=None, p=None, r='infinity',
                   roll=0, xsize=400, ysize=400, zoom=None, spos='faceon', cmap='viridis', vmin=1, vmax=99):
        if center==None:
            center = self.gal.minpotpos.in_units('kpc').value
        L = self.gal.rotation['gas_L']
        t, p = find_rot_ax(L, t, p, spos)
        P = sph.Particles(self.pos, self.mass)
        C = sph.Camera(x=center[0], y=center[1], z=center[2],
                       r=r,t=t, p=p, roll=roll,
                       extent=[-extent,extent,-extent,extent],
                       xsize=xsize, ysize=ysize, zoom=zoom)
        S = sph.Scene(P, Camera=C)
        R = sph.Render(S)
        self.phys_ext = R.get_extent()
        img = R.get_image()
        
        return get_normalized_image(img, vmin, vmax)

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

        


        