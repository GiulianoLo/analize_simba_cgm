"""ReducedRender's angular-momentum frame: faceon must look down the GAS disc's spin, not the stellar minor axis."""
import os, sys
import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_reduced_render():
    """simbanator's package import pulls in yt / fsps / caesar / sphviewer, none of which this test touches: on a machine
    without them (the laptop) a bare stub module stands in, so the frame logic is still verified; on the cluster nothing
    is stubbed."""
    import importlib, re, types
    for _ in range(20):
        try:
            return importlib.import_module('simbanator.visualization.rendering').ReducedRender
        except ModuleNotFoundError as e:
            if e.name in sys.modules or e.name.startswith('simbanator'):
                raise
            sys.modules[e.name] = types.ModuleType(e.name)
        except ImportError as e:                                   # a stub (or a partial install) lacking a name
            m = re.search(r"cannot import name '(\w+)' from '([\w.]+)'", str(e))
            if not m or m.group(2).startswith('simbanator'):
                raise
            mod = sys.modules.get(m.group(2)) or types.ModuleType(m.group(2))
            sys.modules[m.group(2)] = mod
            setattr(mod, m.group(1), types.ModuleType(m.group(1)))
    raise ImportError('could not import simbanator.visualization.rendering')


ReducedRender = _import_reduced_render()

RNG = np.random.default_rng(3)


def _write(path, n_axis, with_vel=True, n_disc=400):
    """A non-rotating stellar spheroid whose principal frame is the identity, plus a thin gas disc spinning about
    n_axis (members), plus a non-member gas cloud far out with random velocities (the CGM that must not matter)."""
    n_axis = np.asarray(n_axis, float); n_axis /= np.linalg.norm(n_axis)
    a = np.cross(n_axis, [0.0, 0.0, 1.0]); a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-3 else np.array([1.0, 0.0, 0.0])
    b = np.cross(n_axis, a)
    r = RNG.uniform(1.0, 6.0, n_disc); phi = RNG.uniform(0, 2 * np.pi, n_disc)
    pos = np.outer(r * np.cos(phi), a) + np.outer(r * np.sin(phi), b) + RNG.normal(0, 0.05, (n_disc, 3))
    vel = 200.0 * np.cross(n_axis, pos / r[:, None]) + RNG.normal(0, 5, (n_disc, 3))          # circular, 200 km/s
    cgm = RNG.normal(0, 30.0, (300, 3)); cvel = RNG.normal(0, 150.0, (300, 3))
    gpos = np.vstack([pos, cgm]); gvel = np.vstack([vel, cvel])
    member = np.concatenate([np.ones(n_disc, bool), np.zeros(300, bool)])
    spos = RNG.normal(0, 1.0, (500, 3)) * np.array([1.0, 0.7, 0.5])                                   # a spheroid, axes = x, y, z
    with h5py.File(path, 'w') as f:
        f.attrs['evecs'] = np.eye(3); f.attrs['redshift'] = 1.0; f.attrs['snap'] = 1; f.attrs['gx'] = 1; f.attrs['rmax_kpc'] = 100.0
        g = f.create_group('gas')
        g.create_dataset('pos', data=gpos); g.create_dataset('m_gas', data=np.full(len(gpos), 1e6)); g.create_dataset('m_H2', data=np.where(member, 5e5, 0.0))
        g.create_dataset('m_dust', data=np.full(len(gpos), 1e3)); g.create_dataset('member', data=member)
        if with_vel:
            g.create_dataset('vel', data=gvel)
        s = f.create_group('star')
        s.create_dataset('pos', data=spos); s.create_dataset('m_star', data=np.full(500, 2e6)); s.create_dataset('member', data=np.ones(500, bool))
        if with_vel:
            s.create_dataset('vel', data=np.zeros((500, 3)))
    return n_axis


def test_h2_frame_points_down_the_gas_spin(tmp_path):
    path = str(tmp_path / 'disc.h5'); n = _write(path, [1.0, 1.0, 1.0])
    r = ReducedRender(path, members_only=False, frame='H2_L')
    assert r.frame == 'H2_L' and r.frame_info['reason'] == 'ok' and r.frame_info['n_used'] == 400
    assert r.frame_info['coherence'] > 0.95                          # a disc: every particle's l is aligned
    _, _, coh_cgm = r.angular_momentum('gas', 'm_gas', rmax=100.0, members=False)
    assert coh_cgm < 0.6                                              # the random cloud dilutes it
    assert abs(np.dot(r.evecs[:, 2], n)) > 0.995                        # column 2 = the spin axis
    assert np.linalg.det(r.evecs) == pytest.approx(1.0, abs=1e-6)       # right-handed
    assert np.allclose(r.evecs.T @ r.evecs, np.eye(3), atol=1e-9)
    disc = np.asarray(r.parts['gas']['member'], bool)
    face = r.project('gas', 'faceon')[disc]; edge = r.project('gas', 'edgeon')[disc]
    assert np.std(face[:, 2]) < 0.1 and np.std(face[:, 0]) > 1.0        # face-on: the disc lies in the image plane
    assert np.std(edge[:, 1]) < 0.1 and np.std(edge[:, 0]) > 1.0        # edge-on: thin along the image y
    assert r.frame_info['tilt_vs_stellar_deg'] == pytest.approx(np.degrees(np.arccos(1 / np.sqrt(3))), abs=1.0)
    # the stored stellar frame is untouched and still the default
    r0 = ReducedRender(path, members_only=False)
    assert r0.frame == 'stellar' and np.allclose(r0.evecs, np.eye(3))
    assert np.std(r0.project('gas', 'faceon')[disc][:, 2]) > 1.0        # in the stellar frame the disc is NOT face-on


def test_frame_falls_back_without_velocities(tmp_path):
    path = str(tmp_path / 'novel.h5'); _write(path, [0.0, 0.0, 1.0], with_vel=False)
    r = ReducedRender(path, frame='H2_L')
    assert r.frame == 'stellar' and 'no velocities' in r.frame_info['reason'] and np.allclose(r.evecs, np.eye(3))
    path2 = str(tmp_path / 'thin.h5'); _write(path2, [0.0, 0.0, 1.0], n_disc=10)
    r2 = ReducedRender(path2, frame='H2_L')
    assert r2.frame == 'stellar' and 'only 10' in r2.frame_info['reason']
    with pytest.raises(ValueError):
        ReducedRender(path, frame='nonsense')


def test_set_evecs_imposes_a_frame(tmp_path):
    path = str(tmp_path / 'disc2.h5'); n = _write(path, [0.0, 1.0, 1.0])
    r = ReducedRender(path, members_only=False, frame='H2_L'); e = r.evecs.copy()
    r2 = ReducedRender(path, members_only=False)                       # stellar frame ...
    r2.set_evecs(e, label='H2_L@sft', info={'n_used': 400})           # ... overridden by the other epoch's spin
    assert r2.frame == 'H2_L@sft' and r2.frame_info['reason'] == 'imposed' and np.allclose(r2.evecs, e)
    with pytest.raises(ValueError):
        r2.set_evecs(np.ones((3, 3)))
