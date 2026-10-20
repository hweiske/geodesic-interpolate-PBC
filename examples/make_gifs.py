"""Run example geodesic interpolations and render them as GIFs with AtomsPlotter.

Usage:  python examples/make_gifs.py [output_dir]

Requires AtomsPlotter (https://doi.org/10.5281/zenodo.23316072) and Pillow.
"""
import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import numpy as np
from AtomsPlotter import atoms_plotter
from PIL import Image
from ase.build import add_adsorbate, bulk, fcc111
from ase.constraints import FixAtoms
from ase.io import read, write

from geodesic_interpolate import interpolate


HERE = os.path.dirname(os.path.abspath(__file__))


class FixedFramePlotter(atoms_plotter):
    """atoms_plotter with a frame shared by all images, so the view does not jump."""
    def __init__(self, frame, **kwargs):
        super().__init__(**kwargs)
        self.shared_frame = frame

    def check_pbc(self):
        periodic = super().check_pbc()
        self.frame = [list(limits) for limits in self.shared_frame]
        self.frame_x, self.frame_y, self.frame_z = self.frame
        return periodic


def shared_frame(images, margin=1.5):
    """Bounding box of all images and their cells."""
    points = [image.positions for image in images]
    for image in images:
        if image.pbc.any():
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            points.append(corners @ np.array(image.cell))
    points = np.concatenate(points)
    return [[lo - margin, hi + margin] for lo, hi in zip(points.min(axis=0), points.max(axis=0))]


def orient_molecule(images):
    """Rotate all images onto the principal axes of the whole path (largest extent along x)."""
    positions = np.concatenate([image.positions for image in images])
    center = positions.mean(axis=0)
    eigenvectors = np.linalg.eigh((positions - center).T @ (positions - center))[1][:, ::-1]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1
    oriented = []
    for image in images:
        image = image.copy()
        image.positions = (image.positions - center) @ eigenvectors
        oriented.append(image)
    return oriented


def wrap_for_display(images, center_atom=None, center=(0.5, 0.5, 0.5)):
    """Translate each atom by the same lattice vector in all images so its mean position lies in
    the cell.  Unlike wrapping every image separately, atoms never jump across the cell.  If
    `center_atom` is given, everything is first shifted so the midpoint of its path is at the
    fractional position `center`."""
    cell = np.array(images[0].cell)
    pbc = images[0].pbc
    path = np.array([image.positions for image in images])
    if center_atom is not None:
        mid = (path[0, center_atom] + path[-1, center_atom]) / 2
        path += np.asarray(center) @ cell - mid
    frac = path.mean(axis=0) @ np.linalg.inv(cell)
    offsets = np.where(pbc, np.floor(frac), 0)
    path -= offsets @ cell
    wrapped = []
    for image, positions in zip(images, path):
        image = image.copy()
        image.positions = positions
        wrapped.append(image)
    return wrapped


def render_gif(images, filename, duration=120, center_atom=None, center=(0.5, 0.5, 0.5), slab=None,
               **options):
    """Render every image with AtomsPlotter and assemble a looping (forward and back) GIF.

    `slab=(lo, hi)` only shows atoms whose mean fractional c coordinate lies in that range.
    """
    if images[0].pbc.any():
        images = wrap_for_display(images, center_atom, center)
    else:
        images = orient_molecule(images)
    if slab is not None:
        frac_c = np.mean([image.get_scaled_positions(wrap=False)[:, 2] for image in images], axis=0)
        shown = (frac_c > slab[0]) & (frac_c < slab[1])
        images = [image[shown] for image in images]
    frame = shared_frame(images)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, image in enumerate(images):
            name = os.path.join(tmp, '%03d' % i)
            plotter = FixedFramePlotter(frame, atoms=image, show=False, format='png', name=name,
                                        auto_orient=False, **options)
            plotter.plot()
            png = Image.open(name + '.png').convert('RGBA')
            background = Image.new('RGBA', png.size, 'white')
            frames.append(Image.alpha_composite(background, png).convert('RGB'))
    frames = frames + frames[-2:0:-1]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print('Wrote', filename)


def al_vacancy_hop():
    """Vacancy migration in bulk Al; the hopping atom crosses the cell boundary."""
    a = 4.05
    initial = bulk('Al', 'fcc', a=a, cubic=True).repeat(2)
    del initial[0]
    final = initial.copy()
    hopping = np.argmin(np.linalg.norm(initial.positions - [2 * a - a / 2, 2 * a - a / 2, 0], axis=1))
    final.positions[hopping] = [0, 0, 0]
    # Put the hopping atom first so the GIF can be centered on it
    order = [hopping] + [i for i in range(len(initial)) if i != hopping]
    return interpolate([initial[order], final[order]], nimages=9)


def h_hop_cu111():
    """H hopping from an fcc to an hcp hollow on Cu(111) across the cell boundary, bottom two
    layers fixed."""
    initial = fcc111('Cu', (3, 3, 3), vacuum=6.0)
    add_adsorbate(initial, 'H', 1.0, 'fcc')
    initial.set_constraint(FixAtoms(indices=[atom.index for atom in initial if atom.tag > 1]))
    final = initial.copy()
    scaled = final.get_scaled_positions()
    scaled[-1, :2] = [-1 / 9, 2 / 9]
    final.set_scaled_positions(scaled)
    return interpolate([initial, final], nimages=9)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'gifs')
    os.makedirs(outdir, exist_ok=True)
    test_cases = os.path.join(HERE, '..', 'test_cases')
    np.random.seed(0)

    paths = {
        # Hop in the top layer, viewed along [001] showing only the two top layers
        'al_vacancy_hop': (al_vacancy_hop(), dict(center_atom=0, center=(0.5, 0.5, 0.875), slab=(0.5, 1))),
        'h_hop_cu111': (h_hop_cu111(), dict()),
        'H+CH4_CH3+H2': (interpolate(read(os.path.join(test_cases, 'H+CH4_CH3+H2.xyz'), ':'), nimages=17),
                         dict(dimension='3D', elev=30, azim=-60, use_bondorders=False)),
        'DielsAlder': (interpolate(read(os.path.join(test_cases, 'DielsAlder.xyz'), ':'), nimages=17),
                       dict(dimension='3D', elev=20, azim=-60)),
    }
    for name, (images, options) in paths.items():
        write(os.path.join(outdir, name + '.traj'), images)
        render_gif(images, os.path.join(outdir, name + '.gif'), **options)


if __name__ == '__main__':
    main()
