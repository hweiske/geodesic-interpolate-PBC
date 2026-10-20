"""Performing geodesic interpolation or smoothing.
Optimize reaction path using geometric information by minimizing path length with metrics defined by
redundant internal coordinates.   Avoids the discontinuity and convergence problems of conventional
interpolation methods by incorporating internal coordinate structure while operating in Cartesian,
avoiding unfeasibility.

Xiaolei Zhu et al, Martinez Group, Stanford University
"""
import logging
import argparse

import numpy as np
from ase.io import read, write

from .ase_interface import atoms_to_path, get_path_fixed_mask, path_to_atoms, write_vasp_neb
from .interpolation import redistribute, smooth_path


logger = logging.getLogger(__name__)


def parse_indices(specs):
    """Parse atom indices given as e.g. `0 3 5-8` (ranges are inclusive)."""
    indices = []
    for spec in specs:
        for part in spec.split(','):
            if '-' in part:
                first, last = part.split('-')
                indices.extend(range(int(first), int(last) + 1))
            elif part:
                indices.append(int(part))
    return indices


def main():
    """Main entry point of the geodesic interpolation package.
    Parse command line arguments then activate the interpolators and smoothers."""
    ps = argparse.ArgumentParser(description="Interpolates between two geometries",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ps.add_argument("filename", type=str, nargs='+', help="File(s) containing geometries in any format ASE "
                    "can read (XYZ, extended XYZ, traj, POSCAR, CIF, ...).  All images of all files are "
                    "used in order.  If the number of images is smaller than the desired number, "
                    "interpolation points will be added.  If the number is greater, subsampling will be "
                    "performed.  The cell and periodic boundary conditions of the first image are used.  "
                    "Coordinates fixed by constraints (e.g. selective dynamics) in all images stay fixed.")
    ps.add_argument("--nimages", type=int, default=17, help="Number of images.")
    ps.add_argument("--sweep", action="store_true", help="Sweep across the path optimizing one image at "
                    "a time, instead of moving all images at the same time.  Default is to perform sweeping "
                    "updates if there are more than 30 atoms.")
    ps.add_argument("--no-sweep", dest='sweep', action="store_false", help="Do not perform sweeping.")
    ps.set_defaults(sweep=None)
    ps.add_argument("--output", default="interpolated.traj", type=str, help="Output filename.  The format "
                    "is determined by ASE from the extension.")
    ps.add_argument("--vasp", nargs='?', const='.', default=None, metavar='DIR', help="Also write VASP NEB "
                    "input, DIR/00/POSCAR ... DIR/NN/POSCAR, with one folder per image.  DIR defaults to "
                    "the current directory.")
    ps.add_argument("--fix", nargs='+', default=None, metavar='INDEX', help="Indices of atoms to keep fixed "
                    "in addition to the constraints in the input, e.g. `--fix 0-15 20` (ranges inclusive).")
    ps.add_argument("--tol", default=2e-3, type=float, help="Convergence tolerance")
    ps.add_argument("--maxiter", default=15, type=int, help="Maximum number of minimization iterations")
    ps.add_argument("--microiter", default=20, type=int, help="Maximum number of micro iterations for "
                    "sweeping algorithm.")
    ps.add_argument("--scaling", default=1.7, type=float, help="Exponential parameter for morse potential")
    ps.add_argument("--friction", default=1e-2, type=float, help="Size of friction term used to prevent "
                    "very large change of geometry.")
    ps.add_argument("--dist-cutoff", dest='dist_cutoff', default=3, type=float, help="Cut-off value for the "
                    "distance between a pair of atoms to be included in the coordinate system.")
    ps.add_argument("--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help="Logging level to adopt [ DEBUG | INFO | WARNING | ERROR ]")
    ps.add_argument("--save-raw", dest='save_raw', default=None, type=str, help="When specified, save the "
                    "raw path after bisections be before smoothing.")
    ps.add_argument("--no-pbc", dest='use_pbc', action="store_false", help="Ignore the cell of periodic "
                    "input files and treat the system as a molecule.")
    ps.add_argument("--max-rounds", dest='max_rounds', default=5, type=int, help="Maximum number of smoothing "
                    "rounds.  After each round, atom pairs that came into contact without being part of the "
                    "coordinate system are added and the path is smoothed again.  1 smooths only once.")
    ps.add_argument("--seed", default=None, type=int, help="Random seed for reproducible results (the bisection "
                    "uses random nudges).  Results also depend on the number of BLAS threads, e.g. "
                    "OMP_NUM_THREADS.")
    ps.add_argument("--wrap", action="store_true", help="Wrap atoms back into the cell in the output.  By "
                    "default periodic paths are written unwrapped so that atoms move continuously.")
    args = ps.parse_args()

    # Setup logging based on designated logging level
    logging.basicConfig(format="[%(module)-12s]%(message)s", level=args.logging)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Read the initial geometries.
    images = [atoms for filename in args.filename for atoms in read(filename, index=':')]
    logger.info('Loaded %d geometries from %s', len(images), ' '.join(args.filename))
    if len(images) < 2:
        raise ValueError("Need at least two initial geometries.")
    symbols, X, cell, pbc = atoms_to_path(images, args.use_pbc)
    if cell is not None:
        logger.info('Using periodic boundary conditions along %s', ''.join(np.array(list('abc'))[pbc]))
    fixed = get_path_fixed_mask(images)
    if args.fix:
        fixed[parse_indices(args.fix)] = True
    if fixed.any():
        logger.info('Keeping %d Cartesian coordinates of %d atoms fixed', fixed.sum(), fixed.any(axis=1).sum())

    def save(filename, path):
        write(filename, path_to_atoms(images, path, fixed, args.wrap))

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(symbols, X, args.nimages, tol=args.tol * 5, cell=cell, pbc=pbc, fixed=fixed)
    if args.save_raw is not None:
        save(args.save_raw, raw)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    current = []
    try:
        smoother = smooth_path(symbols, raw, args.scaling, threshold=args.dist_cutoff, friction=args.friction,
                               sweep=args.sweep, tol=args.tol, max_iter=args.maxiter, micro_iter=args.microiter,
                               max_rounds=args.max_rounds, cell=cell, pbc=pbc, fixed=fixed,
                               callback=current.append)
    finally:
        # Save the smoothed path to output file.  try block is to ensure output is saved if one ^C the
        # process, or there is an error
        logging.info('Saving final path to file %s', args.output)
        save(args.output, current[-1].path if current else raw)
    if args.vasp is not None:
        write_vasp_neb(path_to_atoms(images, smoother.path, fixed, args.wrap), args.vasp)


if __name__ == "__main__":
    main()
