"""Simplified geodesic interpolations module, which uses geodesic lengths as criteria
to add bisection points until point count meet desired number.
Will need another following geodesic smoothing to get final path.
"""
import logging

import numpy as np
from scipy.optimize import least_squares, minimize

from .geodesic import Geodesic
from .coord_utils import get_bond_list, compute_wij, morse_scaler, align_geom, align_path, fixed_mask, get_radii


logger = logging.getLogger(__name__)


def mid_point(atoms, geom1, geom2, tol=1e-2, nudge=0.01, threshold=4, cell=None, pbc=None, fixed=None):
    """Find the Cartesian geometry that has internal coordinate values closest to the average of
    two geometries.

    Simply perform a least-squares minimization on the difference between the current internal
    and the average of the two end points.  This is done twice, using either end point as the
    starting guess.  DON'T USE THE CARTESIAN AVERAGE AS GUESS, THINGS WILL BLOW UP.

    This is used to generate an initial guess path for the later smoothing routine.
    Genenrally, the added point may not be continuous with the both end points, but
    provides a good enough starting guess.

    Random nudges are added to the initial geometry, so running multiple times may not yield
    the same converged geometry. For larger systems, one will never get the same geometry
    twice.  So one may want to perform multiple runs and check which yields the best result.

    Args:
        geom1, geom2:   Cartesian geometry of the end points
        tol:    Convergence tolarnce for the least-squares minimization process
        nudge:  Random nudges added to the initial geometry, which helps to discover different
                solutions.  Also helps in cases where optimal paths break the symmetry.
        threshold:  Threshold for including an atom-pair in the coordinate system
        cell, pbc:  Periodic cell and flags (ASE convention).  The end points must be unwrapped
                    with respect to each other (see `align_path`).
        fixed:      Fixed atoms or Cartesian components, see `Geodesic`.  Fixed coordinates are
                    set to the average of the end points.

    Returns:
        Optimized mid-point which bisects the two endpoints in internal coordinates
    """
    # Process the initial geometries, construct coordinate system and obtain average internals
    geom1, geom2 = np.array(geom1), np.array(geom2)
    fixed = fixed_mask(fixed, len(geom1))
    free = np.ones(geom1.size, dtype=bool) if fixed is None else ~fixed.ravel()
    add_pair = set()
    geom_list = [geom1, geom2]
    # This loop is for ensuring a sufficient large coordinate system.  The interpolated point may
    # have atom pairs in contact that are far away at both end-points, which may cause collision.
    # One can include all atom pairs, but this may blow up for large molecules.  Here the compromise
    # is to use a screened list of atom pairs first, then add more if additional atoms come into
    # contant, then rerun the minimization until the coordinate system is consistant with the
    # interpolated geometry
    while True:
        rijlist, re = get_bond_list(geom_list, threshold=threshold + 1, enforce=add_pair, cell=cell, pbc=pbc)
        scaler = morse_scaler(alpha=0.7, re=re)
        w1, _ = compute_wij(geom1, rijlist, scaler, cell)
        w2, _ = compute_wij(geom2, rijlist, scaler, cell)
        w = (w1 + w2) / 2
        d_min, x_min = np.inf, None
        friction = 0.1 / np.sqrt(geom1.shape[0])
        def target_func(X):
            """Squared difference with reference w0"""
            wx, dwdR = compute_wij(X, rijlist, scaler, cell)
            delta_w = wx - w
            val, grad = 0.5 * np.dot(delta_w, delta_w), np.einsum('i,ij->j', delta_w, dwdR)
            logger.info("val=%10.3f  ", val)
            return val, grad

        # The inner loop performs minimization using either end-point as the starting guess.
        for coef in [0.02, 0.98]:
            x0 = (geom1 * coef + (1 - coef) * geom2).ravel()
            x0 += nudge * np.random.random_sample(x0.shape)
            x0[~free] = ((geom1 + geom2) / 2).ravel()[~free]

            def full(x, x0=x0):
                """Cartesian geometry from the optimized coordinates"""
                X = x0.copy()
                X[free] = x
                return X
            logger.debug('Starting least-squares minimization of bisection point at %7.2f.', coef)
            result = least_squares(lambda x: np.concatenate([compute_wij(full(x), rijlist, scaler, cell)[0] - w, (x-x0[free])*friction]), x0[free],
                                   lambda x: np.vstack([compute_wij(full(x), rijlist, scaler, cell)[1][:, free], np.identity(x.size) * friction]), ftol=tol, gtol=tol)
            x_mid = full(result['x']).reshape(-1, 3)
            # Take the interpolated geometry, construct new pair list and check for new contacts
            new_list = geom_list + [x_mid]
            new_rij, _ = get_bond_list(new_list, threshold=threshold, min_neighbors=0, cell=cell, pbc=pbc)
            extras = set(new_rij) - set(rijlist)
            if extras: 
                logger.info('  Screened pairs came into contact. Adding reference point.')
                # Update pair list then go back to the minimization loop if new contacts are found
                geom_list = new_list
                add_pair |= extras
                break
            # Perform local geodesic optimization for the new image.
            # Upstream passed 0.7 here, which Geodesic silently replaced by 1.7; keep the effective value
            smoother = Geodesic(atoms, [geom1, x_mid, geom2], 1.7, threshold=threshold, log_level=logging.DEBUG, friction=1,
                                cell=cell, pbc=pbc, fixed=fixed)
            smoother.compute_disps()
            width = max([np.sqrt(np.mean((g - smoother.path[1]) ** 2)) for g in [geom1, geom2]])
            dist, x_mid = width + smoother.length, smoother.path[1]
            logger.debug('  Trial path length: %8.3f after %d iterations', dist, result['nfev'])
            if dist < d_min:
                d_min, x_min = dist, x_mid
        else:   # Both starting guesses finished without new atom pairs.  Minimization successful
            break
    return x_min


def redistribute(atoms, geoms, nimages, tol=1e-2, cell=None, pbc=None, fixed=None):
    """Add or remove images so that the path length matches the desired number.

    If the number is too few, new points are added by bisecting the largest RMSD. If too numerous,
    one image is removed at a time so that the new merged segment has the shortest RMSD.

    Args:
        geoms:      Geometry of the original path.
        nimages:    The desired number of images
        tol:        Convergence tolerance for bisection.
        cell, pbc:  Periodic cell and flags (ASE convention).  None for molecules.
        fixed:      Fixed atoms or Cartesian components, see `Geodesic`.  Disables rigid-body alignment.

    Returns:
        An aligned and redistributed path with has the correct number of images.
    """
    geoms = np.asarray(geoms, dtype=float)
    fixed = fixed_mask(fixed, geoms.shape[1])
    rigid = fixed is None
    radii = get_radii(atoms)
    if not rigid:
        moved = np.abs(align_path(geoms, cell, pbc, rigid=False, radii=radii)[1] - geoms[:1])[:, fixed].max()
        if moved > 1e-3:
            logger.warning("Fixed coordinates differ by up to %.3g between the input images; they are "
                           "interpolated linearly.", moved)
    _, geoms = align_path(geoms, cell, pbc, rigid, radii)
    geoms = list(geoms)
    # If there are too few images, add bisection points
    while len(geoms) < nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[1:], geoms)]
        max_i = np.argmax(dists)
        logger.info("Inserting image between %d and %d with Cartesian RMSD %10.3f.  New length:%d",
                    max_i, max_i + 1, dists[max_i], len(geoms) + 1)
        insertion = mid_point(atoms, geoms[max_i], geoms[max_i + 1], tol, cell=cell, pbc=pbc, fixed=fixed)
        _, insertion = align_geom(geoms[max_i], insertion, cell, pbc, rigid, radii)
        geoms.insert(max_i + 1, insertion)
        geoms = list(align_path(geoms, cell, pbc, rigid, radii)[1])
    # If there are too many images, remove points
    while len(geoms) > nimages:
        dists = [np.sqrt(np.mean((g1 - g2) ** 2)) for g1, g2 in zip(geoms[2:], geoms)]
        min_i = np.argmin(dists)
        logger.info("Removing image %d.  Cartesian RMSD of merged section %10.3f",
                    min_i + 1, dists[min_i])
        del geoms[min_i + 1]
        geoms = list(align_path(geoms, cell, pbc, rigid, radii)[1])
    return geoms


def smooth_path(atoms, path, scaling=1.7, threshold=3, friction=1e-2, sweep=None, tol=2e-3, max_iter=15,
                micro_iter=20, max_rounds=5, cell=None, pbc=None, fixed=None, callback=None):
    """Geodesic smoothing of a path, repeated while atoms come into contact that the coordinate
    system does not describe.

    The atom pairs of the coordinate system are chosen from the initial path.  Smoothing can move
    atoms into contact with atoms they have no coordinate with, e.g. a molecule into a surface, since
    nothing in the metric resists that.  After each round, all images are therefore checked for pairs
    closer than `threshold` that are not coordinates yet.  These are added, keeping all previous pairs,
    and the path is smoothed again from the current geometry.

    Args:
        atoms:      Atom symbols.
        path:       Initial path, e.g. from `redistribute`.
        scaling:    Morse alpha of the scaled coordinates.
        threshold:  Distance cut-off for atom pairs in the coordinate system.
        sweep:      Optimize one image at a time.  Defaults to True for more than 35 atoms.
        max_rounds: Maximum number of smoothing rounds.  1 smooths once, as without re-contact checks.
        callback:   Called with the `Geodesic` object at the start of every round, e.g. to save the
                    current path on interruption.

    Returns:
        The `Geodesic` object of the last round; the smoothed path is its `path` attribute.
    """
    if sweep is None:
        sweep = len(atoms) > 35
    enforce = set()
    for iteration in range(max_rounds):
        smoother = Geodesic(atoms, path, scaling, threshold=threshold, friction=friction, cell=cell, pbc=pbc,
                            fixed=fixed, enforce=enforce)
        if callback is not None:
            callback(smoother)
        if sweep:
            smoother.sweep(tol=tol, max_iter=max_iter, micro_iter=micro_iter)
        else:
            smoother.smooth(tol=tol, max_iter=max_iter)
        path = smoother.path
        contacts, _ = get_bond_list(path, threshold=threshold, min_neighbors=0, snapshots=len(path),
                                    cell=cell, pbc=pbc)
        extras = set(contacts) - set(smoother.rij_list)
        if not extras:
            break
        if iteration < max_rounds - 1:
            logger.info("Round %d: %d atom pairs came into contact that are not coordinates.  Smoothing again "
                        "with an extended coordinate system.", iteration + 1, len(extras))
            enforce = set(smoother.rij_list) | extras
    else:
        logger.warning("%d atom pairs in contact are still not coordinates after %d smoothing rounds.",
                       len(extras), max_rounds)
    return smoother
