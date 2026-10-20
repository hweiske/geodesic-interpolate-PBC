import itertools
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from geodesic_interpolate.coord_utils import (
    align_path, compute_wij, compute_wij_sparse, covalent_radius, fixed_mask, get_bond_list, get_radii,
    morse_scaler)
from geodesic_interpolate.fileio import read_xyz
from geodesic_interpolate.geodesic import Geodesic
from geodesic_interpolate.pbc import (
    find_periodic_pairs, minimum_image, normalize_cell, unwrap_geom, unwrap_path, wrap_positions)

from ase.build import add_adsorbate, bulk, fcc111
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read, write
from ase.neighborlist import neighbor_list


TRICLINIC = np.array([[5.0, 0.0, 0.0], [2.0, 4.5, 0.0], [1.0, 1.5, 4.0]])


def _brute_force_pairs(geom, cutoff, cell, pbc, nrep=4):
    """Reference periodic pair search over a large block of images."""
    found = {}
    ranges = [range(-nrep, nrep + 1) if p else [0] for p in pbc]
    for shift in itertools.product(*ranges):
        offset = np.array(shift) @ cell
        for i in range(len(geom)):
            for j in range(i + 1, len(geom)):
                d = np.linalg.norm(geom[i] - geom[j] - offset)
                if d < cutoff:
                    found[(i, j) + shift] = d
    return found


class PeriodicUtilsTest(unittest.TestCase):
    def test_normalize_cell_completes_non_periodic_directions(self):
        cell, pbc = normalize_cell([[4, 0, 0], [0, 5, 0], [0, 0, 0]], [True, True, False])
        np.testing.assert_array_equal(pbc, [True, True, False])
        self.assertGreater(abs(np.linalg.det(cell)), 1)
        self.assertEqual(normalize_cell(np.eye(3) * 10, False), (None, None))
        with self.assertRaises(ValueError):
            normalize_cell([[4, 0, 0], [0, 5, 0], [0, 0, 0]], True)

    def test_minimum_image_is_shortest_in_skewed_cell(self):
        rng = np.random.default_rng(1)
        cell, pbc = normalize_cell(TRICLINIC)
        dvec = rng.uniform(-12, 12, size=(200, 3))
        mic = minimum_image(dvec, cell, pbc)
        shifts = np.array(list(itertools.product(range(-5, 6), repeat=3))) @ cell
        brute = np.min(np.linalg.norm(dvec[:, None] + shifts[None], axis=2), axis=1)
        np.testing.assert_allclose(np.linalg.norm(mic, axis=1), brute)
        # The result differs from the input only by lattice translations
        frac = (mic - dvec) @ np.linalg.inv(cell)
        np.testing.assert_allclose(frac, np.round(frac), atol=1e-10)

    def test_wrap_positions_round_trip(self):
        rng = np.random.default_rng(2)
        cell, pbc = normalize_cell(TRICLINIC, [True, False, True])
        geom = rng.uniform(-20, 20, size=(10, 3))
        wrapped, offsets = wrap_positions(geom, cell, pbc)
        np.testing.assert_allclose(wrapped + offsets @ cell, geom)
        self.assertTrue(np.all(offsets[:, 1] == 0))

    def test_periodic_pairs_match_brute_force(self):
        rng = np.random.default_rng(3)
        for pbc in [True, [True, True, False], [False, True, False]]:
            with self.subTest(pbc=pbc):
                cell, pbc = normalize_cell(TRICLINIC, pbc)
                # Unwrapped positions far outside the cell must give consistent shifts
                geom = rng.uniform(0, 1, size=(6, 3)) @ cell + rng.integers(-3, 4, size=(6, 3)) * pbc @ cell
                found = find_periodic_pairs(geom, 6.0, cell, pbc)
                brute = _brute_force_pairs(geom, 6.0, cell, pbc, nrep=10)
                self.assertEqual(set(found), set(brute))
                for pair, d in brute.items():
                    self.assertAlmostEqual(found[pair], d)

    def test_periodic_pairs_match_ase_neighbor_list(self):
        atoms = bulk('Cu', 'fcc', a=3.6).repeat((2, 1, 1))
        atoms.rattle(0.1, seed=4)
        cell, pbc = normalize_cell(atoms.cell, atoms.pbc)
        found = find_periodic_pairs(atoms.positions, 5.0, cell, pbc)
        i, j, S = neighbor_list('ijS', atoms, 5.0)
        # ASE uses d = x_j + S @ cell - x_i, i.e. the shift belongs to atom j
        ase_pairs = {(int(a), int(b)) + tuple(int(s) for s in shift)
                     for a, b, shift in zip(i, j, S) if a < b}
        self.assertEqual(set(found), ase_pairs)

    def test_unwrap_keeps_bonded_fragments_intact(self):
        # A chain moving by 0.4 cell along b, with its far end moving 0.6 cell: per-atom MIC images
        # the end atoms the other way and tears the chain; bonded unwrapping keeps it intact.
        cell, pbc = normalize_cell(np.diag([6.0, 20.0, 6.0]))
        start = np.array([[3, 10 + 1.4 * k, 3] for k in range(5)], dtype=float)
        end = start.copy()
        end[:, 1] -= 8 + np.arange(5) * 1.0   # chain rotates/stretches slightly while moving
        end[:, 1] += 20 * (end[:, 1] < 0)      # wrapped into the cell
        end = wrap_positions(end, cell, pbc)[0]
        radii = np.full(5, 0.76)
        bonded = unwrap_geom(start, end, cell, pbc, radii)
        mic = start + minimum_image(end - start, cell, pbc)
        self.assertLess(np.linalg.norm(np.diff(bonded, axis=0), axis=1).max(), 1.7)
        self.assertGreater(np.linalg.norm(np.diff(mic, axis=0), axis=1).max(), 10)
        frac = (bonded - end) @ np.linalg.inv(cell)
        np.testing.assert_allclose(frac, np.round(frac), atol=1e-10)
        # Without radii, or for small displacements, unwrapping is plain per-atom MIC
        np.testing.assert_allclose(unwrap_geom(start, end, cell, pbc), mic)
        small = start + 0.3
        np.testing.assert_allclose(unwrap_geom(start, small, cell, pbc, radii), small)

    def test_6cpp_ring_opening_keeps_chain_intact(self):
        start = read("test_cases/6cpp/start.poscar", format='vasp')
        end = read("test_cases/6cpp/end.poscar", format='vasp')
        cell, pbc = normalize_cell(start.cell, start.pbc)
        unwrapped = unwrap_geom(start.positions, end.positions, cell, pbc,
                                get_radii(start.get_chemical_symbols()))
        carbon = [a.index for a in start if a.symbol == 'C']
        d0 = start.get_all_distances(mic=True)
        d1 = end.get_all_distances(mic=True)
        persistent = [(i, j) for i in carbon for j in carbon if i < j and d0[i, j] < 1.7 and d1[i, j] < 1.7]
        self.assertEqual(len(persistent), 41)
        self.assertLess(max(np.linalg.norm(unwrapped[i] - unwrapped[j]) for i, j in persistent), 1.6)
        # The right part of the ring leaves through the boundary at y = 0
        moved = (unwrapped - start.positions)[:, 1]
        self.assertLess(moved.min(), -cell[1, 1] / 2)

    def test_unwrap_path_makes_atoms_continuous(self):
        cell, pbc = normalize_cell(np.eye(3) * 10)
        path = np.array([[[9.8, 5, 5]], [[0.1, 5, 5]], [[0.5, 5, 5]]])
        unwrapped = unwrap_path(path, cell, pbc)
        np.testing.assert_allclose(unwrapped[:, 0, 0], [9.8, 10.1, 10.5])


class PeriodicCoordinatesTest(unittest.TestCase):
    def _molecule_across_boundary(self):
        atoms, coords = read_xyz("test_cases/DielsAlder_interpolated.xyz")
        cell, pbc = normalize_cell(np.eye(3) * 40)
        # Shift so that the molecule straddles the cell corner, then wrap each image independently
        path = np.array(coords) - np.mean(coords[0], axis=0)
        wrapped = np.array([wrap_positions(g, cell, pbc)[0] for g in path])
        return atoms, path, wrapped, cell, pbc

    def test_large_box_reproduces_molecular_coordinates(self):
        atoms, path, wrapped, cell, pbc = self._molecule_across_boundary()
        self.assertGreater(np.ptp(wrapped[:, :, 0]), 30)  # Atoms really are split over the cell
        np.random.seed(0)
        molecular = Geodesic(atoms, path, friction=0.01)
        np.random.seed(0)
        periodic = Geodesic(atoms, wrapped, friction=0.01, cell=cell, pbc=pbc)
        self.assertEqual(sorted(pair[:2] for pair in periodic.rij_list), sorted(molecular.rij_list))
        molecular.compute_disps()
        periodic.compute_disps()
        self.assertAlmostEqual(molecular.length, periodic.length, places=8)

    def test_periodic_sparse_matches_dense_and_finite_difference(self):
        rng = np.random.default_rng(5)
        cell, pbc = normalize_cell(TRICLINIC)
        geom = rng.uniform(0, 1, size=(4, 3)) @ cell
        rij_list, re = get_bond_list(geom, ["C", "H", "O", "N"], threshold=5.0, cell=cell, pbc=pbc)
        self.assertTrue(all(len(pair) == 5 for pair in rij_list))
        self.assertTrue(any(pair[2:] != (0, 0, 0) for pair in rij_list))
        scaler = morse_scaler(re=re, alpha=1.7)
        w_dense, b_dense = compute_wij(geom, rij_list, scaler, cell)
        w_sparse, b_sparse = compute_wij_sparse(geom, rij_list, scaler, cell)
        np.testing.assert_allclose(w_sparse, w_dense)
        np.testing.assert_allclose(b_sparse.toarray(), b_dense)
        direction = rng.normal(size=geom.size)
        eps = 1e-6
        numeric = (compute_wij(geom.ravel() + eps * direction, rij_list, scaler, cell)[0] -
                   compute_wij(geom.ravel() - eps * direction, rij_list, scaler, cell)[0]) / (2 * eps)
        np.testing.assert_allclose(b_dense @ direction, numeric, rtol=1e-5, atol=1e-8)

    def test_periodic_pairs_require_cell(self):
        with self.assertRaisesRegex(ValueError, "require a cell"):
            compute_wij(np.zeros((2, 3)), [(0, 1, 1, 0, 0)], morse_scaler())

    def test_align_path_keeps_fixed_atoms_in_place(self):
        cell, pbc = normalize_cell(np.eye(3) * 10)
        path = np.array([[[1, 1, 1], [9.9, 5, 5]], [[1, 1, 1], [0.4, 5, 5]]])
        _, aligned = align_path(path, cell, pbc)
        np.testing.assert_allclose(aligned[:, 0], path[:, 0])
        np.testing.assert_allclose(aligned[1, 1], [10.4, 5, 5])


class RecontactTest(unittest.TestCase):
    def test_enforced_pairs_are_coordinates(self):
        atoms, coords = read_xyz("test_cases/H+CH4_CH3+H2_interpolated.xyz")
        far = (0, 5)
        geodesic = Geodesic(atoms, coords, threshold=0.5, min_neighbors=0, enforce={far})
        self.assertIn(far, geodesic.rij_list)

    def test_smooth_path_leaves_no_contacts_without_coordinates(self):
        from geodesic_interpolate.interpolation import redistribute, smooth_path

        np.random.seed(0)
        atoms, coords = read_xyz("test_cases/DielsAlder.xyz")
        raw = redistribute(atoms, coords, 7, tol=1e-2)
        threshold = 3
        smoother = smooth_path(atoms, raw, threshold=threshold, max_iter=5, max_rounds=5)
        contacts, _ = get_bond_list(smoother.path, threshold=threshold, min_neighbors=0,
                                    snapshots=len(smoother.path))
        self.assertEqual(set(contacts) - set(smoother.rij_list), set())


class FixedAtomsTest(unittest.TestCase):
    def test_fixed_mask_formats(self):
        self.assertIsNone(fixed_mask(None, 3))
        self.assertIsNone(fixed_mask([], 3))
        np.testing.assert_array_equal(fixed_mask([1], 3), [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        np.testing.assert_array_equal(fixed_mask(np.array([True, False, False]), 3)[:, 1], [1, 0, 0])
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 2] = True
        np.testing.assert_array_equal(fixed_mask(mask, 3), mask)

    def test_covalent_radii_from_ase(self):
        self.assertAlmostEqual(covalent_radius('C'), 0.76)
        self.assertAlmostEqual(covalent_radius('CL'), 1.02)
        self.assertAlmostEqual(covalent_radius('Cu'), 1.32)
        self.assertAlmostEqual(covalent_radius('Xx'), 1.5)

    def test_fixed_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(31)
        atoms = ["C", "H", "O", "N"]
        path = rng.normal(size=(4, len(atoms), 3))
        fixed = np.zeros((4, 3), dtype=bool)
        fixed[0] = True
        fixed[2, 1] = True
        geodesic = Geodesic(atoms, path, scaler=1.0, threshold=10.0, min_neighbors=0, fixed=fixed)
        # No rigid-body alignment with fixed atoms
        np.testing.assert_array_equal(geodesic.path, path)
        start, end, friction = 1, 3, 0.02
        x0 = geodesic.path[start:end].ravel().copy()
        X = geodesic.segment_coords(start, end)
        self.assertEqual(X.size, 2 * (12 - 4))
        direction = rng.normal(size=X.size)
        eps = 1e-6

        def residual(x):
            geodesic.compute_target_func(x, start=start, end=end, x0=x0, friction=friction)
            return geodesic.disps.copy()

        residual(X)
        self.assertEqual(geodesic.grad.shape, (len(geodesic.disps), X.size))
        analytic = geodesic.grad @ direction
        numeric = (residual(X + eps * direction) - residual(X - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)

    def test_smoothing_keeps_fixed_coordinates(self):
        atoms, coords = read_xyz("test_cases/H+CH4_CH3+H2_interpolated.xyz")
        coords = np.array(coords)
        coords[1:-1] += np.random.default_rng(37).normal(scale=0.05, size=coords[1:-1].shape)
        fixed = [0, 1]
        for method in ['smooth', 'sweep']:
            with self.subTest(method=method):
                geodesic = Geodesic(atoms, coords, fixed=fixed)
                original = geodesic.path.copy()
                getattr(geodesic, method)(max_iter=3)
                np.testing.assert_array_equal(geodesic.path[:, fixed], original[:, fixed])
                self.assertGreater(np.abs(geodesic.path - original).max(), 1e-4)


def _hollow_hop():
    """H hopping between hollow sites of Cu(111) across the cell boundary, bottom layers fixed."""
    initial = fcc111('Cu', (3, 3, 3), vacuum=6.0)
    add_adsorbate(initial, 'H', 1.0, 'fcc')
    initial.set_constraint([FixAtoms(indices=[a.index for a in initial if a.tag > 1]),
                          FixCartesian([18], mask=(True, True, False))])
    final = initial.copy()
    # fcc hollow at fractional (1/9, 1/9) to the neighboring hcp hollow at (-1/9, 2/9), which wraps
    # to the other side of the cell
    scaled = final.get_scaled_positions()
    scaled[-1, :2] = [-1 / 9, 2 / 9]
    final.set_scaled_positions(scaled)
    return initial, final


class AseInterfaceTest(unittest.TestCase):
    def test_vacancy_hop_across_cell_boundary(self):
        from geodesic_interpolate import interpolate

        np.random.seed(0)
        a = 4.05
        initial = bulk('Al', 'fcc', a=a, cubic=True).repeat(2)
        del initial[0]  # Vacancy at the origin
        final = initial.copy()
        # Nearest neighbor of the vacancy that sits on the far side of the cell
        target = np.argmin(np.linalg.norm(
            initial.positions - np.array([2 * a - a / 2, 2 * a - a / 2, 0]), axis=1))
        final.positions[target] = [0, 0, 0]
        final.wrap()

        images = interpolate([initial, final], nimages=5)

        self.assertEqual(len(images), 5)
        for image in images:
            np.testing.assert_allclose(np.array(image.cell), np.array(initial.cell))
            self.assertTrue(all(image.pbc))
        # End points are equal to the input up to lattice translations
        for image, ref in [(images[0], initial), (images[-1], final)]:
            frac = (image.positions - ref.positions) @ np.linalg.inv(np.array(ref.cell))
            np.testing.assert_allclose(frac, np.round(frac), atol=1e-8)
        # The hopping atom moves smoothly by about a/sqrt(2) in total, never by a lattice vector
        hop = np.array([image.positions[target] for image in images])
        steps = np.linalg.norm(np.diff(hop, axis=0), axis=1)
        self.assertAlmostEqual(np.linalg.norm(hop[-1] - hop[0]), a / np.sqrt(2), places=5)
        self.assertLess(steps.max(), 1.5)
        # Other atoms only relax slightly
        others = np.delete(np.arange(len(initial)), target)
        disp = np.array([image.positions[others] for image in images]) - initial.positions[others]
        self.assertLess(np.abs(disp).max(), 0.5)

    def test_slab_constraints_are_enforced(self):
        from geodesic_interpolate import interpolate

        np.random.seed(0)
        initial, final = _hollow_hop()
        images = interpolate([initial, final], nimages=5)
        fixed = images[0].constraints[0].get_indices()
        self.assertEqual(len(fixed), 18)
        for image in images:
            # Fixed atoms are exactly where they started (up to lattice translations from unwrapping)
            frac = (image.positions[fixed] - initial.positions[fixed]) @ np.linalg.inv(np.array(initial.cell))
            np.testing.assert_allclose(frac, np.round(frac), atol=1e-10)
            self.assertEqual(image.positions[18, 0], images[0].positions[18, 0])
            self.assertEqual(image.positions[18, 1], images[0].positions[18, 1])
            self.assertEqual(len(image.constraints), 2)
        hop = np.array([image.positions[-1] for image in images])
        self.assertGreater(final.get_scaled_positions()[-1, 0], 0.5)  # Really wrapped across the boundary
        self.assertAlmostEqual(np.linalg.norm(hop[-1, :2] - hop[0, :2]), 2.5527 / np.sqrt(3), places=3)
        self.assertLess(np.linalg.norm(np.diff(hop, axis=0), axis=1).max(), 1.0)

    def test_endpoint_only_constraints_are_not_enforced_along_path(self):
        from geodesic_interpolate import interpolate

        np.random.seed(0)
        initial, final = _hollow_hop()
        # Pin the adsorbate in the initial image only, as for a constrained end-point relaxation
        initial.set_constraint(initial.constraints + [FixAtoms(indices=[len(initial) - 1])])
        images = interpolate([initial, final], nimages=5)
        hop = np.array([image.positions[-1] for image in images])
        self.assertTrue(np.all(np.linalg.norm(np.diff(hop, axis=0), axis=1) > 0.1))
        self.assertEqual(len(images[0].constraints), 3)
        self.assertEqual(len(images[-1].constraints), 2)
        for image in images[1:-1]:
            mask = fixed_mask_of(image)
            self.assertEqual(mask.sum(), 18 * 3 + 2)
            self.assertFalse(mask[-1].any())

    def test_seed_makes_interpolation_reproducible(self):
        from geodesic_interpolate import interpolate

        initial, final = _hollow_hop()
        first = interpolate([initial, final], nimages=4, seed=3)
        np.random.seed(123)  # global state before the call must not matter
        second = interpolate([initial, final], nimages=4, seed=3)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.positions, b.positions)

    def test_cli_writes_traj_and_vasp_folders(self):
        from geodesic_interpolate.__main__ import main

        np.random.seed(0)
        initial, final = _hollow_hop()
        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, 'ends.traj'), [initial, final])
            argv = ['geodesic_interpolate', os.path.join(tmp, 'ends.traj'), '--nimages', '4',
                    '--output', os.path.join(tmp, 'out.traj'), '--vasp', os.path.join(tmp, 'neb'),
                    '--logging', 'WARNING']
            with mock.patch.object(sys, 'argv', argv):
                main()
            path = read(os.path.join(tmp, 'out.traj'), index=':')
            self.assertEqual(len(path), 4)
            self.assertEqual(sorted(os.listdir(os.path.join(tmp, 'neb'))), ['00', '01', '02', '03'])
            for i, image in enumerate(path):
                poscar = read(os.path.join(tmp, 'neb', '%02d' % i, 'POSCAR'))
                self.assertEqual(poscar.get_chemical_symbols(), image.get_chemical_symbols())
                np.testing.assert_allclose(poscar.positions, image.positions, atol=1e-6)
                # Selective dynamics: 18 fully fixed atoms plus x and y of atom 18
                self.assertEqual(fixed_mask_of(poscar).sum(), 18 * 3 + 2)


def fixed_mask_of(atoms):
    from geodesic_interpolate.ase_interface import get_fixed_mask
    return get_fixed_mask(atoms)


if __name__ == "__main__":
    unittest.main()
