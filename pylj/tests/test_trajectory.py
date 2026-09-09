import unittest

import numpy as np
from numpy.testing import assert_allclose

from pylj import placement
from pylj.tests.argon import ARGON
from pylj.trajectory import Trajectory


def frame(box: float = 20e-10, atoms: int = 4):
    return placement.place_square(atoms, (ARGON,), box)


def moved(configuration):
    """The same frame with one atom shifted, so the pair distances differ."""
    position = configuration.position.copy()
    position[0] += [3e-10, 1e-10]
    return configuration.replace(position=position)


class TestTrajectory(unittest.TestCase):
    def test_starts_empty_and_appends_in_order(self):
        trajectory = Trajectory()
        self.assertEqual(len(trajectory), 0)
        first, second = frame(), frame()
        trajectory.append(first)
        trajectory.append(second)
        self.assertEqual(len(trajectory), 2)
        self.assertIs(trajectory[0], first)
        self.assertIs(trajectory[-1], second)
        self.assertEqual(list(trajectory), [first, second])

    def test_slicing_gives_a_trajectory(self):
        trajectory = Trajectory([frame(), frame(), frame()])
        thinned = trajectory[::2]
        self.assertIsInstance(thinned, Trajectory)
        self.assertEqual(len(thinned), 2)

    def test_position_stacks_the_frames(self):
        trajectory = Trajectory([frame(), frame()])
        self.assertEqual(trajectory.position.shape, (2, 4, 2))
        assert_allclose(trajectory.position[1], trajectory[1].position)
        self.assertEqual(Trajectory().position.shape, (0, 0, 2))

    def test_rejects_a_frame_from_a_different_system(self):
        trajectory = Trajectory([frame()])
        with self.assertRaisesRegex(ValueError, "box"):
            trajectory.append(frame(box=30e-10))
        with self.assertRaisesRegex(ValueError, "atoms"):
            trajectory.append(frame(atoms=9))

    def test_rdf_is_the_mean_over_frames(self):
        one = frame()
        other = moved(one)
        trajectory = Trajectory([one, other])
        r, gr = trajectory.rdf(bins=20)
        r_one, gr_one = one.rdf(bins=20)
        _, gr_other = other.rdf(bins=20)
        self.assertFalse(np.allclose(gr_one, gr_other))
        assert_allclose(r, r_one)
        assert_allclose(gr, (gr_one + gr_other) / 2)

    def test_structure_factor_is_the_mean_over_frames(self):
        one = frame()
        other = moved(one)
        trajectory = Trajectory([one, other])
        q, s = trajectory.structure_factor()
        q_one, s_one = one.structure_factor()
        _, s_other = other.structure_factor()
        self.assertFalse(np.allclose(s_one, s_other))
        assert_allclose(q, q_one)
        assert_allclose(s, (s_one + s_other) / 2)

    def test_analyses_refuse_an_empty_trajectory(self):
        with self.assertRaisesRegex(ValueError, "no frames"):
            Trajectory().rdf()
        with self.assertRaisesRegex(ValueError, "no frames"):
            Trajectory().structure_factor()
