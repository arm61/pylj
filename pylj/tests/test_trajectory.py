import unittest

import numpy as np
from numpy.testing import assert_allclose

from pylj import placement
from pylj.configuration import MDConfiguration
from pylj.tests.argon import ARGON
from pylj.trajectory import Trajectory


def frame(box: float = 20e-10, atoms: int = 4):
    return placement.place_square(atoms, (ARGON,), box)


def moved(configuration):
    """The same frame with one atom shifted, so the pair distances differ."""
    position = configuration.positions.copy()
    position[0] += [3e-10, 1e-10]
    return configuration.replace(positions=position)


def md_frame(unwrapped, box=20e-10):
    """An argon MDConfiguration at rest with the given unwrapped positions."""
    unwrapped = np.asarray(unwrapped, dtype=float)
    return MDConfiguration(
        positions=unwrapped % box,
        species=(ARGON,),
        species_index=np.zeros(unwrapped.shape[0], dtype=np.int64),
        box=box,
        velocities=np.zeros_like(unwrapped),
        unwrapped=unwrapped,
    )


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
        self.assertEqual(trajectory.positions.shape, (2, 4, 2))
        assert_allclose(trajectory.positions[1], trajectory[1].positions)
        self.assertEqual(Trajectory().positions.shape, (0, 0, 2))

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

    def test_structure_factor_refuses_a_q_max_below_the_box(self):
        trajectory = Trajectory([frame()])
        with self.assertRaisesRegex(ValueError, "smallest wavevector"):
            trajectory.structure_factor(q_max=8.0)


class TestTimes(unittest.TestCase):
    def test_an_untimed_trajectory_has_no_times(self):
        self.assertIsNone(Trajectory([frame(), frame()]).times)
        self.assertIsNone(Trajectory().times)

    def test_times_are_kept_in_order(self):
        trajectory = Trajectory([frame(), frame()], times=[0.0, 1e-13])
        assert_allclose(trajectory.times, [0.0, 1e-13])
        trajectory.append(frame(), 2e-13)
        assert_allclose(trajectory.times, [0.0, 1e-13, 2e-13])

    def test_a_slice_keeps_its_times(self):
        trajectory = Trajectory([frame(), frame(), frame()], times=[0.0, 1.0, 2.0])
        assert_allclose(trajectory[1:].times, [1.0, 2.0])
        assert_allclose(trajectory[::2].times, [0.0, 2.0])
        self.assertIsNone(Trajectory([frame(), frame()])[1:].times)

    def test_refuses_mixing_timed_and_untimed_frames(self):
        timed = Trajectory([frame()], times=[0.0])
        with self.assertRaisesRegex(ValueError, "time"):
            timed.append(frame())
        with self.assertRaisesRegex(ValueError, "time"):
            Trajectory([], times=[]).append(frame())
        untimed = Trajectory([frame()])
        with self.assertRaisesRegex(ValueError, "time"):
            untimed.append(frame(), 1.0)
        with self.assertRaisesRegex(ValueError, "time"):
            Trajectory().append(frame(), 1.0)

    def test_refuses_times_of_the_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "times"):
            Trajectory([frame(), frame()], times=[0.0])


class TestMSD(unittest.TestCase):
    def test_averages_over_every_origin(self):
        # One atom at x = 0, 1, 3 m. Lag 1 has origins at 0 and 1 with
        # displacements 1 and 2, so (1 + 4) / 2; lag 2 has one origin, 3.
        # A single origin would give 1 at lag 1.
        frames = [md_frame([[x, 0.0]]) for x in (0.0, 1.0, 3.0)]
        lag, msd = Trajectory(frames, times=[0.0, 1.0, 2.0]).msd()
        assert_allclose(lag, [1.0, 2.0])
        assert_allclose(msd, [2.5, 9.0])

    def test_constant_velocity_gives_v_squared_t_squared(self):
        # Two atoms moving at (1, 2) and (-3, 0) m/s, sampled every 0.5 s:
        # every origin gives the same displacement, |v|^2 lag^2, and the
        # mean over atoms is (5 + 9) / 2 = 7 lag^2.
        velocity = np.array([[1.0, 2.0], [-3.0, 0.0]])
        times = np.arange(6) * 0.5
        frames = [md_frame(velocity * t) for t in times]
        lag, msd = Trajectory(frames, times=times).msd()
        assert_allclose(lag, np.arange(1, 6) * 0.5)
        assert_allclose(msd, 7.0 * lag**2)

    def test_max_lag_truncates_and_clamps(self):
        frames = [md_frame([[float(x), 0.0]]) for x in range(6)]
        trajectory = Trajectory(frames, times=np.arange(6) * 2.0)
        lag, _ = trajectory.msd(max_lag=5.0)
        assert_allclose(lag, [2.0, 4.0])
        lag, _ = trajectory.msd(max_lag=1e9)
        assert_allclose(lag, [2.0, 4.0, 6.0, 8.0, 10.0])
        with self.assertRaisesRegex(ValueError, "max_lag"):
            trajectory.msd(max_lag=1.0)
        # An exact multiple of the spacing includes that lag, even where the
        # division lands a hair under the integer in floating point.
        frames = [md_frame([[float(x), 0.0]]) for x in range(30)]
        trajectory = Trajectory(frames, times=np.arange(30) * 1e-14)
        lag, _ = trajectory.msd(max_lag=23e-14)
        self.assertEqual(lag.size, 23)

    def test_refuses_what_it_cannot_measure(self):
        frames = [md_frame([[float(x), 0.0]]) for x in range(3)]
        with self.assertRaisesRegex(ValueError, "times"):
            Trajectory(frames).msd()
        with self.assertRaisesRegex(ValueError, "two frames"):
            Trajectory(frames[:1], times=[0.0]).msd()
        with self.assertRaisesRegex(ValueError, "evenly spaced"):
            Trajectory(frames, times=[0.0, 1.0, 3.0]).msd()
        with self.assertRaisesRegex(ValueError, "evenly spaced"):
            Trajectory(frames, times=[0.0, 0.0, 0.0]).msd()
