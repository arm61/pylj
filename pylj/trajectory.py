"""Sampled configurations, in order."""

from collections.abc import Iterable, Iterator
from typing import overload

import numpy as np
from numpy.typing import NDArray

from pylj.configuration import Configuration
from pylj.scattering import check_q_max, default_q_max, shell_average, wavevectors


class Trajectory:
    """The configurations a simulation has sampled, in order.

    A simulation appends its current configuration each time ``sample`` is
    called. Every frame has the same box and the same number of atoms.
    Indexing gives a :class:`~pylj.configuration.Configuration`; slicing
    gives a ``Trajectory``.

    Args:
        frames: Configurations to start with, in order.
    """

    def __init__(self, frames: Iterable[Configuration] = ()) -> None:
        self._frames: list[Configuration] = []
        for one in frames:
            self.append(one)

    def append(self, configuration: Configuration) -> None:
        """Adds a frame to the end.

        Args:
            configuration: The frame to add.

        Raises:
            ValueError: If the frame's box or number of atoms differs from
                the first frame's.
        """
        if self._frames:
            first = self._frames[0]
            if configuration.box != first.box:
                raise ValueError(
                    f"The frame's box is {configuration.box} m but the trajectory's "
                    f"is {first.box} m"
                )
            if configuration.number_of_atoms != first.number_of_atoms:
                raise ValueError(
                    f"The frame has {configuration.number_of_atoms} atoms but the trajectory "
                    f"has {first.number_of_atoms}"
                )
        self._frames.append(configuration)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Configuration]:
        return iter(self._frames)

    @overload
    def __getitem__(self, index: int) -> Configuration: ...

    @overload
    def __getitem__(self, index: slice) -> "Trajectory": ...

    def __getitem__(self, index: int | slice) -> "Configuration | Trajectory":
        if isinstance(index, slice):
            return Trajectory(self._frames[index])
        return self._frames[index]

    @property
    def position(self) -> NDArray[np.float64]:
        """The positions of every frame, shape ``(frames, N, 2)``, in metres."""
        if not self._frames:
            return np.empty((0, 0, 2))
        return np.stack([one.position for one in self._frames])

    def __repr__(self) -> str:
        if not self._frames:
            return "Trajectory(no frames)"
        atoms = self._frames[0].number_of_atoms
        return f"Trajectory({len(self._frames)} frames, {atoms} atoms)"

    def rdf(
        self, bins: int = 100, r_max: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Averages g(r) over the frames.

        Args:
            bins: The number of bins.
            r_max: The largest distance binned, in metres; by default half
                the box.

        Returns:
            The bin centres, in metres, and the mean g(r) in each bin.

        Raises:
            ValueError: If the trajectory has no frames.
        """
        self._check_frames()
        curves = [one.rdf(bins, r_max) for one in self._frames]
        return curves[0][0], np.mean([gr for _, gr in curves], axis=0)

    def structure_factor(
        self, q_max: float | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Averages the structure factor S(q) over the frames.

        The wavevectors come from the first frame, and every frame is
        evaluated on that one set.

        Args:
            q_max: The largest wavevector magnitude, in 1/m; the default
                comes from :func:`~pylj.scattering.default_q_max` for the
                first frame.

        Returns:
            The wavevector magnitudes, in 1/m, and the mean S(q) at each
            magnitude.

        Raises:
            ValueError: If the trajectory has no frames, or ``q_max`` is
                below ``2 pi / L``.
        """
        self._check_frames()
        first = self._frames[0]
        if q_max is None:
            q_max = default_q_max(first.number_of_atoms, first.box)
        check_q_max(q_max, first.box)
        q, index, shell = wavevectors(first.box, q_max)
        total = np.zeros(q.size)
        for one in self._frames:
            total += shell_average(one.position, first.box, index, shell)
        return q, total / len(self._frames)

    def _check_frames(self) -> None:
        if not self._frames:
            raise ValueError("The trajectory has no frames: call sample() during the run")
