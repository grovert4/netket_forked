# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from netket.experimental.geometry import Cell, SphereN

from .continuous_hilbert import ContinuousHilbert


class Particle(ContinuousHilbert):
    r"""Hilbert space derived from ContinuousHilbert defining N particles
    in continuous space, either in a box (:class:`~netket.experimental.geometry.Cell`)
    or on a sphere (:class:`~netket.experimental.geometry.SphereN`)."""

    def __init__(self, N: int | tuple[int, ...], *, geometry: Cell | SphereN) -> None:
        if not hasattr(N, "__len__"):
            N = (N,)

        if not isinstance(geometry, (Cell, SphereN)):
            raise TypeError(
                "`geometry` must be an instance of `netket.experimental.geometry.Cell` "
                "or `netket.experimental.geometry.SphereN`."
            )

        self._N = sum(N)
        self._n_per_spin = N
        self._geometry = geometry

        if isinstance(geometry, Cell):
            domain = geometry.extent
        else:
            # Placeholder: only encodes the number of coordinates per particle (n + 1)
            domain = (np.inf,) * geometry.embedding_dimension

        super().__init__(domain)

    @property
    def _coords_per_particle(self) -> int:
        if isinstance(self._geometry, Cell):
            return len(self._geometry.extent)
        return self._geometry.embedding_dimension

    @property
    def size(self) -> int:
        return self._N * self._coords_per_particle

    @property
    def n_particles(self) -> int:
        r"""The number of particles"""
        return self._N

    @property
    def n_per_spin(self) -> tuple[int, ...]:
        r"""Number of particles in each spin projection."""
        return self._n_per_spin

    @property
    def geometry(self) -> Cell | SphereN:
        """Geometry of the continuous space."""
        return self._geometry

    @property
    def _attrs(self):
        return (self._N, self.geometry)

    def __repr__(self):
        return f"Particle(N={self.n_particles}, geometry={self.geometry})"