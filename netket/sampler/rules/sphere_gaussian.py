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

import jax
import jax.numpy as jnp

from netket.utils import struct

from .base import MetropolisRule


@struct.dataclass
class SphereGaussianRule(MetropolisRule):
    r"""Metropolis rule for particles on the n-sphere.

    At every step one particle is chosen uniformly, its position ``x`` is
    perturbed in the ambient space with isotropic Gaussian noise and projected
    back onto the sphere:

        x' = (x + sigma * xi) / || x + sigma * xi ||,   xi ~ N(0, I_{n+1}).

    The proposal density depends only on ``x . x'``, hence it is symmetric and
    no correction factor is needed.

    Args:
        sigma: standard deviation of the ambient Gaussian noise (the
            geodesic step size is roughly ``sigma`` for small ``sigma``).
    """

    sigma: float = 1.0

    def transition(rule, sampler, machine, parameters, state, key, r):
        # imported here to avoid circular imports with netket.experimental
        from netket.experimental.geometry import SphereN

        hilb = sampler.hilbert
        if not isinstance(getattr(hilb, "geometry", None), SphereN):
            raise TypeError(
                "SphereGaussianRule requires a `Particle` Hilbert space defined "
                "on a `SphereN` geometry."
            )
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError("SphereGaussianRule is not defined for complex dtypes.")

        n_chains = r.shape[0]
        n_particles = hilb.n_particles
        dim = hilb.geometry.embedding_dimension

        key_idx, key_noise = jax.random.split(key)

        # which particle to move, one per chain
        idx = jax.random.randint(key_idx, (n_chains,), 0, n_particles)

        x = r.reshape(n_chains, n_particles, dim)
        chains = jnp.arange(n_chains)

        chosen = x[chains, idx]  # (n_chains, dim)
        noise = rule.sigma * jax.random.normal(key_noise, chosen.shape, dtype=r.dtype)

        proposal = chosen + noise
        proposal = proposal / jnp.linalg.norm(proposal, axis=-1, keepdims=True)

        x_new = x.at[chains, idx].set(proposal)
        return x_new.reshape(n_chains, hilb.size), None