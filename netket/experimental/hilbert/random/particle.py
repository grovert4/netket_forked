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
from jax import numpy as jnp

import numpy as np

from netket import jax as nkjax
from netket.experimental.geometry import SphereN
from netket.experimental.hilbert import ContinuousHilbert, Particle
from netket.utils.dispatch import dispatch


def take_sub(key, x, n):
    key, subkey = jax.random.split(key)
    ind = jax.random.choice(
        subkey, jnp.arange(0, x.shape[0], 1), replace=False, shape=(n,)
    )
    return x[ind, :]


def _random_state_sphere(hilb: Particle, key, batches: int, dtype):
    """Uniformly distributed points on S^n: normalized Gaussian vectors."""
    dim = hilb.geometry.embedding_dimension
    gaussian = jax.random.normal(
        key,
        shape=(batches, hilb.n_particles, dim),
        dtype=nkjax.dtype_real(dtype),
    )
    norm = jnp.linalg.norm(gaussian, axis=-1, keepdims=True)
    points = gaussian / norm
    return jnp.asarray(points.reshape(batches, hilb.size), dtype=dtype)


def _random_state_box(hilb: Particle, key, batches: int, dtype):
    pbc = np.array(hilb.n_particles * hilb.geometry.pbc)
    boundary = np.tile(pbc, (batches, 1))

    # TODO genearalize to particles with diferent domains
    Ls = np.array(hilb.n_particles * hilb.domain)

    # If we have PBCs this defines the size of the Gaussian noise in the particle positions.
    # Without PBCs this is never needed (initialization with random Gaussian)
    # and we can set a random number (here 1.)
    modulus = np.where(np.equal(pbc, False), 1.0, Ls)

    min_modulus = np.min(modulus)

    # use real dtypes because this does not work with complex ones.
    gaussian = jax.random.normal(
        key, shape=(batches, hilb.size), dtype=nkjax.dtype_real(dtype)
    )

    width = min_modulus / (4.0 * hilb.n_particles)
    # The width gives the noise level. In the periodic case the

@dispatch
def random_state(hilb: Particle, key, batches: int, *, dtype):
    if isinstance(hilb.geometry, SphereN):
        return _random_state_sphere(hilb, key, batches, dtype)
    return _random_state_box(hilb, key, batches, dtype)