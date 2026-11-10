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
r"""Spin-1/2 operators acting on wavefunctions on SU(2)^N = (S^3)^N.

Every spin is a unit quaternion ``q = (q0, q1, q2, q3)`` and the wavefunction
is a function ``psi(q_1, ..., q_N)``.  The Pauli matrices are realised as Lie
derivatives (vector fields on S^3),

    sigma^a_i = -i L^a_i ,   L^a_i f = d/dt f(q_i -> exp(t X_a) * q_i)|_{t=0},

where the three tangent vectors ``A^a(q)`` are given by left-multiplication of
``q`` with the unit quaternions ``i, j, k``.  On the basis
``|up> = q0 + i q3``, ``|down> = q2 + i q1`` they reproduce exactly the Pauli
matrices ``sigma^x, sigma^y, sigma^z``.

Operators are built with the same syntax as the discrete NetKet operators::

    from netket.experimental.operator.spin_manifold import sigmax, sigmaz

    H = 0
    for i in range(N):
        H += -V * sigmaz(hi, i) @ sigmaz(hi, (i + 1) % N) - g * sigmax(hi, i)

The local energy is evaluated with the log-derivative identities

    (L_i^a L_j^b psi) / psi = L_i^a L_j^b log(psi) + (L_i^a log psi)(L_j^b log psi)

which hold for products on two *different* sites.  Products of Paulis on the
same site are reduced with ``s^a s^b = delta_ab + i eps_abc s^c``, so only
terms with at most two Paulis on distinct sites remain.  This is exact on the
spin-1/2 sector (wavefunctions odd and linear in each quaternion).
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Hashable

import jax
import jax.numpy as jnp

from netket.operator import ContinuousOperator
from netket.utils import struct
from netket.utils.types import Array, PyTree

__all__ = ["SpinManifoldOperator", "sigma", "sigmax", "sigmay", "sigmaz"]

_CHANNELS = {"x": 0, "y": 1, "z": 2}
_CH_NAME = "xyz"


def su2_frame(q: Array) -> Array:
    """Tangent frame ``A^a(q)`` of S^3, shape ``(..., 3, 4)``.

    ``A^x, A^y, A^z`` are ``i*q, j*q, k*q`` (quaternion left multiplication).
    """
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ax = jnp.stack([-q1,  q0, -q3,  q2], axis=-1)   # i * q
    ay = jnp.stack([-q2,  q3,  q0, -q1], axis=-1)   # j * q
    az = jnp.stack([-q3, -q2,  q1,  q0], axis=-1)   # k * q
    return jnp.stack([ax, ay, az], axis=-2)         # (x, y, z) order, no remapping

def _pauli_mul(a: int, b: int):
    """sigma^a sigma^b = delta  _ab + i eps_abc sigma^c -> list of (coef, channel|None)."""
    if a == b:
        return [(1.0 + 0j, None)]
    c = 3 - a - b
    sign = 1.0 if (b - a) % 3 == 1 else -1.0
    return [(1j * sign, c)]


def _mul_keys(ka, kb):
    """Product of two term keys (tuples of ``(site, channel)``)."""
    da, db = dict(ka), dict(kb)
    partial = [(1.0 + 0j, ())]
    for s in sorted(set(da) | set(db)):
        if s in da and s in db:
            opts = _pauli_mul(da[s], db[s])
        else:
            opts = [(1.0 + 0j, da[s] if s in da else db[s])]
        partial = [
            (c * oc, key if och is None else key + ((s, och),))
            for c, key in partial
            for oc, och in opts
        ]
    return partial


def _check_hilbert(hilbert):
    from netket.experimental.geometry import SphereN

    geom = getattr(hilbert, "geometry", None)
    if not isinstance(geom, SphereN) or geom.n != 3:
        raise TypeError(
            "SpinManifoldOperator requires a `Particle` Hilbert space on "
            "`SphereN(3)` (one S^3 = SU(2) per spin)."
        )


class SpinManifoldOperator(ContinuousOperator):
    r"""Sum of products of at most two Pauli operators on distinct sites,
    acting on wavefunctions on ``SU(2)^N`` through Lie derivatives.

    Build instances with :func:`sigmax`, :func:`sigmay`, :func:`sigmaz` and the
    usual ``+, -, *, @`` operations.
    """

    _terms: tuple = struct.static_field()

    def __init__(self, hilbert, terms: dict | None = None):
        _check_hilbert(hilbert)
        terms = {} if terms is None else terms
        clean = {k: complex(c) for k, c in terms.items() if c != 0}
        for key in clean:
            if len(key) > 2:
                raise ValueError(
                    "Only terms with at most two Pauli operators on distinct "
                    "sites are supported (quadratic Hamiltonians)."
                )
            for site, _ in key:
                if not 0 <= site < hilbert.n_particles:
                    raise ValueError(f"Site {site} out of range.")
        self._terms = tuple(sorted(clean.items()))
        super().__init__(hilbert, jnp.result_type(complex))

    # ------------------------------------------------------------------ algebra
    @property
    def terms(self) -> dict:
        return dict(self._terms)

    @property
    def is_hermitian(self) -> bool:
        return all(abs(c.imag) < 1e-12 for _, c in self._terms)

    def _new(self, terms):
        return SpinManifoldOperator(self.hilbert, terms)

    def _coerce(self, other):
        if isinstance(other, SpinManifoldOperator):
            if other.hilbert != self.hilbert:
                raise ValueError("Operators act on different Hilbert spaces.")
            return other.terms
        if isinstance(other, numbers.Number):
            return {(): complex(other)}
        return None

    def __add__(self, other):
        o = self._coerce(other)
        if o is None:
            return super().__add__(other)
        out = self.terms
        for k, c in o.items():
            out[k] = out.get(k, 0.0) + c
        return self._new(out)

    __radd__ = __add__

    def __neg__(self):
        return self._new({k: -c for k, c in self._terms})

    def __sub__(self, other):
        o = self._coerce(other)
        if o is None:
            return super().__sub__(other)
        return self + (-self._new(o))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return self._new({k: c * other for k, c in self._terms})
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self * (1.0 / other)
        return NotImplemented

    def __matmul__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        out: dict = {}
        for ka, ca in self._terms:
            for kb, cb in o.items():
                for c, key in _mul_keys(ka, kb):
                    out[key] = out.get(key, 0.0) + ca * cb * c
        return self._new(out)

    def __rmatmul__(self, other):
        o = self._coerce(other)
        if o is None:
            return NotImplemented
        return self._new(o) @ self

    # ------------------------------------------------------------------ kernel
    def _expect_kernel_single(self, logpsi: Callable, params: PyTree, x: Array):
        N = self.hilbert.n_particles
        terms = self.terms

        const = terms.get((), 0.0)
        fields = [(k[0], c) for k, c in terms.items() if len(k) == 1]
        pairs = [(k[0], k[1], c) for k, c in terms.items() if len(k) == 2]

        def f(y):
            return jnp.reshape(logpsi(params, y), ())

        frames = jax.lax.stop_gradient(su2_frame(x.reshape(N, 4)))  # (N, 3, 4)

        def direction(site, ch):
            v = jnp.zeros((N, 4), dtype=x.dtype).at[site].set(frames[site, ch])
            return v.reshape(-1)

        def d1(site, ch):
            return jax.jvp(f, (x,), (direction(site, ch),))[1]

        def d2(site_a, ch_a, site_b, ch_b):
            va = direction(site_a, ch_a)

            def g(y):
                return jax.jvp(f, (y,), (va,))[1]

            return jax.jvp(g, (x,), (direction(site_b, ch_b),))[1]

        res = jnp.asarray(const, dtype=self.dtype)

        if fields:
            sites = jnp.array([s for (s, _), _ in fields])
            chans = jnp.array([a for (_, a), _ in fields])
            coefs = jnp.array([c for _, c in fields], dtype=self.dtype)
            G = jax.vmap(d1)(sites, chans)
            res = res + jnp.sum(-1j * coefs * G)

        if pairs:
            si = jnp.array([a[0] for a, _, _ in pairs])
            ci = jnp.array([a[1] for a, _, _ in pairs])
            sj = jnp.array([b[0] for _, b, _ in pairs])
            cj = jnp.array([b[1] for _, b, _ in pairs])
            coefs = jnp.array([c for _, _, c in pairs], dtype=self.dtype)
            Gi = jax.vmap(d1)(si, ci)
            Gj = jax.vmap(d1)(sj, cj)
            Hij = jax.vmap(d2)(si, ci, sj, cj)
            res = res + jnp.sum(-coefs * (Hij + Gi * Gj))

        return res

    def _expect_kernel(self, logpsi: Callable, params: PyTree, x: Array) -> Array:
        return jax.vmap(
            lambda x_: self._expect_kernel_single(logpsi, params, x_), in_axes=0
        )(x)

    @struct.property_cached(pytree_ignore=True)
    def _attrs(self) -> tuple[Hashable, ...]:
        return (self.hilbert, self._terms)

    def __repr__(self):
        if not self._terms:
            return "SpinManifoldOperator(0)"

        def name(key):
            return "".join(f"s{_CH_NAME[a]}{s}" for s, a in key) or "1"

        def fmt(c):
            return f"{c.real:.6g}" if abs(c.imag) < 1e-14 else f"({c:.6g})"

        body = " + ".join(f"{fmt(c)}*{name(k)}" for k, c in self._terms)
        return f"SpinManifoldOperator({body})"


def sigma(hilbert, site: int, channel: str | int) -> SpinManifoldOperator:
    """Pauli operator ``sigma^channel`` acting on spin ``site``."""
    ch = _CHANNELS[channel] if isinstance(channel, str) else int(channel)
    return SpinManifoldOperator(hilbert, {((int(site), ch),): 1.0})


def sigmax(hilbert, site: int) -> SpinManifoldOperator:
    return sigma(hilbert, site, "x")


def sigmay(hilbert, site: int) -> SpinManifoldOperator:
    return sigma(hilbert, site, "y")


def sigmaz(hilbert, site: int) -> SpinManifoldOperator:
    return sigma(hilbert, site, "z")