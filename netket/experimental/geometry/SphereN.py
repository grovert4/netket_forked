from dataclasses import dataclass
from collections.abc import Sequence

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass(frozen=True)
class SphereN:
    """The n-sphere S^n, embedded in R^(n+1)."""

    n: int

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        object.__setattr__(self, "n", int(n))

    def tree_flatten(self):
        # n is a static integer, so it goes in aux data rather than children
        return (), (self.n,)

    @classmethod
    def tree_unflatten(cls, aux, data):
        (n,) = aux
        return cls(n=n)

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.n

    @property
    def embedding_dimension(self) -> int:
        """Dimension of the ambient space R^(n+1)."""
        return self.n + 1

    def distance(
        self, r1: Sequence[float] | jnp.ndarray, r2: Sequence[float] | jnp.ndarray
    ) -> jnp.ndarray:
        """Geodesic (great-circle) distance between points ``r1`` and ``r2`` on S^n.

        Points are expected to be unit vectors in R^(n+1). Their norm is not
        checked here so that the function stays compatible with ``jax.jit``.
        """

        r1 = jnp.asarray(r1)
        r2 = jnp.asarray(r2)

        # shapes are static, so this check is safe under jit
        if r1.shape[-1] != self.n + 1 or r2.shape[-1] != self.n + 1:
            raise ValueError(
                f"Points must have last dimension n + 1 = {self.n + 1}"
            )

        # angle = 2 * atan2(|r1 - r2|, |r1 + r2|) is accurate for both nearby
        # and nearly antipodal points, unlike arccos(r1 . r2)
        d_minus = jnp.linalg.norm(r1 - r2, axis=-1)
        d_plus = jnp.linalg.norm(r1 + r2, axis=-1)
        return 2.0 * jnp.arctan2(d_minus, d_plus)