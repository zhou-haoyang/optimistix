import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import optimistix.internal as optxi
import pytest


def test_residual_nonarray_no_jit():
    def primal(inputs):
        return (inputs - 2) ** 2, 5

    def rewrite(root, residual, inputs):
        return root**2 + inputs**2

    @jax.grad
    def run(x):
        sol, aux = optxi.implicit_jvp(
            primal, rewrite, x, tags=frozenset(), linear_solver=lx.LU()
        )
        return sol + aux

    run(4.0)


# `fn(y, c) = y**2 + c` has a real root only for `c <= 0`; starting Newton from a huge
# `y0` for `c > 0` diverges to a non-finite root.
def _diverging_fn(y, c):
    return y**2 + c


def test_nonconverged_backward_does_not_raise():
    # Differentiating through a non-converged solve used to raise from the
    # implicit-function-theorem linear solve (and its transpose), even with
    # `throw=False`. It should now return a (finite, masked) cotangent instead of raising.
    solver = optx.Newton(rtol=1e-6, atol=1e-6)

    def solve_one(y0, c):
        return optx.root_find(
            _diverging_fn, solver, y0, args=c, throw=False, max_steps=8
        ).value

    # The middle element diverges to a non-finite root; the others converge.
    y0s = jnp.array([1.0, 1e200, 1.0])
    cs = jnp.array([-4.0, 1.0, -9.0])  # converged roots are 2 and 3

    ys = jax.vmap(solve_one)(y0s, cs)
    assert jnp.isfinite(ys[0]) and jnp.isfinite(ys[2])
    assert not jnp.isfinite(ys[1])

    def loss(scale):
        ys = jax.vmap(lambda y0, c: solve_one(y0, scale * c))(y0s, cs)
        # Mask the diverged element before it contaminates the sum/gradient.
        ys = jnp.where(jnp.isfinite(ys), ys, 0.0)
        return jnp.sum(ys**2)

    grad = eqx.filter_grad(loss)(jnp.asarray(1.0))
    assert jnp.isfinite(grad)

    # The diverged element must not taint the cotangents of the converged ones: the
    # gradient should match a batch that omits the diverging element entirely.
    def loss_good(scale):
        ys = jax.vmap(lambda y0, c: solve_one(y0, scale * c))(
            jnp.array([1.0, 1.0]), jnp.array([-4.0, -9.0])
        )
        return jnp.sum(ys**2)

    assert jnp.allclose(grad, eqx.filter_grad(loss_good)(jnp.asarray(1.0)))


def test_nonconverged_forward_mode_is_nonfinite():
    # Forward-mode autodiff of a non-converged solve should yield a non-finite tangent
    # (rather than raising).
    solver = optx.Newton(rtol=1e-6, atol=1e-6)

    def solve_one(c):
        return optx.root_find(
            _diverging_fn, solver, jnp.asarray(1e200), args=c, throw=False, max_steps=8
        ).value

    primal, tangent = jax.jvp(solve_one, (jnp.asarray(1.0),), (jnp.asarray(1.0),))
    assert not jnp.isfinite(primal)
    assert not jnp.isfinite(tangent)


def test_nonconverged_still_raises_with_throw():
    # `throw=True` must still report the failure (here in the forward pass).
    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    with pytest.raises(Exception):
        optx.root_find(
            _diverging_fn,
            solver,
            jnp.asarray(1e200),
            args=jnp.asarray(1.0),
            throw=True,
            max_steps=8,
        )
