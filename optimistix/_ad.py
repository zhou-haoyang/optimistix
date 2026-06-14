import types
from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.custom_derivatives
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from equinox.internal import ω
from jaxtyping import PyTree

from ._misc import tree_full_like, tree_where


def _is_global_function(x):
    return isinstance(x, types.FunctionType) and x.__closure__ is None


_Inputs = TypeVar("_Inputs")
_Root = TypeVar("_Root")
_Residual = TypeVar("_Residual")


def implicit_jvp(
    fn_primal: Callable[[_Inputs], tuple[_Root, _Residual]],
    fn_rewrite: Callable[[_Root, _Residual, _Inputs], PyTree],
    inputs: _Inputs,
    tags: frozenset[object],
    linear_solver: lx.AbstractLinearSolver,
    throw: bool = True,
):
    """Rewrites gradients via the implicit function theorem.

    **Arguments:**

    - `fn_primal` is a function `inputs -> (root, residual)`.
    - `fn_rewrite` is a function `(root, residual, inputs) -> arbitrary`.
    - `inputs` is some input PyTree of the primal inputs to the computation.
    - `tags`: any Lineax tags (symmetric, diagonal, ...) for the matrix
        `d(fn_rewrite)/d(root)`.
    - `linear_solver`: an `lx.AbstractLinearSolver`, used to solve the linear problem
        on the backward pass.
    - `throw`: whether the backward linear solve should raise an error if it fails. If
        `False` then a failed linear solve returns a non-finite cotangent instead of
        raising. Note that if the primal `root` is non-finite (e.g. the forward solve
        diverged) then -- regardless of `throw` -- the cotangent is masked at those
        entries rather than raising; see `_implicit_impl_jvp` for details.

    Note that due to limitations with JAX's custom autodiff, both `fn_primal` and
    `fn_rewrite` should be global functions (i.e. they should not capture any JAX array
    via closure, even if it does not participate in autodiff).

    **Returns:**

    This function returns `fn_primal(inputs)`. The first output is the output
    primal, whilst the second is auxiliary information.

    The primals have tangents `-(d(fn_rewrite)/d(root))^-1 d(fn_rewrite)/d(inputs)`,
    evaluated at `(root, residual, inputs)`.
    """
    assert _is_global_function(fn_primal)
    assert _is_global_function(fn_rewrite)
    root, residual = _implicit_impl(
        fn_primal, fn_rewrite, inputs, tags, linear_solver, throw
    )
    return root, jtu.tree_map(eqxi.nondifferentiable_backward, residual)


@eqx.filter_custom_jvp
def _implicit_impl(fn_primal, fn_rewrite, inputs, tags, linear_solver, throw):
    del fn_rewrite, tags, linear_solver, throw
    return jtu.tree_map(jnp.asarray, fn_primal(inputs))


def _assert_false(x):
    assert False


def _is_none(x):
    return x is None


def _for_jac(root, args):
    fn_rewrite, residual, inputs = args
    return fn_rewrite(root, residual, inputs)


@_implicit_impl.def_jvp
def _implicit_impl_jvp(primals, tangents):
    fn_primal, fn_rewrite, inputs, tags, linear_solver, throw = primals
    (
        t_fn_primal,
        t_fn_rewrite,
        t_inputs,
        t_tags,
        t_linear_solver,
        t_throw,
    ) = tangents

    jtu.tree_map(
        _assert_false, (t_fn_primal, t_fn_rewrite, t_tags, t_linear_solver, t_throw)
    )
    del t_fn_primal, t_fn_rewrite, t_tags, t_linear_solver, t_throw
    no_tangent = jtu.tree_map(_is_none, t_inputs, is_leaf=_is_none)
    nondiff, diff = eqx.partition(inputs, no_tangent, is_leaf=_is_none)

    root, residual = implicit_jvp(
        fn_primal, fn_rewrite, inputs, tags, linear_solver, throw
    )

    # If the forward solve diverged then `root` is non-finite. Linearising at such a
    # `root` would feed non-finite inputs into the cotangent linear solve -- and, under
    # reverse-mode, into its transpose, which Lineax always solves with `throw=True`
    # (there is nowhere to pipe a result to). So merely threading `throw` into the
    # forward `linear_solve`, or skipping it with a `lax.cond` (which `vmap` turns into
    # a `select` that evaluates both branches anyway), is not enough on its own.
    #
    # Instead we linearise at a sanitised, finite point so the linear solve always
    # receives finite inputs and so cannot raise, and then mask the cotangent back out
    # at the non-finite entries of `root`. (The mask uses `tree_where` rather than a
    # multiply so that on the reverse pass a finite -- in fact zero -- cotangent is fed
    # into the transposed solve, which would otherwise raise on a non-finite input. This
    # does mean a diverged element contributes a zero, rather than non-finite, reverse
    # cotangent.) The forward primal `root` is returned untouched, so the failure stays
    # visible to the caller as a non-finite value, consistent with the forward pass. The
    # mask is element-wise so that, under `vmap`, one diverged batch element does not
    # taint the cotangents of the others.
    root_finite = jtu.tree_map(jnp.isfinite, root)
    safe_root = tree_where(root_finite, root, tree_full_like(root, 0))

    def _for_jvp(_diff):
        _inputs = eqx.combine(_diff, nondiff)
        return fn_rewrite(safe_root, residual, _inputs)

    operator = lx.JacobianLinearOperator(
        _for_jac, safe_root, (fn_rewrite, residual, inputs), tags=tags
    )
    _, jvp_diff = jax.jvp(_for_jvp, (diff,), (t_inputs,))

    solved = (
        -(lx.linear_solve(operator, jvp_diff, linear_solver, throw=throw).value ** ω)
    ).ω
    t_root = tree_where(root_finite, solved, tree_full_like(root, jnp.nan))
    if hasattr(jax.custom_derivatives, "zero_from_primal"):
        t_residual = jax.custom_derivatives.zero_from_primal(  # pyright: ignore[reportGeneralTypeIssues]
            residual, symbolic_zeros=True
        )
    else:
        t_residual = tree_full_like(residual, 0)

    return (root, residual), (t_root, t_residual)
