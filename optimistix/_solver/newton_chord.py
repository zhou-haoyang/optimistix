from collections.abc import Callable
from typing import Any, Generic, TYPE_CHECKING

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, Fn, Out, Y
from .._misc import (
    cauchy_termination,
    filter_cond,
    max_norm,
    tree_dtype,
    tree_full_like,
    tree_where,
)
from .._root_find import AbstractRootFinder
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS
from .gauss_newton import _make_f_info
from .learning_rate import LearningRate


def _small(diffsize: Scalar) -> Bool[Array, " "]:
    # TODO(kidger): make a more careful choice here -- the existence of this
    # function is pretty ad-hoc.
    resolution = 10 ** (2 - jnp.finfo(diffsize.dtype).precision)
    return diffsize < resolution


def _diverged(rate: Scalar) -> Bool[Array, " "]:
    return jnp.invert(jnp.isfinite(rate)) | (rate > 2)


def _converged(factor: Scalar, tol: float) -> Bool[Array, " "]:
    return (factor > 0) & (factor < tol)


class _NewtonChordState(eqx.Module, Generic[Y, Aux]):
    # Search infrastructure (mirrors _GaussNewtonState)
    first_step: Bool[Array, ""]
    y_eval: Y                                          # proposed next point
    search_state: Any                                  # SearchState (type-erased)
    # Accepted-point info
    f_info: FunctionInfo.ResidualJac                   # replaces old `f: PyTree[Array]`
    aux: Aux
    # For Hairer/Wanner (cauchy_termination=False)
    diff: Y
    diffsize: Scalar
    diffsize_prev: Scalar
    # Status
    result: RESULTS
    step: Scalar
    accept: Bool[Array, ""]
    y_diff: Y
    # Chord only: fixed Jacobian at y0 (Newton: None)
    linear_state: tuple[lx.AbstractLinearOperator, PyTree] | None


class _NoAux(eqx.Module):
    fn: Callable

    def __call__(self, y, args):
        out, aux = self.fn(y, args)
        del aux
        return out


class _AbstractNewtonChord(AbstractRootFinder[Y, Out, Aux, _NewtonChordState[Y, Aux]]):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    kappa: float = 1e-2
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    cauchy_termination: bool = True
    search: AbstractSearch = LearningRate(1.0)

    _is_newton: AbstractClassVar[bool]

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _NewtonChordState[Y, Aux]:
        jac_mode = options.get("jac", "fwd")
        del options

        if self._is_newton:
            linear_state = None
            # Build dummy f_info using the same Jacobian structure that step() will use
            f_info_struct, _ = eqx.filter_eval_shape(
                _make_f_info, fn, y, args, tags, jac_mode
            )
            f_info = tree_full_like(f_info_struct, 0, allow_static=True)
        else:
            # Chord: compute fixed Jacobian once at initial y
            jac = lx.JacobianLinearOperator(_NoAux(fn), y, args, tags=tags)
            jac = lx.linearise(jac)
            init_later_state = self.linear_solver.init(jac, options={})
            init_later_state = lax.stop_gradient(init_later_state)
            linear_state = (jac, init_later_state)
            # Build dummy f_info using the chord Jacobian so the static structure
            # matches what accepted() will produce in step()
            f_info = FunctionInfo.ResidualJac(tree_full_like(f_struct, 0), jac)
            f_info_struct = eqx.filter_eval_shape(lambda: f_info)

        dtype = tree_dtype(f_struct)
        return _NewtonChordState(
            first_step=jnp.array(True),
            y_eval=y,
            search_state=self.search.init(y, f_info_struct),
            f_info=f_info,
            aux=tree_full_like(aux_struct, 0),
            diff=tree_full_like(y, jnp.inf),
            diffsize=jnp.array(jnp.inf, dtype=dtype),
            diffsize_prev=jnp.array(1.0, dtype=dtype),
            result=RESULTS.successful,
            step=jnp.array(0),
            accept=jnp.array(False),
            y_diff=tree_full_like(y, 0),
            linear_state=linear_state,
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonChordState[Y, Aux],
        tags: frozenset[object],
    ) -> tuple[Y, _NewtonChordState[Y, Aux], Aux]:
        lower = options.get("lower")
        upper = options.get("upper")
        jac_mode = options.get("jac", "fwd")
        del options

        # ------------------------------------------------------------------
        # Part 1: Evaluate fn at the current proposed point (state.y_eval)
        # ------------------------------------------------------------------
        if self._is_newton:
            f_eval_info, aux_eval = _make_f_info(
                fn, state.y_eval, args, tags, jac_mode
            )
            # Jaxpr identity trick: the jaxpr inside FunctionLinearOperator is
            # compared by identity in filter_cond. Combine dynamic arrays from the
            # freshly computed Jacobian with static parts from the stored f_info so
            # both branches of filter_cond see the same static structure.
            dynamic = eqx.filter(f_eval_info.jac, eqx.is_array)
            static = eqx.filter(state.f_info.jac, eqx.is_array, inverse=True)
            jac_combined = eqx.combine(dynamic, static)
            f_eval_info = eqx.tree_at(lambda f: f.jac, f_eval_info, jac_combined)
        else:
            # Chord: only evaluate the residual; reuse the fixed Jacobian from init.
            # f_eval_info only needs as_min() for the search, so Residual suffices.
            fx_eval, aux_eval = fn(state.y_eval, args)
            f_eval_info = FunctionInfo.Residual(fx_eval)

        # ------------------------------------------------------------------
        # Part 2: Ask the search whether to accept state.y_eval
        # ------------------------------------------------------------------
        step_size, accept, search_result, search_state = self.search.step(
            state.first_step,
            y,
            state.y_eval,
            state.f_info,
            f_eval_info,
            state.search_state,
        )

        # ------------------------------------------------------------------
        # Part 3: Accept / reject via filter_cond
        # ------------------------------------------------------------------
        def accepted(linear_state):
            new_y = state.y_eval

            if self._is_newton:
                # Solve the linear system at the accepted point
                sol = lx.linear_solve(
                    f_eval_info.jac,
                    f_eval_info.residual,
                    self.linear_solver,
                    throw=False,
                )
                new_f_info = f_eval_info  # ResidualJac at the accepted point
            else:
                # Chord: solve using the fixed Jacobian (precomputed in init)
                jac_chord, chord_lin_state = linear_state  # pyright: ignore
                chord_lin_state = lax.stop_gradient(chord_lin_state)
                sol = lx.linear_solve(
                    jac_chord,
                    fx_eval,
                    self.linear_solver,
                    state=chord_lin_state,
                    throw=False,
                )
                # Wrap residual + chord Jacobian so f_info keeps type ResidualJac,
                # matching the rejected branch and state initialisation.
                new_f_info = FunctionInfo.ResidualJac(fx_eval, jac_chord)

            diff = sol.value
            y_diff = (new_y**ω - y**ω).ω
            return (new_y, new_f_info, aux_eval, diff, y_diff, RESULTS.promote(sol.result))

        def rejected(linear_state):
            del linear_state
            return (y, state.f_info, state.aux, state.diff, state.y_diff, state.result)

        new_y, new_f_info, new_aux, diff, y_diff, step_result = filter_cond(
            accept, accepted, rejected, state.linear_state
        )

        # ------------------------------------------------------------------
        # Part 4: Compute next proposed point y_eval = new_y - step_size * diff
        # ------------------------------------------------------------------
        y_eval = (new_y**ω - step_size * diff**ω).ω
        if lower is not None:
            y_eval = jtu.tree_map(lambda a, b: jnp.clip(a, min=b), y_eval, lower)
        if upper is not None:
            y_eval = jtu.tree_map(lambda a, b: jnp.clip(a, max=b), y_eval, upper)

        # ------------------------------------------------------------------
        # Part 5: Compute diffsize for Hairer/Wanner termination tracking
        # ------------------------------------------------------------------
        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        with jax.numpy_dtype_promotion("standard"):
            diffsize = self.norm((diff**ω / scale**ω).ω)

        result = RESULTS.where(
            search_result == RESULTS.successful, step_result, search_result
        )

        prev_aux = tree_where(state.first_step, new_aux, state.aux)
        new_state = _NewtonChordState(
            first_step=jnp.array(False),
            y_eval=y_eval,
            search_state=search_state,
            f_info=new_f_info,
            aux=new_aux,
            diff=diff,
            diffsize=jnp.asarray(diffsize, dtype=state.diffsize.dtype),
            diffsize_prev=state.diffsize,
            result=result,
            step=state.step + 1,
            accept=accept,
            y_diff=y_diff,
            linear_state=state.linear_state,
        )
        return new_y, new_state, prev_aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonChordState[Y, Aux],
        tags: frozenset[object],
    ):
        del fn, args, options, tags
        # Only check termination when the last step was accepted.
        # On rejection the search will propose a new step_size; no convergence yet.
        if self.cauchy_termination:
            # For root-finding we compare f(y) against 0, not against f(y_prev).
            # Only atol matters in f-space (rtol * 0 = 0).
            new_y = (y**ω + state.y_diff**ω).ω
            terminate = state.accept & cauchy_termination(
                self.rtol,
                self.atol,
                self.norm,
                new_y,
                state.y_diff,
                jtu.tree_map(jnp.zeros_like, state.f_info.residual),
                state.f_info.residual,
            )
        else:
            new_y = (y**ω + state.y_diff**ω).ω
            scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
            with jax.numpy_dtype_promotion("standard"):
                new_diffsize = self.norm((state.diff**ω / scale**ω).ω)
            rate = new_diffsize / state.diffsize_prev
            factor = new_diffsize * rate / (1 - rate)
            at_least_two = state.step >= 2
            small = _small(new_diffsize)
            diverged = _diverged(rate)
            converged = _converged(factor, self.kappa)
            terminate = state.accept & at_least_two & (small | diverged | converged)
        return terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _NewtonChordState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}


class Newton(_AbstractNewtonChord[Y, Out, Aux]):
    """Newton's method for root finding. Also sometimes known as Newton--Raphson.

    Unlike the SciPy implementation of Newton's method, the Optimistix version also
    works for vector-valued (or PyTree-valued) `y`.

    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`.
    """

    _is_newton = True


class Chord(_AbstractNewtonChord[Y, Out, Aux]):
    """The Chord method of root finding.

    This is equivalent to the Newton method, except that the Jacobian is computed only
    once at the initial point `y0`, and then reused throughout the computation. This is
    a useful way to cheapen the solve, if `y0` is expected to be a good initial guess
    and the target function does not change too rapidly. (For example this is the
    standard technique used in implicit Runge--Kutta methods, when solving differential
    equations.)

    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. Should be a
        PyTree of arrays each broadcastable to the corresponding element of `y`. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_newton = False


_init_doc = """**Arguments:**

- `rtol`: Relative tolerance for terminating the solve.
- `atol`: Absolute tolerance for terminating the solve.
- `norm`: The norm used to determine the difference between two iterates in the
    convergence criteria. Should be any function `PyTree -> Scalar`. Optimistix
    includes three built-in norms: [`optimistix.max_norm`][],
    [`optimistix.rms_norm`][], and [`optimistix.two_norm`][].
- `kappa`: A tolerance for early convergence check when `cauchy_termination=False`.
- `linear_solver`: The linear solver used to compute the Newton step.
- `cauchy_termination`: When `True`, use the Cauchy termination condition, that
    two adjacent iterates should have a small difference between them. This is usually
    the standard choice when solving general root finding problems. When `False`, use
    a procedure which attempts to detect slow convergence, and quickly fail the solve
    if so. This is useful when iteratively performing the root-find, refining the
    target problem for those which fail. This comes up when solving differential
    equations with adaptive step sizing and implicit solvers. The exact procedure is as
    described in Section IV.8 of Hairer & Wanner, "Solving Ordinary Differential
    Equations II".
- `search`: The search strategy used to determine the step size and whether to accept
    a proposed iterate. Defaults to [`optimistix.LearningRate`]`(1.0)`, which always
    accepts with a full Newton step. Can be set to e.g.
    [`optimistix.BacktrackingArmijo`]`()` for a line search (Newton only; Chord uses
    an approximate gradient based on its fixed initial Jacobian).
"""

Newton.__init__.__doc__ = _init_doc
Chord.__init__.__doc__ = _init_doc
