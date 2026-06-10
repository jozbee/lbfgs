"""Stripped lbfgs implementation, in jax.

Notes
-----
From Nocedal and Wright (2006):
* Equation (3.59): cubic interpolation formula
* Algorithm 3.6: zoom subroutine (for line search)
* Algorithm 3.5: line search
* Equation (7.20): scaling for H_k^0
* Algorithm 7.4: efficient matrix vector product
* Algorithm 7.5: L-BFGS
"""

from __future__ import annotations

import copy
import dataclasses
import typing as tp
import jax
import jax.numpy as jnp


# fun(params, x) -> (value, grad)
fun_tp: tp.TypeAlias = tp.Callable[
    [tp.Any, jax.Array], tuple[jax.Array, jax.Array]
]


def _static_field() -> tp.Any:
    return dataclasses.field(metadata=dict(static=True))


def _dyn_field() -> tp.Any:
    return dataclasses.field()


def cubic_interp(
    alpha0: jax.Array,
    phi0: jax.Array,
    phip0: jax.Array,
    alpha1: jax.Array,
    phi1: jax.Array,
    phip1: jax.Array,
) -> jax.Array:
    r"""Cubic interpolation for line search.
    
    Notes
    -----
    Cf. equation (3.59) from [NW06].
    Correspondence:
    * alpha0 -> $alpha_{i - 1}$
    * phi0 -> $\phi(\alpha_{i - 1})$
    * phip0 -> $\phi'(\alpha_{i - 1})$
    * alpha1 -> $alpha_i$
    * phi1 -> $\phi(\alpha_i)$
    * phip1 -> $\phi'(\alpha_i)$
    Returns: $alpha_{i + 1}$.
    """
    d1 = phip0 + phip1 - 3.0 * (phi0 - phi1) / (alpha0 - alpha1)
    disc = d1**2 - phip0 * phip1  # discriminant
    d2 = jax.lax.cond(
        disc > 0.0,
        lambda: jnp.sign(alpha1 - alpha0) * jnp.sqrt(disc),
        lambda: 0.0,
    )
    frac = (phip1 + d2 - d1) / (phip1 - phip0 + 2.0 * d2)
    return alpha1 - (alpha1 - alpha0) * frac


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ParamsZoom:
    """Zoom parameters (documented in the `Notes` section of `zoom`)."""
    c1: float = _static_field() # e.g., 10**-4
    c2: float = _static_field() # e.g., 0.9
    max_iter: int = _static_field() # e.g., `np.iinfo(np.int64).max`
    fun: fun_tp = _static_field()
    fun_params: jax.Array = _dyn_field()
    x0: jax.Array = _dyn_field()
    p: jax.Array = _dyn_field()


def zoom(
    params: ParamsZoom,
    phi_zero: jax.Array,
    phip_zero: jax.Array,
    grad_f_hi: jax.Array,
    alpha_lo: jax.Array,  # e.g., 0.0
    phi_lo: jax.Array,
    phip_lo: jax.Array,
    alpha_hi: jax.Array,  # e.g., 1.0
    phi_hi: jax.Array,
    phip_hi: jax.Array,
    unroll: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Zoom line search.

    Notes
    -----
    Cf. Algorithm 3.6 from [NW06].
    Note that `phi` evaluates both the function and its derivative,
    simultaneously.
    Correspondence:
    * c1 -> $c_1$
    * c2 -> $c_2$
    * fun -> $\phi(\alpha) = fun(fun_params, x_0 + \alpha p)$
    * phi_zero -> $\phi(0)$
    * phip_zero -> $\phi'(0)$
    * grad_f_zero -> $\nabla f(x_0)$
    * alpha_lo -> $\alpha_{\mathrm{lo}}$
    * phi_lo -> $\phi(\alpha_{\mathrm{lo}})$
    * phip_lo -> $\phi'(\alpha_{\mathrm{lo}})$
    * alpha_hi -> $\alpha_{\mathrm{hi}}$
    * phi_hi -> $\phi(\alpha_{\mathrm{hi}})$
    * phip_hi -> $\phi'(\alpha_{\mathrm{hi}})$
    Returns: $(\alpha_*, \phi(\alpha_*), \phi'(\alpha_*))$

    In practice, `alpha_lo` is 0.0, which means that the calling structure has
    some repetition, i.e., `phi_lo == phi_zero`, `phip_lo == phip_zero`.
    """

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class ZoomState:
        alpha_lo: jax.Array
        phi_lo: jax.Array
        phip_lo: jax.Array
        alpha_hi: jax.Array
        phi_hi: jax.Array
        phip_hi: jax.Array

        alpha_j: jax.Array
        phi_j: jax.Array
        phip_j: jax.Array
        grad_f_j: jax.Array
        iter: jax.Array

        # we always enforce at least one iteration, if max_iter > 0
        # seems to be productive in practice, but a little more expensive...
        # more subtle stopping criteria seems subtle...
        is_done: jax.Array

        def update(self, **kwargs) -> "ZoomState":
            res = copy.copy(self)  # shallow
            res.__dict__.update(**kwargs)
            return res

    def cond_fun(state: ZoomState):
        return (~state.is_done) & (state.iter < params.max_iter)

    def body_fun(state: ZoomState):
        s = state
        p = params
        s.iter += 1
        s.alpha_j = cubic_interp(
            s.alpha_lo, s.phi_lo, s.phip_lo,
            s.alpha_hi, s.phi_hi, s.phip_hi
        )
        s.phi_j, s.grad_f_j = p.fun(
            p.fun_params, p.x0 + s.alpha_j * p.p
        )
        s.phip_j = jnp.dot(s.grad_f_j, p.p)  # directional derivative

        """
        # the following is transcribed into jax, below

        if phi_j >= phi_zero + c1 * alpha_j * phip_zero or phi_j >= phi_lo:
            alpha_hi = alpha_j
            phi_hi = phi_j
            phip_hi = phip_j
        else:
            if abs(phip_j) <= -c2 * phip_zero:
                is_done = True
            else:
                if phip_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                    phi_hi = phi_lo
                    phip_hi = phip_lo
                alpha_lo = alpha_j
                phi_lo = phi_j
                phip_lo = phip_j
        """

        wolfe1 = (
            (s.phi_j >= phi_zero + p.c1 * s.alpha_j * phip_zero) |
            (s.phi_j >= s.phi_lo)
        )
        wolfe2 = jnp.abs(s.phip_j) <= -p.c2 * phip_zero
        flip_hi = s.phip_j * (s.alpha_hi - s.alpha_lo) >= 0

        j2hi = dict(
            alpha_hi=s.alpha_j, phi_hi=s.phi_j, phip_hi=s.phip_j
        )
        lo2hi = dict(
            alpha_hi=s.alpha_lo, phi_hi=s.phi_lo, phip_hi=s.phip_lo
        )
        j2lo = dict(
            alpha_lo=s.alpha_j, phi_lo=s.phi_j, phip_lo=s.phip_j
        )

        s = jax.lax.cond(
            wolfe1,
            lambda: s.update(**j2hi),  # wolfe1
            lambda: jax.lax.cond(
                wolfe2,
                lambda: s.update(_is_done=jnp.array(True)),  # wolfe2
                lambda: jax.lax.cond(
                    flip_hi,
                    lambda: s.update(**lo2hi, **j2lo),  # flip
                    lambda: s.update(**j2lo),  # default
                ),
            ),
        )

        return s

    zoom_state = ZoomState(
        alpha_lo=alpha_lo,
        phi_lo=phi_lo,
        phip_lo=phip_lo,
        alpha_hi=alpha_hi,
        phi_hi=phi_hi,
        phip_hi=phip_hi,
        alpha_j=alpha_hi,
        phi_j=phi_hi,
        phip_j=phip_hi,
        grad_f_j=grad_f_hi,
        iter=jnp.array(0),
        is_done=jnp.array(False),
    )
    if not unroll:
        res_state = jax.lax.while_loop(cond_fun, body_fun, zoom_state)
    else:
        res_state = zoom_state
        for _ in range(params.max_iter):
            res_state = body_fun(res_state)

    # also handles case when `is_done == False`
    alpha_star = res_state.alpha_j
    phi_star = res_state.phi_j
    grad_f_star = res_state.grad_f_j
    return alpha_star, phi_star, grad_f_star


def gamma_scale(s0: jax.Array, y0: jax.Array) -> jax.Array:
    r"""Scaling factor for L-BFGS matrices.

    Notes
    -----
    Cf. equation (7.20) from [NW06].
    Correspondence:
    * s0 -> $s_{k - 1}$
    * y0 -> $y_{k - 1}$
    Returns: $\gamma_k$.
    """
    assert len(s0.shape) == 1 and len(y0.shape) == 1
    assert s0.shape == y0.shape
    return jnp.dot(s0, y0) / jnp.dot(y0, y0)


def hess_vec_product(
    q: jax.Array,
    s: jax.Array,
    y: jax.Array,
    rho: jax.Array,
    m: jax.Array,
    unroll: bool = False,
) -> jax.Array:
    r"""Efficient L-BFGS matrix vector product.

    Notes
    -----
    Cf. Algorithm 7.4 from [NW06].
    Correspondence:
    * q -> $q = \nabla f_k$
    * s[i] -> $s_{k - m}$
    * y[i] -> $y_{k - m}$
    * rho[i] -> $\rho_{k - m} = 1 / (y_{k - m}^T s_k)$
    * m -> $k$ (and the $m = k$, so really BFGS)
    * r -> $r$
    Return
    """
    assert len(s.shape) == 2 and s.shape == y.shape
    assert len(rho.shape) == 1 and rho.shape[0] == s.shape[0]
    assert len(q.shape) == 1 and q.shape[0] == s.shape[1]
    # assert m >= 1
    # assert s.shape[0] >= m

    def backward_cond(state):
        i, _, _ = state
        return i >= 0  # break, when negative

    def backward_body(state):
        i, q, alpha = state
        alpha_i = rho[i] * jnp.dot(s[i], q)
        q = q - alpha_i * y[i]
        return i - 1, q, alpha.at[i].set(alpha_i)

    def forward_cond(state):
        i, _ = state
        return i <= m - 2

    def forward_body(state):
        i, r = state
        beta = rho[i] * jnp.dot(y[i], r)
        r = r + s[i] * (alpha[i] - beta)
        return i + 1, r

    alpha = jnp.zeros(s.shape[0] - 1)
    gamma_k = gamma_scale(s[m - 1], y[m - 1])

    if not unroll:
        _, q, alpha = jax.lax.while_loop(
            backward_cond, backward_body, (m - 2, q, alpha)
        )
        r = gamma_k * q
        _, r = jax.lax.while_loop(forward_cond, forward_body, (jnp.array(0), r))
    else:
        # unrolling these for loops is somewhat subtle in jax's back
        #  propogation autodiff
        # namely, we need to perform extra loops than desired in general
        # so, we ensure that extra loop iterations add zero
        m_unroll = s.shape[0]

        def zeroed(arr):
            m_indices = jnp.arange(m_unroll) <= m - 2
            if len(arr.shape) == 2:
                m_indices = jnp.squeeze(
                    jnp.tile(m_indices.reshape(-1, 1), reps=(1, arr.shape[1]))
                )
            return jnp.where(m_indices, arr, 0.0)

        s = zeroed(s)
        y = zeroed(y)
        rho = zeroed(rho)
        for i in range(m_unroll - 2, -1, -1):
            _, q, alpha = backward_body((i, q, alpha))
        r = gamma_k * q
        for i in range(0, m_unroll - 2 + 1):
            _, r = forward_body((i, r))

    return r


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class OptParamsLBFGS:
    """Optimization parameters for L-BFGS ipmlementation.

    Attributes
    ----------
    fun :
        Optimization function, with calling structure `fun(params, x)`.
        (The parameters position is required, even if there are no parameters.)
    max_iter :
        Maximum number of L-BFGS iterations.
        General recommendation: 16.
    max_ls :
        Maximum number of zoom line search iterations.
        If `max_ls > 0`, then at least one interpolation is always guaranteed.
        General recommendation: 8.
    tol :
        Termination tolerance.
        If the norm of the gradient is less than `tol`, then terminate.
        General recommendation: 1e-5.
    c1 :
        Zoom search paramter for strong Wolfe conditions.
        Cf. equation (3.7a) from [NW06].
        Note that $0 < c_1 < c_2 < 1$.
        General recommendation: 1e-4.
    c2 :
        Zoom search paramter for strong Wolfe conditions.
        Cf. equation (3.7b) from [NW06].
        Note that $0 < c_1 < c_2 < 1$.
        General recommendation: 0.9.
    """

    fun: fun_tp = _static_field()
    max_iter: int = _static_field()
    max_ls: int = _static_field()
    tol: float = _static_field()
    c1: float = _static_field()
    c2: float = _static_field()


def lbfgs(
    opt_params: OptParamsLBFGS,
    x0: jax.Array,
    fun_params: jax.Array,
    unroll: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Stripped LBFGS routine for jax.

    In reality, this is just a BFGS routine with minimal error checking.
    The algorithm is NOT meant to be robust.
    It is designed for real-time applications.
    Essentially, the algorithm is designed to be a glorified gradient descent,
    but considers some initial curvature.

    Parameters
    ----------
    opt_params :
        Optimization parameters, cf. `OptParamsLBFGS` for documentation.
        This includes the optimization function.
    x0 :
        Initial guess.
    fun_params :
        Parameters for cost function in `opt_params`.
        The cost function must have parameters, even if they are ignored.
    unroll :
        False to check gradient condition for early exit.
        True to use static-iterations, for reverse-mode differentation.

    Returns
    -------
    The triple (minimizer, value at minimizer, gradient at minimizer).

    Notes
    -----
    Usually `unroll` should be set to False.
    Allowing `unroll == True` allows reverse-mode differentiation, which is
    desired for approximate minimization calls.
    Namely, implementing an implicit function theorem for the lbfgs routine
    would be more accurate for computing derivatives, unless the lbfgs routine
    returns results that are far from the (local) minimizer.
    """
    assert len(x0.shape) == 1 and x0.size >= 1
    assert opt_params.tol > 0
    assert opt_params.max_iter >= 1

    m = opt_params.max_iter

    iter = jnp.array(0)
    fun0, grad0 = opt_params.fun(fun_params, x0)
    s = jnp.zeros(shape=(m, x0.size))
    y = jnp.zeros(shape=(m, x0.size))
    rho = jnp.zeros(shape=(m,))

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class LBFGSState:
        x0: jax.Array
        fun0: jax.Array
        grad0: jax.Array
        iter: jax.Array
        s: jax.Array
        y: jax.Array
        rho: jax.Array

    def while_cond(state: LBFGSState) -> jax.Array:
        st = state
        grad_cond = jnp.dot(st.grad0, st.grad0) >= opt_params.tol**2
        return (st.iter < m) & grad_cond

    def while_body(state: LBFGSState) -> LBFGSState:
        st = state
        p1 = -hess_vec_product(st.grad0, st.s, st.y, st.rho, st.iter, unroll)
        p1 = jax.lax.cond(
            st.iter == 0,
            lambda: -st.grad0 / jnp.linalg.norm(st.grad0),
            lambda: p1,
        )

        def phi(alpha: float | jax.Array) -> tuple[jax.Array, jax.Array]:
            return opt_params.fun(fun_params, st.x0 + alpha * p1)

        alpha_lo = jnp.array(0.0)
        phi_zero, grad_f_zero = phi(alpha_lo)
        phip_zero = jnp.dot(grad_f_zero, p1)
        alpha_hi = jnp.array(1.0)
        phi_hi, grad_f_hi = phi(alpha_hi)
        phip_hi = jnp.dot(grad_f_hi, p1)

        
        alpha1, fun1, grad1 = zoom(
            params=ParamsZoom(
                c1=opt_params.c1,
                c2=opt_params.c2,
                max_iter=opt_params.max_ls,
                fun=opt_params.fun,
                fun_params=fun_params,
                x0=st.x0,
                p=p1,
            ),
            phi_zero=phi_zero,
            phip_zero=phip_zero,
            grad_f_hi=grad_f_hi,
            alpha_lo=alpha_lo,
            phi_lo=phi_zero,
            phip_lo=phip_zero,
            alpha_hi=alpha_hi,
            phi_hi=phi_hi,
            phip_hi=phip_hi,
            unroll=unroll,
        )

        x1 = st.x0 + alpha1 * p1
        st.s = st.s.at[st.iter].set(x1 - st.x0)
        st.y = st.y.at[st.iter].set(grad1 - st.grad0)
        rho_iter = 1.0 / jnp.dot(st.s[st.iter], st.y[st.iter])
        st.rho = st.rho.at[st.iter].set(rho_iter)

        # minimal error checking: abort early if there is a problem
        st.x0, st.fun0, st.grad0, st.iter = jax.lax.cond(
            jnp.isnan(rho_iter) | jnp.any(jnp.isnan(x1)),
            lambda: (st.x0, st.fun0, st.grad0, m),
            lambda: (x1, fun1, grad1, st.iter + 1),
        )

        return st

    state0 = LBFGSState(x0, fun0, grad0, iter, s, y, rho)
    if not unroll:
        res = jax.lax.while_loop(while_cond, while_body, state0)
    if unroll:
        res = state0
        for _ in range(opt_params.max_iter):
            res = while_body(res)
    return res.x0, res.fun0, res.grad0
