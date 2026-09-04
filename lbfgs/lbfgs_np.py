"""Simple lbfgs implementation, in numpy.

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

import warnings
import dataclasses
import numpy as np
import typing as tp

floating: tp.TypeAlias = float | np.floating


def cubic_interp(
    alpha0: floating,
    phi0: floating,
    phip0: floating,
    alpha1: floating,
    phi1: floating,
    phip1: floating,
) -> floating:
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
    if disc > 0:
        d2 = np.sign(alpha1 - alpha0) * np.sqrt(d1**2 - phip0 * phip1)
    else:
        d2 = 0
    frac = (phip1 + d2 - d1) / (phip1 - phip0 + 2.0 * d2)
    return alpha1 - (alpha1 - alpha0) * frac


def zoom(
    # params
    c1: floating,  # e.g., 10**-4
    c2: floating,  # e.g., 0.9
    max_iter: int,  # e.g., `np.iinfo(np.int64).max`
    phi: tp.Callable[[floating], tuple[floating, floating, np.ndarray]],
    phi_zero: floating,
    phip_zero: floating,
    grad_f_hi: np.ndarray,
    # updates
    alpha_lo: floating,  # e.g., 0.0
    phi_lo: floating,
    phip_lo: floating,
    alpha_hi: floating,  # e.g., 1.0
    phi_hi: floating,
    phip_hi: floating,
) -> tuple[floating, floating, np.ndarray]:
    r"""Zoom line search.

    Notes
    -----
    Cf. Algorithm 3.6 from [NW06].
    Note that `phi` evaluates both the function and its derivative,
    simultaneously.
    Correspondence:
    * c1 -> $c_1$
    * c2 -> $c_2$
    * phi -> $\alpha \mapsto (\phi(\alpha), \phi'(\alpha), (\nabla f)(x_0 + \alpha p))$
      * $\phi(\alpha) = f(x_0 + \alpha p)$
      * $\phi'(\alpha) = p \cdot (\nabla f)(x_0 + \alpha p)$
      * Namely, we return an intermediate result, because returning  the vector
         $\nabla f$ is desirable for the algorithm, for efficiency
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
    alpha_j = 1.0
    phi_j = phi_hi
    phip_j = phip_hi
    grad_f_j = grad_f_hi
    iter = 0

    # we always enforce at least one iteration, if max_iter > 0
    # seems to be productive in practice, but a little more expensive...
    # more subtle stopping criteria seems subtle...
    is_done = False

    while not is_done and iter < max_iter:
        iter += 1
        alpha_j = cubic_interp(
            alpha_lo, phi_lo, phip_lo, alpha_hi, phi_hi, phip_hi)
        phi_j, phip_j, grad_f_j = phi(alpha_j)

        if phi_j >= phi_zero + c1 * alpha_j * phip_zero or phi_j >= phi_lo:
            alpha_hi = alpha_j
            phi_hi = phi_j
            phip_hi = phip_j
        else:
            if np.abs(phip_j) <= -c2 * phip_zero:
                is_done = True
            else:
                if phip_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                    phi_hi = phi_lo
                    phip_hi = phip_lo
                alpha_lo = alpha_j
                phi_lo = phi_j
                phip_lo = phip_j

    # also handles case when `is_done == False`
    alpha_star, phi_star, grad_f_star = alpha_j, phi_j, grad_f_j
    return alpha_star, phi_star, grad_f_star


def gamma_scale(s0: np.ndarray, y0: np.ndarray) -> floating:
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
    return np.dot(s0, y0) / np.dot(y0, y0)


def hess_vec_product(
    q: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    rho: np.ndarray,
    m: int,
) -> np.ndarray:
    r"""Efficient L-BFGS matrix vector product.

    Notes
    -----
    Cf. Algorithm 7.4 from [NW06].
    Correspondence:
    * q -> $q = \nabla f_k$
    * s[i] -> $s_{k - m}$
    * y[i] -> $y_{k - m}$
    * rho[i] -> $\rho_{k - m} 1 / (y_{k - m}^T s_k)$
    * m -> $k$ (and the $m = k$, so really BFGS)
    * r -> $r$
    Return
    """
    assert len(s.shape) == 2 and s.shape == y.shape
    assert len(rho.shape) == 1 and rho.shape[0] == s.shape[0]
    assert len(q.shape) == 1 and q.shape[0] == s.shape[1]
    assert m >= 1
    assert s.shape[0] >= m

    alpha = np.empty(shape=(m - 1,), dtype=float)
    for i in range(m - 2, -1, -1):  # k - 1, ..., k - m
        alpha[i] = rho[i] * np.dot(s[i], q)
        q = q - alpha[i] * y[i]

    gamma_k = gamma_scale(s[m - 1], y[m - 1])
    r = gamma_k * q
    for i in range(m - 1):
        beta = rho[i] * np.dot(y[i], r)
        r = r + s[i] * (alpha[i] - beta)

    return r


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

    fun: tp.Callable[[np.ndarray, np.ndarray], tuple[floating, np.ndarray]]
    max_iter: int
    max_ls: int
    tol: floating
    c1: floating
    c2: floating


def lbfgs(
    opt_params: OptParamsLBFGS,
    x0: np.ndarray,
    fun_params: np.ndarray,
) -> tuple[np.ndarray, floating, np.ndarray]:
    """Stripped LBFGS routine for numpy.

    In reality, this is just a BFGS routine with minimal error checking.
    The algorithm is not meant to be robust.
    It is designed for real time applications.
    Essentially, the algorithm is designed to be a glorified gradient descent.

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

    Returns
    -------
    The triple (minimizer, value at minimizer, gradient at minimizer).

    Notes
    -----
    One call evaluates `opt_params.fun` (value and gradient together) at
    most

        1 + (1 + max_ls) * max_iter

    times: once for the initial point, then per L-BFGS iteration once at
    the unit step `alpha = 1` and up to `max_ls` times inside `zoom`.
    The line search reuses `fun0`/`grad0` for `phi(0)` instead of
    re-evaluating it, which would make the count
    `1 + (2 + max_ls) * max_iter`.
    Unlike the jax version, `zoom` here can stop early on the strong Wolfe
    conditions, and the gradient tolerance can stop the outer loop early,
    so the count is an upper bound.
    """
    assert len(x0.shape) == 1 and x0.size >= 1
    assert opt_params.tol > 0
    assert opt_params.max_iter >= 1

    m = opt_params.max_iter

    iter = 0
    fun0, grad0 = opt_params.fun(fun_params, x0)
    s = np.empty(shape=(m, x0.size))
    y = np.empty(shape=(m, x0.size))
    rho = np.empty(shape=(m,))

    while iter < m and np.dot(grad0, grad0) >= opt_params.tol**2:
        if iter == 0:
            # first iteration: scaled gradient descent
            p1 = -grad0
            p1 = p1 / np.linalg.norm(p1)  # important enough to get a line
        else:
            p1 = -hess_vec_product(grad0, s, y, rho, iter)

        def phi(alpha):
            res = opt_params.fun(fun_params, x0 + alpha * p1)
            return res[0], np.dot(res[1], p1), res[1]

        # phi(0) *is* (fun0, grad0): `x0 + 0.0 * p1` is bitwise `x0` and
        # `fun` is deterministic, so re-evaluating it would cost one call
        # per iteration for the same result.  Here that result is bitwise
        # the same, because the arithmetic around it is unchanged; the
        # jax version compiles the loop as a whole, where dropping the
        # evaluation can still move the accepted step by a few ulp.
        phi_zero = fun0
        phip_zero = np.dot(grad0, p1)
        alpha_hi = 1.0
        phi_hi, phip_hi, grad_f_hi = phi(alpha_hi)
        c1, c2, max_ls = opt_params.c1, opt_params.c2, opt_params.max_ls
        zoom_params0 = [c1, c2, max_ls, phi, phi_zero, phip_zero, grad_f_hi]
        zoom_params1 = [0.0, phi_zero, phip_zero, alpha_hi, phi_hi, phip_hi]
        alpha1, fun1, grad1 = zoom(*(zoom_params0 + zoom_params1))

        x1 = x0 + alpha1 * p1

        s[iter] = x1 - x0
        y[iter] = grad1 - grad0
        rho[iter] = 1.0 / np.dot(s[iter], y[iter])  # might warn?

        if np.any(np.isnan(x1)):
            # break before something bad happens
            # there is minimal error checking, on purpose
            warnings.warn(
                f"detected nan in lbfgs, iter={iter}", category=UserWarning
            )
            break

        x0 = x1
        fun0 = fun1
        grad0 = grad1
        iter += 1

    return x0, fun0, grad0
