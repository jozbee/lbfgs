"""Simple lbfgs implementation, in numpy.

Notes
----------
From Nocedal and Wright (2006):
* Equation (3.59): cubic interpolation formula
* Algorithm 3.6: zoom subroutine (for line search)
* Algorithm 3.5: line search
* Equation (7.20): scaling for H_k^0
* Algorithm 7.4: efficient matrix vector product
* Algorithm 7.5: L-BFGS
"""

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
    d2 = np.sign(alpha1 - alpha0) * np.sqrt(d1**2 - phip0 * phip1)
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
    grad_f_zero: np.ndarray,
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
    """
    alpha_j = np.nan
    phi_j = np.nan
    grad_f_j = np.nan * grad_f_zero
    iter = 0
    is_done = False

    while not is_done and iter < max_iter:
        iter += 1
        alpha_j = cubic_interp(
            alpha_lo, phi_lo, phip_lo, alpha_hi, phi_hi, phip_hi)
        phi_j, phip_j, grad_f_j = phi(alpha_j)

        if phi_j > phi_zero + c1 * alpha_j * phip_zero or phi_j >= phi_lo:
            alpha_hi = alpha_j
            phi_hi = phi_j
            phip_hi = phip_j
        else:
            if np.abs(phip_j) <= -c2 * phip_zero:
                is_done = True
            else:
                if phip_j * (alpha_hi - alpha_lo):
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
    assert len(rho.shape) == 1 and rho.shape[0] == s.shape[1]
    assert len(q.shape) == 1 and q.shape[0] == s.shape[1]
    assert m >= 1
    assert s.shape[0] > m

    alpha = np.empty(shape=(m,), dtype=float)
    for i in range(m - 2, 1, -1):  # k - 1, ..., k - m
        alpha[i] = rho[i] * np.dot(s[i], q)
        q = q - alpha[i] * y[i]
    
    gamma_k = gamma_scale(s[m], y[m])
    r = gamma_k * q
    for i in range(m - 1):
        beta = rho[i] * np.dot(y[i], r)
        r += s[i] * (alpha[i] - beta)

    return r


def lbfgs(
    # params
    fun: tp.Callable[[np.ndarray, np.ndarray], tuple[floating, np.ndarray]],
    max_iter: int,
    tol: floating,
    c1: floating,
    c2: floating,
    max_ls: int,
    # updates
    x0: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, floating, np.ndarray]:
    assert len(x0.shape) == 1 and x0.size >= 1
    assert tol > 0
    assert max_iter >= 1

    m = max_iter

    iter = 0
    fun0, grad0 =  fun(params, x0)
    s = np.empty(shape=(m, x0.size))
    y = np.empty(shape=(m, x0.size))
    rho = np.empty(shape=(x0.size,))

    while iter < m and np.dot(grad0, grad0) >= tol**2:
        if iter == 0:
            p1 = -grad0
        else:
            p1 = -hess_vec_product(grad0, s, y, rho, iter)

        def phi(alpha):
            res = fun(params, x0 + alpha * p1)
            return res[0], np.dot(res[1], p1), res[1]

        phi_zero = fun0
        phip_zero = np.dot(grad0, p1)
        grad_f_zero = grad0
        zoom_params = [c1, c2, max_ls, phi, phi_zero, phip_zero, grad_f_zero]
        # alpha_lo, phi_lo, phip_lo
        zoom_params.extend([0.0, phi_zero, phip_zero])
        # alpha_hi, phi_hi, phip_hi
        zoom_params.extend([1.0, *phi(1.0)])
        alpha1, fun1, grad1 = zoom(*zoom_params)
        x1 = x0 + alpha1 * p1

        s[iter] = x1 - x0
        y[iter] = grad1 - grad0
        rho[iter] = 1.0 / np.dot(s[iter], y[iter])

        x0 = x1
        fun0 = fun1
        grad0 = grad1
        iter += 1

    return x0, fun0, grad0
