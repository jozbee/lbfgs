"""Tests for stripped lbfgs, numpy version."""

# import pytest
import functools
import numpy as np
import lbfgs.lbfgs_np as lbfgs_np
import scipy.optimize as sci_opt
import scipy.interpolate as sci_interp


def cubic_coeffs(alpha0, phi0, phip0, alpha1, phi1, phip1):
    A = np.array([
        [1.0, alpha0, alpha0**2, alpha0**2],
        [0.0, 1.0, 2.0 * alpha0, 3.0 * alpha0**2],
        [1.0, alpha1, alpha1**2, alpha1**3],
        [0.0, 1.0, 2.0 * alpha1, 3.0 * alpha1**2],
    ])
    b = np.array([phi0, phip0, phi1, phip1])
    return np.flip(np.linalg.solve(A, b))


def min_cubic_interval(alpha0, phi0, phip0, alpha1, phi1, phip1):
    coeffs = cubic_coeffs(alpha0, phi0, phip0, alpha1, phi1, phip1)
    polyval = functools.partial(np.polyval, coeffs)
    res = sci_opt.minimize(polyval, x0=(alpha0 + alpha1) * 0.5, bounds=[(alpha0, alpha1)])
    return res.x


def test_cubic_interp():
    data0 = [0.0, 0.0, -1.0, 1.0, 0.0, -1.0]
    data1 = [1.0, 1.0, -1.0, 1.5, 2.0, 0.5]
    data2 = [0.5, -1.0, -0.5, 1.0, -3.0, 0.0]

    for i, data in enumerate([data0, data1, data2]):
        res = lbfgs_np.cubic_interp(*data)
        check = min_cubic_interval(*data)
        assert np.isclose(res, check), f"index: {i}"


def phi_template(
    poly: sci_interp.CubicSpline,
    alpha: float | np.floating
) -> tuple[float, float, np.ndarray]:
    return float(poly(alpha, nu=0)), float(poly(alpha, nu=1)), np.array(poly(alpha, nu=1))


def test_zoom():
    poly0 = sci_interp.CubicSpline(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, -0.2, -0.3, 0.0, 0.1, 0.0]
    )
    phi0 = functools.partial(phi_template, poly0)

    poly1 = sci_interp.CubicSpline(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        [0.0, -0.1, 0.1, -0.2, 0.2, -0.3]
    )
    phi1 = functools.partial(phi_template, poly1)

    poly2 = sci_interp.CubicSpline(
        [0.0, 0.1, 0.5, 0.6, 0.7, 1.2],
        [1.0, 0.0, 0.2, 0.5, 0.4, 0.5]
    )
    phi2 = functools.partial(phi_template, poly2)

    for i, phi in enumerate([phi0, phi1, phi2]):
        c1 = 10**-4
        c2 = 0.9
        res = lbfgs_np.zoom(
            c1=c1,
            c2=c2,
            max_iter=4,
            phi=phi,
            phi_zero=phi(0.0)[0],
            phip_zero=phi(0.0)[1],
            grad_f_zero=phi(0.0)[2],
            alpha_lo=0.0,
            phi_lo=phi(0.0)[0],
            phip_lo=phi(0.0)[1],
            alpha_hi=1.0,
            phi_hi=phi(1.0)[0],
            phip_hi=phi(1.0)[1],
        )
        err_str = f"index = {i}"
        assert phi(res[0])[0] <= phi(0.)[0] + c1 * res[0] * phi(0.)[1], err_str
        assert abs(phi(res[0])[1]) <= -c2 * phi(0.)[1], err_str


def hess_vec_product(
    q: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    m: int,
):
    I = np.identity(n=s.shape[1])  # noqa: E741
    H = np.dot(s[m - 1], y[m - 1]) / np.dot(y[m - 1], y[m - 1]) * I
    for i in range(m - 1):
        rho = 1. / np.dot(y[i], s[i])
        V = I - rho * y[i].reshape(-1, 1) @ s[i].reshape(1, -1)
        H = V.T @ H @ V + rho * s[i].reshape(-1, 1) @ s[i].reshape(1, -1)
    return H @ q


def test_hess_vec_product():
    rng = np.random.default_rng(42)

    n = 20
    m = 4
    s = rng.uniform(-1, 1, size=n * m).reshape(n, m)
    y = rng.uniform(-1, 1, size=n * m).reshape(n, m)
    rho = np.array([1.0 / np.dot(s[i], y[i]) for i in range(n)])
    q = rng.uniform(-1, 1, size=m)

    m_prod = 5
    res = lbfgs_np.hess_vec_product(q, s, y, rho, m=m_prod)
    check = hess_vec_product(q, s, y, m=m_prod)

    assert np.allclose(res, check)
