"""Tests for stripped lbfgs, numpy version."""

# import pytest
import functools
import numpy as np
import jax
import jax.numpy as jnp
import lbfgs.lbfgs_np as lbfgs_np
import scipy.optimize as sci_opt
import scipy.interpolate as sci_interp

jax.config.update("jax_enable_x64", True)


def cubic_coeffs(alpha0, phi0, phip0, alpha1, phi1, phip1):
    A = np.array([
        [1.0, alpha0, alpha0**2, alpha0**3],
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
            grad_f_hi=phi(1.0)[2],
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


fun_np_counter = 0


def test_lbfgs():
    rng = np.random.default_rng(67)

    sol_rosenbrock = np.array([1.0, 1.0])
    x0_rosenbrock = sol_rosenbrock + rng.uniform(-1, 1, 2)

    def rosenbrock(x: jax.Array) -> jax.Array:
        assert len(x.shape) == 1 and x.size == 2
        x1, x2 = x
        f1 = 10.0 * (x2 - x1**2)
        f2 = 1.0 - x1
        return jnp.squeeze(f1**2 + f2**2)

    sol_freudenstein_roth = np.array([5.0, 4.0])
    x0_freudenstein_roth = sol_freudenstein_roth + rng.uniform(-1, 1, 2)

    def freudenstein_roth(x: jax.Array) -> jax.Array:
        x1, x2 = x
        f1 = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2
        f2 = -29.0 + x1 + ((x2 + 1.0) * x2 - 14.0) * x2
        return jnp.squeeze(f1**2 + f2**2)

    sol_brown = np.array([1.0e6, 2.0e-6])
    x0_brown = np.array([1.0, 1.0]) + rng.uniform(-1, 1, 2)

    def brown(x: jax.Array) -> jax.Array:
        x1, x2 = x
        f1 = x1 - 1e6
        f2 = x2 - 2.0 * 1e-6
        f3 = x1 * x2 - 2.0
        return jnp.squeeze(f1**2 + f2**2 + f3**2)

    problems = [
        (sol_rosenbrock, x0_rosenbrock, rosenbrock),
        (sol_freudenstein_roth, x0_freudenstein_roth, freudenstein_roth),
        (sol_brown, x0_brown, brown),
    ]

    for sol, x0, fun in problems:
        fun_jax = jax.value_and_grad(fun)

        def fun_np(_: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
            global fun_np_counter
            fun_np_counter += 1
            res = fun_jax(x)
            return float(res[0]), np.array(res[1])

        fun_np_counter = 0
        opt_params = lbfgs_np.OptParamsLBFGS(
            fun=fun_np,
            max_iter=16,
            max_ls=2,
            tol=1e-12,
            c1=1e-4,
            c2=0.9,
        )
        res = lbfgs_np.lbfgs(
            opt_params=opt_params,
            x0=x0,
            fun_params=np.array([]),
        )
        assert np.allclose(res[0], sol), f"{fun.__name__}, {x0}"


def test_fun_eval_count():
    """`1 + (1 + max_ls) * max_iter` bounds the number of evaluations.

    The numpy `zoom` honours the strong Wolfe exit, so the bound is not
    tight; it is exceeded as soon as `phi(0)` is re-evaluated instead of
    reusing `fun0` / `grad0`.
    """
    rng = np.random.default_rng(67)
    x0 = np.array([1.0, 1.0]) + rng.uniform(-1, 1, 2)

    def rosenbrock(x: jax.Array) -> jax.Array:
        x1, x2 = x
        return jnp.squeeze((10.0 * (x2 - x1**2)) ** 2 + (1.0 - x1) ** 2)

    fun_jax = jax.value_and_grad(rosenbrock)

    for max_iter in (1, 3):
        for max_ls in (1, 2):
            calls = 0

            def fun_np(
                _: np.ndarray, x: np.ndarray
            ) -> tuple[float, np.ndarray]:
                nonlocal calls
                calls += 1
                val, grad = fun_jax(x)
                return float(val), np.array(grad)

            lbfgs_np.lbfgs(
                opt_params=lbfgs_np.OptParamsLBFGS(
                    fun=fun_np,
                    max_iter=max_iter,
                    max_ls=max_ls,
                    tol=1e-12,
                    c1=1e-4,
                    c2=0.9,
                ),
                x0=x0,
                fun_params=np.array([]),
            )

            msg = f"max_iter={max_iter}, max_ls={max_ls}"
            assert 1 + max_iter <= calls, msg
            assert calls <= 1 + (1 + max_ls) * max_iter, msg
