"""Tests for stripped lbfgs, numpy version."""

# import pytest
import functools
import numpy as np
import jax
import jax.numpy as jnp
import lbfgs.lbfgs as lbfgs
import scipy.optimize as sci_opt

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


###########
# helpers #
###########


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


#################
# opt functions #
#################


def rosenbrock(x: jax.Array) -> jax.Array:
    assert len(x.shape) == 1 and x.size == 2
    x1, x2 = x
    f1 = 10.0 * (x2 - x1**2)
    f2 = 1.0 - x1
    return jnp.squeeze(f1**2 + f2**2)


def freudenstein_roth(x: jax.Array) -> jax.Array:
    x1, x2 = x
    f1 = -13.0 + x1 + ((5.0 - x2) * x2 - 2.0) * x2
    f2 = -29.0 + x1 + ((x2 + 1.0) * x2 - 14.0) * x2
    return jnp.squeeze(f1**2 + f2**2)


def brown(x: jax.Array) -> jax.Array:
    x1, x2 = x
    f1 = x1 - 1e6
    f2 = x2 - 2.0 * 1e-6
    f3 = x1 * x2 - 2.0
    return jnp.squeeze(f1**2 + f2**2 + f3**2)


rng = np.random.default_rng(67)

sol_rosenbrock = np.array([1.0, 1.0])
x0_rosenbrock_np = sol_rosenbrock + rng.uniform(-1, 1, 2)
sol_freudenstein_roth = np.array([5.0, 4.0])
x0_freudenstein_roth_np = sol_freudenstein_roth + rng.uniform(-1, 1, 2)
sol_brown = np.array([1.0e6, 2.0e-6])
x0_brown_np = np.array([1.0, 1.0]) + rng.uniform(-1, 1, 2)

x0_rosenbrock = jnp.array(x0_rosenbrock_np)
x0_freudenstein_roth = jnp.array(x0_freudenstein_roth_np)
x0_brown = jnp.array(x0_brown_np)


#########
# tests #
#########


def test_cubic_interp():
    data0 = jnp.array([0.0, 0.0, -1.0, 1.0, 0.0, -1.0])
    data1 = jnp.array([1.0, 1.0, -1.0, 1.5, 2.0, 0.5])
    data2 = jnp.array([0.5, -1.0, -0.5, 1.0, -3.0, 0.0])

    cubic_interp = jax.jit(lbfgs.cubic_interp)
    for i, data in enumerate([data0, data1, data2]):
        res = cubic_interp(*data)
        check = min_cubic_interval(*data)
        assert np.isclose(res, check), f"index: {i}"


def test_zoom():
    zoom = jax.jit(lbfgs.zoom)

    tests = [
        (x0_rosenbrock, rosenbrock),
        (x0_freudenstein_roth, freudenstein_roth),
        (x0_brown, brown),
    ]

    for iter, (x0, fun_base) in enumerate(tests):
        grad0 = jax.grad(fun_base)(x0)
        grad0 = -grad0 / np.linalg.norm(grad0)

        fun_val_grad = jax.value_and_grad(fun_base)

        def fun(_: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
            val, grad = fun_val_grad(x)
            return val, grad

        def phi(
            alpha: float | jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            val, grad = fun(jnp.array([]), x0 + alpha * grad0)
            return val, jnp.dot(grad, grad0), grad

        c1 = 1e-4
        c2 = 0.9
        params = lbfgs.ParamsZoom(
            c1=c1,
            c2=c2,
            max_iter=16,
            fun=fun,
            fun_params=jnp.array([]),
            x0=x0,
            p=grad0,
        )

        res = zoom(
            params=params,
            phi_zero=phi(0.0)[0],
            phip_zero=phi(0.0)[1],
            grad_f_hi=phi(1.0)[2],
            alpha_lo=jnp.array(0.0),
            phi_lo=phi(0.0)[0],
            phip_lo=phi(0.0)[1],
            alpha_hi=jnp.array(1.0),
            phi_hi=phi(1.0)[0],
            phip_hi=phi(1.0)[1],
        )
        msg = f"iter={iter}"
        assert phi(res[0])[0] <= phi(0.)[0] + c1 * res[0] * phi(0.)[1], msg
        assert abs(phi(res[0])[1]) <= -c2 * phi(0.)[1], msg


def test_hess_vec_product():
    rng = np.random.default_rng(42)
    n = 20
    m = 4
    jax_hess_vec_product = jax.jit(lbfgs.hess_vec_product)

    for iter in range(3):
        s = rng.uniform(-1, 1, size=n * m).reshape(n, m)
        y = rng.uniform(-1, 1, size=n * m).reshape(n, m)
        rho = np.array([1.0 / np.dot(s[i], y[i]) for i in range(n)])
        q = rng.uniform(-1, 1, size=m)

        m_prod = 5
        res = jax_hess_vec_product(q, s, y, rho, m=m_prod)
        check = hess_vec_product(q, s, y, m=m_prod)

        assert np.allclose(res, check), f"iter={iter}"


def test_lbfgs():

    problems = [
        (sol_rosenbrock, x0_rosenbrock, rosenbrock),
        (sol_freudenstein_roth, x0_freudenstein_roth, freudenstein_roth),
        (sol_brown, x0_brown, brown),
    ]

    for sol, x0, fun in problems:
        fun_jax = jax.value_and_grad(fun)

        @jax.jit
        def funner(_: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
            """`fun_jax`, but with empty params"""
            return fun_jax(x)

        res = lbfgs.lbfgs(
            opt_params=lbfgs.OptParamsLBFGS(
                fun=funner,
                max_iter=16,
                max_ls=2,
                tol=1e-12,
                c1=1e-4,
                c2=0.9,
            ),
            x0=x0,
            fun_params=jnp.array([]),
        )

        assert np.allclose(res[0], sol), f"{fun.__name__}, {x0}"
