"""Tests for stripped lbfgs, numpy version."""

# import pytest
import dataclasses
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

    def loop_test(jax_hess_vec_product):
        for iter in range(3):
            s = rng.uniform(-1, 1, size=n * m).reshape(n, m)
            y = rng.uniform(-1, 1, size=n * m).reshape(n, m)
            rho = np.array([1.0 / np.dot(s[i], y[i]) for i in range(n)])
            q = rng.uniform(-1, 1, size=m)

            m_prod = 5
            res = jax_hess_vec_product(q, s, y, rho, m=m_prod)
            check = hess_vec_product(q, s, y, m=m_prod)

            assert np.allclose(res, check), f"iter={iter}"

    loop_test(jax.jit(lbfgs.hess_vec_product))
    loop_test(jax.jit(functools.partial(lbfgs.hess_vec_product, unroll=True)))


def test_lbfgs():

    problems = [
        (sol_rosenbrock, x0_rosenbrock, rosenbrock, False),
        (sol_rosenbrock, x0_rosenbrock, rosenbrock, True),
        (sol_freudenstein_roth, x0_freudenstein_roth, freudenstein_roth, False),
        (sol_freudenstein_roth, x0_freudenstein_roth, freudenstein_roth, True),
        (sol_brown, x0_brown, brown, False),
        (sol_brown, x0_brown, brown, True),
    ]

    for sol, x0, fun, unroll in problems:
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
                unroll=unroll,
            ),
            x0=x0,
            fun_params=jnp.array([]),
        )

        assert np.allclose(res[0], sol), f"{fun.__name__}, {x0}, {unroll}"


def lbfgs_reeval(
    opt_params: lbfgs.OptParamsLBFGS,
    x0: jax.Array,
    fun_params: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """`lbfgs.lbfgs` as it was before `phi(0)` was reused, as an oracle.

    The only difference to `lbfgs.lbfgs` is the marked line: the line
    search re-evaluates the objective at `alpha_lo == 0.0` instead of
    reusing `st.fun0` / `st.grad0`.  Because `x0 + 0.0 * p1` is bitwise
    `x0` and the objective is deterministic, both must return bitwise
    identical results.
    """
    m = opt_params.max_iter
    fun0, grad0 = opt_params.fun(fun_params, x0)
    s = jnp.zeros(shape=(m, x0.size))
    y = jnp.zeros(shape=(m, x0.size))
    rho = jnp.zeros(shape=(m,))

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class State:
        x0: jax.Array
        fun0: jax.Array
        grad0: jax.Array
        iter: jax.Array
        s: jax.Array
        y: jax.Array
        rho: jax.Array

    def while_cond(st: State) -> jax.Array:
        grad_cond = jnp.dot(st.grad0, st.grad0) >= opt_params.tol**2
        return (st.iter < m) & grad_cond

    def while_body(st: State) -> State:
        p1 = -lbfgs.hess_vec_product(
            st.grad0, st.s, st.y, st.rho, st.iter, opt_params.unroll
        )
        init_norm = opt_params.init_norm
        p1 = jax.lax.cond(
            st.iter == 0,
            lambda: -st.grad0 / jnp.linalg.norm(st.grad0) * init_norm,
            lambda: p1,
        )

        def phi(alpha: float | jax.Array) -> tuple[jax.Array, jax.Array]:
            return opt_params.fun(fun_params, st.x0 + alpha * p1)

        alpha_lo = jnp.array(0.0)
        phi_zero, grad_f_zero = phi(alpha_lo)  # <- the removed evaluation
        phip_zero = jnp.dot(grad_f_zero, p1)
        alpha_hi = jnp.array(1.0)
        phi_hi, grad_f_hi = phi(alpha_hi)
        phip_hi = jnp.dot(grad_f_hi, p1)

        alpha1, fun1, grad1 = lbfgs.zoom(
            params=lbfgs.ParamsZoom(
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
            unroll=opt_params.unroll,
        )

        x1 = st.x0 + alpha1 * p1
        st.s = st.s.at[st.iter].set(x1 - st.x0)
        st.y = st.y.at[st.iter].set(grad1 - st.grad0)
        rho_iter = 1.0 / jnp.dot(st.s[st.iter], st.y[st.iter])
        st.rho = st.rho.at[st.iter].set(rho_iter)
        st.x0, st.fun0, st.grad0, st.iter = jax.lax.cond(
            jnp.isnan(rho_iter) | jnp.any(jnp.isnan(x1)),
            lambda: (st.x0, st.fun0, st.grad0, m),
            lambda: (x1, fun1, grad1, st.iter + 1),
        )
        return st

    state0 = State(x0, fun0, grad0, jnp.array(0), s, y, rho)
    if not opt_params.unroll:
        res = jax.lax.while_loop(while_cond, while_body, state0)
    else:
        res = state0
        for _ in range(m):
            res = while_body(res)
    return res.x0, res.fun0, res.grad0


def test_fun_eval_count():
    """One call costs `1 + (1 + max_ls) * max_iter` evaluations."""
    fun_val_grad = jax.value_and_grad(rosenbrock)

    for max_iter in (1, 3):
        for max_ls in (1, 2):
            calls = 0

            def fun(_: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
                nonlocal calls
                calls += 1
                return fun_val_grad(x)

            # `disable_jit` makes the `lax` loops run as python loops, so
            # the python counter sees every evaluation.
            with jax.disable_jit():
                lbfgs.lbfgs(
                    opt_params=lbfgs.OptParamsLBFGS(
                        fun=fun,
                        max_iter=max_iter,
                        max_ls=max_ls,
                        tol=1e-12,
                        c1=1e-4,
                        c2=0.9,
                        unroll=False,
                    ),
                    x0=x0_rosenbrock,
                    fun_params=jnp.array([]),
                )

            expect = 1 + (1 + max_ls) * max_iter
            msg = f"max_iter={max_iter}, max_ls={max_ls}"
            assert calls == expect, msg


def test_lbfgs_unchanged_by_phi_zero_reuse():
    """Reusing `phi(0)` leaves the returned iterate bitwise unchanged."""
    fun_val_grad = jax.value_and_grad(rosenbrock)

    def fun(_: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return fun_val_grad(x)

    settings = [
        (2, 1, False),
        (3, 2, False),
        (4, 1, False),
        (2, 1, True),
        (3, 2, True),
    ]
    for max_iter, max_ls, unroll in settings:
        opt_params = lbfgs.OptParamsLBFGS(
            fun=fun,
            max_iter=max_iter,
            max_ls=max_ls,
            tol=1e-12,
            c1=1e-4,
            c2=0.9,
            unroll=unroll,
        )
        res = lbfgs.lbfgs(
            opt_params=opt_params,
            x0=x0_rosenbrock,
            fun_params=jnp.array([]),
        )
        ref = lbfgs_reeval(
            opt_params=opt_params,
            x0=x0_rosenbrock,
            fun_params=jnp.array([]),
        )
        msg = f"max_iter={max_iter}, max_ls={max_ls}, unroll={unroll}"
        for got, want, name in zip(res, ref, ["x", "fun", "grad"]):
            assert np.array_equal(got, want), f"{name}: {msg}"
