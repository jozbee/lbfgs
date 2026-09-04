Stripped LBFGS aglorithm for realtime iterations.

Cost of one `lbfgs` call
------------------------
The objective (value and gradient together) is evaluated

    1 + (1 + max_ls) * max_iter

times: once for the initial point, then per L-BFGS iteration once at the
unit step `alpha = 1` and `max_ls` times inside the `zoom` line search.
The count used to be `1 + (2 + max_ls) * max_iter`, because the line
search re-evaluated `phi(0)` although `fun0`/`grad0` already held exactly
that value (`x0 + 0.0 * p == x0` and the objective is deterministic).
Both `lbfgs.lbfgs` and `lbfgs.lbfgs_np` now reuse them.  The reuse is
mathematically exact and strictly cheaper.  It is bitwise exact in
`lbfgs_np`, whose arithmetic is unchanged; `lbfgs.lbfgs` drops an
operation from the compiled program, so the directional derivative, and
with it the accepted line-search step, can move by a few ulp -- up to
`1e-8` relative on the iterate of the MPC problem this was written for.
The count is an upper bound: in both versions `zoom` returns as soon as
the strong Wolfe conditions hold, and the outer loop returns as soon as
the gradient tolerance is met.
