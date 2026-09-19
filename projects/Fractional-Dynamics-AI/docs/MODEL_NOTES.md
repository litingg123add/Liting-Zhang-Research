# Model notes

## Reduced travelling-wave equation

The public numerical code is based on the reduced equation used in the accepted
LNETLM study:

```text
P U'' + T U' + Q U² - c U = 0
```

Introducing `V = U'` gives

```text
U' = V
V' = -(T/P)V - (Q/P)U² + (c/P)U
```

For the unforced system and `Q != 0`, the fixed points are

```text
(0, 0) and (c/Q, 0).
```

The repository classifies these points using the eigenvalues of the full
Jacobian rather than relying only on a sign shortcut.

## Numerical method

`src/rk38.py` implements the four-stage fourth-order Runge-Kutta 3/8 rule used
in the numerical-verification section of the research:

```text
y[n+1] = y[n] + h/8 (k1 + 3k2 + 3k3 + k4)
```

with stage locations at `0`, `1/3`, `2/3` and `1` of each step.

The accepted study reported numerical agreement at approximately the
`10^-6` level for a representative analytical solution. The public code here
focuses on the reduced dynamical system and reproducible diagnostics; it does
not reproduce every symbolic mGREM solution family.

## Parameter example

A useful reference case is

```text
P = 1, T = 1, Q = -2, c = 4.
```

The equilibria are `(0, 0)` and `(-2, 0)`. Eigenvalue analysis classifies
`(0, 0)` as a saddle and `(-2, 0)` as asymptotically stable.

## Periodic forcing

For sensitivity studies the code can add

```text
A sin(omega t)
```

to the second equation. In that non-autonomous setting, the repository offers
trajectory separation, a finite-time Lyapunov diagnostic, and a stroboscopic
section sampled once per forcing period.

A positive finite-time separation rate by itself is **not** treated as a
stand-alone proof of chaos.

## Bridge to Scientific Machine Learning

`src/trajectory_dataset.py` generates sequence datasets directly from the
governing dynamics. The train/test design holds out parameter values rather
than randomly mixing points from the same trajectory. This supports later
comparisons of FNO, DeepONet, PINN/fPINN and sequence baselines under unseen
dynamical regimes.
