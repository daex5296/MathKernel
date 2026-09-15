# =============================================================================
# MathKernel - ODEs and a scoped PDE solver
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""ODEs and a scoped PDE solver.

- Symbolic ODEs via sympy.dsolve with classification (SYMBOLIC).
- Numeric IVP: own adaptive Dormand-Prince RK45 on mpmath
  (NUMERIC_HIGH_PRECISION) with per-step error reporting; njit float64 fast
  path (NUMERIC); optional scipy solve_ivp cross-check (`sci` extra).
- IVP ensembles (parameter sweeps / many initial conditions) batch onto the
  GPU via a CuPy RawKernel — one thread per trajectory, fixed-step RK4 with
  the RHS compiled from MathIR to C. CPU fallback: process pool of njit RK4.
- PDE scope is deliberately honest: 1D heat equation via explicit FTCS
  finite differences (NUMERIC evidence only), njit stencil with a CuPy path.
"""
from __future__ import annotations

import math

from .models import Expr
from .numerics import compile_float64, emit_c, emit_float64
from .parallel import process_map, resolve_workers

# Dormand-Prince coefficients
_DP_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0]
_DP_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
         [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
         [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]]
_DP_B = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
_DP_B4 = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200,
          187 / 2100, 1 / 40]


def dsolve_symbolic(rhs, y_var: str, x_var: str, ics: dict | None = None):
    """Solve dy/dx = rhs(x, y) symbolically. Returns (solution, classification)."""
    import sympy as sp
    x = sp.Symbol(x_var)
    f = sp.Function(y_var)
    # in the rhs, the bare dependent-variable symbol means y(x)
    rhs = rhs.subs(sp.Symbol(y_var), f(x))
    equation = sp.Eq(f(x).diff(x), rhs)
    classification = list(sp.classify_ode(equation, f(x)))
    solution = sp.dsolve(equation, f(x), ics=ics or None)
    return solution, classification


def rk45_mpmath(f, t0: str, y0: list[str], t1: str, tol: float = 1e-10,
                max_steps: int = 100_000, dps: int = 50, *,
                rtol: float | None = None, atol: float | None = None,
                max_step: float | None = None,
                t_eval: list[str | float] | None = None,
                dense_output: bool = False) -> dict:
    """Adaptive RK45 at arbitrary precision with optional sampled output.

    Requested points are evaluated from cubic-Hermite interpolants over
    accepted steps, so they do not trigger separate integrations.  The
    interpolation is reported separately because it has no certified error
    bound.
    """
    import mpmath as mp
    mp.mp.dps = dps
    t = mp.mpf(t0)
    y = [mp.mpf(v) for v in y0]
    target = mp.mpf(t1)
    direction = 1 if target >= t else -1
    rtol_mp = mp.mpf(str(tol if rtol is None else rtol))
    atol_mp = mp.mpf(str(tol if atol is None else atol))
    if rtol_mp <= 0 or atol_mp <= 0 or not mp.isfinite(rtol_mp) or not mp.isfinite(atol_mp):
        raise ValueError("rtol and atol must be positive and finite")
    max_step_mp = mp.inf if max_step is None else mp.mpf(str(max_step))
    if max_step_mp <= 0 or (max_step is not None and not mp.isfinite(max_step_mp)):
        raise ValueError("max_step must be positive and finite")
    span = abs(target - t)
    h = direction * min(span / 100, mp.mpf("0.1"), max_step_mp)
    if h == 0:
        h = mp.mpf("0.01") * direction

    requested = None
    if t_eval is not None:
        requested = [mp.mpf(str(value)) for value in t_eval]
        if not requested:
            raise ValueError("t_eval must contain at least one time")
        if any((value - t) * direction < 0 or (value - target) * direction > 0
               for value in requested):
            raise ValueError("t_eval values must lie inside t_span")
        if any((b - a) * direction <= 0 for a, b in zip(requested, requested[1:])):
            raise ValueError("t_eval values must be strictly ordered in the integration direction")
    trajectory_t = [t] if dense_output and requested is None else []
    trajectory_y = [list(y)] if dense_output and requested is None else []
    sampled_t: list = []
    sampled_y: list[list] = []
    request_index = 0
    if target == t and requested is not None:
        sampled_t = list(requested)
        sampled_y = [list(y) for _ in requested]
        request_index = len(requested)

    def interpolate(query, left_t, left_y, left_f, right_t, right_y, right_f):
        width = right_t - left_t
        s = (query - left_t) / width
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        return [h00 * left_y[i] + h10 * width * left_f[i] +
                h01 * right_y[i] + h11 * width * right_f[i]
                for i in range(len(left_y))]
    steps = 0
    rejected = 0
    max_err = mp.mpf(0)
    n = len(y)
    while (t - target) * direction < 0:
        if steps + rejected >= max_steps:
            raise ValueError(f"rk45 exceeded max_steps={max_steps}")
        if abs(h) > max_step_mp:
            h = direction * max_step_mp
        if (t + h - target) * direction > 0:
            h = target - t
        k = []
        k.append(f(t, y))
        for stage in range(1, 6):
            ys = [y[i] + h * sum(_DP_A[stage][j] * k[j][i] for j in range(stage))
                  for i in range(n)]
            k.append(f(t + _DP_C[stage] * h, ys))
        y5 = [y[i] + h * sum(_DP_B[j] * k[j][i] for j in range(6)) for i in range(n)]
        ks = k + [f(t + h, y5)]
        y4 = [y[i] + h * sum(_DP_B4[j] * ks[j][i] for j in range(7)) for i in range(n)]
        errors = [abs(y5[i] - y4[i]) for i in range(n)]
        err = max(errors)
        max_err = max(max_err, err)
        error_norm = max(errors[i] /
                         (atol_mp + rtol_mp * max(abs(y[i]), abs(y5[i])))
                         for i in range(n))
        if error_norm <= 1:
            old_t, old_y, old_f = t, y, k[0]
            t += h
            y = y5
            steps += 1
            new_f = ks[-1]
            if requested is not None:
                while request_index < len(requested) and (requested[request_index] - t) * direction <= 0:
                    query = requested[request_index]
                    values = old_y if query == old_t else (y if query == t else
                        interpolate(query, old_t, old_y, old_f, t, y, new_f))
                    sampled_t.append(query); sampled_y.append(list(values))
                    request_index += 1
            elif dense_output:
                trajectory_t.append(t); trajectory_y.append(list(y))
        else:
            rejected += 1
        factor = mp.mpf("0.9") * (1 / (error_norm + mp.mpf("1e-300"))) ** mp.mpf("0.2")
        h *= min(mp.mpf(5), max(mp.mpf("0.2"), factor))
    out = {"t": mp.nstr(t, 30), "y": [mp.nstr(v, 30) for v in y],
           "method": "rk45-mpmath", "solver_status": "converged",
           "accepted_steps": steps, "steps": steps, "rejected_steps": rejected,
           "controls": {"rtol": str(rtol_mp), "atol": str(atol_mp),
                        "max_step": None if max_step is None else str(max_step_mp),
                        "max_steps": max_steps, "dps": dps},
           "error_estimates": {
               "integration": {"method": "dormand-prince-embedded-4-5",
                               "max_local_absolute_error": mp.nstr(max_err, 8),
                               "certified": False},
               "interpolation": {"method": "cubic-hermite",
                                 "estimate": None, "certified": False}},
           "max_local_error": mp.nstr(max_err, 5), "dps": dps}
    times = sampled_t if requested is not None else trajectory_t
    states = sampled_y if requested is not None else trajectory_y
    if requested is not None or dense_output:
        out["trajectory"] = {"t": [mp.nstr(value, 30) for value in times],
                             "y": [[mp.nstr(value, 30) for value in row] for row in states],
                             "requested": requested is not None,
                             "interpolation": "cubic-hermite"}
    return out


def compile_rhs_float64(rhs_irs: list[Expr], y_vars: list[str], t_var: str = "t",
                        njit: bool = False):
    """Compile a first-order system y' = F(t, y) to f(t, y, out)."""
    parts = []
    for i, ir in enumerate(rhs_irs):
        var_map = {t_var: "t"}
        var_map.update({v: f"y[{j}]" for j, v in enumerate(y_vars)})
        parts.append(f"    out[{i}] = {emit_float64(ir, var_map)}")
    source = "def _rhs(t, y, out):\n" + "\n".join(parts) + "\n"
    namespace: dict = {"math": math}
    exec(source, namespace)
    fn = namespace["_rhs"]
    if njit:
        try:
            from numba import njit as _njit
            return _njit(cache=False)(fn)
        except Exception:
            return None
    return fn


def rk4_float64(f, t0: float, y0: list[float], t1: float, steps: int) -> list[float]:
    """Fixed-step RK4 (float64). f(t, y, out) writes derivatives into out."""
    import numpy as np
    y = np.asarray(y0, dtype=np.float64)
    n = y.size
    h = (t1 - t0) / steps
    out = np.zeros(n)
    k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
    yt = np.zeros(n)
    t = t0
    for s in range(steps):
        f(t, y, k1)
        yt[:] = y + 0.5 * h * k1
        f(t + 0.5 * h, yt, k2)
        yt[:] = y + 0.5 * h * k2
        f(t + 0.5 * h, yt, k3)
        yt[:] = y + h * k3
        f(t + h, yt, k4)
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t0 + (s + 1) * h
    return y.tolist()


def rk4_float64_solution(f, t0: float, y0: list[float], t1: float, steps: int,
                         *, t_eval: list[float] | None = None,
                         dense_output: bool = False) -> dict:
    """Fixed-step RK4 with optional cubic-Hermite trajectory sampling."""
    import numpy as np
    if steps < 1:
        raise ValueError("steps must be positive")
    direction = 1 if t1 >= t0 else -1
    requested = None if t_eval is None else [float(v) for v in t_eval]
    if requested is not None:
        if not requested:
            raise ValueError("t_eval must contain at least one time")
        if any(not math.isfinite(v) or (v - t0) * direction < 0 or (v - t1) * direction > 0
               for v in requested):
            raise ValueError("t_eval values must be finite and lie inside t_span")
        if any((b - a) * direction <= 0 for a, b in zip(requested, requested[1:])):
            raise ValueError("t_eval values must be strictly ordered in the integration direction")
    if t0 == t1:
        row = [repr(float(v)) for v in y0]
        out = {"t": repr(float(t1)), "y": row, "method": "rk4-float64",
               "solver_status": "converged", "accepted_steps": 0, "steps": 0,
               "rejected_steps": 0, "controls": {"steps": steps},
               "error_estimates": {
                   "integration": {"method": None, "estimate": None, "certified": False},
                   "interpolation": {"method": "cubic-hermite", "estimate": None,
                                     "certified": False}}}
        if requested is not None or dense_output:
            times = requested if requested is not None else [t0]
            out["trajectory"] = {"t": [repr(float(v)) for v in times],
                                 "y": [list(row) for _ in times],
                                 "requested": requested is not None,
                                 "interpolation": "cubic-hermite"}
        return out
    y = np.asarray(y0, dtype=np.float64)
    h = (t1 - t0) / steps
    n = y.size
    k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
    yt = np.zeros(n); f_right = np.zeros(n)
    accepted_t = [float(t0)] if dense_output and requested is None else []
    accepted_y = [y.tolist()] if dense_output and requested is None else []
    sampled_t: list[float] = []; sampled_y: list[list[float]] = []; q = 0
    t = float(t0)
    for index in range(steps):
        old_t, old_y = t, y.copy()
        f(t, y, k1)
        yt[:] = y + 0.5 * h * k1; f(t + 0.5 * h, yt, k2)
        yt[:] = y + 0.5 * h * k2; f(t + 0.5 * h, yt, k3)
        yt[:] = y + h * k3; f(t + h, yt, k4)
        y += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t0 + (index + 1) * h
        if requested is not None:
            f(t, y, f_right)
            while q < len(requested) and (requested[q] - t) * direction <= 0:
                query = requested[q]
                s = (query - old_t) / h
                row = ((2*s**3-3*s**2+1)*old_y + (s**3-2*s**2+s)*h*k1 +
                       (-2*s**3+3*s**2)*y + (s**3-s**2)*h*f_right)
                sampled_t.append(query); sampled_y.append(row.tolist()); q += 1
        elif dense_output:
            accepted_t.append(float(t)); accepted_y.append(y.tolist())
    out = {"t": repr(float(t1)), "y": [repr(float(v)) for v in y],
           "method": "rk4-float64", "solver_status": "converged",
           "accepted_steps": steps, "steps": steps, "rejected_steps": 0,
           "controls": {"steps": steps},
           "error_estimates": {
               "integration": {"method": None, "estimate": None, "certified": False},
               "interpolation": {"method": "cubic-hermite", "estimate": None,
                                 "certified": False}}}
    times = sampled_t if requested is not None else accepted_t
    states = sampled_y if requested is not None else accepted_y
    if requested is not None or dense_output:
        out["trajectory"] = {"t": [repr(v) for v in times],
                             "y": [[repr(v) for v in row] for row in states],
                             "requested": requested is not None,
                             "interpolation": "cubic-hermite"}
    return out


def solve_ivp_float64(f, t0: float, y0: list[float], t1: float, *, method: str,
                      rtol: float, atol: float, max_step: float | None,
                      max_steps: int, t_eval: list[float] | None = None,
                      dense_output: bool = False) -> dict:
    """SciPy adaptive float64 route, kept optional behind the ``sci`` extra."""
    try:
        import numpy as np
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise ValueError("adaptive float64 ODE methods require the 'sci' extra") from exc
    allowed = {"RK23": "RK23", "RK45": "RK45", "DOP853": "DOP853",
               "RADAU": "Radau", "BDF": "BDF", "LSODA": "LSODA"}
    key = method.upper()
    if key not in allowed:
        raise ValueError(f"unsupported adaptive float64 method: {method}")
    canonical = allowed[key]
    if rtol <= 0 or atol <= 0 or not math.isfinite(rtol) or not math.isfinite(atol):
        raise ValueError("rtol and atol must be positive and finite")
    if max_step is not None and (max_step <= 0 or not math.isfinite(max_step)):
        raise ValueError("max_step must be positive and finite")
    direction = 1 if t1 >= t0 else -1
    requested = None if t_eval is None else np.asarray(t_eval, dtype=float)
    if requested is not None:
        if requested.ndim != 1 or requested.size == 0 or not np.all(np.isfinite(requested)):
            raise ValueError("t_eval must be a nonempty finite one-dimensional sequence")
        if np.any((requested - t0) * direction < 0) or np.any((requested - t1) * direction > 0):
            raise ValueError("t_eval values must lie inside t_span")
        if np.any(np.diff(requested) * direction <= 0):
            raise ValueError("t_eval values must be strictly ordered in the integration direction")
    if t0 == t1:
        row = [repr(float(v)) for v in y0]
        out = {"t": repr(float(t1)), "y": row, "method": canonical,
               "solver_status": "converged", "message": "zero-length interval",
               "accepted_steps": 0, "rejected_steps": 0, "function_evaluations": 0,
               "controls": {"rtol": rtol, "atol": atol, "max_step": max_step,
                            "max_steps": max_steps},
               "error_estimates": {
                   "integration": {"method": "solver-controlled-local-error",
                                   "estimate": "0", "certified": False},
                   "interpolation": {"method": f"{canonical}-continuous-extension",
                                     "estimate": "0", "certified": False}}}
        if requested is not None or dense_output:
            times = requested.tolist() if requested is not None else [t0]
            out["trajectory"] = {"t": [repr(float(v)) for v in times],
                                 "y": [list(row) for _ in times],
                                 "requested": requested is not None,
                                 "interpolation": f"{canonical}-continuous-extension"}
        return out

    def rhs(t, y):
        out = np.empty_like(y)
        f(t, y, out)
        return out

    solution = solve_ivp(rhs, (t0, t1), np.asarray(y0, dtype=float), method=canonical,
                         rtol=rtol, atol=atol,
                         max_step=math.inf if max_step is None else max_step,
                         dense_output=requested is not None or dense_output)
    accepted = max(0, len(solution.t) - 1)
    if accepted > max_steps:
        raise ValueError(f"adaptive solver exceeded max_steps={max_steps}")
    out = {"t": repr(float(solution.t[-1])),
           "y": [repr(float(v)) for v in solution.y[:, -1]],
           "method": canonical, "solver_status": "converged" if solution.success else "failed",
           "message": solution.message, "accepted_steps": accepted,
           "rejected_steps": None, "function_evaluations": int(solution.nfev),
           "controls": {"rtol": rtol, "atol": atol, "max_step": max_step,
                        "max_steps": max_steps},
           "error_estimates": {
               "integration": {"method": "solver-controlled-local-error",
                               "estimate": None, "certified": False},
               "interpolation": {"method": f"{canonical}-continuous-extension",
                                 "estimate": None, "certified": False}}}
    if requested is not None:
        values = solution.sol(requested)
        out["trajectory"] = {"t": [repr(float(v)) for v in requested],
                             "y": [[repr(float(v)) for v in values[:, i]]
                                   for i in range(values.shape[1])],
                             "requested": True,
                             "interpolation": f"{canonical}-continuous-extension"}
    elif dense_output:
        out["trajectory"] = {"t": [repr(float(v)) for v in solution.t],
                             "y": [[repr(float(v)) for v in solution.y[:, i]]
                                   for i in range(solution.y.shape[1])],
                             "requested": False,
                             "interpolation": f"{canonical}-continuous-extension"}
    return out


def _rk4_njit():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True)
    def rk4(f, t0, y0, t1, steps):
        import numpy as np
        y = y0.copy()
        n = y.shape[0]
        h = (t1 - t0) / steps
        k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
        yt = np.zeros(n)
        t = t0
        for s in range(steps):
            f(t, y, k1)
            for i in range(n):
                yt[i] = y[i] + 0.5 * h * k1[i]
            f(t + 0.5 * h, yt, k2)
            for i in range(n):
                yt[i] = y[i] + 0.5 * h * k2[i]
            f(t + 0.5 * h, yt, k3)
            for i in range(n):
                yt[i] = y[i] + h * k3[i]
            f(t + h, yt, k4)
            for i in range(n):
                y[i] += (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            t = t0 + (s + 1) * h
        return y

    return rk4


# --- GPU ensemble (CuPy RawKernel, one thread per trajectory) -----------------------

def ensemble_rk4_gpu(rhs_irs: list[Expr], y_vars: list[str], t0: float, t1: float,
                     y0s: list[list[float]], steps: int, t_var: str = "t") -> dict:
    """Integrate many trajectories on the GPU. The RHS is compiled from MathIR
    to C and inlined into an NVRTC kernel. Returns endpoints per trajectory."""
    from .koopman import _xp
    xp = _xp(prefer_gpu=True)
    if xp.__name__ != "cupy":
        raise ValueError("GPU ensemble requires a working CuPy/CUDA installation")
    dim = len(rhs_irs)

    def stage(k_name: str, t_expr: str) -> str:
        lines = []
        for i, ir in enumerate(rhs_irs):
            var_map = {t_var: "t_"}
            var_map.update({v: f"y[{j}]" for j, v in enumerate(y_vars)})
            lines.append(f"        {k_name}[{i}] = {emit_c(ir, var_map)};")
        return "\n".join(lines)

    source = f"""
extern "C" __global__
void rk4_ensemble(double* yall, int n_traj, int dim, double t0, double t1, int steps) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n_traj) return;
    double* yi = yall + idx * dim;
    double h = (t1 - t0) / steps;
    double t = t0;
    double k1[16], k2[16], k3[16], k4[16], yt[16];
    for (int s = 0; s < steps; s++) {{
        {{ double* y = yi; double t_ = t;
{stage("k1", "t")} }}
        for (int i = 0; i < dim; i++) yt[i] = yi[i] + 0.5*h*k1[i];
        {{ double* y = yt; double t_ = t + 0.5*h;
{stage("k2", "t+h/2")} }}
        for (int i = 0; i < dim; i++) yt[i] = yi[i] + 0.5*h*k2[i];
        {{ double* y = yt; double t_ = t + 0.5*h;
{stage("k3", "t+h/2")} }}
        for (int i = 0; i < dim; i++) yt[i] = yi[i] + h*k3[i];
        {{ double* y = yt; double t_ = t + h;
{stage("k4", "t+h")} }}
        for (int i = 0; i < dim; i++)
            yi[i] += (h/6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        t += h;
    }}
}}
"""
    if dim > 16:
        raise ValueError("GPU ensemble supports at most 16 state variables")
    kernel = xp.RawKernel(source, "rk4_ensemble")
    n_traj = len(y0s)
    if any(len(row) != dim for row in y0s):
        raise ValueError("every initial condition must match the system dimension")
    data = xp.asarray(y0s, dtype=xp.float64).reshape(-1)
    block = 128
    grid = (n_traj + block - 1) // block
    kernel((grid,), (block,), (data, n_traj, dim, float(t0), float(t1), int(steps)))
    out = data.get().reshape(n_traj, dim)
    return {"endpoints": out.tolist(), "trajectories": n_traj, "steps": steps,
            "engine_tier": "numeric-gpu"}


def _ensemble_cpu_job(args) -> list[float]:
    rhs_srcs, y_vars, t0, t1, y0, steps = args
    from .parser import parse_math
    irs = [parse_math(s) for s in rhs_srcs]
    f = compile_rhs_float64(irs, y_vars, njit=False)
    return rk4_float64(f, t0, y0, t1, steps)


def ensemble_rk4_cpu(rhs_sources: list[str], y_vars: list[str], t0: float, t1: float,
                     y0s: list[list[float]], steps: int,
                     workers: int | None = None) -> dict:
    """CPU ensemble fallback: process pool of float64 RK4 trajectories."""
    resolved = resolve_workers(workers)
    jobs = [(rhs_sources, y_vars, t0, t1, y0, steps) for y0 in y0s]
    results = process_map(_ensemble_cpu_job, jobs, workers=resolved)
    return {"endpoints": results, "trajectories": len(y0s), "steps": steps,
            "engine_tier": "numeric-cpu", "workers": resolved}


# --- 1D heat equation (explicit FTCS) ------------------------------------------------

def heat_ftcs_float64(u0: list[float], alpha: float, dx: float, dt: float,
                      steps: int) -> list[float]:
    """u_t = alpha * u_xx on a fixed grid, Dirichlet-zero boundaries.
    Stability requires r = alpha*dt/dx^2 <= 1/2 (checked)."""
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(f"FTCS unstable: r = {r:.4f} > 1/2; reduce dt or alpha")
    u = list(u0)
    n = len(u)
    for _ in range(steps):
        new = [0.0] * n
        for i in range(1, n - 1):
            new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
        u = new
    return u


def _heat_njit():
    try:
        from numba import njit
    except ImportError:
        return None

    import numpy as np

    @njit(cache=True)
    def heat(u, r, steps):
        n = u.shape[0]
        out = np.zeros(n)
        for _ in range(steps):
            for i in range(1, n - 1):
                out[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
            u, out = out, u
            out[0] = 0.0
            out[n - 1] = 0.0
        return u

    return heat


def heat_ftcs(u0: list[float], alpha: float, dx: float, dt: float, steps: int,
              prefer_gpu: bool = True) -> dict:
    """1D heat equation with tier dispatch: CuPy stencil > njit > Python."""
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(f"FTCS unstable: r = {r:.4f} > 1/2; reduce dt or alpha")
    if prefer_gpu:
        from .koopman import _xp
        xp = _xp(prefer_gpu=True)
        if xp.__name__ == "cupy":
            u = xp.asarray(u0, dtype=xp.float64)
            for _ in range(steps):
                new = xp.zeros_like(u)
                new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
                u = new
            return {"u": u.get().tolist(), "engine_tier": "numeric-gpu",
                    "stability_r": r}
    kernel = _heat_njit()
    if kernel is not None:
        import numpy as np
        u = kernel(np.asarray(u0, dtype=np.float64), r, steps)
        return {"u": u.tolist(), "engine_tier": "njit", "stability_r": r}
    return {"u": heat_ftcs_float64(u0, alpha, dx, dt, steps),
            "engine_tier": "python-fallback", "stability_r": r}
