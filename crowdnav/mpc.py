# crowdnav/mpc.py
from __future__ import annotations
import casadi as ca
import numpy as np

def _wrap_angle(x):
    return ca.atan2(ca.sin(x), ca.cos(x))

def _f_unicycle(x, u, T):
    px, py, th = x[0], x[1], x[2]
    v, omg     = u[0], u[1]
    pxn  = px + T * v * ca.cos(th)
    pyn  = py + T * v * ca.sin(th)
    thn  = _wrap_angle(th + T * omg)
    return ca.vertcat(pxn, pyn, thn)

def build_mpc_solver_random_obs(max_obs=0, max_static=8, N: int = 10, T: float = 0.1):
    """
    Returns (solver, f_dyn, T, N).

    Decision vars:
      U in R^{2N}  -> [v0, ω0, v1, ω1, ..., v_{N-1}, ω_{N-1}]

    Parameters P (order matters):
      [ x0(3),
        path_xy(2N),
        theta_ref(N),
        static_flat(2*max_static),     # (sx0, sy0, sx1, sy1, ...)
        weights(6),                    # [w_track, w_goal, w_smooth, w_obs, w_speed, w_theta]
        v_des(1)
      ]
    """

    nx, nu = 3, 2

    # decision
    U = ca.SX.sym('U', nu * N)

    # parameters (include static obstacles)
    P = ca.SX.sym('P', 3 + 2*N + N + 2*max_static + 6 + 1)
    off = 0
    x0         = P[off:off+3];            off += 3
    path_flat  = P[off:off+2*N];          off += 2*N
    th_ref     = P[off:off+N];            off += N
    static_flat= P[off:off+2*max_static]; off += 2*max_static
    w_vec      = P[off:off+6];            off += 6
    v_des      = P[off];                  off += 1

    w_track, w_goal, w_smooth, w_obs, w_speed, w_theta = \
        w_vec[0], w_vec[1], w_vec[2], w_vec[3], w_vec[4], w_vec[5]

    # handy slices for vectorized static cost
    sx = static_flat[0::2]   # (max_static,)
    sy = static_flat[1::2]

    # rollout (single-shooting)
    xk = x0
    J  = 0
    u_prev = None

    # small epsilon for obstacle barrier
    eps = 0.05  # ~ (22 cm)^2 in denominator; tune

    for k in range(N):
        uk = U[2*k:2*k+2]  # [v, ω]

        # waypoint tracking (position)
        rx = path_flat[2*k]
        ry = path_flat[2*k+1]
        ex = xk[0] - rx
        ey = xk[1] - ry
        J += w_track * (ex*ex + ey*ey)

        # heading tracking
        e_th = _wrap_angle(xk[2] - th_ref[k])
        J += w_theta * (e_th * e_th)

        # static obstacle repulsion (inverse-square style)
        dx  = xk[0] - sx
        dy  = xk[1] - sy
        dsq = dx*dx + dy*dy
        # if a slot is padded with large 1e6, dsq is ~1e12 and contributes ~0
        # distances from predicted state xk to static points (sx, sy)
        d   = ca.sqrt(dsq)

        # --- Two-zone obstacle cost ---
        # Far-field: gentle inverse-square that never goes to zero
        eps_far   = 0.08**2          # sets how "wide" the far field is (~8 cm)
        w_far     = 0.15             # gentle weight (tune)
        J += w_obs * w_far * ca.sum1(1.0 / (dsq + eps_far))

        # Near-field: hard wall (quartic hinge)
        robot_rad = 0.09
        inflate   = 0.06   # was 0.10 → make r_infl = 0.15 m
        w_near    = 3.0    # was 6.0 → still strong but less brutal
        r_infl    = robot_rad + inflate
        viol      = ca.fmax(0, r_infl - d)
        J += w_obs * w_near * ca.sum1(viol**4)



        # speed bias
        J += w_speed * (v_des - uk[0])**2

        # input smoothing
        if u_prev is not None:
            du = uk - u_prev
            J += w_smooth * ca.dot(du, du)
        u_prev = uk

        # propagate one step
        xk = _f_unicycle(xk, uk, T)

    # terminal (pull to last ref waypoint in window)
    rxN = path_flat[2*(N-1)]
    ryN = path_flat[2*(N-1)+1]
    eN  = ca.vertcat(xk[0] - rxN, xk[1] - ryN)
    J  += w_goal * ca.dot(eN, eN)

    nlp = {'x': U, 'f': J, 'p': P}
    opts = {"ipopt": {"print_level": 0, "max_iter": 200, "sb": "yes"}, "print_time": 0}
    solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, opts)

    f_dyn = lambda x, u: _f_unicycle(x, u, T)
    return solver, f_dyn, T, N
