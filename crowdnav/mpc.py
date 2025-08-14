import casadi as ca
K_turn = 5
# def build_mpc_solver_random_obs(T=0.1, N=10, max_obs=4, max_static=4):
#     """Build MPC solver for obstacle avoidance."""
#     x = ca.SX.sym('x')
#     y = ca.SX.sym('y')
#     theta = ca.SX.sym('theta')
#     states = ca.vertcat(x, y, theta)
#     v = ca.SX.sym('v')
#     omega = ca.SX.sym('omega')
#     controls = ca.vertcat(v, omega)
#     rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
#     f = ca.Function('f', [states, controls], [rhs])
#     U = ca.SX.sym('U', 2, N)
#     X = ca.SX.sym('X', 3, N + 1)
#     # P = ca.SX.sym('P', 3 + 2*N + 2*N*max_obs + 2*max_static + 4)
    
#     # new P size: 3 + 2*N (xy path) +   N (θ_ref) + 2*N*max_obs + 2*max_static + 5 weights
#     P = ca.SX.sym('P',
#         3
#       + 2*N
#       + N
#       + 2*N*max_obs
#       + 2*max_static
#       + 5
#     )
    
#     Q_goal    = P[-5]   # goal‐tracking
#     Q_smooth  = P[-4]   # control smoothness
#     Q_obs     = P[-3]   # obstacle avoidance
#     Q_speed   = P[-2]   # speed regularization
#     Q_theta   = P[-1]   # new heading‐error weight

    
    
#     obj = 0
#     g = [X[:, 0] - P[0:3]]
#     # before the loop, compute this once:
#     base_theta = 3 + 2*N 

#     for k in range(N):
        
#         st = X[:, k]
#         u  = U[:, k]

#         # pull out the reference heading:
#         theta_ref_k = P[base_theta + k]

#         if k == 0:
#             # ── FIRST STEP: only pay heading error, no position cost ──
#             obj += Q_theta * (st[2] - theta_ref_k)**2
#         else:
#             # ── NORMAL: position + heading cost ──
#             ref = ca.vertcat(P[3 + 2*k], P[3 + 2*k + 1])
#             obj += Q_goal  * ca.sumsqr(st[0:2] - ref)
#             obj += Q_theta * (st[2] - theta_ref_k)**2
            
        
#         st = X[:, k]
#         u = U[:, k]
#         # extract the k-th waypoint (x,y)
#         ref = ca.vertcat(P[3 + 2*k], P[3 + 2*k + 1])
#         # extract the k-th reference heading (θ_ref)
#         base_theta = 3 + 2*N
#         theta_ref_k = P[base_theta + k]
        
#         obj += Q_goal * ca.sumsqr(st[0:2] - ref)
#         # penalize heading misalignment
#         # wrap the error into [–π, π] then square it
#         err = ca.fmod(st[2] - theta_ref_k + ca.pi, 2*ca.pi) - ca.pi
#         obj += Q_theta * err**2
#         if k > 0:
#             obj += Q_smooth * ca.sumsqr(u - U[:, k-1])
#         obj += Q_speed * u[0]**2
#         base = 3 + 2*N + N
#         for i in range(max_obs):
#             obs_pos = ca.vertcat(P[base + 2*(N*i + k)], P[base + 2*(N*i + k)+1])
#             dist = ca.norm_2(st[0:2] - obs_pos)
#             obj += Q_obs * (1.0 / (dist + 1e-2))**2
#         static_base = base + 2*N*max_obs
#         for j in range(max_static):
#             s_x = P[static_base + 2*j]
#             s_y = P[static_base + 2*j + 1]
#             dist_s = ca.norm_2(st[0:2] - ca.vertcat(s_x, s_y))
#             obj += Q_obs * (1.0 / (dist_s + 1e-2))**2
#         st_next = X[:, k+1]
#         f_val = f(st, u)
#         g.append(st_next - (st + T * f_val))
        
        
        
#     opt_vars = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
#     nlp_prob = {'f': obj, 'x': opt_vars, 'g': ca.vertcat(*g), 'p': P}
#     opts = {'ipopt.print_level': 0, 'print_time': 0}
#     solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
#     return solver, f, T, N



def build_mpc_solver_random_obs(T=0.1, N=10, max_obs=4, max_static=4):
    x, y, theta = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('theta')
    states      = ca.vertcat(x, y, theta)
    v, omega    = ca.SX.sym('v'), ca.SX.sym('omega')
    controls    = ca.vertcat(v, omega)
    rhs         = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta), omega)
    f           = ca.Function('f', [states, controls], [rhs])

    U = ca.SX.sym('U', 2, N)
    X = ca.SX.sym('X', 3, N+1)
    P = ca.SX.sym('P',
        3              # initial state
      + 2*N            # waypoints
      +   N            # headings
      + 2*N*max_obs    # moving‐obs
      + 2*max_static   # static obs
      +   5            # weights
    )


    Q_goal   = P[-5]
    Q_smooth = P[-4]
    Q_obs    = P[-3]
    Q_speed  = P[-2]
    Q_theta  = P[-1]

    obj = 0
    g   = [X[:,0] - P[0:3]]

    base_theta = 3 + 2*N
    base_obs   = base_theta + N

    for k in range(N):
        st = X[:,k]
        u  = U[:,k]

        # references
        ref_xy      = P[3+2*k : 3+2*k+2]
        theta_ref_k = P[base_theta + k]

        if k < K_turn:
            # first step: only align heading
            err0 = ca.fmod(st[2] - theta_ref_k + ca.pi, 2*ca.pi) - ca.pi
            obj += Q_theta * err0**2
            obj += 1e-3 * u[1]**2    # tiny ω‐penalty to force non‐zero ω when err0≠0
        else:
            # from step 1 onward: full cost
            err_xy = st[0:2] - ref_xy
            err0   = ca.fmod(st[2] - theta_ref_k + ca.pi, 2*ca.pi) - ca.pi
            obj   += Q_goal   * ca.sumsqr(err_xy)
            obj   += Q_theta  * err0**2
            obj   += Q_speed  * u[0]**2
            obj   += Q_smooth * ca.sumsqr(u - U[:,k-1])

        # obstacle avoidance
        for i in range(max_obs):
            obs_pos = P[base_obs + 2*(N*i + k) : base_obs + 2*(N*i + k) + 2]
            dist    = ca.norm_2(st[0:2] - obs_pos)
            obj    += Q_obs * (1.0/(dist + 1e-2))**2

        static_base = base_obs + 2*N*max_obs
        for j in range(max_static):
            s_pos  = P[static_base + 2*j : static_base + 2*j + 2]
            dist_s = ca.norm_2(st[0:2] - s_pos)
            obj   += Q_obs * (1.0/(dist_s + 1e-2))**2

        # dynamics constraint
        st_next = X[:,k+1]
        f_val   = f(st, u)
        g.append(st_next - (st + T*f_val))

    opt_vars = ca.vertcat(ca.reshape(U, -1, 1),
                         ca.reshape(X, -1, 1))
    nlp_prob = {'f': obj,
                'x': opt_vars,
                'g': ca.vertcat(*g),
                'p': P}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob,
                       {'ipopt.print_level': 0, 'print_time': 0})
    return solver, f, T, N
