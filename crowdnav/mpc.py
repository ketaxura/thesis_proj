import casadi as ca

def build_mpc_solver_random_obs(T=0.1, N=10, max_obs=4, max_static=4):
    """Build MPC solver for obstacle avoidance."""
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    v = ca.SX.sym('v')
    omega = ca.SX.sym('omega')
    controls = ca.vertcat(v, omega)
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
    f = ca.Function('f', [states, controls], [rhs])
    U = ca.SX.sym('U', 2, N)
    X = ca.SX.sym('X', 3, N + 1)
    P = ca.SX.sym('P', 3 + 2*N + 2*N*max_obs + 2*max_static + 4)
    Q_goal = P[-4]
    Q_smooth = P[-3]
    Q_obs = P[-2]
    Q_speed = P[-1]
    obj = 0
    g = [X[:, 0] - P[0:3]]
    for k in range(N):
        st = X[:, k]
        u = U[:, k]
        ref = ca.vertcat(P[3 + 2*k], P[3 + 2*k + 1])
        obj += Q_goal * ca.sumsqr(st[0:2] - ref)
        if k > 0:
            obj += Q_smooth * ca.sumsqr(u - U[:, k-1])
        obj += Q_speed * u[0]**2
        base = 3 + 2*N
        for i in range(max_obs):
            obs_pos = ca.vertcat(P[base + 2*(N*i + k)], P[base + 2*(N*i + k)+1])
            dist = ca.norm_2(st[0:2] - obs_pos)
            obj += Q_obs * (1.0 / (dist + 1e-2))**2
        static_base = base + 2*N*max_obs
        for j in range(max_static):
            s_x = P[static_base + 2*j]
            s_y = P[static_base + 2*j + 1]
            dist_s = ca.norm_2(st[0:2] - ca.vertcat(s_x, s_y))
            obj += Q_obs * (1.0 / (dist_s + 1e-2))**2
        st_next = X[:, k+1]
        f_val = f(st, u)
        g.append(st_next - (st + T * f_val))
    opt_vars = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
    nlp_prob = {'f': obj, 'x': opt_vars, 'g': ca.vertcat(*g), 'p': P}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    return solver, f, T, N