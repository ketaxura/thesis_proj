# crowdnav/planning/mpc_runner.py
import numpy as np
import casadi as ca



def _nnz_size(tup_or_int):
    # CasADi returns (rows, cols) tuples for sizes in some builds.
    if isinstance(tup_or_int, tuple):
        r, c = tup_or_int
        return int(r * c)
    return int(tup_or_int)

def _wrap(a):
    # wrap-to-pi
    return (a + np.pi) % (2*np.pi) - np.pi

def _rollout_guess(x0, U, T, N):
    # x0: [x,y,θ]; U: flat [v0,ω0, v1,ω1, ...], length 2N
    X = np.zeros((N+1, 3), dtype=float)
    X[0] = x0
    for k in range(N):
        v = U[2*k + 0]
        w = U[2*k + 1]
        x, y, th = X[k]
        X[k+1, 0] = x + T * v * np.cos(th)
        X[k+1, 1] = y + T * v * np.sin(th)
        X[k+1, 2] = _wrap(th + T * w)
    return X


def _wrap_angle_np(a):
    return np.arctan2(np.sin(a), np.cos(a))

class MPCRunner:
    def __init__(self, solver, T, N, f_dyn=None):
        self.solver = solver
        self.T = float(T)
        self.N = int(N)
        self.f_dyn = f_dyn
        self._U_prev = None

        # decision vector length
        try:
            self.nx_dec = _nnz_size(self.solver.size_in('x0'))
        except Exception:
            # fallback for some nlpsol wrappers
            self.nx_dec = _nnz_size(self.solver.size_out('x'))

        # number of constraint equations (if any)
        try:
            self.ng_con = _nnz_size(self.solver.size_out('g'))
        except Exception:
            self.ng_con = 0

        # Optional: print available input names once at DEBUG to sanity-check
        try:
            in_names = [self.solver.name_in(i) for i in range(self.solver.n_in())]
            print(f"[MPCRunner] solver inputs: {in_names}")  # or use your logger.debug
        except Exception:
            pass

    def _seed_controls(self, v_des):
        """Build a length-2N control warm start."""
        N = self.N
        if self._U_prev is not None and self._U_prev.size == 2*N:
            u0 = np.copy(self._U_prev)
            u0[:-2] = u0[2:]
            u0[-2:] = u0[-4:-2]
        else:
            u0 = np.zeros(2*N, dtype=float)
            for k in range(N):
                u0[2*k]   = v_des
                u0[2*k+1] = 0.0
        return u0

    def _rollout_states(self, x0, U):
        """
        Roll forward N steps with f_dyn if provided, else unicycle Euler model.
        Returns X list length N+1 (including x0).
        """
        X = [np.asarray(x0, dtype=float).ravel()]
        xk = X[0].copy()
        T  = self.T
        for k in range(self.N):
            v, omg = float(U[2*k]), float(U[2*k+1])
            if self.f_dyn is not None:
                # f_dyn is casadi-style; evaluate numerically
                xk = np.array(self.f_dyn(ca.DM(xk), ca.DM([v, omg]))).astype(float).ravel()
            else:
                # fallback unicycle
                x, y, th = xk
                x  = x + T * v * np.cos(th)
                y  = y + T * v * np.sin(th)
                th = _wrap_angle_np(th + T * omg)
                xk = np.array([x, y, th], dtype=float)
            X.append(xk)
        return np.array(X)  # shape (N+1, 3)




    def solve(self, state, path_xy, theta_ref,
            obs_traj, static_flat, weights,
            v_min_soft, v_max, omega_max):
        """
        Solve the MPC and return (v, omega, U, debug).
        Hard-enforce v_min_soft via lbx/ubx on the control entries.
        """
        N = self.N
        state = np.asarray(state, dtype=float).ravel()
        path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
        M = path_xy.shape[0]
        if M < N:
            pad = np.repeat(path_xy[-1:,:], N - M, axis=0)
            path_xy = np.vstack([path_xy, pad])
        elif M > N:
            path_xy = path_xy[:N]
            
            
        theta_ref = np.asarray(theta_ref, dtype=float).reshape(-1)
        if theta_ref.shape[0] < N:
            theta_ref = np.concatenate([theta_ref, np.repeat(theta_ref[-1], N - theta_ref.shape[0])])
        elif theta_ref.shape[0] > N:
            theta_ref = theta_ref[:N]

        # ---- Pack parameter vector exactly as expected by build_mpc_solver_random_obs() ----
        # P = [ x0(3), path_xy(2N), theta_ref(N), weights(6), v_des(1) ]
        
        
        
        v_des = 0.8 * float(v_max)

        # static_flat is already padded to 2*max_static in env.step()
        static_vec = np.asarray(static_flat, float).ravel()

        P = np.concatenate([
            np.asarray(state, float).ravel(),        # 3
            np.asarray(path_xy, float).ravel(),      # 2N
            np.asarray(theta_ref, float).ravel(),    # N
            static_vec,                              # 2*max_static
            np.asarray(weights, float).ravel(),      # 6
            np.array([v_des], float)                 # 1
        ]).astype(float)

        # (Optional but handy) assert the size matches the solver's expectation
        try:
            p_len = int(np.prod(self.solver.size_in('p')))
        except Exception:
            p_len = len(P)
        assert P.size == p_len, f"P has {P.size} elems but solver expects {p_len}"

        # ---- Warm start for controls U ----
        if self._U_prev is not None and self._U_prev.size == 2*N:
            u0 = self._U_prev.copy()
            # shift one step (optional); here we just reuse as-is
        else:
            u0 = np.zeros(2*N, dtype=float)
            for k in range(N):
                u0[2*k]   = v_des
                u0[2*k+1] = 0.0

        # ---- Figure out decision layout ----
        want   = self.nx_dec
        only_U = (2*N == want)
        XU     = (3*(N+1) + 2*N == want)   # [X(:); U(:)]

        # ---- Build x0_dec and bounds lbx/ubx ----
        if only_U:
            # decision is U only
            x0_dec = u0

            # bounds directly on U entries
            lbx = np.empty_like(x0_dec)
            ubx = np.empty_like(x0_dec)
            for k in range(N):
                lbx[2*k]   = float(v_min_soft)
                ubx[2*k]   = float(v_max)
                lbx[2*k+1] = -float(omega_max)
                ubx[2*k+1] =  float(omega_max)

        elif XU:
            # multiple-shooting: pack [X(0..N), U(0..N-1)]
            # Make a simple rollout for X warm start
            X_seed = [state.copy()]
            xk = state.copy()
            T  = self.T
            for k in range(N):
                v, omg = float(u0[2*k]), float(u0[2*k+1])
                # minimalist unicycle rollout (consistent with solver model)
                x, y, th = xk
                x  = x + T * v * np.cos(th)
                y  = y + T * v * np.sin(th)
                th = np.arctan2(np.sin(th + T*omg), np.cos(th + T*omg))  # wrap
                xk = np.array([x, y, th], dtype=float)
                X_seed.append(xk)
            X_seed = np.asarray(X_seed)  # (N+1,3)

            x0_dec = np.concatenate([X_seed.ravel(), u0], axis=0)

            # states free; controls bounded at the tail
            lbx = -np.inf * np.ones_like(x0_dec)
            ubx =  np.inf * np.ones_like(x0_dec)

            # tail indices for U inside [X;U]
            base = 3*(N+1)
            for k in range(N):
                i_v = base + 2*k
                i_w = base + 2*k + 1
                lbx[i_v] = float(v_min_soft)
                ubx[i_v] = float(v_max)
                lbx[i_w] = -float(omega_max)
                ubx[i_w] =  float(omega_max)

        else:
            raise RuntimeError(
                f"Unexpected solver decision size {want}. "
                f"Expected 2N={2*N} (single-shooting) or 3*(N+1)+2N={3*(N+1)+2*N} (multiple-shooting)."
            )

        # ---- Call solver ----
        kwargs = dict(x0=x0_dec, p=P, lbx=lbx, ubx=ubx)
        if self.ng_con > 0:
            # if your NLP has equality/inequality constraints g, set them here
            kwargs.update(lbg=[0.0]*self.ng_con, ubg=[0.0]*self.ng_con)

        sol = self.solver(**kwargs)
        Xout = np.array(sol['x']).squeeze()

        # ---- Extract the first control to apply ----
        if only_U:
            U = Xout
        else:
            U = Xout[-2*N:]  # controls live at the tail

        self._U_prev = U.copy()
        v_cmd     = float(U[0])
        omega_cmd = float(U[1])

        stats = self.solver.stats() if hasattr(self.solver, "stats") else {}
        mpc_dbg = {
            "solver_ok": True,
            "solver_iters": stats.get("iter_count", -1),
            "U_minmax_v": (
                float(U[0::2].min()) if U.size else 0.0,
                float(U[0::2].max()) if U.size else 0.0
            ),
            "v_min_soft_used": float(v_min_soft),
            "v_max_used": float(v_max),
            "omega_max_used": float(omega_max),
        }
        return v_cmd, omega_cmd, U, mpc_dbg
