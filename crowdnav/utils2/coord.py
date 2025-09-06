# crowdnav/utils2/coord.py
import numpy as np

def grid_to_world(idx, res, shape):
    half_x = res * shape[1] / 2.0
    half_y = res * shape[0] / 2.0
    r, c = idx
    x = (c + 0.5) * res - half_x
    y = (r + 0.5) * res - half_y
    return np.array([x, y])

def world_to_grid(xy, res, shape):
    half_x = res * shape[1] / 2.0
    half_y = res * shape[0] / 2.0
    c = int((xy[0] + half_x)/res)
    r = int((xy[1] + half_y)/res)
    r = np.clip(r, 0, shape[0]-1); c = np.clip(c, 0, shape[1]-1)
    return int(r), int(c)

def wrap_angle(a):
    import numpy as np
    return (a + np.pi) % (2*np.pi) - np.pi



def _coord_self_test(grid_shape, res):
    import numpy as np
    from .coord import world_to_grid as w2g, grid_to_world as g2w

    H, W = grid_shape
    cx = (W * res) / 2.0
    cy = (H * res) / 2.0

    # 1) Origin should map to center cell (or one of the two center cells if even dims)
    rc0 = w2g((0.0, 0.0), res, (H, W))
    xy0 = g2w(rc0, res, (H, W))
    err0 = np.linalg.norm(np.asarray(xy0) - np.array([0.0, 0.0]))
    print(f"[COORD] origin→rc={rc0} rc→xy={xy0} err={err0:.4f}")
    assert err0 <= res * 0.75, "Origin round-trip too large → center/offset logic wrong"

    # 2) Unit step along +x should bump col by +1; +y should bump row by +1
    rc_xp = w2g((res, 0.0), res, (H, W))
    rc_yp = w2g((0.0, res), res, (H, W))
    print(f"[COORD] +x→rc={rc_xp}  +y→rc={rc_yp}")
    # deltas measured from origin cell
    dr_x = rc_xp[0] - rc0[0]; dc_x = rc_xp[1] - rc0[1]
    dr_y = rc_yp[0] - rc0[0]; dc_y = rc_yp[1] - rc0[1]
    assert (dr_x, dc_x) == (0, 1), f"+x should be (0,+1) cols; got {(dr_x, dc_x)}"
    assert (dr_y, dc_y) == (1, 0), f"+y should be (+1,0) rows; got {(dr_y, dc_y)}"

    # 3) Random round-trip accuracy
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.uniform(-cx + 0.1, cx - 0.1)
        y = rng.uniform(-cy + 0.1, cy - 0.1)
        rc = w2g((x, y), res, (H, W))
        xy = g2w(rc, res, (H, W))
        err = np.linalg.norm(np.asarray(xy) - np.array([x, y]))
        assert err <= np.sqrt(2)*res, f"Round-trip > cell diag: {err:.3f} at {(x,y)}"

    print("[COORD] ✅ conventions consistent")
