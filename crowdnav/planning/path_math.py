# crowdnav/planning/path_math.py
import numpy as np

def _cumlen(poly: np.ndarray) -> np.ndarray:
    """Cumulative arc-length (meters) along a 2D polyline."""
    if poly.shape[0] <= 1:
        return np.array([0.0], dtype=float)
    seg = np.diff(poly, axis=0)
    d = np.linalg.norm(seg, axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])

def resample_equal(poly: np.ndarray, step: float) -> np.ndarray:
    """
    Resample a polyline at ~equal arc-length spacing 'step' (meters).
    Keeps endpoints; linear interpolation on each segment.
    """
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    S = _cumlen(poly)
    L = float(S[-1])
    if L < 1e-9:
        return poly[:1].copy()
    sgrid = np.arange(0.0, L + 0.5*step, step)
    out = []
    j = 0
    for s in sgrid:
        while j + 1 < len(S) and S[j+1] < s:
            j += 1
        if j + 1 >= len(S):
            out.append(poly[-1]); continue
        denom = max(S[j+1] - S[j], 1e-9)
        t = (s - S[j]) / denom
        out.append((1.0 - t) * poly[j] + t * poly[j+1])
    return np.asarray(out, dtype=float)

def project_to_polyline(xy: np.ndarray, poly: np.ndarray):
    """
    Orthogonally project 'xy' onto 'poly' (equal-spaced polyline).
    Returns: (s_star, proj_xy, seg_idx, tangent_unit)
    """
    xy = np.asarray(xy, dtype=float)
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    S = _cumlen(poly)
    best = (1e18, 0.0, poly[0], 0, np.array([1.0, 0.0]))
    for i in range(len(poly) - 1):
        a = poly[i]; b = poly[i+1]; ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            # degenerate; treat as a point
            d2 = float(np.dot(xy - a, xy - a))
            if d2 < best[0]:
                best = (d2, float(S[i]), a, i, np.array([1.0, 0.0]))
            continue
        t = float(np.dot(xy - a, ab) / L2)
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        p = a + t * ab
        d2 = float(np.dot(xy - p, xy - p))
        if d2 < best[0]:
            s_here = float(S[i] + t * (S[i+1] - S[i]))
            tan = ab / np.sqrt(L2)
            best = (d2, s_here, p, i, tan)
    _, s_star, proj, seg_idx, tan = best
    return s_star, proj.astype(float), int(seg_idx), tan.astype(float)

def sample_forward(poly_equal: np.ndarray, s_star: float, s_ahead: float, N: int, ds: float):
    """
    Return N points starting from arc-length s_star + s_ahead, spaced by ds.
    Assumes 'poly_equal' was created by resample_equal with step≈ds (or close).
    """
    poly_equal = np.asarray(poly_equal, dtype=float).reshape(-1, 2)
    step = ds if ds > 1e-9 else 0.05
    idx0 = int(round((s_star + s_ahead) / step))
    idxs = np.clip(idx0 + np.arange(N), 0, len(poly_equal) - 1)
    return poly_equal[idxs]
