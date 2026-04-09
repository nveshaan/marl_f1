import numpy as np
from scipy.optimize import minimize

def compute_optimal_line(track_points, width=6.0):
    # ==========================================
    # FIX 1: AGGRESSIVE TRACK CLEANING
    # ==========================================
    # Gym stacks multiple points at the finish line. 
    # We only keep points that are actually moving forward.
    clean_points = [track_points[0]]
    for p in track_points[1:]:
        # Only keep the point if it's at least 1.0 unit away from the previous one
        if np.linalg.norm(p - clean_points[-1]) > 1.0:
            clean_points.append(p)
    track_points = np.array(clean_points)

    # Final check: Does the very last point overlap the starting line?
    if np.linalg.norm(track_points[-1] - track_points[0]) < 1.0:
        track_points = track_points[:-1]

    n = len(track_points)

    # ==========================================
    # MATH & OPTIMIZATION
    # ==========================================
    # 1. Tangents and Normals
    tangents = np.roll(track_points, -1, axis=0) - np.roll(track_points, 1, axis=0)
    normals = np.array([-tangents[:, 1], tangents[:, 0]]).T
    
    # Adding 1e-8 prevents the math from exploding if any points are too close
    normals /= (np.linalg.norm(normals, axis=1)[:, np.newaxis] + 1e-8) 

    # 2. Objective Function
    def objective(alpha):
        path = track_points + (alpha[:, np.newaxis] * normals * width)
        
        # 3-point stencil for curvature
        p_prev = np.roll(path, 1, axis=0)
        p_next = np.roll(path, -1, axis=0)
        dd = p_prev - 2*path + p_next
        
        return np.sum(dd**2)

    # 3. Constraints
    bounds = [(-0.9, 0.9) for _ in range(n)]
    init_alpha = np.zeros(n)

    # 4. Solve (Increased tolerance so it ignores micro-wobbles)
    res = minimize(objective, init_alpha, bounds=bounds, method='L-BFGS-B', tol=1e-3)

    # ==========================================
    # FIX 2: PLOT ALIGNMENT
    # ==========================================
    optimal_path = track_points + (res.x[:, np.newaxis] * normals * width)
    
    # Append the very first point back to the end so Matplotlib closes the circle!
    optimal_path = np.vstack([optimal_path, optimal_path[0]])
    
    return optimal_path