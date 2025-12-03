import numpy as np
from hybrid_planner import _check_line_numba
from utils.ate import fast_compute_errors

def test_check_line_numba():
    print("Testing _check_line_numba...")
    grid_dim = 100
    global_map = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    threshold = 0.6
    
    # Add an obstacle
    global_map[50, 50] = 0.8
    
    # Test line passing through obstacle
    # Start (40, 40) -> End (60, 60) passes through (50, 50)
    blocked = _check_line_numba(global_map, 40, 40, 60, 60, grid_dim, threshold)
    assert blocked == True
    print("Blocked path detected correctly.")
    
    # Test clear path
    # Start (40, 45) -> End (60, 45) should be clear
    blocked = _check_line_numba(global_map, 40, 45, 60, 45, grid_dim, threshold)
    assert blocked == False
    print("Clear path detected correctly.")
    print("_check_line_numba passed!")

def test_fast_compute_errors():
    print("Testing fast_compute_errors...")
    # Create dummy trajectories (N, 7)
    # x, y, z, qx, qy, qz, qw
    n = 10
    traj_est = np.zeros((n, 7), dtype=np.float64)
    traj_gt = np.zeros((n, 7), dtype=np.float64)
    
    # Fill with some data
    for i in range(n):
        traj_est[i, :3] = [i, i, 0]
        traj_gt[i, :3] = [i + 1, i, 0] # 1m error in X
        
        # Identity quaternion
        traj_est[i, 3:] = [0, 0, 0, 1]
        traj_gt[i, 3:] = [0, 0, 0, 1]
        
    p_errs, r_errs = fast_compute_errors(traj_est, traj_gt)
    
    assert len(p_errs) == n
    assert len(r_errs) == n
    
    # Check position error (should be 1.0 for all)
    assert np.allclose(p_errs, 1.0)
    print("Position errors correct.")
    
    # Check rotation error (should be 0.0)
    assert np.allclose(r_errs, 0.0)
    print("Rotation errors correct.")
    print("fast_compute_errors passed!")

if __name__ == "__main__":
    test_check_line_numba()
    test_fast_compute_errors()
