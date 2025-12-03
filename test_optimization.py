import numpy as np
from mapping.occupancy_grid import fast_raycast
from mapping.fusion_server import fast_fusion_update

def test_fast_raycast():
    print("Testing fast_raycast...")
    grid_dim = 100
    grid = np.zeros((grid_dim, grid_dim), dtype=np.uint8)
    center = 50
    
    # Test a few rays
    end_r = np.array([50, 60, 40], dtype=np.int32)
    end_c = np.array([60, 50, 40], dtype=np.int32)
    
    fast_raycast(grid, center, center, end_r, end_c)
    
    # Check if lines are drawn (set to 1)
    assert grid[50, 55] == 1
    assert grid[55, 50] == 1
    print("fast_raycast passed!")

def test_fast_fusion_update():
    print("Testing fast_fusion_update...")
    grid_dim = 100
    log_odds_map = np.zeros((grid_dim, grid_dim), dtype=np.float32)
    
    # Dummy local indices
    local_indices_r = np.array([0, 0, 1], dtype=np.int32)
    local_indices_c = np.array([0, 1, 0], dtype=np.int32)
    
    gx, gy = 0.0, 0.0
    gyaw = 0.0
    grid_size = 1.0
    max_x = 50.0
    min_y = -50.0
    local_center = 1 # 3x3 grid, center at 1
    update_val = 0.5
    
    fast_fusion_update(log_odds_map, local_indices_r, local_indices_c, 
                      gx, gy, gyaw, grid_size, max_x, min_y, grid_dim, 
                      local_center, update_val)
    
    # Check if map updated
    # Center of global map is at index (50, 50) roughly
    # With gx=0, gy=0, vehicle is at center.
    # Local (0,0) is top-left of local grid.
    # If local_center is 1, then (0,0) is (-1, -1) relative to vehicle.
    # Global X = x_veh + gx = (1-0)*1 + 0 = 1
    # Global Y = y_veh + gy = (0-1)*1 + 0 = -1
    # Global R = (50 - 1) / 1 = 49
    # Global C = (-1 - (-50)) / 1 = 49
    
    print(f"Value at (49, 49): {log_odds_map[49, 49]}")
    assert log_odds_map[49, 49] == 0.5
    print("fast_fusion_update passed!")

if __name__ == "__main__":
    test_fast_raycast()
    test_fast_fusion_update()
