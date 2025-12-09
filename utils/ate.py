'''
UC Riverside
EE260: Introduction to Self-Driving Stack
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from numba import njit

@njit
def fast_compute_errors(traj_est, traj_gt):
    n = len(traj_est)
    # Pre-allocate output arrays for speed
    pos_errors = np.zeros(n, dtype=np.float64)
    rot_errors = np.zeros(n, dtype=np.float64)

    for i in range(n):
        est = traj_est[i]
        gt = traj_gt[i]

        # Position Error (Euclidean)
        dx = est[0] - gt[0]
        dy = est[1] - gt[1]
        dz = est[2] - gt[2]
        pos_errors[i] = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Rotation Error (Quaternion Dot Product)
        # Assuming [w, x, y, z] layout. 
        # Note: The original code used est[3:] which is 4 elements.
        # We need to be careful about layout. CARLA usually provides (x, y, z, roll, pitch, yaw) or similar.
        # But ATE class reshapes to (N, 7). 
        # Let's assume 3:7 are quaternion (qx, qy, qz, qw) or similar.
        # The original code did: q_est = est[3:]
        
        q_est = est[3:]
        q_gt = gt[3:]
        
        # Manual normalization to avoid numpy overhead inside loop
        norm_est = np.sqrt(np.sum(q_est**2))
        norm_gt = np.sqrt(np.sum(q_gt**2))
        
        if norm_est > 1e-6 and norm_gt > 1e-6:
            # Dot product
            dot = 0.0
            for j in range(4):
                dot += (q_est[j]/norm_est) * (q_gt[j]/norm_gt)
            
            dot = np.abs(dot)
            if dot > 1.0: dot = 1.0
            
            rot_errors[i] = 2.0 * np.arccos(dot)
        else:
            rot_errors[i] = 0.0
            
    return pos_errors, rot_errors

class AbsoluteTrajectoryError:
    def __init__(self, traj_gt, traj_est):
        # Handle Data Shapes: Ensure (N, 7) format
        self.traj_gt = np.array(traj_gt)
        if self.traj_gt.size > 0:
            self.traj_gt = self.traj_gt.reshape(-1, 7)
            
        self.traj_est = np.array(traj_est)
        if self.traj_est.size > 0:
            self.traj_est = self.traj_est.reshape(-1, 7)
            
        self.traj_err = None
        
        # Align trajectories immediately upon initialization
        self.align_trajectories()

    def align_trajectories(self):
        """
        Aligns the estimated trajectory to the ground truth using Umeyama alignment 
        (Rotation + Translation). 
        """
        if len(self.traj_est) < 2 or len(self.traj_gt) < 2:
            print("ATE WARNING: Not enough points to align trajectories.")
            return

        # 1. Truncate to matching length
        n = min(len(self.traj_est), len(self.traj_gt))
        self.traj_est = self.traj_est[:n]
        self.traj_gt = self.traj_gt[:n]

        # 2. Extract Positions (3 x N)
        gt_xyz = self.traj_gt[:, :3].T
        est_xyz = self.traj_est[:, :3].T

        # 3. Center the trajectories
        gt_centroid = np.mean(gt_xyz, axis=1).reshape(3, 1)
        est_centroid = np.mean(est_xyz, axis=1).reshape(3, 1)

        gt_centered = gt_xyz - gt_centroid
        est_centered = est_xyz - est_centroid

        # 4. Compute Rotation Matrix (SVD)
        H = est_centered @ gt_centered.T
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # 5. Apply Alignment
        aligned_xyz = (R @ est_centered) + gt_centroid
        self.traj_est[:, :3] = aligned_xyz.T

    def compute_trajectory_error(self):
        """Compute ATE between aligned trajectories"""
        # Ensure arrays are float64 for Numba
        
        # DEBUG: Print shapes
        # print(f"DEBUG: traj_est shape: {self.traj_est.shape}")
        # print(f"DEBUG: traj_gt shape: {self.traj_gt.shape}")

        if len(self.traj_est.shape) == 1:
            self.traj_est = self.traj_est.reshape(-1, 7)
        if len(self.traj_gt.shape) == 1:
            self.traj_gt = self.traj_gt.reshape(-1, 7)

        est = self.traj_est.astype(np.float64)
        gt = self.traj_gt.astype(np.float64)
        
        # Verify Numba-compatible array (C-contiguous usually helps)
        est = np.ascontiguousarray(est)
        gt = np.ascontiguousarray(gt)
        
        p_errs, r_errs = fast_compute_errors(est, gt)
        
        # Convert back to dictionary format if needed, or just store arrays
        self.traj_err = []
        for i in range(len(p_errs)):
            self.traj_err.append({'position_error': p_errs[i], 'rotation_error': r_errs[i]})
        
        return self.traj_err
    
    def get_statistics(self):
        """Return RMSE and other stats"""
        if not self.traj_err:
            return {
                'rmse_position': 0.0, 'mean_position': 0.0, 'median_position': 0.0, 'std_position': 0.0, 'min_position': 0.0, 'max_position': 0.0,
                'rmse_rotation': 0.0, 'mean_rotation': 0.0, 'median_rotation': 0.0, 'std_rotation': 0.0, 'min_rotation': 0.0, 'max_rotation': 0.0
            }

        pos_errors = [e['position_error'] for e in self.traj_err]
        rot_errors = [e['rotation_error'] for e in self.traj_err]
        
        return {
            'rmse_position': np.sqrt(np.mean(np.square(pos_errors))),
            'mean_position': np.mean(pos_errors),
            'median_position': np.median(pos_errors),
            'std_position': np.std(pos_errors),
            'min_position': np.min(pos_errors),
            'max_position': np.max(pos_errors),
            
            'rmse_rotation': np.sqrt(np.mean(np.square(rot_errors))),
            'mean_rotation': np.mean(rot_errors),
            'median_rotation': np.median(rot_errors), # Fixed: Added back
            'std_rotation': np.std(rot_errors),       # Fixed: Added back
            'min_rotation': np.min(rot_errors),
            'max_rotation': np.max(rot_errors)
        }

    def plot_traj(self, save_path=None):
        """Plot the aligned trajectories"""
        if len(self.traj_est) == 0:
            print("ATE WARNING: No estimated trajectory to plot.")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        est = self.traj_est[:, :3]
        gt = self.traj_gt[:, :3]
        
        ax.plot(gt[:,0], gt[:,1], gt[:,2], 'b--', label='Ground Truth', linewidth=2)
        ax.plot(est[:,0], est[:,1], est[:,2], 'r-', label='Estimated (Aligned)', linewidth=2)
        
        ax.scatter(gt[0,0], gt[0,1], gt[0,2], c='green', s=100, label='Start')
        ax.scatter(gt[-1,0], gt[-1,1], gt[-1,2], c='black', marker='x', s=100, label='End')
        
        stats = self.get_statistics()
        ax.set_title(f"Trajectory Comparison\nRMSE Pos: {stats.get('rmse_position', 0):.2f}m")
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.legend()
        
        all_pts = np.vstack((est, gt))
        max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2.0
        mid = (all_pts.max(0) + all_pts.min(0)) * 0.5
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    pass