# MNE-Lite Distributed SLAM Architecture
# Exports for mapping module

# Core components (no cv2 dependency)
from .submap import Submap
from .submap_manager import SubmapManager
from .loop_closure import LoopClosureDetector, LoopClosure, compute_icp_2d
from .pose_graph import PoseGraph, PoseNode, PoseEdge

# FusionServer imports (requires cv2)
# Import these directly when needed:
#   from mapping.fusion_server_mne import FusionServerMNELite
#   from mapping.fusion_server import FusionServer
