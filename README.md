

# Project: Multi-Agent Map Fusion & Cooperative Planning

## Project Goal
This project implements a multi-agent cooperative system in CARLA. It features two autonomous agents that explore the environment, generating local occupancy grids from LiDAR data. These local maps are fused into a central global map using a log-odds Bayesian update approach. A `FusionServer` manages this global state, which is then shared back to the agents for cooperative path planning. The system includes a `HybridRoutePlanner` that allows agents to detect and avoid obstacles discovered by other agents, and a real-time HUD visualization of the fused global map.

## How to Run

### Prerequisite
Ensure you have a conda environment set up with the necessary dependencies (CARLA, numpy, pygame, opencv-python).

```bash
conda create -n carla_openpcdet python=3.7
conda activate carla_openpcdet
pip install numpy pygame opencv-python
```

### Step 1: Start the CARLA Server
Before running the python client, you must have the CARLA server running. Navigate to your CARLA installation folder and run:

```bash
# Standard run (with window)
./CarlaUE4.sh

# Headless mode (no window)
./CarlaUE4.sh -RenderOffScreen

# Low-power mode (if you experience lag)
DRI_PRIME=1 ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -opengl
```

### Step 2: Run the Simulation
To run the multi-agent simulation:

```bash
# Activate your conda environment
conda activate carla_openpcdet

# Run the script
python3 multi_agent_control.py
```

## Controls & Usage
- **TAB**: Switch camera view between Agent A and Agent B.
- **ESC** or **Ctrl+Q**: Quit the simulation.
- **N**: Switch between sensor data types (LiDAR, Semantic LiDAR, Camera).

## Visualization
The HUD displays a real-time overlay of the global map in the bottom-right corner:
- **Black**: Free Space
- **White**: Occupied Space
- **Gray**: Unknown Space
- **Blue Dot**: Agent 0
- **Red Dot**: Agent 1

## Key Components
- **`multi_agent_control.py`**: Main entry point, handles simulation loop and agents.
- **`fusion_server.py`**: Manages the global log-odds occupancy grid.
- **`mapping.py`**: Handles local LiDAR-to-grid mapping for each agent.
- **`hybrid_planner.py`**: Checks the global map for path blockages.
