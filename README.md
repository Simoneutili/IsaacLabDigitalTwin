# Digital Twin of TIAGo Robotic Manipulator

This repository contains the code and configuration files for a thesis project that develops a high-fidelity digital twin of the TIAGo robotic manipulator. The project leverages a real-to-sim approach combined with advanced Reinforcement Learning techniques to optimize the dynamic parameters of the simulated model in real time, ensuring that its behavior closely mirrors that of the physical robot.

## Project Overview

This work presents an architecture that:
- **Enhances Simulation Fidelity:** Improves the simulation's accuracy by online optimization of dynamic parameters.
- **Real-Time Parameter Optimization:** Uses a Reinforcement Learning (RL) algorithm to adaptively estimate and optimize joint parameters, minimizing velocity errors.
- **Curriculum Learning:** Gradually calibrates the seven joint parameters of the robot arm during training.
- **Algorithm Comparison:** Compares state-of-the-art Deep RL algorithms (SAC and PPO) via Stable-Baselines3.
- **Hyperparameter Tuning:** Implements a pipeline to fine-tune learning curves and increase cumulative rewards.
- **Integration of Simulation and Reality:** Validates the Digital Twin using Gazebo simulated TIAGo as a surrogate for the real robot (Sim2Sim).
- - **Transfer Learning via Domain Randomization:** Leverages domain randomization to bridge the simulation-to-reality gap. The policy trained on Gazebo ground truth is effectively transferred to the real robot.

## Repository Structure

- **tiago_public_ws/**: ROS2 workspace for the TIAGo simulation.
- **scripts/TIAGO_RL/**: Contains RL environment definition, training scripts, hyperparameter tuning modules, and deployment procedures.
- **IsaacLabTiagoExtension/**: Integration components for NVIDIA Isaac Lab with the TIAGo simulation.

## Installation and Setup

### 1. Install the ROS2 TIAGo Simulation Workspace

Clone the repository and navigate to the workspace:
  
    cd ~/Repository_name/tiago_public_ws


Build and Source the Workspace

    colcon build --symlink-install
    source ~/Repository_name/tiago_public_ws/install/setup.bash

## 2. Run Gazebo TIAGo ROS2 Simulation

Launch the Gazebo simulation, which acts as a surrogate for the real robot:

```bash
ros2 launch tiago_gazebo tiago_gazebo.launch.py is_public_sim:=True [arm_type:=no-arm]
```

## 3. Activate the Bridge Node Between Isaac Lab and Gazebo

Establish communication between the simulation and Isaac Lab by running:

```bash
ros2 run tiago_gazebo bridge_velocities_reset.py
```

## Running the RL and Digital Twin Training

_All commands below should be executed from inside the `IsaacLabTiagoExtension` directory._

### 4. Hyperparameter Tuning Process

Run the hyperparameter tuning process using the provided configuration file and job class:

```bash
python scripts/TIAGO_RL/tiago_RL/DigitalTwin/hyperparameter_tuning/tuner.py \
    --run_mode local \
    --cfg_file TIAGO_RL/tiago_RL/DigitalTwin/hyperparameter_tuning/tiago_cfg_file.py \
    --cfg_class TiAGO_SB3SAC_JobCfg
```

### 5. TensorBoard Metrics Visualization

Launch TensorBoard to monitor training metrics:

```bash
tensorboard --logdir logs/sb3/Isaac-DigitalTwin-TiAGO-v0
```

### 6. Training the RL Agent

Before training, pick the optimal set of hyperparameters and update the agent YAML file located in:

```
scripts/TIAGO_RL/tiago_RL/DigitalTwin/agents
```

#### Training Options

**Training from Scratch:**

```bash
python scripts/TIAGO_RL/tiago_RL/ISAACLAB_TEST/SB3/deploy.py --task Isaac-DigitalTwin-TiAGO-v0 --num_envs 256
```

**Training with a Pretrained Model:**

```bash
python TIAGO_RL/tiago_RL/ISAACLAB_TEST/SB3/pretrain.py --task Isaac-DigitalTwin-TiAGO-v0 --num_envs 64 --headless --checkpoint_path /home/simone/IsaacLab/logs/sb3/Isaac-DigitalTwin-TiAGO-v0/best/model.zip
```

### 7. Deploying the Trained Policy

After training, deploy the policy for real-time testing:

```bash
python TIAGO_RL/tiago_RL/ISAACLAB_TEST/SB3/deploy.py --task Isaac-DigitalTwin-TiAGO-v0 --num_envs 1
```

## Requirements

Before running the project, ensure you have the following installed:

- **ROS2**
- **Gazebo**
- **NVIDIA Isaac Sim & Isaac Lab**
- **Stable-Baselines3**
- **Python 3.x**

