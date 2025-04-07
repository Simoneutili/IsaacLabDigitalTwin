# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import the required modules
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import torch
import os
import gymnasium as gym
import numpy as np
from omni.isaac.lab.assets import Articulation, ArticulationCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg
import seaborn as sns

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
import matplotlib.pyplot as plt


# Pre-defined configs
root_path = "/home/simone/Desktop/TESI/MYVERSION/TiagoExtension/data/robots/urdf"
urdf_path = os.path.join(root_path, "tiago.urdf")
usd_path = os.path.join(os.path.dirname(__file__), "tiago.usd")

TIAGO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=urdf_path,
        usd_dir=usd_path,
        merge_fixed_joints=True,
        fix_base=True,
        make_instanceable=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
        ), articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0), joint_pos={
            "arm_1_joint": 0.5,
            "arm_2_joint": 0.5, 
            "arm_3_joint": 0.5, 
            "arm_4_joint": 0.5,
            "arm_5_joint": 0.0, 
            "arm_6_joint": 0.0, 
            "arm_7_joint": 0.0
        }
    ),
    actuators={
        "torso": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=500.0),
        "tiago_arm": ImplicitActuatorCfg(joint_names_expr=["arm_[1-7]_joint"], stiffness=0.0, damping=500.0),
    },
)

# Define Tiago environment configuration


# ROS2 Publisher Node for Isaac Sim

class ReferenceVelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher_ = self.create_publisher(JointState, "/reference_velocities", 10)  # Use JointState
        self.msg = JointState()
        self.msg.name = [f"arm_{i+1}_joint" for i in range(7)]  # Joint names for 7-DOF

    def publish_velocity(self, velocities):
        if rclpy.ok():  # Check if the ROS context is still valid
            self.msg.velocity = velocities.tolist()  # Set velocities
            self.publisher_.publish(self.msg)
            # self.get_logger().info(f'Publishing velocities: {self.msg.velocity}')
        else:
            self.get_logger().warn('ROS context is not ok, cannot publish.')


class TiagoJointPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('tiago_joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'isaaclab_velocities', 10)
        self.joint_names = joint_names

    def publish_joint_states(self, velocities):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.velocity = velocities.tolist()  # Convert tensor to list
        # You can also send efforts if needed, otherwise leave it empty
        self.publisher_.publish(msg)

import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState

class GazeboVelocitySubscriber(Node):
    def __init__(self, env):
        super().__init__('gazebo_velocity_subscriber')
        self.env = env
        self.joint_state_names = [
            "arm_1_joint",
            "arm_2_joint",
            "arm_3_joint",
            "arm_4_joint",
            "arm_5_joint",
            "arm_6_joint",
            "arm_7_joint",
        ]
        # Map joint names to indices for quick lookup
        self.joint_index_map = {name: idx for idx, name in enumerate(self.joint_state_names)}
        
        # Create a subscription to the /joint_states topic
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.velocity_callback, 10
        )

    def velocity_callback(self, msg: JointState):
        """Callback to receive and store velocity data from Gazebo."""
        if len(msg.velocity) == 0:
            self.get_logger().warn("Received JointState message with no velocity data.")
            return

        # Initialize a tensor to store the 7 joint velocities for the current environment
        single_gazebo_velocity = torch.zeros(len(self.joint_state_names))

        # Map the received joint velocities to the appropriate indices
        for name, velocity in zip(msg.name, msg.velocity):
            if name in self.joint_index_map:
                idx = self.joint_index_map[name]
                single_gazebo_velocity[idx] = velocity

        # Expand single environment velocity to all environments
        self.env.gazebo_velocities = single_gazebo_velocity.unsqueeze(0).repeat(self.env.num_envs, 1).to(self.env.device)  # Shape: [num_envs, 7]
        #print("gazebo velocity",self.env.gazebo_velocities)
        # Logging for debugging
        # self.get_logger().info(f"Gazebo velocities: {single_gazebo_velocity.tolist()}")
        
"""
def plot_damping_friction_distribution(damping_samples, friction_samples, num_dofs=7):
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(nrows=num_dofs, ncols=2, figsize=(14, 3 * num_dofs))

    def remove_outliers(data, threshold=0.1):
        # Remove outliers using the Z-score method.
        mean = np.mean(data)
        std_dev = np.std(data)
        return data[np.abs((data - mean) / std_dev) < threshold]

    for i in range(num_dofs):
        filtered_damping = remove_outliers(damping_samples[:, i])
        filtered_friction = remove_outliers(friction_samples[:, i])

        # Improved KDE with adjusted bandwidth
        sns.histplot(filtered_damping, bins=100, kde=True, kde_kws={"bw_adjust": 0.5}, color='b', ax=axes[i, 0])
        sns.rugplot(filtered_damping, color='black', ax=axes[i, 0])  # Rug plot for reference
        axes[i, 0].set_xlabel('Damping Value', fontsize=14)
        axes[i, 0].set_ylabel('Density', fontsize=14)

        sns.histplot(filtered_friction, bins=200, kde=True, kde_kws={"bw_adjust": 0.5}, color='r', ax=axes[i, 1])
        sns.rugplot(filtered_friction, color='black', ax=axes[i, 1])
        axes[i, 1].set_xlabel('Friction Value', fontsize=14)
        axes[i, 1].set_ylabel('Density', fontsize=14)

        # Add a single centered title for both histograms of the joint
        fig.text(0.5, (num_dofs - i - 0.5) / num_dofs, f'Joint {i+1}', fontsize=18, fontweight='bold', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Prevent overlap
    plt.show()


    print("Damping and friction probability distributions generated.")
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def moving_average(data, window_size=10):
    """Computes a simple moving average for smoothing the time series."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def silverman_bandwidth(data):
    """Computes Silverman's rule of thumb for KDE bandwidth."""
    if len(data) < 2:
        return 0.5  # Fallback for small datasets
    std_dev = np.std(data)
    n = len(data)
    return 1.5 * std_dev * (n ** -0.2)

def remove_outliers_iqr(data, k=1.5):
    """Removes outliers using the Interquartile Range (IQR) method."""
    Q1 = np.percentile(data, 25)  # 25th percentile (Q1)
    Q3 = np.percentile(data, 75)  # 75th percentile (Q3)
    IQR = Q3 - Q1  # Interquartile Range
    lower_bound = Q1 - k * IQR  # Lower bound for outliers
    upper_bound = Q3 + k * IQR  # Upper bound for outliers
    return data[(data >= lower_bound) & (data <= upper_bound)]  # Filter data

def log_transform(data):
    """Applies log transformation to reduce skewness."""
    #return np.log1p(data)  # log1p avoids log(0) errors
    return data

def plot_parameters_analysis(damping_samples, friction_samples, num_dofs=7):
    """Plots distributions and boxplots of damping and friction parameters using IQR for outlier removal."""
    sns.set_theme(style="whitegrid")

    # Create figure for KDE and Histograms
    fig, axes = plt.subplots(nrows=num_dofs, ncols=2, figsize=(16, 4 * num_dofs))

    for i in range(num_dofs):
        # Apply IQR-based filtering
        filtered_damping = remove_outliers_iqr(np.array(damping_samples[i]))
        filtered_friction = remove_outliers_iqr(np.array(friction_samples[i]))

        # Apply log transformation to reduce skewness
        transformed_damping = log_transform(filtered_damping)
        transformed_friction = log_transform(filtered_friction)

        # Compute KDE bandwidth
        bw_damping = silverman_bandwidth(filtered_damping)
        bw_friction = silverman_bandwidth(filtered_friction)

        # Plot KDE & Histogram for Damping
        sns.histplot(transformed_damping, bins=30, kde=True, kde_kws={"bw_adjust": bw_damping}, color='b', ax=axes[i, 0])
        sns.rugplot(transformed_damping, color='b', ax=axes[i, 0])
        axes[i, 0].set_xlabel('Log(Damping Value)', fontsize=14)
        axes[i, 0].set_ylabel('Density', fontsize=14)
        axes[i, 0].set_title(f'Joint {i+1}: Damping Distribution', fontsize=16)

        # Plot KDE & Histogram for Friction
        sns.histplot(transformed_friction, bins=30, kde=True, kde_kws={"bw_adjust": bw_friction}, color='r', ax=axes[i, 1])
        sns.rugplot(transformed_friction, color='r', ax=axes[i, 1])
        axes[i, 1].set_xlabel('Log(Friction Value)', fontsize=14)
        axes[i, 1].set_ylabel('Density', fontsize=14)
        axes[i, 1].set_title(f'Joint {i+1}: Friction Distribution', fontsize=16)

    plt.tight_layout()
    plt.show()

    # Create boxplots for Damping and Friction
    # Set theme and font scale
    # Set theme and font scale
    sns.set_theme(style="whitegrid", font_scale=1.2)  # Reduced font scale

    # Create figure for boxplots with a more squared shape
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 5))  # Reduced height to 5 for a more squared shape

    # Customize boxplot appearance
    boxplot_kwargs = {
        "linewidth": 2.8,  # Thicker lines for boxplots
        "flierprops": dict(marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.5),  # Outlier markers
        "palette": "viridis",  # Professional color palette
    }

    # Boxplot for Damping (log-transformed)
    sns.boxplot(
        data=[log_transform(remove_outliers_iqr(np.array(damping_samples[i]))) for i in range(num_dofs)],
        ax=axes[0],
        **boxplot_kwargs
    )
    axes[0].set_xticklabels([f'Giunto {i+1}' for i in range(num_dofs)], fontsize=13)  # Reduced x-tick label font size
    axes[0].set_ylabel('Damping', fontsize=26)  # Adjusted y-label font size
    axes[0].set_title('Distribuzione di probabilità Damping', fontsize=28, pad=20)  # Adjusted title font size

    # Boxplot for Friction (log-transformed)
    sns.boxplot(
        data=[log_transform(remove_outliers_iqr(np.array(friction_samples[i]))) for i in range(num_dofs)],
        ax=axes[1],
        **boxplot_kwargs
    )
    axes[1].set_xticklabels([f'Giunto {i+1}' for i in range(num_dofs)], fontsize=13)  # Reduced x-tick label font size
    axes[1].set_ylabel('Friction', fontsize=26)  # Adjusted y-label font size
    axes[1].set_title('Distribuzione di probabilità Friction', fontsize=28, pad=20)  # Adjusted title font size

    # Apply logarithmic scaling to the y-axis for both boxplots
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')

    # Customize axes and gridlines
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=22, width=2, length=5)  # Adjusted tick font size
        ax.tick_params(axis='both', which='minor', labelsize=15, width=4, length=20)  # Adjusted minor tick font size
        ax.grid(True, linestyle='--', linewidth=2.0, alpha=0.8)  # Add gridlines
        for spine in ax.spines.values():  # Thicker spines (borders)
            spine.set_linewidth(2.5)

    # Adjust layout to make the plot more squared
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)  # Add horizontal space between subplots
    plt.show()
    # ================== HEATMAP ==================
    # Create DataFrame correctly with time steps as index

    # Convert data to a NumPy array and transpose to get shape (num_samples, num_joints)
    # Example data (replace with your actual data)

    # Convert to matrices

    sns.set_theme(style="whitegrid", font_scale=1.5)

    # Convert to matrices
    damping_matrix = np.array(damping_samples)  
    friction_matrix = np.array(friction_samples)  

    # Create DataFrames
    df_damping = pd.DataFrame(damping_matrix.T, columns=[f"Damping_Joint_{i+1}" for i in range(num_dofs)])
    df_friction = pd.DataFrame(friction_matrix.T, columns=[f"Friction_Joint_{i+1}" for i in range(num_dofs)])

    # Compute correlation matrices
    damping_correlation = df_damping.corr()
    friction_correlation = df_friction.corr()

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Define color bar tick labels
    cbar_fontsize = 22  # Increase font size for color bar

    # Damping correlation heatmap
    heatmap1 = sns.heatmap(
        damping_correlation,
        cmap="coolwarm",
        linewidths=0.5,
        cbar=True,
        annot=True,
        fmt=".2f",
        ax=axes[0],
        annot_kws={"size": 15},  # Increase annotation text size
    )
    axes[0].set_title("Correlazione Damping tra i giunti ", fontsize=26, pad=20)  
    axes[0].set_xticklabels([f"Giunto {i+1}" for i in range(num_dofs)], rotation=45, ha='right', fontsize=20)
    axes[0].set_yticklabels([f"Giunto {i+1}" for i in range(num_dofs)], rotation=0, fontsize=20)

    # Increase color bar tick labels
    heatmap1.collections[0].colorbar.ax.tick_params(labelsize=cbar_fontsize)

    # Friction correlation heatmap
    heatmap2 = sns.heatmap(
        friction_correlation,
        cmap="coolwarm",
        linewidths=0.5,
        cbar=True,
        annot=True,
        fmt=".2f",
        ax=axes[1],
        annot_kws={"size": 15},  
    )
    axes[1].set_title("Correlazione Friction tra i giunti", fontsize=26, pad=20)
    axes[1].set_xticklabels([f"Giunto {i+1}" for i in range(num_dofs)], rotation=45, ha='right', fontsize=20)
    axes[1].set_yticklabels([f"Giunto {i+1}" for i in range(num_dofs)], rotation=0, fontsize=20)

    # Increase color bar tick labels
    heatmap2.collections[0].colorbar.ax.tick_params(labelsize=cbar_fontsize)

    # Adjust layout
    plt.tight_layout()
    plt.show()


    # ================== PARAMETER EVOLUTION OVER TIME ==================
    """Plots parameter evolution over time with improved y-axis scaling for better visibility of lower values."""
    window_size = 5  # Define window size for moving average
    fig, axes = plt.subplots(nrows=num_dofs, ncols=1, figsize=(12, 3 * num_dofs))

    for i in range(num_dofs):
        ax = axes[i]

        # Extract time-series data
        time_steps = np.arange(len(damping_samples[i]))
        damping_series = np.array(damping_samples[i])
        friction_series = np.array(friction_samples[i])

        # Compute smoothed versions
        smoothed_damping = moving_average(damping_series, window_size)
        smoothed_friction = moving_average(friction_series, window_size)

        # Adjust time steps for smoothed data
        smooth_time_steps = np.linspace(0, len(time_steps), len(smoothed_damping))

        # Plot original time series
        ax.plot(time_steps, damping_series, label="Damping (Raw)", color='blue', alpha=0.3, linestyle="dotted")
        ax.plot(time_steps, friction_series, label="Friction (Raw)", color='red', alpha=0.3, linestyle="dotted")

        # Plot smoothed time series
        ax.plot(smooth_time_steps, smoothed_damping, label="Damping (Smoothed)", color='blue', linewidth=2)
        ax.plot(smooth_time_steps, smoothed_friction, label="Friction (Smoothed)", color='red', linewidth=2)

        # Set Y-axis to logarithmic scale
        ax.set_yscale('log')

        # Adjust Y-axis limits to highlight lower values
        ymin = min(np.min(damping_series), np.min(friction_series))
        ymax = np.percentile(np.concatenate((damping_series, friction_series)), 95)  # Use 95th percentile to remove outliers
        ax.set_ylim(ymin * 0.9, ymax * 1.1)  # Provide small margins to avoid cutting data

        ax.set_xlabel("Time Step", fontsize=14)
        ax.set_ylabel("Parameter Value (log scale)", fontsize=14)
        ax.set_title(f"Joint {i+1}: Damping & Friction Evolution Over Time", fontsize=16)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()




@configclass
class TiagoEnvCfg(DirectRLEnvCfg):
    decimation = 1  # Increase for faster training  
    episode_length_s = 3.0  # Increase if needed for better learning
    # action_scale = 1.0  # Scale for action adjustments
    action_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05]  # Scale for each DOF
    scaling = 1.0  # Scaling factor for the action space

    # NORMALIZED ACTION SPACE !!
    action_space = gym.spaces.Box(low=-1, high=1, shape=(28,))  # Stiffness and damping adjustments
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)
    state_space = 0
    #events: EventCfg = EventCfg()
    #num_envs = 128  # Increase for faster learning
    # seed: int = -1  # Add this line to properly initialize the seed attribute

    # simulation
    sim = SimulationCfg(dt=1 / 100, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = TIAGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0, replicate_physics=True)
    """
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )
    """
    # reward scales
    rew_scale_alive = 0.05  # Reward for staying alive
    rew_scale_terminated = -0.1  # Reward for termination
    rew_scale_error = -4.0  # Reward for minimizing error
    rew_scale_gazebo_error = -1.0  # Reward for minimizing velocity error between Isaac and Gazebo
    stiffness_penalty_scale = 0.5  # Penalty for large stiffness changes
    negative_param_scale = 0.8  # Penalty for negative stiffness/damping values

import threading
import random
class TiagoEnv(DirectRLEnv):
    cfg: TiagoEnvCfg

    def __init__(self, cfg: TiagoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize ROS 2
        rclpy.init()

        # ROS 2 nodes for joint state data
        self.joint_publisher = TiagoJointPublisher([f"arm_{i}_joint" for i in range(1, 8)])  # Assuming 7 arm joints
        self.gazebo_subscriber = GazeboVelocitySubscriber(self)  # passing the environment instance to the subscriber
        self.reference_publisher = ReferenceVelocityPublisher()
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.gazebo_subscriber,), daemon=True)
        self.spin_thread.start()
        self.active_dof = 6 # Initialize the active DOF index
        self.dof_group = self.active_dof + 1  # Initialize the DOF group size
        self._joint_ind = [self.robot.find_joints(f"arm_{i}_joint")[0][0] for i in range(1, 8)]
        print("ARM INDICES",self._joint_ind) 
        #self._joint_indices = [self._joint_ind[self.active_dof]]
        self._joint_indices = self._joint_ind
        print("last dof index",self._joint_indices)
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        self.total_error = torch.zeros((self.num_envs), device=self.device)
        
        self.duration = 3.0
        self.samples = 100.0
        self.num_dofs = 7
        self.total_joints = 24  # Total number of joints in the robot
        self.frequency = 2.0
        self.max_value = 0.1
        self.min_value = -self.max_value
        self.velocities = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.gazebo_velocities = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.gazebo_velocity_error = torch.zeros((self.num_envs, self.dof_group), device=self.device)
        self.dof_gains = torch.ones((self.num_envs, self.num_dofs), device=self.device)  # Store gains for all DOFs

        # Define buffers to store historical trajectory data (initialize these in your class)
        self.isaac_trajectory_buffer = torch.zeros((self.num_envs, 100, self.dof_group), device=self.device)  # Buffer for Isaac velocities
        self.gazebo_trajectory_buffer = torch.zeros((self.num_envs, 100, self.dof_group), device=self.device)  # Buffer for Gazebo velocities
        
        # Initial threshold and parameters for dynamic adjustment
        self.initial_error_threshold = 5.0  # Starting threshold
        self.min_error_threshold = 0.01  # Minimum threshold to reach over time
        self.training_progress_rate = 1e-3  # Adjust rate per timestep
        self.dt = 0  # Initialize time step (for sampling sinusoidal signal)
        self.friction_min = 1e-3 # Define range
        self.damping_min = 1e-3  # Define range
        #self.friction_max = torch.tensor([1000.0, 1000.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0]).to('cuda')
        
        # Initialize lists to store data for all environments
        self.reference_signals_list = [[] for _ in range(self.num_dofs)]
        self.actual_velocities_list = [[] for _ in range(self.num_dofs)]
        self.realrobot_velocities_list = [[] for _ in range(self.num_dofs)]
        self.errors_list = [[] for _ in range(self.num_dofs)]
        self.damping_samples = [[] for _ in range(self.num_dofs)]
        self.friction_samples = [[] for _ in range(self.num_dofs)]
        self.plot_interval = 150  # Define after how many steps to plot data

        self.COUNTER = 0
        self.period_steps = 500
        self.shape = 0
        self.collect = 0
        
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Clone the environment for multiple instances, filter collisions and add the robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["tiago"] = self.robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        return self.robot
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.dt % (20 * self.period_steps) == 0 and False:
            print("Changing signal shape")
            print("Shape:", self.shape)
            if self.shape > 2:
                self.shape = 0
            self.max_value = round(random.uniform(0.05, 0.2), 2)
            self.min_value = -self.max_value

        self.actions = actions.clone() * self.cfg.scaling  # actions shape: (num_envs, 7, 2)
        num_joints = len(self._joint_indices)  # Number of joints to be updated
        
        friction_scale = (self.actions[:, :7] + 1) / 2 * 3 
        damping_scale = (self.actions[:, 7:14] + 1) / 2 * 3
        self.friction_max = torch.pow(10, friction_scale).to(self.device)  # Element-wise 10^friction_scale
        self.damping_max = torch.pow(10, damping_scale).to(self.device)    # Element-wise 10^damping_scale

        new_friction = (self.actions[:, 14:21] + 1) / 2 * (self.friction_max - self.friction_min) + self.friction_min
        new_damping = (self.actions[:, 21:28] + 1) / 2 * (self.damping_max - self.damping_min) + self.damping_min

        # ACTION MASKING
        selected_friction = new_friction[:, :num_joints]
        selected_damping = new_damping[:, :num_joints]

        self.robot.write_joint_damping_to_sim(selected_damping, self._joint_indices)
        self.robot.write_joint_friction_to_sim(selected_friction, self._joint_indices)
        #print("Friction:", selected_friction[0, 5:])
        #print("Damping:", selected_damping[0, 5:])

        # Gather probability distribution when all 7 joints are active


    def _apply_action(self) -> None:
        self.COUNTER += 1
        # No direct action application, we only adjust stiffness/damping
        velocity_signal = self.generate_sinusoidal_signal()
        t_idx = self._sim_step_counter % velocity_signal.size(1)
        self.velocities = velocity_signal[:, t_idx, :]  # Velocity for all DOFs
        # Scale velocities with the gains for all DOFs
        scaled_velocities = self.velocities

        # Create a zero tensor for all joint velocities
        full_joint_velocities = torch.zeros((self.num_envs, self.total_joints), device='cuda')

        # Assign the scaled velocities only to arm joints
        full_joint_velocities[:, self._joint_ind] = scaled_velocities
        #print("full joint velocities", full_joint_velocities)

        # Now set the full_joint_velocities for the robot, but only modify arm joint velocities
        self.robot.set_joint_velocity_target(target=full_joint_velocities)
        
        self.reference_publisher.publish_velocity(self.velocities[0].cpu().numpy()) # Publish reference velocities to ROS2 topic
        
        # Get the velocities for a single environment (env 0 for example)
        self.vel = self.robot.data.joint_vel[:, self._joint_ind]
        joint_velocities = self.vel[0].cpu().numpy()
        # Publish joint state data to the ROS2 topic
        #if np.absolute(joint_velocities) < self.joint_vel_limits:
        self.joint_publisher.publish_joint_states(joint_velocities)
        
        # During each step, append the velocities to the buffers
        # Loop through all environments to append trajectory data
        self.current_step = self.dt  # Get current step for this environment
        # Update trajectory buffers for each environment individually
        self.isaac_trajectory_buffer[:, self.current_step, :] = self.robot.data.joint_vel[:, self._joint_indices]
        self.gazebo_trajectory_buffer[:, self.current_step, :] = self.gazebo_velocities[:, :self.dof_group]
        self.dt += 1  # Increment time step for sampling sinusoidal signal
        
        
        # TESTING PLOT !!!!
        if self.COUNTER > 1000:  # Ensure collection starts after warmup
            # If it's the first sample of a new batch, clear all buffers
            if len(self.reference_signals_list[0]) == 0:  
                print("Clearing buffers before new collection cycle")

                # Initialize buffers for velocity data
                self.reference_signals_list = [[] for _ in range(self.num_dofs)]
                self.actual_velocities_list = [[] for _ in range(self.num_dofs)]
                self.realrobot_velocities_list = [[] for _ in range(self.num_dofs)]
                self.errors_list = [[] for _ in range(self.num_dofs)]

                # Initialize buffers for damping and friction if not already done
                if not hasattr(self, 'damping_samples'):
                    self.damping_samples = [[] for _ in range(self.num_dofs)]
                    self.friction_samples = [[] for _ in range(self.num_dofs)]

            # Append new velocity data
            for idx in range(self.num_dofs):
                self.reference_signals_list[idx].append(self.velocities[0, idx].cpu().item())
                self.actual_velocities_list[idx].append(self.robot.data.joint_vel[0, self._joint_indices[idx]].cpu().item())
                self.realrobot_velocities_list[idx].append(self.gazebo_velocities[0, idx].cpu().item())

                error = self.gazebo_velocities[0, idx] - self.robot.data.joint_vel[0, self._joint_indices[idx]]
                self.errors_list[idx].append(error.cpu().item())

            # Append new damping and friction data
            damping_sample = self.robot.data.joint_damping[0, self._joint_indices].clone()
            friction_sample = self.robot.data.joint_friction[0, self._joint_indices].clone()

            for i in range(self.num_dofs):
                self.damping_samples[i].append(damping_sample[i].item())
                self.friction_samples[i].append(friction_sample[i].item())

            # Generate plots every plot_interval samples
            if len(self.reference_signals_list[0]) >= self.plot_interval:  
                print(f"Generating plots at COUNTER: {self.COUNTER}, Episode Length: {len(self.reference_signals_list[0])}")

                # Convert lists to NumPy arrays for velocity plots
                self.reference_signals = [np.array(self.reference_signals_list[i]) for i in range(self.num_dofs)]
                self.actual_velocities = [np.array(self.actual_velocities_list[i]) for i in range(self.num_dofs)]
                self.realrobot_velocities = [np.array(self.realrobot_velocities_list[i]) for i in range(self.num_dofs)]
                self.errors = [np.array(self.errors_list[i]) for i in range(self.num_dofs)]

                # Convert lists to NumPy arrays for damping/friction plots
                damping_samples = [np.array(self.damping_samples[i]) for i in range(self.num_dofs)]
                friction_samples = [np.array(self.friction_samples[i]) for i in range(self.num_dofs)]

                # Generate both plots
                self.plot_data()
                #plot_parameters_analysis(damping_samples, friction_samples, self.num_dofs)

                # Explicitly clear buffers after plotting
                print("Resetting buffers after plotting")
                self.reference_signals_list = [[] for _ in range(self.num_dofs)]
                self.actual_velocities_list = [[] for _ in range(self.num_dofs)]
                self.realrobot_velocities_list = [[] for _ in range(self.num_dofs)]
                self.errors_list = [[] for _ in range(self.num_dofs)]
                self.damping_samples = [[] for _ in range(self.num_dofs)]
                self.friction_samples = [[] for _ in range(self.num_dofs)]


    def plot_data(self):
        """
        Plots velocity comparisons for all joints in a single figure with subplots, ensuring a professional and academic layout.

        This function visualizes:
        - Reference velocity (blue, hatched line)
        - Real robot velocity (green, dashed line)
        - Simulated velocity (red, solid line)

        Each joint has its own subplot with correctly scaled time values based on a 100 Hz sampling rate.
        """
        print("Plotting data...")
        sns.set_theme(style="whitegrid")

        # Create a figure with subplots for each joint
        fig, axes = plt.subplots(self.num_dofs, 1, figsize=(12, 3 * self.num_dofs), sharex=False)  # Remove sharex

        # Generate the time vector with 100 Hz sampling rate (1 sample every 1/100s)
        episode_length = len(self.reference_signals[0])  # Assuming all signals have the same length
        time_vector = np.arange(0, episode_length / 100, 1 / 100)  # Time in seconds

        # Ensure time_vector matches the signal length
        if len(time_vector) > episode_length:
            time_vector = time_vector[:episode_length]

        for i in range(self.num_dofs):
            ax = axes[i] if self.num_dofs > 1 else axes  # Handle single subplot case

            # Plot velocity curves
            ax.plot(time_vector, self.reference_signals[i], color='blue', linewidth=2.5, linestyle='dashed')  
            ax.plot(time_vector, self.realrobot_velocities[i], color='green', linewidth=2.5,)
            ax.plot(time_vector, self.actual_velocities[i], color='red', linewidth=2.5, alpha=0.6)

            # Labels and formatting
            ax.set_ylabel('Velocità [rad/s]', fontsize=14)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=14)
            #ax.legend(fontsize=12, loc='upper right', frameon=True)  
            #ax.set_title(f'Giunto {i+1}', fontsize=20)

            # Set x-axis labels and tick positions for all subplots
            xtick_positions = np.linspace(0, episode_length / 100, num=6)
            ax.set_xticks(xtick_positions)  
            ax.set_xticklabels([f"{x:.2f}" for x in xtick_positions], fontsize=12)
            #ax.legend(['Riferimento', 'Gazebo', 'Isaac Lab'], fontsize=12)
            #ax.set_xlabel("Tempo [s]", fontsize=14)  # Ensure all subplots have x-axis labels

        # Adjust layout for clarity
        plt.tight_layout()
        plt.show()


    def _get_observations(self) -> dict:
        # Gather joint velocities (current observation)
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        joint_velocities_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        joint_velocities_padded[:, :self.dof_group] = self.joint_vel  # Zero inactive DOFs

        # Reference velocities (target velocities from reference signal)
        reference_velocities_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        reference_velocities_padded[:, :self.dof_group] = self.velocities[:, :self.dof_group]  # Zero inactive DOFs

        # Gazebo velocities
        gazebo_velocities_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        gazebo_velocities_padded[:, :self.dof_group] = self.gazebo_velocities[:, :self.dof_group]  # Zero inactive DOFs

        # Gazebo velocity error
        gazebo_velocity_error_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        gazebo_velocity_error_padded[:, :self.dof_group] = self.gazebo_velocity_error  # Zero inactive DOFs

        current_damping = self.robot.data.joint_damping[:, self._joint_indices]
        current_damping_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        current_damping_padded[:, :self.dof_group] = current_damping  # Zero inactive DOFs
        
        current_friction = self.robot.data.joint_friction[:, self._joint_indices]
        current_friction_padded = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        current_friction_padded[:, :self.dof_group] = current_friction  # Zero inactive DOFs
        
        # Create a binary mask for active DOFs
        active_dof_mask = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        active_dof_mask[:, :self.dof_group] = 1  # 1 for active joints, 0 for inactive

        # Concatenate the mask with observations
        obs = torch.cat((
            joint_velocities_padded,
            reference_velocities_padded,
            gazebo_velocities_padded,
            gazebo_velocity_error_padded,
            active_dof_mask
        ), dim=-1)

        return {"policy": obs}
    
    def calculate_error(self) -> torch.Tensor:          
        # Assuming the velocities are in range [-1, 1]
        error = torch.norm(self.gazebo_velocities[:, :self.dof_group] - self.robot.data.joint_vel[:, self._joint_indices], dim=1) # Calculate the error
        max_error = self.max_value - self.min_value  # Define a meaningful maximum error threshold
        normalized_error = error / max_error  # Normalize the error
        return normalized_error  # Ensure the error is between 0 and 1    
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = self.compute_rewards(
            self.cfg.rew_scale_gazebo_error,
        )
        return total_reward
    
    # @torch.jit.script   is used to compile the function to TorchScript for performance optimization in terms of speed
    def compute_rewards(self,
                        rew_scale_gazebo_error,
                        ):
        gazebo_velocities = self.gazebo_velocities[:, :self.dof_group] 
        
        # Create a mask for DOFs with zero Gazebo velocity (same for all environments)
        zero_velocity_mask_dof = (gazebo_velocities[0, :] == 0)  # Shape: (dof_group,)
        # Broadcast this mask across all environments
        zero_velocity_mask = zero_velocity_mask_dof.unsqueeze(0).expand(self.num_envs, -1)  # Shape: (num_envs, dof_group)

        # Error between Isaac Sim joint velocities and Gazebo joint velocities
        gazebo_error = self.calculate_error()  # Shape: (num_envs,)
        reward_gazebo_error = rew_scale_gazebo_error * gazebo_error

        # ================================
        # Dynamic Velocity Error (Isaac-Gazebo) Threshold Shaping
        # ================================
        #print("gazebo velocities shape",gazebo_velocities.shape)
        #print("robot velocities shape",self.robot.data.joint_vel[:, self._joint_indices].shape)
        self.gazebo_velocity_error = torch.abs(gazebo_velocities - self.robot.data.joint_vel[:, self._joint_indices])

        # Define fractions for dynamic thresholds (e.g., 5%, 10%, 15% of current velocity)
        
        threshold_fractions = torch.tensor([0.05, 0.10, 0.15], device=self.device)
        
        # Dynamic thresholds computed based on actual Gazebo velocities
        min_threshold = torch.tensor(0.006, device=self.device)  # Convert to tensor on the same device
        dynamic_thresholds = torch.maximum(
            torch.abs(gazebo_velocities) * threshold_fractions[:, None, None],
            min_threshold
        )  # Shape: (3, num_envs, num_dofs)

        # Rewards for hitting different thresholds
        gazebo_rewards_for_thresholds = torch.tensor([0.4, 0.3, 0.2], device=self.device)
        gazebo_penalty_large_error = -0.1  # Increased penalty for large errors

        reward_gazebo_tensor = torch.zeros(self.gazebo_velocity_error.shape[0]).to(self.device)  # Shape: (num_envs)
        for i in range(len(threshold_fractions)):
            gazebo_reward_mask = (self.gazebo_velocity_error < dynamic_thresholds[i]).float().to(self.device)
            gazebo_reward_mask[zero_velocity_mask] = 0
            #print("gazebo reward mask shape",gazebo_reward_mask.shape)
            reward_gazebo_tensor += gazebo_reward_mask.sum(dim=1) * gazebo_rewards_for_thresholds[i]

        # Penalty for environments with large errors in any DOF
        large_error_mask = (self.gazebo_velocity_error >= dynamic_thresholds[-1]).any(dim=1).float().to(self.device)
        penalty_large_error = large_error_mask * gazebo_penalty_large_error
        # ================================
        
        # Total reward combining all components
        total_reward = (     
            reward_gazebo_error +  
            reward_gazebo_tensor +
            penalty_large_error
        ).to(self.device)

        #print("CURRENT ACTIVE DOF",self.active_dof)
        
        # return total_reward
        #reward = - self.gazebo_velocity_error.abs().sum(dim=1)/self.dof_group  # Shape: (num_envs,)
        # ================================
        #print("reward",reward)
        #print(reward.shape)
        reward = - self.gazebo_velocity_error.abs().sum(dim=1)/self.num_dofs  # Shape: (num_envs,)
        return reward  # Combine all rewards and penalties

    # ================================
    # CURRICULUM LEARNING MANAGER
    # ================================
    def update_dof(self):
        # Update the joint indices for the last DOF
        self.active_dof += 1
        if self.active_dof >= self.num_dofs - 1:
            self.active_dof = self.num_dofs - 1
        self.dof_group = self.active_dof + 1
        self._joint_indices = self._joint_ind[:self.dof_group]
        #print("ACTIVE DOFS",self._joint_indices)
        
        #print(" ACTIVE DOF",self.active_dof)
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Update target error threshold based on training progress using episode length
        self.target_error_threshold = torch.maximum(
            torch.full((self.num_envs,), self.min_error_threshold, device=self.device),
            self.initial_error_threshold - self.training_progress_rate * self.episode_length_buf
        )
        # Fetch current joint positions for limit checking
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_indices]

        # Standard done conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Determine out of bounds using dynamic threshold
        out_of_bounds = torch.any(self.gazebo_velocity_error > self.target_error_threshold.unsqueeze(1), dim=1) # Check if any error exceeds the threshold
        # Once a trajectory length is reached, compute correlation
        if self.current_step == 100 - 1:  # Check if buffer is full
            numerator = torch.sum((self.isaac_trajectory_buffer) * (self.gazebo_trajectory_buffer), dim=1)

            denominator = torch.sqrt(torch.sum((self.isaac_trajectory_buffer)**2, dim=1)) * torch.sqrt(torch.sum((self.gazebo_trajectory_buffer)**2, dim=1))
            
            print("NUMERATOR FOR LAST ADDED JOINT:", numerator[0])
            print("DENOMINATOR FOR LAST ADDED JOINT:", denominator[0])
            correlation = numerator / (denominator + 1e-6)  # Avoid division by zero
            print("CORRELATION FOR LAST ADDED JOINT:", torch.median(correlation))
            if self.COUNTER % 800 == 0 and False:
                # Create subplots with better distribution

                # Define the number of rows and columns for subplots
                nrows = 2  # Two rows
                ncols = 4  # Four columns (4 plots in the first row, 3 plots in the second row)

                # Create subplots with two rows
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))

                # Set global font size and line thickness
                plt.rcParams.update({
                    'font.size': 20,          # Increase text size
                    'axes.labelsize': 22,     # Increase label size
                    'axes.titlesize': 20,     # Increase title size
                    'lines.linewidth': 2.0,   # Increase line thickness
                    'scatter.marker': 'o',    # Define scatter marker
                    'grid.linewidth': 1.5     # Increase grid thickness
                })

                # Flatten the axes array for easier indexing
                axes = axes.flatten()

                for i in range(self.num_dofs):
                    # Extract velocity data for the current joint
                    isaac_velocities = self.isaac_trajectory_buffer[:, :, i].flatten().cpu().numpy()
                    gazebo_velocities = self.gazebo_trajectory_buffer[:, :, i].flatten().cpu().numpy()

                    # Plot scatter plot for the current joint
                    axes[i].scatter(isaac_velocities, gazebo_velocities, alpha=0.5, s=100)  # Increase marker size
                    axes[i].plot([min(isaac_velocities), max(isaac_velocities)], 
                                [min(isaac_velocities), max(isaac_velocities)], 
                                color='red', linestyle='--', linewidth=3)  # Thicker line for y = x

                    # Set labels and title
                    axes[i].set_xlabel("Velocità Isaac Lab [rad/s]", fontsize=22)
                    axes[i].set_ylabel("Velocità Gazebo [rad/s]", fontsize=22)
                    axes[i].set_title(f"Giunto {i + 1}", fontsize=25)
                    axes[i].grid(True, linestyle='--', linewidth=1.5)  # Thicker grid lines
                    # Increase tick label size for both x and y axes
                    axes[i].tick_params(axis='both', labelsize=18)  # Set tick label size

                # Hide unused subplots (if any)
                for i in range(self.num_dofs, nrows * ncols):
                    axes[i].axis('off')  # Turn off unused subplots

                # Adjust layout for better spacing
                plt.tight_layout()
                plt.show()
            # Switch DOF if only the last joint reaches the threshold
            if torch.median(correlation) > 0.9:
                self.update_dof()
            self.isaac_trajectory_buffer = torch.zeros((self.num_envs, 100, self.dof_group), device=self.device)  # Buffer for Isaac velocities
            self.gazebo_trajectory_buffer = torch.zeros((self.num_envs, 100, self.dof_group), device=self.device)  # Buffer for Gazebo velocities
            self.dt = 0  # Reset time step for the next trajectory   
        return time_out, out_of_bounds


    def _reset_idx(self, env_ids: torch.Tensor | None):
        # self.dt = 0  # Reset time step for the next trajectory
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        steps_buffer = self.episode_length_buf[env_ids]
        
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        # Reset joint positions and velocities for the new episode

        # Get the default root state for the new episode
        default_root_state = self.robot.data.default_root_state[env_ids]
        
        # Adjust the root state based on the environment origins
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        
        # iterate over the environments for tensor broadcasting of specific joint indices
        joint_pos = torch.stack([self.robot.data.default_joint_pos[env_id, self._joint_indices] for env_id in env_ids])
        joint_vel = torch.stack([self.robot.data.default_joint_vel[env_id, self._joint_indices] for env_id in env_ids])
        # joint_vel = self.robot.data.default_joint_vel[env_ids, self._joint_indices]
        
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_indices, env_ids)
        

        
    # Assuming the original simulation class and plotting functions remain as is
    def generate_sinusoidal_signal(self):
        time = torch.linspace(0, self.duration, int(self.duration * self.samples), device="cuda").unsqueeze(-1)  # Time vector (samples, 1)
        repeated_time = time.repeat(1, self.num_dofs)  # Repeat the time for each joint
        signal = []
        # Generate sinusoidal signal for each DOF (velocity control)
        
        signal.append(
            (torch.sin(
                2 * torch.pi * self.frequency * repeated_time) / 2 + 0.5) \
                * (self.max_value - self.min_value) + self.min_value
        )
        signal.append(
            torch.where(
                torch.sign(torch.sin(2 * torch.pi * self.frequency * repeated_time)) > 0,
                self.max_value,
                self.min_value
            )
        )
        
        period = 1/self.frequency
        signal.append((self.max_value - self.min_value) * torch.abs((repeated_time % period) - (period / 2)) * 2 / period + self.min_value)
        # Repeat the signal for each environment (num_envs, samples, num_dofs)
        signal = signal[self.shape].unsqueeze(0).repeat(self.num_envs, 1, 1)  # Shape: (num_envs, samples, num_dofs)
        return signal

