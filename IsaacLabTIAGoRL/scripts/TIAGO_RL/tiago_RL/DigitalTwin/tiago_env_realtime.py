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
        "tiago_arm": ImplicitActuatorCfg(joint_names_expr=["arm_[1-7]_joint"], stiffness=0.0, damping=5000.0),
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
        self.current_time = time.time()

    def publish_velocity(self, velocities):
        self.interval_time = time.time() - self.current_time
        print("Interval time", self.interval_time)
        self.current_time = time.time()
        self.msg.velocity = velocities.tolist()  # Set velocities
        self.publisher_.publish(self.msg)


class TiagoJointPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('tiago_joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'isaaclab_velocities', 10)
        self.joint_names = joint_names

    def publish_joint_states(self, velocities):
        msg = JointState()
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
            JointState, '/real_velocities', self.velocity_callback, 10
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
    # reward scales
    rew_scale_alive = 0.05  # Reward for staying alive
    rew_scale_terminated = -0.1  # Reward for termination
    rew_scale_error = -4.0  # Reward for minimizing error
    rew_scale_gazebo_error = -1.0  # Reward for minimizing velocity error between Isaac and Gazebo
    stiffness_penalty_scale = 0.5  # Penalty for large stiffness changes
    negative_param_scale = 0.8  # Penalty for negative stiffness/damping values

import threading
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        self.active_dof = 6  # Initialize the active DOF index
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
        self.samples = 1000.0
        self.num_dofs = 7
        self.total_joints = 24  # Total number of joints in the robot
        #self.frequency = 1 / 1.38 + 0.01  # Frequency for the sinusoidal signal
        self.frequency = 0.7352941176 * 20 * 1.05
        self.duration = 6 * (1 / self.frequency)  # Duration for the sinusoidal signal
        self.max_value = 0.1
        self.min_value = -self.max_value
        self.velocities = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.gazebo_velocities = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.gazebo_velocity_error = torch.zeros((self.num_envs, self.dof_group), device=self.device)
        self.dof_gains = torch.ones((self.num_envs, self.num_dofs), device=self.device)  # Store gains for all DOFs

        # Define buffers to store historical trajectory data (initialize these in your class)
        self.isaac_trajectory_buffer = torch.zeros((self.num_envs, 200, self.dof_group), device=self.device)  # Buffer for Isaac velocities
        self.gazebo_trajectory_buffer = torch.zeros((self.num_envs, 200, self.dof_group), device=self.device)  # Buffer for Gazebo velocities

        self.plot_interval = 5000  # Define after how many steps to plot data
        self.last_plot_time = 0   # To track when the last plot occurred
        
        # Initial threshold and parameters for dynamic adjustment
        self.initial_error_threshold = 5.0  # Starting threshold
        self.min_error_threshold = 0.01  # Minimum threshold to reach over time
        self.training_progress_rate = 1e-3  # Adjust rate per timestep
        self.dt = 0  # Initialize time step (for sampling sinusoidal signal)
        self.friction_min = 1e-3 # Define range
        self.damping_min = 1e-3  # Define range
        #self.friction_max = torch.tensor([1000.0, 1000.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0]).to('cuda')
        self.friction_max = 500.0
        self.damping_max = 500.0
        
        # Define the action scale tensor
        dof_action_scale = torch.tensor(self.cfg.action_scale, device=self.device).unsqueeze(0)  # Shape: (1, 3)
        self.dof_action_scale = dof_action_scale.repeat(self.num_envs, 1)  # Shape: (256, 3)
        
        # Initialize lists to store data for all environments
        self.reference_signals_list = [[] for _ in range(self.num_dofs)]
        self.actual_velocities_list = [[] for _ in range(self.num_dofs)]
        self.realrobot_velocities_list = [[] for _ in range(self.num_dofs)]
        self.errors_list = [[] for _ in range(self.num_dofs)]
        self.damping_samples = [[] for _ in range(self.num_dofs)]
        self.friction_samples = [[] for _ in range(self.num_dofs)]
        self.plot_interval = 150  # Define after how many steps to plot data

        self.COUNTER = 0
        self.period_steps = 50
        self.shape = 0
        self.time_step = 1/50
        self.last_update_time = time.time()
        self.sampling_counter = 0
        # Start background thread for velocity control at 50Hz
        self.velocity_signal = self.generate_sinusoidal_signal()
        self.running = True
        self.sampling_counter = 0
        self.random_factor = 1.0

        self.apply_thread = threading.Thread(target=self.run_velocity_control, daemon=True)
        self.apply_thread.start()

    def run_velocity_control(self):
        """Runs velocity control at a fixed 50Hz with drift compensation."""
        expected_interval = self.time_step
        next_time = time.perf_counter() + expected_interval

        while self.running:
            start_time = time.perf_counter()

            # Perform the velocity update
            self.update_velocities()

            # Calculate the time taken for the update
            update_duration = time.perf_counter() - start_time

            # Sleep until the next expected time
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Log the actual interval for analysis
            current_interval = time.perf_counter() - start_time
            print(f"ACTUAL INTERVAL: {current_interval:.6f} seconds")

            # Update the next expected time
            next_time += expected_interval

    def update_velocities(self):
        """Handles velocity updates at 50Hz, moved from apply_action."""
        if self.dt % 300 == 0:
            print("Changing signal shape")
            print("Shape:", self.shape)
            if self.shape > 2:
                self.shape = 0
            #self.max_value = round(random.uniform(0.05, 0.2), 2)
            self.random_factor = round(random.uniform(0.5, 2), 2) 
            self.max_value = self.random_factor * 0.1
            # if self.max_value == 0.1:
            #     self.max_value = 0.2
            # else:
            #     self.max_value = 0.1
            self.min_value = -self.max_value

        velocity_signal = self.generate_sinusoidal_signal()
        t_idx = self.sampling_counter % velocity_signal.size(1)
        self.velocities = velocity_signal[:, t_idx, :]  # Get velocity for all DOFs
        self.sampling_counter += 1

        # Scale velocities with the gains for all DOFs
        scaled_velocities = self.velocities  # Apply necessary scaling if needed

        # Create a zero tensor for all joint velocities
        full_joint_velocities = torch.zeros((self.num_envs, self.total_joints), device='cuda')

        # Assign the scaled velocities only to arm joints
        full_joint_velocities[:, self._joint_ind] = scaled_velocities

        self.robot.set_joint_velocity_target(target=full_joint_velocities)
        self.reference_publisher.publish_velocity(self.velocities[0].cpu().numpy())

    def stop(self):
        """Gracefully stop the background thread."""
        self.running = False
        self.apply_thread.join()
        
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

        #self.robot.write_joint_damping_to_sim(selected_damping, self._joint_indices)
        #self.robot.write_joint_friction_to_sim(selected_friction, self._joint_indices)


    def _apply_action(self) -> None:
        self.COUNTER += 1
        # No direct action application, we only adjust stiffness/damping
            
            # During each step, append the velocities to the buffers
            # Loop through all environments to append trajectory data
        self.current_step = self.dt  # Get current step for this environment
        # Update trajectory buffers for each environment individually
        self.vel = self.robot.data.joint_vel[:, self._joint_ind] if self.robot else torch.zeros_like(self.velocities)
        joint_velocities = self.vel[0].cpu().numpy()
        self.joint_publisher.publish_joint_states(joint_velocities)
        self.isaac_trajectory_buffer[:, self.current_step, :] = self.robot.data.joint_vel[:, self._joint_indices]
        self.gazebo_trajectory_buffer[:, self.current_step, :] = self.gazebo_velocities[:, :self.dof_group]
        self.dt += 1  # Increment time step for sampling sinusoidal signal
       

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
        gazebo_velocities_padded[:, :self.dof_group] = self.gazebo_velocities[:, :self.dof_group] * self.random_factor  # Zero inactive DOFs

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
        self.gazebo_velocity_error = torch.abs(gazebo_velocities * self.random_factor - self.robot.data.joint_vel[:, self._joint_indices])

        reward = - self.gazebo_velocity_error.abs().sum(dim=1)/self.dof_group  # Shape: (num_envs,)
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
        if self.current_step == 200 - 1:  # Check if buffer is full
            numerator = torch.sum((self.isaac_trajectory_buffer) * (self.gazebo_trajectory_buffer), dim=1)

            denominator = torch.sqrt(torch.sum((self.isaac_trajectory_buffer)**2, dim=1)) * torch.sqrt(torch.sum((self.gazebo_trajectory_buffer)**2, dim=1))
            
            print("NUMERATOR FOR LAST ADDED JOINT:", numerator[0])
            print("DENOMINATOR FOR LAST ADDED JOINT:", denominator[0])
            correlation = numerator / (denominator + 1e-6)  # Avoid division by zero
            print("CORRELATION FOR LAST ADDED JOINT:", torch.median(correlation))

            # Switch DOF if only the last joint reaches the threshold
            if torch.median(correlation) > 0.8:
                self.update_dof()
            self.isaac_trajectory_buffer = torch.zeros((self.num_envs, 200, self.dof_group), device=self.device)  # Buffer for Isaac velocities
            self.gazebo_trajectory_buffer = torch.zeros((self.num_envs, 200, self.dof_group), device=self.device)  # Buffer for Gazebo velocities
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

