#Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=7, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import torch
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from omni.isaac.lab.assets import Articulation, ArticulationCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from stable_baselines3 import A2C, PPO
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize


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
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={"tiago_arm": ImplicitActuatorCfg(joint_names_expr=["arm_[1-7]_joint"],stiffness=10.0, damping=50.0)},
)

# Define Tiago environment configuration


@configclass
class TiagoEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 15.0  # Increase if needed for better learning
    action_scale = 0.1  # Scale for action adjustments
    action_space = gym.spaces.Box(low=-10, high=1e4, shape=(7, 2))  # Stiffness and damping adjustments
    observation_space = 14  # Joint positions and velocities
    state_space = 0
    num_envs = 3
    # seed: int = -1  # Add this line to properly initialize the seed attribute

    # simulation
    sim = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = TIAGO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0, replicate_physics=True)

    # reward scales
    rew_scale_alive = 2.0
    rew_scale_terminated = -0.5  # Reward for termination
    rew_scale_error = -2  # Reward for minimizing error
    stiffness_penalty_scale = 0.5  # Penalty for large stiffness changes
    negative_param_scale = 1.0  # Penalty for negative stiffness/damping values


class TiagoEnv(DirectRLEnv):
    cfg: TiagoEnvCfg

    def __init__(self, cfg: TiagoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # self._simulation = Simulation()
        self._joint_indices = [self.robot.find_joints(f"arm_{i}_joint")[0][0] for i in range(1, 8)]
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        self.total_error = torch.zeros((self.cfg.num_envs), device=self.device)

        self.duration = 15.0
        self.samples = 100.0
        self.num_dofs = 7
        self.frequency = 0.5
        self.min_value = -0.4
        self.max_value = 0.4
        # self.dt = 0  # Initialize time step (for sampling sinusoidal signal)
        
        # Initialize lists to store data for all environments
        self.reference_signals = [[[] for _ in range(self.num_dofs)] for _ in range(self.num_envs)]
        self.actual_velocities = [[[] for _ in range(self.num_dofs)] for _ in range(self.num_envs)]
        self.errors = [[[] for _ in range(self.num_dofs)] for _ in range(self.num_envs)]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Clone the environment for multiple instances, filter collisions and add the robot
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["tiago"] = self.robot
        
        return self.robot
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # self.dt += 1  # Increment time step for sampling sinusoidal signal
        self.actions = actions.clone()  # actions shape: (num_envs, 7, 2)

        # Split the action tensor into stiffness and damping parts
        stiffness_change = self.actions[:, :, 0]  # first coloumn for stiffness (scaled)
        damping_change = self.actions[:, :, 1]  # last coloumn for damping (scaled)
        print("stiffness change shape", stiffness_change.shape)
        print("damping change shape", damping_change.shape)

        # Access current stiffness and damping for all joints
        current_stiffness = self.robot.data.joint_stiffness[:, self._joint_indices]
        current_damping = self.robot.data.joint_damping[:, self._joint_indices]

        # Calculate new stiffness and damping values for all joints (element-wise addition)
        new_stiffness = current_stiffness + stiffness_change
        new_damping = current_damping + damping_change

        # Write new stiffness and damping values to the simulation for all joints at once
        self.robot.write_joint_stiffness_to_sim(new_stiffness, self._joint_indices)
        self.robot.write_joint_damping_to_sim(new_damping, self._joint_indices)
    """
    def _pre_physics_step(self, actions: torch.Tensor, velocities) -> None:
        self.velocities = velocities
        self.actions = actions.clone()
        # Adjust stiffness and damping based on actions
        stiffness_change = actions[:, 0] * 100.0  # Scale action for stiffness
        damping_change = actions[:, 1] * 50.0  # Scale action for damping
        for i, joint_id in enumerate(self._joint_indices):
            current_stiffness = self.robot.data.joint_stiffness[:,joint_id]
            current_damping = self.robot.data.joint_damping[:,joint_id]
            self.robot.write_joint_stiffness_to_sim(current_stiffness + stiffness_change[:, i], joint_id)
            self.robot.write_joint_damping_to_sim(current_damping + damping_change[:, i], joint_id)
    """

    def _apply_action(self) -> None:
        # No direct action application, we only adjust stiffness/damping
        # self.velocities = self._simulation.command_transfer()
        velocity_signal = generate_sinusoidal_signal(self.frequency, self.duration, self.samples, self.num_envs, self.num_dofs, self.min_value, self.max_value)
        t_idx = self._sim_step_counter % velocity_signal.size(1)
        self.velocities = velocity_signal[:, t_idx, :]
        # self.velocities = self._simulation.joint_velocities
        if self.velocities is None:
            print("[ERROR]: Velocities are not set, skipping action application.")
            print("shape required", self.robot.data.joint_vel[:, self._joint_indices].shape)
        print("velocities capted", self.velocities)
        self.robot.set_joint_velocity_target(target=self.velocities, joint_ids=self._joint_indices)
        
        # TESTING PLOT !!!!
        
        for env_idx in range(self.num_envs):
            for idx in range(self.num_dofs):
                # Store reference velocities and actual velocities for all environments
                self.reference_signals[env_idx][idx].append(self.velocities[env_idx, idx].cpu().item())
                self.actual_velocities[env_idx][idx].append(self.robot.data.joint_vel[env_idx, self._joint_indices[idx]].cpu().item())
                error = self.velocities[env_idx, idx] - self.robot.data.joint_vel[env_idx, self._joint_indices[idx]]
                self.errors[env_idx][idx].append(error.cpu().item())
        
        
    """
    def _get_observations(self) -> dict:
        obs = torch.cat((
            self.joint_pos,
            self.joint_vel,
        ), dim=-1)
        return {"policy": obs}
    """
    def _get_observations(self) -> dict:
        # Gather joint velocities (current observation)
        joint_velocities = self.joint_vel

        # Gather current stiffness and damping values
        current_stiffness = self.robot.data.joint_stiffness[:, self._joint_indices]
        current_damping = self.robot.data.joint_damping[:, self._joint_indices]

        # Reference velocities (target velocities from reference signal)
        reference_velocities = self.velocities  # This holds the reference velocity generated by the sinusoidal signal

        # Velocity error (optional)
        velocity_error = reference_velocities - joint_velocities

        # Concatenate all observations (adjust as necessary)
        obs = torch.cat((
            joint_velocities,  # Current joint velocities
            reference_velocities,  # Reference joint velocities
            current_stiffness,  # Current stiffness values
            current_damping,  # Current damping values
            velocity_error  # Velocity error (optional)
        ), dim=-1)

        return {"policy": obs}
    
    def calculate_error(self) -> torch.Tensor:          
        # Assuming the velocities are in range [-1, 1]
        error = torch.norm(self.velocities - self.robot.data.joint_vel[:, self._joint_indices], dim=1)
        max_error = 1.0  # Define a meaningful maximum error threshold
        normalized_error = error / max_error  # Normalize the error
        return torch.clamp(normalized_error, 0, 1)  # Ensure the error is between 0 and 1    
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = self.compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_error,
            self.cfg.stiffness_penalty_scale,
            self.reset_terminated,
        )
        return total_reward
    
    # @torch.jit.script # torch.jit.script is used to compile the function to TorchScript for performance optimization in terms of speed
    def compute_rewards(self,
                        rew_scale_alive,
                        rew_scale_terminated, 
                        rew_scale_error,
                        stiff_penalty_scale,
                        reset_terminated, 
                        ):
        # Calculate the error between reference and actual joint positions
        self.total_error = self.calculate_error()
        rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
        rew_termination = rew_scale_terminated * reset_terminated.float()
        # reward_alive = rew_scale_alive * torch.ones(self.cfg.num_envs, device=self.device)
        reward_error = rew_scale_error * self.total_error
        # Ensure stiffness penalties are summed over the correct dimension
        stiffness_penalty = torch.abs(self.actions[:, :, 0])  # Shape: (num_envs, 7, 2) 
        stiff_penalty = -stiff_penalty_scale * torch.sum(stiffness_penalty, dim=1)  # Shape: (num_envs,)
        
        # Negative stiffness penalty
        negative_stiff_penalty = -self.cfg.negative_param_scale * torch.sum(torch.relu(self.actions[:, :, 0]), dim=1)  # Shape: (num_envs,)
        
        # Negative damping penalty
        negative_damp_penalty = -self.cfg.negative_param_scale * torch.sum(torch.relu(self.actions[:, :, 1]), dim=1)  # Ensure this covers the remaining actions
        
        print("Shapes:")
        print("rew_alive shape:", rew_alive.shape)
        print("rew_termination shape:", rew_termination.shape)
        print("reward_error shape:", reward_error.shape)
        print("stiff_penalty shape:", stiff_penalty.shape)
        print("negative_stiff_penalty shape:", negative_stiff_penalty.shape)
        print("negative_damp_penalty shape:", negative_damp_penalty.shape)
        
        total_reward = (
            rew_alive + 
            rew_termination + 
            reward_error +  # Ensure this matches (num_envs,)
            stiff_penalty + 
            negative_stiff_penalty + 
            negative_damp_penalty
        )
        print("Total reward shape:", total_reward.shape)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_indices]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_indices]
        
        time_out = self.episode_length_buf >= self.max_episode_length -1
        # out_of_bounds = self.total_error < 0.01  # Define a meaningful error threshold
        # check if any joint velocity is greater than a limit
        out_of_bounds = torch.any(self.joint_vel > 7.0, dim=1) 
        # print("Time out:", time_out)
        # print("Out of bounds shape:", out_of_bounds.shape)
        return time_out, out_of_bounds
    
    def plot_data(self, env_ids):
        print("Plotting data...")
        # time_vector = np.linspace(0, self.duration, int(self.duration * self.samples), endpoint=False)
        
        for i in range(self.num_dofs):
            fig, axes = plt.subplots(4, self.num_envs, figsize=(20, 12))  # 4 rows for 4 plots, num_envs columns

            for env_idx in env_ids:
                
                # By multiplying the number of action steps in the episode (self.episode_length_buf)
                # by both the simulation time step (dt) and the decimation value, you get the total episode length in seconds.
                # The total simulation time per action step is:
                # Total Time per Action Step=dt×decimation where dt is self.cfg.sim.dt (simulation step duration).
                
                duration = self.episode_length_buf[env_idx].cpu() * (self.cfg.sim.dt * self.cfg.decimation) # Calculate the duration of the episode
                episode_length = len(self.reference_signals[env_idx][i])
                time_vector = np.linspace(0, duration, episode_length, endpoint=False)
                # Row 1: Reference Velocity
                axes[0, env_idx].plot(time_vector, self.reference_signals[env_idx][i], label='Reference Velocity', color='blue')
                axes[0, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Reference Velocity')
                axes[0, env_idx].set_xlabel('Time [s]')
                axes[0, env_idx].set_ylabel('Velocity')
                axes[0, env_idx].grid()
                axes[0, env_idx].legend()

                # Row 2: Actual Velocity
                axes[1, env_idx].plot(time_vector, self.actual_velocities[env_idx][i], label='Actual Velocity', color='red')
                axes[1, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Actual Velocity')
                axes[1, env_idx].set_xlabel('Time [s]')
                axes[1, env_idx].set_ylabel('Velocity')
                axes[1, env_idx].grid()
                axes[1, env_idx].legend()

                # Row 3: Comparison of Reference vs Actual Velocity
                axes[2, env_idx].plot(time_vector, self.reference_signals[env_idx][i], label='Reference Velocity', color='blue')
                axes[2, env_idx].plot(time_vector, self.actual_velocities[env_idx][i], label='Actual Velocity', color='red')
                axes[2, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Velocity Comparison')
                axes[2, env_idx].set_xlabel('Time [s]')
                axes[2, env_idx].set_ylabel('Velocity')
                axes[2, env_idx].grid()
                axes[2, env_idx].legend()

                # Row 4: Velocity Error
                axes[3, env_idx].plot(time_vector, self.errors[env_idx][i], label='Error', color='green')
                axes[3, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Velocity Error')
                axes[3, env_idx].set_xlabel('Time [s]')
                axes[3, env_idx].set_ylabel('Error')
                axes[3, env_idx].grid()
                axes[3, env_idx].legend()

            plt.tight_layout()  # Adjust the layout
        plt.show()
    

    def _reset_idx(self, env_ids: torch.Tensor | None):
        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # plotting data
        if self.dt > 0:
            plt.close()
            # Convert collected data to NumPy arrays
            self.reference_signals = [[np.array(self.reference_signals[env_idx][i]) for i in range(self.num_dofs)] for env_idx in env_ids]
            self.actual_velocities = [[np.array(self.actual_velocities[env_idx][i]) for i in range(self.num_dofs)] for env_idx in env_ids]
            self.errors = [[np.array(self.errors[env_idx][i]) for i in range(self.num_dofs)] for env_idx in env_ids]

            self.plot_data(env_ids)

        super()._reset_idx(env_ids)
        
        # Reset joint positions and velocities for the new episode

        # Get the default root state for the new episode
        default_root_state = self.robot.data.default_root_state[env_ids]
        
        # Adjust the root state based on the environment origins
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Reset joint positions and velocities for the new episode
        # joint_pos = self.robot.data.default_joint_pos[env_ids, self._joint_indices]
        
        # iterate over the environments for tensor broadcasting of specific joint indices
        joint_pos = torch.stack([self.robot.data.default_joint_pos[env_id, self._joint_indices] for env_id in env_ids])
        joint_vel = torch.stack([self.robot.data.default_joint_vel[env_id, self._joint_indices] for env_id in env_ids])
        # joint_vel = self.robot.data.default_joint_vel[env_ids, self._joint_indices]
        
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, self._joint_indices, env_ids)
        
        # plotting data
        if self.dt > 0:
            # Convert collected data to NumPy arrays
            self.reference_signals = [[np.array(self.reference_signals[env_idx][i]) for i in range(self.num_dofs)] for env_idx in env_ids]
            self.actual_velocities = [[np.array(self.actual_velocities[env_idx][i]) for i in range(self.num_dofs)] for env_idx in env_ids]
            self.errors = [[np.array(self.errors[env_idx][i]) for i in range(self.num_dofs)] for env_idx in env_ids]

            self.plot_data(env_ids)


# Assuming the original simulation class and plotting functions remain as is

def generate_sinusoidal_signal(frequency, duration, samples, num_envs, num_dofs, min_value, max_value):
    time = torch.linspace(0, duration, int(duration * samples), device="cuda").unsqueeze(-1)  # Time vector (samples, 1)
    repeated_time = time.repeat(1, num_dofs)  # Repeat the time for each joint

    # Generate sinusoidal signal for each DOF (velocity control)
    signal = (torch.sin(2 * torch.pi * frequency * repeated_time) / 2 + 0.5) \
                * (max_value - min_value) + min_value

    # Repeat the signal for each environment (num_envs, samples, num_dofs)
    signal = signal.unsqueeze(0).repeat(num_envs, 1, 1)  # Shape: (num_envs, samples, num_dofs)
    return signal


class Simulation:
    def __init__(self):
        self.num_dofs = 7
        self.duration = 10.0
        self.samples = 100.0
        
        # not sure about this because the robot is not defined here 
        # self.robot = Articulation(TIAGO_CFG)


    def run_simulator(self, env: DirectRLEnv, model):
        # tiago_env = TiagoEnv(TiagoEnvCfg())
        # self.robot = env.robot()
        #self.robot = env._setup_scene()
        # sim_dt = sim.get_physics_dt()        
        self.num_envs = env.num_envs
        count = 0

        """  
        joint_indices = []
        for i in range(self.num_dofs):
            joint_idx, _ = robot.find_joints("arm_" + str(i + 1) + "_joint")
            joint_indices.append(joint_idx)
        """    
        # joint_ids = [idx[0] for idx in joint_indices]
        
        # Get joint indices for all 7 joints 
        # joint_indices = [self.robot.find_joints(f"arm_{i}_joint")[0][0] for i in range(1, 8)]
        
        # Initialize lists to store data for all environments
        # reference_signals = [[[] for _ in range(self.num_dofs)] for _ in range(self.num_envs)]
        # actual_velocities = [[[] for _ in range(self.num_dofs)] for _ in range(self.num_envs)]
        # errors = [[[] for _ in range(self.num_dofs)] for _ in range(self.num_envs)]
        
        score = 0
        obs = env.reset()
        # Simulation loop
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                
                """
                for env_idx in range(self.num_envs):
                    for idx in range(self.num_dofs):
                        # Store reference velocities and actual velocities for all environments
                        reference_signals[env_idx][idx].append(self.joint_velocities[env_idx, idx].cpu().item())
                        actual_velocities[env_idx][idx].append(self.robot.data.joint_vel[env_idx, joint_indices[idx]].cpu().item())
                        error = self.joint_velocities[env_idx, idx] - self.robot.data.joint_vel[env_idx, joint_indices[idx]]
                        errors[env_idx][idx].append(error.cpu().item())
                """
            
                # Run the simulation and learning algorithm
                
                actions, _state = model.predict(obs, deterministic=True)  # Get actions from the RL policy
                obs, rewards, done, info = env.step(actions)
                score += rewards
                # env.plot_data()  # Plot data if required
                print(f"[INFO]: Episode complete, rewards: {score}")

            #count += 1
            #if count >= self.duration * self.samples:
            #    break
        """
        # Convert collected data to NumPy arrays
        self.reference_signals = [[np.array(reference_signals[env_idx][i]) for i in range(self.num_dofs)] for env_idx in range(self.num_envs)]
        self.actual_velocities = [[np.array(actual_velocities[env_idx][i]) for i in range(self.num_dofs)] for env_idx in range(self.num_envs)]
        self.errors = [[np.array(errors[env_idx][i]) for i in range(self.num_dofs)] for env_idx in range(self.num_envs)]
        """

    def plot_data(self):
        print("Plotting data...")
        time_vector = np.linspace(0, self.duration, int(self.duration * self.samples), endpoint=False)
        
        for i in range(self.num_dofs):
            fig, axes = plt.subplots(4, self.num_envs, figsize=(20, 12))  # 4 rows for 4 plots, num_envs columns

            for env_idx in range(self.num_envs):
                # Row 1: Reference Velocity
                axes[0, env_idx].plot(time_vector, self.reference_signals[env_idx][i], label='Reference Velocity', color='blue')
                axes[0, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Reference Velocity')
                axes[0, env_idx].set_xlabel('Time [s]')
                axes[0, env_idx].set_ylabel('Velocity')
                axes[0, env_idx].grid()
                axes[0, env_idx].legend()

                # Row 2: Actual Velocity
                axes[1, env_idx].plot(time_vector, self.actual_velocities[env_idx][i], label='Actual Velocity', color='red')
                axes[1, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Actual Velocity')
                axes[1, env_idx].set_xlabel('Time [s]')
                axes[1, env_idx].set_ylabel('Velocity')
                axes[1, env_idx].grid()
                axes[1, env_idx].legend()

                # Row 3: Comparison of Reference vs Actual Velocity
                axes[2, env_idx].plot(time_vector, self.reference_signals[env_idx][i], label='Reference Velocity', color='blue')
                axes[2, env_idx].plot(time_vector, self.actual_velocities[env_idx][i], label='Actual Velocity', color='red')
                axes[2, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Velocity Comparison')
                axes[2, env_idx].set_xlabel('Time [s]')
                axes[2, env_idx].set_ylabel('Velocity')
                axes[2, env_idx].grid()
                axes[2, env_idx].legend()

                # Row 4: Velocity Error
                axes[3, env_idx].plot(time_vector, self.errors[env_idx][i], label='Error', color='green')
                axes[3, env_idx].set_title(f'Env {env_idx+1} - Joint {i+1} Velocity Error')
                axes[3, env_idx].set_xlabel('Time [s]')
                axes[3, env_idx].set_ylabel('Error')
                axes[3, env_idx].grid()
                axes[3, env_idx].legend()

            plt.tight_layout()  # Adjust the layout
        plt.show()


class Simulation:

    def __init__(self):
        
        self.duration = 10.0
        self.samples = 100.0
        self.num_dofs = 7
        self.joint_velocities = torch.zeros((5, self.num_dofs), device="cuda")

    def command_transfer(self):
        if self.joint_velocities is None:
            print("[ERROR]: Joint velocities not set.")
        return self.joint_velocities

    def run_simulator(self, env: DirectRLEnv, env_cfg: TiagoEnvCfg):
        # simulate physics      
        num_envs = env.num_envs
        count = 0
        frequency = 0.5
        min_value = -0.4
        max_value = 0.4
        velocity_signal = generate_sinusoidal_signal(frequency, self.duration, self.samples, num_envs, self.num_dofs, min_value, max_value)
        
        while simulation_app.is_running():
            with torch.inference_mode():
                # reset
                if count % 500 == 0:
                    count = 0
                    env.reset()
                    print("-" * 80)
                    print("[INFO]: Resetting environment...")

                t_idx = count % velocity_signal.size(1)
                # self.joint_velocities = velocity_signal[:, t_idx, :]
                # print("joint velocities reference", self.joint_velocities)
                # print("velocity signal shape", self.joint_velocities.shape)
                # sample random actions
                actions = torch.randn_like(env.actions)
                # step the environment
                obs, score, _, _, _ = env.step(actions)
                # print current orientation of pole
                policy_obs = obs["policy"]
                print("[Env 0]: Pole joint: ", policy_obs)
                print("[Env 0]: Reward: ", score[0])
                # update counter
                count += 1

def main():
    #Main function.
    # parse the arguments
    env_cfg = TiagoEnvCfg()
    # setup base environment
    env = TiagoEnv(cfg = env_cfg)
    simulation = Simulation()
    # Run the simulation
    simulation.run_simulator(env, env_cfg)
    # close the environment
    env.close()

if __name__ == "__main__":    
    # Run the main function
    main()    
    # Close the simulation app
    simulation_app.close()
