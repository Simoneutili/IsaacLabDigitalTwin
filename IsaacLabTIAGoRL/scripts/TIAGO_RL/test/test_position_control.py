# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims."""



import argparse

from omni.isaac.lab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from omni.isaac.lab.sim import SimulationContext
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, ActuatorBaseCfg, DCMotorCfg

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import random



class TiagoJointPublisher(Node):
    def __init__(self, joint_names):
        super().__init__('tiago_joint_publisher')
        self.publisher_ = self.create_publisher(JointState, 'joint_configuration', 10)
        self.joint_names = joint_names

    def publish_joint_states(self, velocities):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = velocities.tolist()  # Convert tensor to list
        # You can also send efforts if needed, otherwise leave it empty
        self.publisher_.publish(msg)


class ReferenceVelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher_ = self.create_publisher(JointState, "/velocity_trajectory", 10)  # Use JointState
        self.msg = JointState()
        self.msg.name = [f"arm_{i+1}_joint" for i in range(7)]  # Joint names for 7-DOF
    def publish_velocity(self, velocities):
        if rclpy.ok():  # Check if the ROS context is still valid
            self.msg.position = velocities.tolist()  # Set velocities
            self.publisher_.publish(self.msg)
            # self.get_logger().info(f'Publishing velocities: {self.msg.velocity}')
        else:
            self.get_logger().warn('ROS context is not ok, cannot publish.')
        

def generate_sinusoidal_signal(frequency, duration, samples, num_envs, num_dofs, min_value, max_value):
    time = torch.linspace(0, duration, int(duration * samples), device="cuda").unsqueeze(-1)  # Time vector (samples, 1)
    repeated_time = time.repeat(1, num_dofs)  # Repeat the time for each joint

    # Generate sinusoidal signal for each DOF (velocity control)
    signal = (torch.sin(2 * torch.pi * frequency * repeated_time) / 2 + 0.5) \
                * (max_value - min_value) + min_value
    """
    signal = torch.where(
        torch.sign(torch.sin(2 * torch.pi * frequency * repeated_time)) > 0,
        max_value,
        min_value
    )
    """
    #period = 1/frequency
    #signal = (max_value - min_value) * torch.abs((repeated_time % period) - (period / 2)) * 2 / period + min_value
    # Repeat the signal for each environment (num_envs, samples, num_dofs)
    signal = signal.unsqueeze(0).repeat(num_envs, 1, 1)  # Shape: (num_envs, samples, num_dofs)
    return signal

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
        override_joint_dynamics=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
        ), articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 1.0, 1.0, 1.0),
    ),
    actuators={"tiago": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=500.0, damping=250.0)},
    #actuators={"tiago": DCMotorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=40)}
)

@configclass
class TiagoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    tiago: ArticulationCfg = TIAGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    

class Simulation:
    def __init__(self):
        self.num_dofs = 7
        self.total_joints = 24
        self.duration = 1000.0
        self.samples = 100.0
        self.joint_publisher = TiagoJointPublisher([f"arm_{i}_joint" for i in range(1, 8)])  # Assuming 7 arm joints
        self.reference_publisher = ReferenceVelocityPublisher()
        self.friction = torch.zeros(self.num_dofs).to('cuda')

    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        sim_dt = sim.get_physics_dt()        
        num_envs = scene.num_envs
        robot = scene["tiago"]
        count = 0

        frequency = 2.0
        max_value = 0.1
        min_value = - max_value
        target_publish_rate = 100  # Desired publish frequency in Hz
        publish_interval = 1.0 / target_publish_rate  # Time between publications in seconds
        up = 0
        
        joint_indices = []
        for i in range(self.num_dofs):
            joint_idx, _ = robot.find_joints("arm_" + str(i + 1) + "_joint")
            joint_indices.append(joint_idx)
        #joint_ids = [idx[0] for idx in joint_indices]
        
        joint_ids = [robot.find_joints(f"arm_{i}_joint")[0][0] for i in range(1, 8)]

        # Initialize lists to store data for all environments
        reference_signals = [[[] for _ in range(self.num_dofs)] for _ in range(num_envs)]
        actual_velocities = [[[] for _ in range(self.num_dofs)] for _ in range(num_envs)]
        errors = [[[] for _ in range(self.num_dofs)] for _ in range(num_envs)]

        # Generate the sinusoidal velocity signals
        velocity_signal = generate_sinusoidal_signal(frequency, self.duration, self.samples, num_envs, self.num_dofs, min_value, max_value)
        #damping = torch.logspace(25, 50, steps=7).to('cuda')

        #effort = torch.rand(7).to('cuda') * 10
        #robot.write_joint_armature_to_sim(armature, joint_ids)
        #robot.write_joint_effort_limit_to_sim(effort, joint_ids)
        #robot.write_joint_friction_to_sim(friction, joint_ids)
        #robot.write_joint_damping_to_sim(10, [joint_ids[4]])
        #robot.write_joint_friction_to_sim(100, [joint_ids[4]])
        # Simulation loop
        next_publish_time = time.time()  # Initialize the time for the next publish
        
        while simulation_app.is_running():
            #armature = torch.rand(self.num_dofs).to('cuda') * 5
            #robot.write_joint_armature_to_sim(armature, joint_ids) 
            if count % 200 == 0:
                #self.max_value = round(random.uniform(0.05, 0.3), 2)
                if max_value == 0.1:
                    max_value = 0.2
                    min_value = -max_value
                else:
                    max_value = 0.1
                    min_value = -max_value
                effort = torch.rand(7).to('cuda') * 1000
                #robot.write_joint_armature_to_sim(effort, joint_ids)
                #robot.write_joint_effort_limit_to_sim(100, joint_ids)
                
                #max_value = round(random.uniform(0.1, 0.4), 2)
                #min_value = -max_value
                #friction = torch.rand(7).to('cuda') * 0.2
                #frictions =  10
                #frictionss = torch.rand(2).to('cuda') 

                up += 1
                #print("Friction:", friction)
                #sprint("Damping:", damping)
                #self.friction = 50 * torch.ones(self.num_dofs).to('cuda')
                #robot.write_joint_friction_to_sim(self.friction, joint_ids)
                # joint_velocities = torch.randint(-1, 2, (num_envs, 7))
                
            velocity_signal = generate_sinusoidal_signal(frequency, self.duration, self.samples, num_envs, self.num_dofs, min_value, max_value)
            t_idx = count % velocity_signal.size(1)
            joint_velocities = torch.zeros((num_envs, self.num_dofs), device='cuda') + 0.5
            joint_velocities[:,2] = velocity_signal[:, t_idx, 0]
            print("Joint positions:", joint_velocities)
            #print("Joint Velocities:", joint_velocities)
            # Generate a tensor with random integers between -1 and 1
            
            #print("joint indices", joint_ids)
            #print("Friction:", self.friction)
            """
            for env_idx in range(num_envs):
                for idx in range(self.num_dofs):
                    # Store reference velocities and actual velocities for all environments
                    reference_signals[env_idx][idx].append(joint_velocities[env_idx, idx].cpu().item())
                    actual_velocities[env_idx][idx].append(robot.data.joint_vel[env_idx, joint_ids[idx]].cpu().item())
                    error = joint_velocities[env_idx, idx] - robot.data.joint_vel[env_idx, joint_ids[idx]]
                    errors[env_idx][idx].append(error.cpu().item())
            """
            
            # Create a zero tensor for all joint velocities
            full_joint_velocities = torch.zeros((num_envs, self.total_joints), device='cuda')

            # Assign the velocity only to arm joints using joint_ids
            # Ensure joint_velocities is on the same device as full_joint_velocities
            full_joint_velocities[:, joint_ids] = joint_velocities.float().to(full_joint_velocities.device)
            print("full_joint_velocities:", full_joint_velocities)      
         # Publish velocities to ROS2 at 100 Hz
            if time.time() >= next_publish_time:
                next_publish_time += publish_interval  # Update for the next publish
                
                # Now set the full_joint_velocities for the robot, but only modify arm joint velocities
                robot.set_joint_position_target(target=full_joint_velocities)
                effort = robot.data.applied_torque[:, joint_ids]
                #print("Effort:", effort)
                #print("Friction:", friction)
                joint_positions = robot.data.joint_pos[:, joint_ids]
                joint_vel = robot.data.joint_pos[0, joint_ids]
                joint_vel = joint_vel.cpu().numpy()
                # Publish joint state data to the ROS2 topic
                self.joint_publisher.publish_joint_states(joint_vel)
                
                self.reference_publisher.publish_velocity(joint_velocities[0].cpu().numpy())  # Publish reference velocities to ROS2 topic

            
            # Print the joint velocity target buffer
            # Assuming the joint velocity target buffer is stored in self.robot.data.joint_velocity_target
           # print("Joint Velocity Target Buffer:")
            #print(robot.data.joint_vel_target)
            print("Actual Joint Velocities:")
            print(robot.data.joint_vel[:, joint_ids])
            
            self.num_envs = num_envs
            scene.write_data_to_sim()
            sim.step()
            count += 1
            scene.update(sim_dt)
            if count >= self.duration * self.samples:
                break
        
        # Convert collected data to NumPy arrays
        self.reference_signals = [[np.array(reference_signals[env_idx][i]) for i in range(self.num_dofs)] for env_idx in range(num_envs)]
        self.actual_velocities = [[np.array(actual_velocities[env_idx][i]) for i in range(self.num_dofs)] for env_idx in range(num_envs)]
        self.errors = [[np.array(errors[env_idx][i]) for i in range(self.num_dofs)] for env_idx in range(num_envs)]

    def plot_data(self):
        print("Plotting data...")
        time_vector = np.linspace(0, self.duration, int(self.duration * self.samples), endpoint=False)

        for i in range(self.num_dofs):
            fig, axes = plt.subplots(4, self.num_envs, figsize=(20, 12))  # 4 rows for 4 plots, num_envs columns
            
            # If only one environment, axes may be 1D, so we adjust for that
            if self.num_envs == 1:
                axes = np.expand_dims(axes, axis=-1)  # Make it 2D for consistent indexing
                
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

def main():
    # Main function.
    # Initialize ROS 2
    rclpy.init()
    #physics = sim_utils.PhysxCfg(solver_type=2, max_position_iteration_count=50, max_velocity_iteration_count=50)  
    sim_cfg = sim_utils.SimulationCfg(dt = 1/100, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_cfg = TiagoSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    simulation = Simulation()
    simulation.run_simulator(sim, scene)
    # shutdown rclpy
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()