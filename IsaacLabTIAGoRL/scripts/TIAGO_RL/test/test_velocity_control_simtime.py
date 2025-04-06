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
import threading


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
        fix_base=True,
        override_joint_dynamics=True,
        make_instanceable=False,
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
    actuators={"tiago": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=500.0)},
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
    def __init__(self, scene: InteractiveScene):
        self.num_dofs = 7
        self.total_joints = 24
        self.duration = 1000.0
        self.samples = 100.0
        self.joint_publisher = TiagoJointPublisher([f"arm_{i}_joint" for i in range(1, 8)])  # Assuming 7 arm joints
        self.reference_publisher = ReferenceVelocityPublisher()   
        self.robot = scene["tiago"]
        self._joint_ind = [self.robot.find_joints(f"arm_{i}_joint")[0][0] for i in range(1, 8)]
        self.num_envs = scene.num_envs
        self.frequency = 2.0
        self.max_value = 0.1
        self.min_value = - self.max_value
        self.velocity_signal = generate_sinusoidal_signal(self.frequency, self.duration, self.samples, self.num_envs, self.num_dofs, self.min_value, self.max_value)

    def run_simulator(self, sim: sim_utils.SimulationContext, scene: InteractiveScene):
        sim_dt = sim.get_physics_dt()        
        count = 0
        num_envs = self.num_envs
        joint_ids = self._joint_ind
        current_time = time.time()
        while simulation_app.is_running():
            print(f"Time: {time.time() - current_time}")
            current_time = time.time()
            if count % 200 == 0:
                #self.max_value = round(random.uniform(0.05, 0.3), 2)
                # if max_value == 0.1:
                #     max_value = 0.2
                #     min_value = -max_value
                # else:
                #     max_value = 0.1
                #     min_value = -max_value
                effort = torch.rand(7).to('cuda') * 1000
                #robot.write_joint_armature_to_sim(effort, joint_ids)
                #robot.write_joint_effort_limit_to_sim(100, joint_ids)
                friction = torch.tensor([7.939, 9.0, 20.0, 237.0, 278.0, 100.2315, 100.2810]).to('cuda')
                #robot.write_joint_friction_to_sim(friction, joint_ids)
                damping = torch.tensor([141.358, 121.0, 363.0, 82.0, 300.0, 300.0, 300.0]).to('cuda') 
                #robot.write_joint_damping_to_sim(damping, joint_ids)
                
            velocity_signal = self.velocity_signal
            t_idx = count % velocity_signal.size(1)
            self.velocities = velocity_signal[:, t_idx, :]  # Get velocity for all DOFs

            # Scale velocities with the gains for all DOFs
            scaled_velocities = self.velocities  # Apply necessary scaling if needed

            # Create a zero tensor for all joint velocities
            full_joint_velocities = torch.zeros((self.num_envs, self.total_joints), device='cuda')

            # Assign the scaled velocities only to arm joints
            full_joint_velocities[:, self._joint_ind] = scaled_velocities

            self.robot.set_joint_velocity_target(target=full_joint_velocities)
            self.reference_publisher.publish_velocity(self.velocities[0].cpu().numpy())
            self.vel = self.robot.data.joint_vel[:, self._joint_ind] if self.robot else torch.zeros_like(self.velocities)
            joint_velocities = self.vel[0].cpu().numpy()
            self.joint_publisher.publish_joint_states(joint_velocities)
            print("Actual Joint Velocities:")
            print(self.robot.data.joint_vel[:, joint_ids])
            
            self.num_envs = num_envs
            scene.write_data_to_sim()
            sim.step()
            count += 1
            scene.update(sim_dt)
            if count >= self.duration * self.samples:
                break
        

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
    simulation = Simulation(scene)
    simulation.run_simulator(sim, scene)
    # shutdown rclpy
    rclpy.shutdown()

if __name__ == "__main__":
    main()
    simulation_app.close()