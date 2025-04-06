# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import importlib.util
import os
import sys
from time import sleep

import ray
import util
from ray import air, tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.repeater import Repeater
import pickle

"""
This script breaks down an aggregate tuning job, as defined by a hyperparameter sweep configuration,
into individual jobs (shell commands) to run on the GPU-enabled nodes of the cluster.
By default, one worker is created for each GPU-enabled node in the cluster for each individual job.
To use more than one worker per node (likely the case for multi-GPU machines), supply the
num_workers_per_node argument.

Each hyperparameter sweep configuration should include the workflow,
runner arguments, and hydra arguments to vary.

This assumes that all workers in a cluster are homogeneous. For heterogeneous workloads,
create several heterogeneous clusters (with homogeneous nodes in each cluster),
then submit several overall-cluster jobs with :file:`../submit_job.py`.
KubeRay clusters on Google GKE can be created with :file:`../launch.py`

To report tune metrics on clusters, a running MLFlow server with a known URI that the cluster has
access to is required. For KubeRay clusters configured with :file:`../launch.py`, this is included
automatically, and can be easily found with with :file:`grok_cluster_with_kubectl.py`

Usage:

python TIAGO_RL/tiago_RL/ISAACLAB_TEST/hyperparameter_tuning/tuner.py \
    --run_mode local \
    --cfg_file TIAGO_RL/tiago_RL/ISAACLAB_TEST/hyperparameter_tuning/tiago_cfg_file.py \
    --cfg_class TiAGO_SB3SAC_JobCfg

TENSORBOARD METRICS VISUALIZATION: 

tensorboard --logdir /home/simone/IsaacLab/logs/sb3/Isaac-DigitalTwin-TiAGO-v0




"""

DOCKER_PREFIX = "/workspace/isaaclab/"
BASE_DIR = os.path.expanduser("~")
PYTHON_EXEC = "./isaaclab.sh -p"
WORKFLOW = "TIAGO_RL/tiago_RL/ISAACLAB_TEST/SB3/train.py"
#WORKFLOW = "TIAGO_RL/tiago_RL/ISAACLAB_TEST/RL_GAMES/train.py"
NUM_WORKERS_PER_NODE = 2  # needed for local parallelism


class IsaacLabTuneTrainable(tune.Trainable):
    """Isaac Lab Trainable class for Population-Based Training (PBT)."""

    def setup(self, config: dict) -> None:
        """Initializes the experiment setup with given configuration."""
        self.data = None
        self.invoke_cmd = util.get_invocation_command_from_cfg(
            cfg=config, python_cmd=PYTHON_EXEC, workflow=WORKFLOW
        )
        print(f"[INFO]: Recovered invocation command: {self.invoke_cmd}")
        self.experiment = None
        self.proc = None
        self.experiment_name = None
        self.isaac_logdir = None
        self.tensorboard_logdir = None

    def reset_config(self, new_config: dict) -> bool:
        """Allows resetting the configuration dynamically."""
        self.setup(new_config)
        return True

    def step(self) -> dict:
        """Performs a single training step and reports metrics."""
        if self.experiment is None:
            # Start experiment
            print(f"[INFO]: Starting experiment with command: {self.invoke_cmd}...")
            experiment = util.execute_job(
                self.invoke_cmd,
                identifier_string="",
                extract_experiment=True,
                persistent_dir=BASE_DIR,
            )
            self.experiment = experiment
            self.proc = experiment["proc"]
            self.experiment_name = experiment["experiment_name"]
            self.isaac_logdir = experiment["logdir"]
            self.tensorboard_logdir = os.path.join(self.isaac_logdir, self.experiment_name)
            self.done = False

        if self.proc is None:
            raise ValueError("[ERROR]: Could not start trial process.")

        proc_status = self.proc.poll()
        if proc_status is not None:
            self.data["done"] = True
            print(f"[INFO]: Process finished with status {proc_status}, returning data...")
        else:
            # Load TensorBoard logs
            data = util.load_tensorboard_logs(self.tensorboard_logdir)

            while data is None:
                sleep(2)  # Avoid excessive polling
                data = util.load_tensorboard_logs(self.tensorboard_logdir)

            if self.data is not None:
                while util._dicts_equal(data, self.data):
                    sleep(2)  # Wait for new data
                    data = util.load_tensorboard_logs(self.tensorboard_logdir)

            print(f"[INFO]: Reporting metrics: {data}")

            while "rollout/ep_rew_mean" not in data:
                sleep(2)
                data = util.load_tensorboard_logs(self.tensorboard_logdir)
                print(f"[INFO]: Waiting for 'rollout/ep_rew_mean' in {data}")

            # Report metrics to Ray Tune
            tune.report(reward=data["rollout/ep_rew_mean"])
            self.data = data
            self.data["done"] = False

        return self.data

    def save_checkpoint(self, checkpoint_dir):
        """Save training state for Population-Based Training (PBT)."""
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")

        checkpoint_data = {
            "invoke_cmd": self.invoke_cmd,
            "experiment_name": self.experiment_name,
            "tensorboard_logdir": self.tensorboard_logdir,
            "training_data": self.data,
        }

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        print(f"[INFO]: Checkpoint saved in {checkpoint_dir}")
        return checkpoint_dir  # ✅ Return the checkpoint directory, not the file path

    def load_checkpoint(self, checkpoint_dir):
        """Load training state from a checkpoint."""
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        self.invoke_cmd = checkpoint_data["invoke_cmd"]
        self.experiment_name = checkpoint_data["experiment_name"]
        self.tensorboard_logdir = checkpoint_data["tensorboard_logdir"]
        self.data = checkpoint_data["training_data"]

        print(f"[INFO]: Checkpoint loaded from {checkpoint_dir}")

    def default_resource_request(self):
        """Specifies resource requirements for Ray Tune."""
        resources = util.get_gpu_node_resources(one_node_only=True)
        if NUM_WORKERS_PER_NODE != 1:
            print("[WARNING]: Splitting node into more than one worker")

        return tune.PlacementGroupFactory(
            [{"CPU": resources["CPU"] / NUM_WORKERS_PER_NODE, "GPU": resources["GPU"] / NUM_WORKERS_PER_NODE}],
            strategy="STRICT_PACK",
        )


def invoke_tuning_run(cfg: dict, args: argparse.Namespace) -> None:
    """Invoke an Isaac-Ray tuning run.

    Log either to a local directory or to MLFlow.
    Args:
        cfg: Configuration dictionary extracted from job setup
        args: Command-line arguments related to tuning.
    """
    # Allow for early exit
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"

    print("[WARNING]: Not saving checkpoints, just running experiment...")
    print("[INFO]: Model parameters and metrics will be preserved.")
    print("[WARNING]: For homogeneous cluster resources only...")
    # Get available resources
    resources = util.get_gpu_node_resources()
    print(f"[INFO]: Available resources {resources}")

    if not ray.is_initialized():
        ray.init(
            address=args.ray_address,
            log_to_driver=True,
            num_gpus=len(resources),
        )

    print(f"[INFO]: Using config {cfg}")

    # Configure the search algorithm and the repeater
	# Define Population-Based Training hyperparameter mutations
    hyperparam_mutations = {
        "agent.learning_rate": tune.loguniform(1e-5, 1e-2),  # Learning rate mutation range
        "agent.batch_size": [64, 128, 256, 512],  # Batch size mutation choices
        "agent.gamma": tune.uniform(0.9, 0.999),  # Discount factor
        "agent.ent_coef": tune.loguniform(1e-3, 1e-1),  # Entropy coefficient
        "agent.tau": tune.loguniform(1e-3, 0.1),  # Soft update coefficient
        "agent.train_freq": [1, 10, 100],  # Train frequency mutation choices
        "agent.gradient_steps": tune.randint(1, 41),  # Gradient steps mutation
        "agent.buffer_size": [50000, 100000, 500000],  # Replay buffer sizes
    }

	# Define PBT Scheduler (Adaptive Hyperparameter Optimization)
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",  # Use training iterations for adaptation
        metric="rollout/ep_rew_mean",  # Optimize for highest reward
        mode="max",  # Maximize reward
        perturbation_interval=5,  # Every 5 training steps, adapt hyperparameters
        resample_probability=0.25,  # Probability of resampling a new hyperparameter
        hyperparam_mutations=hyperparam_mutations,  # Defined mutations
    )

    if args.run_mode == "local":  # Standard config, to file
        run_config = air.RunConfig(
            storage_path="/tmp/ray",
            name=f"IsaacRay-{args.cfg_class}-tune",
            verbose=1,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=0,  # Disable periodic checkpointing
                checkpoint_at_end=False,  # Disable final checkpoint
            ),
        )

    elif args.run_mode == "remote":  # MLFlow, to MLFlow server
        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=args.mlflow_uri,
            experiment_name=f"IsaacRay-{args.cfg_class}-tune",
            save_artifact=False,
            tags={"run_mode": "remote", "cfg_class": args.cfg_class},
        )

        run_config = ray.train.RunConfig(
            name="mlflow",
            storage_path="/tmp/ray",
            callbacks=[mlflow_callback],
            checkpoint_config=ray.train.CheckpointConfig(checkpoint_frequency=0, checkpoint_at_end=False),
        )
    else:
        raise ValueError("Unrecognized run mode.")

    # Configure the tuning job
    tuner = tune.Tuner(
        IsaacLabTuneTrainable,
        param_space=cfg,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler,
            num_samples=args.num_samples,
            reuse_actors=True,
        ),
        run_config=run_config,
    )

    # Execute the tuning
    tuner.fit()

    # Save results to mounted volume
    if args.run_mode == "local":
        print("[DONE!]: Check results with tensorboard dashboard")
    else:
        print("[DONE!]: Check results with MLFlow dashboard")


class JobCfg:
    """To be compatible with :meth: invoke_tuning_run and :class:IsaacLabTuneTrainable,
    at a minimum, the tune job should inherit from this class."""

    def __init__(self, cfg: dict):
        """
        Runner args include command line arguments passed to the task.
        For example:
        cfg["runner_args"]["headless_singleton"] = "--headless"
        cfg["runner_args"]["enable_cameras_singleton"] = "--enable_cameras"
        """
        assert "runner_args" in cfg, "No runner arguments specified."
        """
        Task is the desired task to train on. For example:
        cfg["runner_args"]["--task"] = tune.choice(["Isaac-Cartpole-RGB-TheiaTiny-v0"])
        """
        assert "--task" in cfg["runner_args"], "No task specified."
        """
        Hydra args define the hyperparameters varied within the sweep. For example:
        cfg["hydra_args"]["agent.params.network.cnn.activation"] = tune.choice(["relu", "elu"])
        """
        assert "hydra_args" in cfg, "No hyperparameters specified."
        self.cfg = cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune Isaac Lab hyperparameters.")
    parser.add_argument("--ray_address", type=str, default="auto", help="the Ray address.")
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="Tiago_cfg_file.py",
        required=False,
        help="The relative filepath where a hyperparameter sweep is defined",
    )
    parser.add_argument(
        "--cfg_class",
        type=str,
        default="CartpoleRGBNoTuneJobCfg",
        required=False,
        help="Name of the hyperparameter sweep class to use",
    )
    parser.add_argument(
        "--run_mode",
        choices=["local", "remote"],
        default="local",
        help=(
            "Set to local to use ./isaaclab.sh -p python, set to "
            "remote to use /workspace/isaaclab/isaaclab.sh -p python"
        ),
    )
    parser.add_argument(
        "--workflow",
        default=None,  # populated with RL Games
        help="The absolute path of the workflow to use for the experiment. By default, RL Games is used.",
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        required=False,
        help="The MLFlow Uri.",
    )
    parser.add_argument(
        "--num_workers_per_node",
        type=int,
        default=2,
        help="Number of workers to run on each GPU node. Only supply for parallelism on multi-gpu nodes",
    )

    parser.add_argument("--metric", type=str, default="rewards/time", help="What metric to tune for.")

    parser.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="What to optimize the metric to while tuning",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="How many hyperparameter runs to try total.",
    )
    parser.add_argument(
        "--repeat_run_count",
        type=int,
        default=1,
        help="How many times to repeat each hyperparameter config.",
    )

    args = parser.parse_args()
    NUM_WORKERS_PER_NODE = args.num_workers_per_node
    print(f"[INFO]: Using {NUM_WORKERS_PER_NODE} workers per node.")
    if args.run_mode == "remote":
        BASE_DIR = DOCKER_PREFIX  # ensure logs are dumped to persistent location
        PYTHON_EXEC = DOCKER_PREFIX + PYTHON_EXEC[2:]
        if args.workflow is None:
            WORKFLOW = DOCKER_PREFIX + WORKFLOW
        else:
            WORKFLOW = args.workflow
        print(f"[INFO]: Using remote mode {PYTHON_EXEC=} {WORKFLOW=}")

        if args.mlflow_uri is not None:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            from ray.air.integrations.mlflow import MLflowLoggerCallback
        else:
            raise ValueError("Please provide a result MLFLow URI server.")
    else:  # local
        PYTHON_EXEC = os.getcwd() + "/" + PYTHON_EXEC[2:]
        if args.workflow is None:
            WORKFLOW = os.getcwd() + "/" + WORKFLOW
        else:
            WORKFLOW = args.workflow
        BASE_DIR = os.getcwd()
        print(f"[INFO]: Using local mode {PYTHON_EXEC=} {WORKFLOW=}")
    file_path = args.cfg_file
    class_name = args.cfg_class
    print(f"[INFO]: Attempting to use sweep config from {file_path=} {class_name=}")
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    print(f"[INFO]: Successfully imported {module_name} from {file_path}")
    if hasattr(module, class_name):
        ClassToInstantiate = getattr(module, class_name)
        print(f"[INFO]: Found correct class {ClassToInstantiate}")
        instance = ClassToInstantiate()
        print(f"[INFO]: Successfully instantiated class '{class_name}' from {file_path}")
        cfg = instance.cfg
        print(f"[INFO]: Grabbed the following hyperparameter sweep config: \n {cfg}")
        invoke_tuning_run(cfg, args)

    else:
        raise AttributeError(f"[ERROR]:Class '{class_name}' not found in {file_path}")
