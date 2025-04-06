import pathlib
import sys
from ray import tune
import util
import tuner
import json
import shlex

# Ensure proper path resolution
CUR_DIR = pathlib.Path(__file__).parent
UTIL_DIR = CUR_DIR.parent
sys.path.extend([str(UTIL_DIR), str(CUR_DIR)])


class SB3SAC_JobCfg(tuner.JobCfg):
    """
    Configuration for tuning SAC hyperparameters using StableBaselines3 in a flat YAML format.
    The __init__ constructor takes three Boolean arguments:
      - vary_env_count: whether to tune the number of environments.
      - vary_activation_fn: whether to tune the activation function.
      - vary_net_arch: whether to tune the MLP network architecture.
    """
    def __init__(self, cfg: dict = {}, vary_env_count: bool = False, vary_activation_fn: bool = False, vary_net_arch: bool = False):
        # Ensure both runner_args and hydra_args exist (even if empty)
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        
        # Basic runner arguments
        cfg["runner_args"]["headless_singleton"] = "--headless"
        # cfg["runner_args"]["enable_cameras_singleton"] = "--enable_cameras"

        # Specify the task and default number of environments
        cfg["runner_args"]["--num_envs"] = 512  # default

        # Flat hyperparameter definitions for SAC:
        cfg["hydra_args"]["agent.learning_rate"] = tune.loguniform(1e-4, 1e-2)  # Log scale
        cfg["hydra_args"]["agent.batch_size"] = tune.choice([64, 128, 256, 512])  # Categorical
        cfg["hydra_args"]["agent.gamma"] = tune.uniform(0.9, 0.999)  # Continuous range
        cfg["hydra_args"]["agent.ent_coef"] = tune.loguniform(1e-3, 2e-1)  # Log scale
        cfg["hydra_args"]["agent.tau"] = tune.loguniform(1e-3, 0.1)  # Target smoothing coefficient
        cfg["hydra_args"]["agent.buffer_size"] = tune.choice([50000, 100000, 500000])
        
        # 🔹 Vary environment count if enabled
        if vary_env_count:
            env_counts = [2**x for x in range(7, 10)]  # e.g., 64 to 512
            cfg["runner_args"]["--num_envs"] = tune.choice(env_counts)

        # 🔹 Handle activation function selection

        if vary_activation_fn:
            
            def get_net_arch():
                """Generates a random network architecture with 2-3 layers."""
                num_layers = tune.randint(2, 4).sample()  # 2 to 3 hidden layers
                return [tune.randint(16, 256).sample() for _ in range(num_layers)]
            
            # Select activation function and net_arch dynamically
            activation_fn = tune.choice(["relu", "elu", "tanh"]).sample()
            net_arch = tune.sample_from(lambda _: get_net_arch()).sample()

            # Convert net_arch list to string with no spaces for Hydra compatibility
            net_arch_str = str(net_arch).replace(" ", "")

            # Construct the properly formatted string for Hydra
            hydra_arg = f'"{{activation_fn:{activation_fn},net_arch:{net_arch_str}}}"'

            # Assign to Hydra args
            cfg["hydra_args"]["agent.policy_kwargs"] = shlex.quote(hydra_arg)
        
        super().__init__(cfg)


class SB3PPO_JobCfg(tuner.JobCfg):
    """
    Configuration for tuning SAC hyperparameters using StableBaselines3 in a flat YAML format.
    The __init__ constructor takes three Boolean arguments:
      - vary_env_count: whether to tune the number of environments.
      - vary_activation_fn: whether to tune the activation function.
      - vary_net_arch: whether to tune the MLP network architecture.
    """
    def __init__(self, cfg: dict = {}, vary_env_count: bool = False, vary_activation_fn: bool = False, vary_net_arch: bool = False):
        # Ensure both runner_args and hydra_args exist (even if empty)
        cfg = util.populate_isaac_ray_cfg_args(cfg)
        
        # Basic runner arguments
        cfg["runner_args"]["headless_singleton"] = "--headless"
        # cfg["runner_args"]["enable_cameras_singleton"] = "--enable_cameras"

        # Specify the task and default number of environments
        cfg["runner_args"]["--num_envs"] = 512  # default

        # Flat hyperparameter definitions for SAC:
        cfg["hydra_args"]["agent.learning_rate"] = tune.loguniform(1e-5, 1e-3)  # Log scale
        cfg["hydra_args"]["agent.batch_size"] = tune.choice([32, 64, 128, 256, 512])  # Categorical
        cfg["hydra_args"]["agent.gamma"] = tune.uniform(0.9, 0.9999)  # Continuous range
        cfg["hydra_args"]["agent.ent_coef"] = tune.loguniform(1e-4, 1e-1)  # Log scale
        #cfg["hydra_args"]["agent.n_steps"] = tune.choice([128, 256, 512, 1024, 2048])  # Categorical
        cfg["hydra_args"]["agent.n_steps"] = 64  # default
        cfg["hydra_args"]["agent.n_epochs"] = tune.randint(3, 20)  # Integer range
        
        # 🔹 Vary environment count if enabled
        if vary_env_count:
            env_counts = [2**x for x in range(7, 10)]  # e.g., 64 to 512
            cfg["runner_args"]["--num_envs"] = tune.choice(env_counts)

        # 🔹 Handle activation function selection

        if vary_activation_fn:
            
            def get_net_arch():
                """Generates a random network architecture with 2-3 layers."""
                num_layers = tune.randint(2, 4).sample()  # 2 to 3 hidden layers
                return [tune.randint(16, 256).sample() for _ in range(num_layers)]
            
            # Select activation function and net_arch dynamically
            activation_fn = tune.choice(["relu", "elu", "tanh"]).sample()
            net_arch = tune.sample_from(lambda _: get_net_arch()).sample()

            # Convert net_arch list to string with no spaces for Hydra compatibility
            net_arch_str = str(net_arch).replace(" ", "")

            # Construct the properly formatted string for Hydra
            hydra_arg = f'"{{activation_fn:{activation_fn},net_arch:{net_arch_str}}}"'

            # Assign to Hydra args
            cfg["hydra_args"]["agent.policy_kwargs"] = shlex.quote(hydra_arg)
        
        super().__init__(cfg)
        
class RLGAMESPPO_JobCfg(tuner.JobCfg):
    """In order to be compatible with :meth: invoke_tuning_run, and
    :class:IsaacLabTuneTrainable , configurations should
    be in a similar format to this class. This class can vary env count/horizon length,
    CNN structure, and MLP structure. Broad possible ranges are set, the specific values
    that work can be found via tuning. Tuning results can inform better ranges for a second tuning run.
    These ranges were selected for demonstration purposes. Best ranges are run/task specific."""

    @staticmethod
    def _get_batch_size_divisors(batch_size: int, min_size: int = 128) -> list[int]:
        """Get valid batch divisors to combine with num_envs and horizon length"""
        divisors = [i for i in range(min_size, batch_size + 1) if batch_size % i == 0]
        return divisors if divisors else [min_size]

    def __init__(self, cfg={}, vary_env_count: bool = False, vary_cnn: bool = False, vary_mlp: bool = False):
        cfg = util.populate_isaac_ray_cfg_args(cfg)

        # Basic configuration
        cfg["runner_args"]["headless_singleton"] = "--headless"
        #cfg["runner_args"]["enable_cameras_singleton"] = "--enable_cameras"
        cfg["hydra_args"]["agent.params.config.max_epochs"] = 200

        if vary_env_count:  # Vary the env count, and horizon length, and select a compatible mini-batch size
            # Check from 512 to 8196 envs in powers of 2
            # check horizon lengths of 8 to 256
            # More envs should be better, but different batch sizes can improve gradient estimation
            env_counts = [2**x for x in range(7, 10)]
            horizon_lengths = [2**x for x in range(2, 7)]

            selected_env_count = tune.choice(env_counts)
            selected_horizon = tune.choice(horizon_lengths)

            cfg["runner_args"]["--num_envs"] = selected_env_count
            cfg["hydra_args"]["agent.params.config.horizon_length"] = selected_horizon

            def get_valid_batch_size(config):
                num_envs = config["runner_args"]["--num_envs"]
                horizon_length = config["hydra_args"]["agent.params.config.horizon_length"]
                total_batch = horizon_length * num_envs
                divisors = self._get_batch_size_divisors(total_batch)
                return divisors[0]

            cfg["hydra_args"]["agent.params.config.minibatch_size"] = tune.sample_from(get_valid_batch_size)

        if vary_cnn:  # Vary the depth, and size of the layers in the CNN part of the agent
            # Also varies kernel size, and stride.
            num_layers = tune.randint(2, 3)
            cfg["hydra_args"]["agent.params.network.cnn.type"] = "conv2d"
            cfg["hydra_args"]["agent.params.network.cnn.activation"] = tune.choice(["relu", "elu"])
            cfg["hydra_args"]["agent.params.network.cnn.initializer"] = "{name:default}"
            cfg["hydra_args"]["agent.params.network.cnn.regularizer"] = "{name:None}"

            def get_cnn_layers(_):
                layers = []
                size = 64  # Initial input size

                for _ in range(num_layers.sample()):
                    # Get valid kernel sizes for current size
                    valid_kernels = [k for k in [3, 4, 6, 8, 10, 12] if k <= size]
                    if not valid_kernels:
                        break

                    kernel = int(tune.choice([str(k) for k in valid_kernels]).sample())
                    stride = int(tune.choice(["1", "2", "3", "4"]).sample())
                    padding = int(tune.choice(["0", "1"]).sample())

                    # Calculate next size
                    next_size = ((size + 2 * padding - kernel) // stride) + 1
                    if next_size <= 0:
                        break

                    layers.append({
                        "filters": tune.randint(16, 32).sample(),
                        "kernel_size": str(kernel),
                        "strides": str(stride),
                        "padding": str(padding),
                    })
                    size = next_size

                return layers

            cfg["hydra_args"]["agent.params.network.cnn.convs"] = tune.sample_from(get_cnn_layers)

        if vary_mlp:  # Vary the MLP structure; neurons (units) per layer, number of layers,

            max_num_layers = 6
            max_neurons_per_layer = 128
            if "env.observations.policy.image.params.model_name" in cfg["hydra_args"]:
                # By decreasing MLP size when using pretrained helps prevent out of memory on L4
                max_num_layers = 3
                max_neurons_per_layer = 32
            if "agent.params.network.cnn.convs" in cfg["hydra_args"]:
                # decrease MLP size to prevent running out of memory on L4
                max_num_layers = 2
                max_neurons_per_layer = 32

            num_layers = tune.randint(1, max_num_layers)

            def get_mlp_layers(_):
                return [tune.randint(4, max_neurons_per_layer).sample() for _ in range(num_layers.sample())]

            cfg["hydra_args"]["agent.params.network.mlp.units"] = tune.sample_from(get_mlp_layers)
            cfg["hydra_args"]["agent.params.network.mlp.initializer.name"] = tune.choice(["default"]).sample()
            cfg["hydra_args"]["agent.params.network.mlp.activation"] = tune.choice(
                ["relu", "tanh", "sigmoid", "elu"]
            ).sample()

        super().__init__(cfg)
