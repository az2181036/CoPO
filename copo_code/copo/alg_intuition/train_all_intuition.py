from copo.callbacks import MultiAgentDrivingCallbacks
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import get_rllib_compatible_env
from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

from ray import tune


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # Setup config
    # We set the stop criterion to 2M environmental steps! Since PPO in single OurEnvironment converges at around 20k steps.
    stop = int(100_0000)

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_rllib_compatible_env(MultiAgentIntersectionEnv),
        env_config=dict(
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]),
            start_seed=5000,
        ),

        # ===== Resource =====
        # So we need 0.2(num_cpus_per_worker) * 5(num_workers) + 1(num_cpus_for_driver) = 2 CPUs per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,
    )

    # Launch training
    train(
        "QMIX",
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,  # Don't call get_ippo_config here!
        num_gpus=args.num_gpus,
        num_seeds=1,
        # test_mode=True,
        # custom_callback=MultiAgentDrivingCallbacks(api_key_file="~/.netrc", project="metadrive-intuition"),
        custom_callback=MultiAgentDrivingCallbacks,
    )
