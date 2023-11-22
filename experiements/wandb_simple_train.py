import argparse
import yaml
import os
from decouple import Config, RepositoryEnv

from gpuscheduler import HyakScheduler

parser = argparse.ArgumentParser(description='Compute script.')
parser.add_argument('--dry', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--p', type=float, default=1.0, help='Probability with which to select a configuration.')
args = parser.parse_args()

config = Config(RepositoryEnv(".env"))

# sbatch details
gpus = 1
cmd = "wandb agent --count 1 "
name = f"mae_tiny_imagenet_initial_exp"
cores_per_job = 5
mem = 64
time_hours = 8
time_minutes = 0
constraint = ""
exclude = ""

repo = config("GIT_HOME")
change_dir = config("GIT_HOME")
scheduler = HyakScheduler(verbose=args.verbose, use_wandb=True, exp_name=name)
ckpt_base_dir = config("LOG_HOME")
logfolder = os.path.join(ckpt_base_dir, name)
sweep_config_path = config("SWEEP_CONFIG_BASE_PATH")
num_runs = 10

# default commands and args
base_flags = [
    "${env}",
    "python",
    "main_pretrain.py",
    "--use_wandb",
    f"--project_name={name}",
    f"--output_dir={logfolder}",
    f"--log_dir={logfolder}",
    "${args}"  # use args from configuration as command arguments
]

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "train_loss"},
    "parameters":
    {
        "batch_size": {"values": [128]},
        "epochs": {"values": [100]},
        "model": {"values": ["mae_vit_base_patch16"]},
        "input_size": {"values": [64]},
        "mask_ratio": {"values": [0.3, 0.5, 0.75]},
        "weight_decay": {"values": [0.05]},
        "blr": {"max": 1e-3, "min": 1e-4},
        "warmup_epochs": {"values": [40]},
        "dataset": {"values": ["tiny_imagenet"]},
        "data_group": {"values": [1]},
        "num_workers": {"values": [5]},
        "data_subset": {"values": [1.0]}
    },
    "command": base_flags
}

# Create log folder
if not os.path.exists(logfolder):
    print(f"Creating {logfolder}")
    os.makedirs(logfolder)

# remove previous sweep output if one exists
sweep_out_file = f'{logfolder}/sweepid.txt'
if os.path.exists(sweep_out_file):
    os.remove(sweep_out_file)

# dump sweep config for main to read
with open(f"{sweep_config_path}/{name}.yaml", "w") as config_file:
    yaml.dump(sweep_configuration, config_file)

# add job to scheduler
for i in range(num_runs):
    scheduler.add_job(logfolder, change_dir, [cmd], time_hours, cores=cores_per_job, mem=mem, constraint=constraint, exclude=exclude, time_minutes=time_minutes, gpus=gpus)

if args.dry:
    pass
else:
    scheduler.run_jobs(begin=None, single_process=True)
