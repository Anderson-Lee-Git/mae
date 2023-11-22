import os
import subprocess
import shlex
import time
import copy
import hashlib
from os.path import join

from decouple import Config, RepositoryEnv

MEM_THRESHOLD_AVAILABLE = 300
UTILIZARTION_THERESHOLD_AVAILABLE = 5

def execute_and_return(strCMD):
    proc = subprocess.Popen(shlex.split(strCMD), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out, err = out.decode("UTF-8").strip(), err.decode("UTF-8").strip()
    return out, err

# https://github.com/TimDettmers/sched/blob/master/gpuscheduler/core.py#L156
class HyakScheduler(object):
    def __init__(self, verbose=False, use_gres=False, use_wandb=False, exp_name=None):
        print(os.curdir)
        self.jobs = []
        self.verbose = verbose
        self.remap = {}
        self.use_gres = use_gres
        self.use_wandb = use_wandb
        self.exp_name = exp_name
        self.config = Config(RepositoryEnv(".env"))

    def update_host_config(self, name, mem_threshold, util_threshold):
        pass

    def add_job(self, path, work_dir, cmds, time_hours, gpus=1, mem=32, cores=6, constraint='', exclude='', time_minutes=0):
        self.jobs.append([path, work_dir, cmds, time_hours, gpus, mem, cores, constraint, exclude, time_minutes])
        if self.verbose:
            print('#SBATCH --time={0:02d}:{1:02d}:00'.format(time_hours, time_minutes))

    def run_jobs(self, as_array=True, sleep_delay_seconds=0, single_process=False, log_id=None, skip_cmds=0, comment=None, begin=None, gpus_per_node=8, requeue=False, requeue_length_hours=4):
        array_preamble = []

        strval = self.exp_name
        if not isinstance(strval, str): strval = strval[0]
        array_id = hashlib.md5(strval.encode('utf-8')).hexdigest() if log_id is None else log_id

        array_file = join(self.config('SCRIPT_HISTORY'), 'array_init_{0}.sh'.format(array_id))
        array_job_list = join(self.config('SCRIPT_HISTORY'), 'array_jobs_{0}.sh'.format(array_id))
        script_list = []
        print('processing cmds...')
        file_contents = ''
        for i, (path, work_dir, cmds, time_hours, gpus, mem, cores, constraint, exclude, time_minutes) in enumerate(self.jobs):
            if not as_array:
                if i % 10 == 0 and i > 0: print('Processing cmd no ', i)
            nodes = gpus // gpus_per_node
            nodes += 1 if (gpus % gpus_per_node) > 0 else 0
            if nodes == 0: nodes = 1
            gpus = gpus_per_node if gpus > gpus_per_node else gpus
            if not isinstance(cmds, list): cmds = [cmds]
            lines = []
            script_file = join(self.config('SCRIPT_HISTORY'), 'init_{0}_{1}.sh'.format(array_id, i))

            script_list.append(script_file)
            log_path = join(join(self.config('LOG_HOME'), path))
            lines.append('#!/bin/bash')
            lines.append('#')
            lines.append('#SBATCH --job-name={0}'.format(script_file))
            if self.config("ACCOUNT") != '':
                lines.append('#SBATCH --account={0}'.format(self.config("ACCOUNT")))
            lines.append('#SBATCH --partition={0}'.format(self.config("PARTITION")))
            lines.append('#')
            lines.append('#SBATCH --nodes={0}'.format(nodes))
            if single_process:
                lines.append('#SBATCH --ntasks-per-node=1')
                lines.append('#SBATCH --cpus-per-task={0}'.format(cores*(gpus if gpus != 0 else 1)))
            else:
                lines.append('#SBATCH --ntasks-per-node={0}'.format(gpus if gpus != 0 else 1))
                lines.append('#SBATCH --cpus-per-task={0}'.format(cores))
            lines.append('#SBATCH --time={0:02d}:{1:02}:00'.format(time_hours, time_minutes))
            if gpus > 0:
                if self.use_gres:
                    lines.append('#SBATCH --gres=gpu:{0}'.format(gpus))
                else:
                    lines.append('#SBATCH --gpus-per-node={0}'.format(gpus))
            lines.append('#SBATCH --mem={0}G'.format(mem))
            lines.append('#SBATCH --requeue')
            if len(constraint) > 0:
                lines.append('#SBATCH --constraint={0}'.format(constraint))
            if exclude != '':
                lines.append('#SBATCH --exclude={0}'.format(exclude))
            if comment is not None:
                lines.append('#SBATCH --comment={0}'.format(comment))
            if begin is not None:
                lines.append('#SBATCH --begin={0}'.format(begin))

            lines.append('#')
            lines.append('#SBATCH --open-mode=truncate')
            lines.append('#SBATCH --chdir={0}'.format(join(self.config("GIT_HOME"), work_dir)))
            lines.append('#SBATCH --output={0}'.format(join(log_path, array_id + '_{0}.log'.format(i))))
            lines.append('#SBATCH --error={0}'.format(join(log_path, array_id + '_{0}.err'.format(i))))
            lines.append('')
            lines.append('export PATH=$PATH:{0}'.format(join(self.config('ANACONDA_HOME'), 'bin')))
            for cmd_no, cmd in enumerate(cmds[skip_cmds:]):
                lines.append(cmd)

            if len(array_preamble) == 0:
                array_preamble = copy.deepcopy(lines[:-(1*len(cmds[skip_cmds:]) + 1)])
                array_preamble[2] = '#SBATCH --job-name={0}'.format(array_job_list)
                array_preamble[-3] = '#SBATCH --output={0}'.format(join(log_path, array_id + '_%a.log'))
                array_preamble[-2] = '#SBATCH --error={0}'.format(join(log_path, array_id + '_%a.err'))
                array_preamble.append('#SBATCH --array=0-{0}'.format(len(self.jobs)-1))
                array_preamble.append('')
                array_preamble.append('export PATH=$PATH:{0}'.format(join(self.config('ANACONDA_HOME'), 'bin')))

            if not os.path.exists(log_path):
                print('Creating {0}'.format(log_path))
                os.makedirs(log_path)

            if not os.path.exists(self.config('SCRIPT_HISTORY')):
                print('Creating {0}'.format(self.config('SCRIPT_HISTORY')))
                os.makedirs(self.config('SCRIPT_HISTORY'))


            if not as_array:
                print('Writing job file to: {0}'.format(script_file))
                with open(script_file, 'w') as f:
                    for line in lines:
                        f.write('{0}\n'.format(line))
                if not requeue:
                    time.sleep(0.05)
                    out, err = execute_and_return('sbatch {0}'.format(script_file))
                    if err != '':
                        print(err)
                else:
                    num_requeues = int((time_hours+(time_minutes/60)+requeue_length_hours-0.01)//requeue_length_hours)
                    bid, err = execute_and_return('sbatch --parsable {0}'.format(script_file))
                    for j in range(num_requeues-1):
                        print(num_requeues, bid)
                        if err != '':
                            print(err)
                            break
                        time.sleep(0.05)
                        bid, err = execute_and_return(f'sbatch --parsable --dependency=afterany:{bid} {script_file}')

        if as_array:
            print('creating array...')
            array_lines = []
            array_lines.append('')
            array_lines.append('')
            array_lines.append('echo $SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID'.format(cmd_no))
            for i, (path, work_dir, cmds, time_hours, gpus, mem, cores, constraint, exclude, time_minutes) in enumerate(self.jobs):
                bare_script_file = join(self.config('SCRIPT_HISTORY'), 'init_bare_{0}_{1}.sh'.format(array_id, i))
                bare_lines = []
                bare_lines.append('#!/bin/bash')
                bare_lines.append('#')
                bare_lines.append('export PATH=$PATH:{0}'.format(join(self.config('ANACONDA_HOME'), 'bin')))
                for cmd_no, cmd in enumerate(cmds[skip_cmds:]):
                    if self.use_wandb:
                        bare_lines.append('echo "starting wandb agent.."')
                        if i == 0:
                            # run sweep on gpu 0
                            bare_lines.append("nvidia-smi")
                            bare_lines.append("nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9 2> /dev/null")
                            bare_lines.append('echo SLURM_PROCID=$SLURM_PROCID')
                            bare_lines.append(f'if [[ $SLURM_PROCID -eq {0} ]] && [ ! -f "{log_path}/sweepid.txt" ]')
                            bare_lines.append('then')
                            bare_lines.append(f'\t wandb sweep --project {self.exp_name} {self.config("SWEEP_CONFIG_BASE_PATH")}/{self.exp_name}.yaml 2>&1 | tee "{log_path}/sweepid.txt"')
                            bare_lines.append('fi')
                        else:
                            bare_lines.append('\t echo "Checking to see that sweep file exists and contains the sweepid"')
                            bare_lines.append(f'until [ -f "{log_path}/sweepid.txt" ] && [ ! -z `cat "{log_path}/sweepid.txt" | grep "wandb agent" | cut -d" " -f 8` ];')
                            bare_lines.append('do')
                            bare_lines.append('\t echo "Sweep ID not set, waiting for wandb sweep to start..."')
                            bare_lines.append('\t sleep 5')
                            bare_lines.append('done')
                        bare_lines.append(f'SWEEPID=`cat "{log_path}/sweepid.txt" | grep "wandb agent" | cut -d" " -f 8`')
                        bare_lines.append('echo "sweepid=$SWEEPID"')
                        bare_lines.append('echo SLURM_PROCID=$SLURM_PROCID')
                        bare_lines.append('export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256')
                        bare_lines.append('if [[ $SLURM_PROCID -eq {0} ]]'.format(0))
                        bare_lines.append('then')
                        bare_lines.append('export MASTER_PORT=`shuf -i 2000-65000 -n 1`')
                        bare_lines.append('\t echo MASTER_PORT=$MASTER_PORT')
                        bare_lines.append('\t ' + cmd + '$SWEEPID')
                        bare_lines.append('fi')
                    else:
                        bare_lines.append(cmd)
                with open(bare_script_file, 'w') as f:
                    for line in bare_lines:
                        f.write('{0}\n'.format(line))

                if i == 0:
                    array_lines.append('if [[ $SLURM_ARRAY_TASK_ID -eq {0} ]]'.format(i))
                    array_lines.append('then')
                else:
                    array_lines.append('elif [[ $SLURM_ARRAY_TASK_ID -eq {0} ]]'.format(i))
                    array_lines.append('then')
                if sleep_delay_seconds > 0:
                    array_lines.append('\t sleep {0}'.format(i*sleep_delay_seconds))

                array_lines.append('\t srun bash ' + bare_script_file)

            array_lines.append('else')
            array_lines.append('\t echo $SLURM_ARRAY_TASK_ID')
            array_lines.append('fi')


            array_lines = array_preamble + array_lines
            print('Writing array file to: {0}'.format(array_file))
            with open(array_file, 'w') as f:
                for line in array_lines:
                    f.write('{0}\n'.format(line))

            print('Writing job list to: {0}'.format(array_job_list))
            with open(array_job_list, 'w') as f:
                for line in script_list:
                    f.write('{0}\n'.format(line))

            if not requeue:
                out, err = execute_and_return('sbatch {0}'.format(array_file))
                if err != '':
                    print(err)
            else:
                raise NotImplementedError('Requeue does not work for array jobs!')