from mdshark import config
import os
import sys
import subprocess
import submitit
from submitit.core.utils import FailedJobError
import logging

assert sys.version_info >= (3, 6)


try:
    import coloredlogs
    import verboselogs

    logger = verboselogs.VerboseLogger('mdshark')
    # level = 'SPAM'
    level = 'DEBUG'
    # logger.setLevel(level)
    coloredlogs.install(fmt='%(asctime)s %(message)s',
                        level=level, logger=logger)

except ModuleNotFoundError:

    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger('mdshark')
    logger.warning(
        "Modules coloredlogs and/or verboselogs not found, using original logging module.")
    logger.setLevel(logging.DEBUG)


def do_nothing():
    pass


def run_submit_multi(cmd, cwd, env_setup, wait_complete=True):

    command = (config.environment_setup[env_setup]
               + ["set -e", f"cd {cwd}"]
               + cmd
               + ["set +e"])

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        timeout_min=4*60,
        mem_gb=80,
        cpus_per_task=36,
        slurm_setup=command,
        slurm_additional_parameters=config.slurm_additional_parameters)

    function = do_nothing

    job = executor.submit(function)

    if not wait_complete:
        return job

    output = job.result()
    # print(job.stderr())
    # print(job.stdout())
    return job


def run_submit(cmd, cwd, env_setup, wait_complete=True):

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        timeout_min=4*60,
        mem_gb=80,
        cpus_per_task=36,
        slurm_setup=config.environment_setup[env_setup],
        slurm_additional_parameters=config.slurm_additional_parameters)

    if isinstance(cmd, list):
        pass
    else:
        cmd = cmd.split()

    function = submitit.helpers.CommandFunction(cmd, verbose=True, cwd=cwd)

    job = executor.submit(function)

    if not wait_complete:
        return job

    output = job.result()
    # print(job.stderr())
    # print(job.stdout())
    return job


def run(cmd):
    try:
        my_env = os.environ.copy()
        # print(my_env)
        subprocess.run(cmd, shell=True, check=True, env=my_env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    except subprocess.CalledProcessError as e:

        print(f"The system command")
        print(f"    {cmd}")
        print(f"resulted in an error.")
        print("")
        print(f" -- Working directory (at the moment):")
        print(f"{os.getcwd()}")
        print("")
        print(f" -- captured stdout:")
        if e.stdout is not None:
            print(e.stdout.decode('utf-8'))
        print("")
        print(f" -- captured stderr:")
        if e.stderr is not None:
            print(e.stderr.decode('utf-8'))

        raise(e)


def run_popen(command, cwd=None, env=None, verbose=False):
    """
    run compatible with submitit
    """
    full_command = command

    if env is None:
        env = os.environ.copy()
    if cwd is None:
        cwd = os.getcwd()
    
    logger.spam(f"Executing command:")
    logger.spam(f"    {' '.join(full_command)}")
    logger.spam(f" -- location:")
    logger.spam(f"    {os.getcwd()}")

    outlines: List[str] = []
    with subprocess.Popen(
        full_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        cwd=cwd,
        env=env,
    ) as process:
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, b""):
                if not line:
                    break
                outlines.append(line.decode().strip())
                if verbose:
                    print(outlines[-1], flush=True)
        except Exception as e:
            process.kill()
            process.wait()
            raise FailedJobError(
                "Job got killed for an unknown reason.") from e
        stderr = process.communicate()[1]  # we already got stdout
        stdout = "\n".join(outlines)
        retcode = process.poll()
        if stderr and (retcode or verbose):
            print(stderr.decode(), file=sys.stderr)
            print(stdout)
        if retcode:
            subprocess_error = subprocess.CalledProcessError(
                retcode, process.args, output=stdout, stderr=stderr
            )
            raise FailedJobError(stderr.decode()) from subprocess_error

    if logger.isEnabledFor(verboselogs.SPAM):
            logger.spam(f" -- captured stdout:")
            print(stdout)

    return stdout


def get_default_executor(env_setup):
    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        timeout_min=4*60,
        mem_gb=85,
        cpus_per_task=36,
        slurm_setup=config.environment_setup[env_setup],
        slurm_additional_parameters=config.slurm_additional_parameters)

    return executor
