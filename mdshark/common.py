import os
import sys
import subprocess
import logging

assert sys.version_info >= (3, 6)

try:
    import coloredlogs, verboselogs

    logger = verboselogs.VerboseLogger('mdshark')
    # level = 'SPAM'
    level = 'DEBUG'
    # logger.setLevel(level)
    coloredlogs.install(fmt='%(asctime)s %(message)s', level=level, logger=logger)

except ModuleNotFoundError:
    
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger = logging.getLogger('mdshark')
    logger.warning("Modules coloredlogs and/or verboselogs not found, using original logging module.")
    logger.setLevel(logging.DEBUG)


def run(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True, 
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