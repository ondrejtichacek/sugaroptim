import os
import subprocess

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
        print(e.stdout.decode('utf-8'))
        print("")
        print(f" -- captured stderr:")
        print(e.stderr.decode('utf-8'))
        
        raise(e)