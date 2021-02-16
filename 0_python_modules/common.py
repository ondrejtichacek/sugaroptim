import subprocess

def run(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True, 
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    except subprocess.CalledProcessError as e:
        
        print(f"The system command\n{cmd}\nresulted in an error.")
        
        print(f"captured stdout:")
        print(e.stdout.decode('utf-8'))
        
        print(f"captured stderr:")
        print(e.stderr.decode('utf-8'))
        
        raise(e)