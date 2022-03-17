import socket
import re
import pathlib
from pathlib import Path

default_path = {
    # gromacs
    'gmx': 'gmx',

    # gromacs-plumed
    'mdrun_plumed': 'mdrun_plumed',
    'plumed': 'plumed',
}

host = socket.gethostname()

if host == 'dummy_host':
    path = {
        # gromacs
        'gmx': Path('/path/to/gmx'),

        # gromacs-plumed
        'mdrun_plumed': Path('/path/to/mdrun_plumed'),
        'plumed': Path('/path/to/plumed'),
    }
elif host == 'lucy':
    path = {
        # gromacs
        'gmx': Path('/usr/local/gromacs/gromacs-2020.5/bin/gmx'),

        # gromacs-plumed
        'mdrun_plumed': Path('/usr/bin/mdrun_plumed'),
        'plumed': Path('/usr/bin/plumed'),
    }
elif (host == 'login1.aurum.ccf.uochb.local'
        or re.match(r'ai\d+\.aurum\.ccf\.uochb\.local', host)):

    path = default_path

    path['gmx'] = 'gmx'
    path['gmx make_ndx'] = 'gmx make_ndx'
    path['mdrun_plumed'] = 'gmx mdrun'

    environment_setup = {
        'gromacs': [
            "source /opt/uochb/soft/spack/20211108-git/share/spack/setup-env.sh",
            "spack load gromacs+plumed",
            "spack load cuda",
            "spack load fftw~mpi",
            # "source /opt/uochb/soft/spack/20191115/share/spack/setup-env.sh",
            # "spack env activate gromacs2019",
            # "spack env activate gromacs2020-nompi",
            # "source /opt/uochb/soft/spack/latest/share/spack/setup-env.sh",
            # "spack env activate gromacs2020",
        ],

        'plumed': [
            "source /opt/uochb/soft/spack/20211108-git/share/spack/setup-env.sh",
            "spack load gromacs+plumed",
            "spack load cuda",
            "spack load fftw~mpi",
            # "source /opt/uochb/soft/spack/20191115/share/spack/setup-env.sh",
            # "spack env activate gromacs2019-plumed",
            # "spack env activate gromacs2020-nompi",
            # "source /opt/uochb/soft/spack/latest/share/spack/setup-env.sh",
            # "spack env activate gromacs2020",
        ],

        'gaussian': None,
        # 'gaussian': [
        #     "export g16root=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml",
        #     "export LD_LIBRARY_PATH=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16",
        #     "export GAUSS_EXEDIR=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16",
        #     "export GAUSS_SCRDIR=/dev/shm",
        #     "source /opt/uochb/soft/anaconda/202007/bin/activate",
        #     "conda activate sugaroptim",
        # ]
    }
else:
    raise(ValueError(f"Host {host} not set up."))


slurm_additional_parameters = {
    'qos': 'backfill',
    'partition': 'cpu,scpu,bfill',
    # 'reservation': 'ondra',
}


cluster_bash_header = {}

cluster_bash_header['gaussian_o1'] = (
    """
#SBATCH -J g1
#SBATCH --qos=normal
#SBATCH --partition=cpu,scpu,bfill
#SBATCH --time=0-12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --exclusive
#SBATCH --mem=80gb

export g16root=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml
export LD_LIBRARY_PATH=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16
export GAUSS_EXEDIR=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16
export GAUSS_SCRDIR=/dev/shm

source /opt/uochb/soft/anaconda/202007/bin/activate
conda activate sugaroptim
"""
)

cluster_bash_header['gaussian_o2'] = (
    """
#SBATCH -J g1
#SBATCH --qos=backfill
#SBATCH --partition=cpu,scpu,bfill
#SBATCH --time=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --exclusive
#SBATCH --mem=80gb

export g16root=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml
export LD_LIBRARY_PATH=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16
export GAUSS_EXEDIR=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16
export GAUSS_SCRDIR=/dev/shm

source /opt/uochb/soft/anaconda/202007/bin/activate
conda activate sugaroptim
"""
)

cluster_bash_header['gaussian_o3'] = (
    """
#SBATCH -J g1
#SBATCH --qos=backfill
#SBATCH --partition=cpu,scpu,bfill
#SBATCH --time=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --exclusive
#SBATCH --mem=80gb

export g16root=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml
export LD_LIBRARY_PATH=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16
export GAUSS_EXEDIR=/opt/uochb/soft/gaussian/gaussian16_a03/arch/amd64-pgi_12.10-acml/g16
export GAUSS_SCRDIR=/dev/shm

source /opt/uochb/soft/anaconda/202007/bin/activate
conda activate sugaroptim
"""
)
