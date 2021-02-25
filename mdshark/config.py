import pathlib
from pathlib import Path

path = {
    'gmx': Path('/usr/local/gromacs/gromacs-2020.5/bin/gmx'), # location of gmx
    'mdrun_plumed': Path('/usr/bin/mdrun_plumed'), # location of mdrun_plumed
}

cluster_bash_header = {}

cluster_bash_header['gaussian_o1'] = (
"""
#SBATCH -J g1
#SBATCH --qos=normal
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