# sugaroptim



### Installation

```
conda create -n sugaroptim
conda activate sugaroptim
conda install -c conda-forge python matplotlib jupyter mdtraj verboselogs coloredlogs tqdm submitit joblib pylint

pip install -e .
```

### Setup
Modify paths to gromacs, plumed, etc. in `mdshark/config.py` by editing the `dummy_host`.

### Example usage

```
source /opt/uochb/soft/spack/20211108-git/share/spack/setup-env.sh
spack load gromacs+plumed
spack load cuda
spack load fftw~mpi
python run.py

```