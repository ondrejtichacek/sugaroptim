# sugaroptim



### Installation

```
conda create -n sugaroptim
conda activate sugaroptim
conda install -c conda-forge python matplotlib jupyter mdtraj verboselogs coloredlogs tqdm submitit joblib pylint

pip install -e .
```


### Example usage

```
source /opt/uochb/soft/spack/20210305/share/spack/setup-env.sh
spack load gromacs
python run.py

```