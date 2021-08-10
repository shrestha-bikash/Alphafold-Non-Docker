## Non Docker Alphafold
This is the modified version of Alphafold 2 that does not require docker.

In this pipeline, the program accepts an alignment file in a3m format. (Note: This program does not generate MSA files). It helps to evaluate the given MSA file using alphafold prediction results.


## Install conda using miniconda if not installed already
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

## Create conda environment
```
conda create --name <env_name> python==3.8
conda update -n base conda
```

## Activate conda environment and install dependencies
```
conda activate <env_name>

conda install -y -c conda-forge openmm==7.5.1 cudnn==8.0.4 cudatoolkit==11.0.3 pdbfixer==1.7
conda install -y -c bioconda hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04
```

## Install other alphafold dependencies using pip
```
pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow==2.5.0
```

### Change jaxlib version for alphafold
```
pip install --upgrade jax jaxlib==0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Download Alphafold parameters
```
wget https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar -P <path_to_params_dir>

tar --extract --verbose --file=<path_to_params_dir>/alphafold_params_2021-07-14.tar

rm <path_to_params_dir>/alphafold_params_2021-07-14.tar
```

## OpenMM Patch
`Note: <path_to_alphafold> should the directory path where the alphafold directory is located`

```
cd ~/miniconda3/envs/<env_name>/lib/python3.8/site-packages/
patch -p0 < <path_to_alphafold>/docker/openmm.patch
```

## How to run?
```
bash run_alphafold.sh -d <path_to_params_dir> -o <output_dir> -m model_1,model_2,model_3,model_4,model_5 -f <path_to_fasta> -s <path_to_a3m_file> -t 2019-05-14
```
- -m: at least one model name must be provided
- -t: template date (refer alphafold github repository for more details)

### Note
This codebase runs on `CUDA 10.1`. This was tried and tested in `Ubuntu 18.04.4 LTS` and the hardware specifications of the server are as follow:
1. Dual 4215R 3.2GHz CPUs
2. 128 GB RAM
3. NVIDIA Quadro RTX 6000 GUPs each with 24GB memory

### Acknowledgement
To create this code base, the actual alphafold repository [https://github.com/deepmind/alphafold] was used as well as another repository from https://github.com/kalininalab/alphafold_non_docker was referenced.
