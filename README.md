# Video Stabilization
## POVa project

Video statbilization project for the computer vision course using PyTorch.

## Setup

This project includes the file [`env.yml`](env.yml) that includes all the
dependencies needed to run. This project uses a 
[miniconda](https://docs.conda.io/projects/miniconda/en/latest/) environment.

In order to get install everything you just need to run the following commands:

```bash
conda env create --file=env.yml     # Creates the env (only needed once) 
conda activate stabilization        # Activates the env
```

## Usage

First, ensure that you have the environment activated using:

```bash
conda activate stabilization
```

The next step is to execute the program using the command below, which will
print the help instructions.

```bash
python src/stab.py --help
```

You may also try the gaussian stabilization version with:

```bash
python src/gauss_stab.py --help
```

---
Authors:

- [Guilherme Franco](mailto:xfranc01@stud.fit.vutbr.cz)
- [Miguel Rodrigues](mailto:xboave00@stud.fit.vutbr.cz)

