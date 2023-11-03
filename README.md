# Learning Part Motion of Articulated Objects Using Spatially Continuous Neural Implicit Representations

Yushi Du*, Ruihai Wu*, Yan Shen, Hao Dong

BMVC 2023

[Project](https://yushi-du.github.io/PartMotion/)

We released the data generation code of Learning Part Motion of Articulated Objects Using Spatially 
Continuous Neural Implicit Representations here

## Installation

1. Create a conda environment and install required packages.

```bash
conda env create -f conda_env_gpu.yaml -n PMotion
```

You can change the `pytorch` and `cuda` version by yourself in conda_env_gpu.yaml.

2. Build ConvONets dependents by running `python scripts/convonet_setup.py build_ext --inplace`.

3. Unzip the data under the repo's root into `./data`.

## Training

```Python
# single GPU
python run.py experiment=Door_emd

# multiple GPUs
python run.py trainer.gpus=4 +trainer.accelerator='ddp' experiment=Door_emd

```

## Testing

```Python
# only support single GPU
python run_test.py experiment=Door_emd trainer.resume_from_checkpoint=/path/to/trained/model/
```

## Tuning

```Python
# both single and multiple GPUs
python run_tune.py experiment=Door_emd trainer.resume_from_checkpoint=/path/to/trained/model/
```

## Other instructions

You'll also need to follow the instructions [here](https://github.com/daerduoCarey/PyTorchEMD) to set up the Earth Mover 
Distance mentioned in our paper.

## Citation

If you consider this paper useful, please consider citing:

```Python
@inproceedings{du2023learning,
  title={Learning Part Motion of Articulated Objects Using Spatially Continuous Neural Implicit Representations},
  author={Du, Yushi and Wu, Ruihai and Shen, Yan and Dong, Hao},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2023}
}
```