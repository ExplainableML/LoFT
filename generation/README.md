## Setup

Complete the `local.yaml` file.

## Run

You can generate images of LoFT, DataDream-class, or DataDream-dataset methods with the following commands. The example is on the eurosat dataset. It will generate 500 images for each class. You can change the number of generated images by changing `$NIPC` in the `run.sh` file.

```bash
### LoFT

CUDA_VISIBLE_DEVICES=0 bash run.sh eurosat loft
```

```bash
### DataDream-class

CUDA_VISIBLE_DEVICES=0 bash run.sh eurosat datadream-cls
```

```bash
### DataDream-dataset

CUDA_VISIBLE_DEVICES=0 bash run.sh eurosat datadream-dset
```
