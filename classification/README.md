## Setup

Complete the `local.yaml` file.

## Run

You can train the classifier with synthetic images of LoFT, DataDream-class, or DataDream-dataset with the following commands. The example is on the eurosat dataset. It will use 500 images for each class as a training set. You can change the number of images by changing `$NIPC` in the `run.sh` file.

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
