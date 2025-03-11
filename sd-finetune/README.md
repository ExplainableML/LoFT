## Setup

Complete the `local.yaml` file.

## Run

You can run LoFT, DataDream-class, and DataDream-dataset with the following commands. The example is on the eurosat dataset (which has 10 classes) with 16-shot setting.

```bash
### LoFT

for CLASS_IDX in {0..9}; do
  for INSTANCE_IDX in {0..15}; do
    CUDA_VISIBLE_DEVICES=0 bash run.sh eurosat $CLASS_IDX $INSTANCE_IDX
  done
done
```

```bash
### DataDream-class

for CLASS_IDX in {0..9}; do
  CUDA_VISIBLE_DEVICES=0 bash run.sh eurosat $CLASS_IDX None
done
```

```bash
### DataDream-dataset

CUDA_VISIBLE_DEVICES=0 bash run.sh eurosat None None
```
