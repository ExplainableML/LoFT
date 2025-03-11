# Synthetic Data Generation with Few-shot Guidance

This repository contains the codebase of a series of projects on synthetic data generation with few-shot guidance. 

* [LoFT: LoRA-fused Dataset Generation with Few-shot Guidance](TBD), Arxiv.
* [DataDream: Few-shot Guided Dataset Generation](https://arxiv.org/pdf/2407.10910), in ECCV, 2024.


## Preliminary Setup
We use [Stable-Diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) as a base diffusion model.

Also, few-shot real data should be formed in the following way. Assuming we use 16-shot, each data file should be located in the path `PATH_TO_REAL_FEWSHOT/$DATASET/shot$N_SHOT_seed$FEWSHOT_SEED/$CLASS_NAME/$FILE`. The list of `$CLASS_NAME` For each `$DATASET` can be found in `sd-finetune/util.py` file. For instance.
```bash
📂 data
|_📂 real_train_fewshot
  |_📂 imagenet
    |_📂 shot16_seed0
      |_📂 abacus
        |_📄 n02666196_17944.JPEG
        |_📄 n02666196_10754.JPEG
        |_📄 n02666196_10341.JPEG
        ...
        |_📄 n02666196_16649.JPEG
      |_📂 clothes iron
      |_📂 great white shark
      |_📂 goldfish
      |_📂 tench
      ...
  |_📂 eurosat
    |_📂 shot16_seed0
      |_📂 AnnualCrop
      |_📂 Forest
      ...
```


## Step

You can run LoFT, DataDream-class, and DataDream-dataset methods by following the process below.
1. Install the necessary dependencies in `requirements.txt`.
2. **Finetune diffusion model**: Follow the instructions in the `sd-finetune` folder.
3. **Dataset generation**: Follow the instructions in the `generation` folder.
4. **Train Classification with synthetic data**: Follow the instructions in the `classification` folder.

## Citation

If you use this code in your research, please kindly cite the following papers

```bibtex
@article{kim2025loft,
TBD
}

@article{kim2024datadream,
  title={DataDream: Few-shot Guided Dataset Generation},
  author={Kim, Jae Myung and Bader, Jessica and Alaniz, Stephan and Schmid, Cordelia and Akata, Zeynep},
  journal={arXiv preprint arXiv:2407.10910},
  year={2024}
}
```
