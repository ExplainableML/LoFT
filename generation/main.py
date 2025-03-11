import json
import os
import random
from os.path import join as ospj
from typing import Tuple
import itertools
from typing import Union
from PIL import Image

import fire
import numpy as np
import torch
import torchvision as tv
import yaml
from tqdm import tqdm

from util import (
    batch_iteration, 
    make_dirs,
    set_seed,
    CLASSNAMES,
    TEMPLATES_SMALL,
)


def set_local():
    yaml_file = "local.yaml"
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    return args_local



def get_pipe(model_dir, device, is_tqdm):
    # CUDA_VISIBLE_DEVICES issue
    # https://discuss.pytorch.org/t/cuda-visible-device-is-of-no-use/10018
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=not is_tqdm)
    return pipe


def get_prompt_embeds(pipe, prompts, device):
    text_inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if (
        hasattr(pipe.text_encoder.config, "use_attention_mask")
        and pipe.text_encoder.config.use_attention_mask
    ):
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def get_dataset_name_for_template(dataset):
    dataset_name = {
        "imagenet": "",
        "imagenet_100": "",
        "pets": "pet ",
        "fgvc_aircraft": "aircraft ",
        "cars": "car ",
        "eurosat": "satellite ",
        "dtd": "texture ",
        "flowers102": "flower ",
        "food101": "food ",
        "sun397": "scene ",
        "caltech101": "",
    }[dataset]
    return dataset_name


class GenerateImage:
    def __init__(
        self,
        pipe,
        device,
        method,
        guidance_scale,
        num_inference_steps,
        n_image_per_class,
        save_dir,
        count_start,
        bs,
        n_shot,
        dataset,
        fewshot_seed,
        sd_lora_dir,
        loft_interpolation_weight,
    ):
        self.pipe = pipe
        self.device = device
        self.method = method
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.n_image_per_class = n_image_per_class
        self.save_dir = save_dir
        self.count_start = count_start
        self.bs = bs
        self.n_shot = n_shot
        self.dataset = dataset
        self.dataset_name = get_dataset_name_for_template(dataset)
        self.fewshot_seed = fewshot_seed
        self.sd_lora_dir = sd_lora_dir
        self.loft_interpolation_weight = loft_interpolation_weight

        self.run = {
            "loft": self.run_loft,
            "datadream-cls": self.run_datadream,
            "datadream-dset": self.run_datadream,
        }[method]


    def set_save_dir(self, classname, meta_data):
        save_dir = ospj(self.save_dir, classname)
        make_dirs(save_dir)
        with open(ospj(save_dir, "meta_data.json"), "w") as f:
            json.dump(meta_data, f, indent=2)
        return save_dir


    def save_data(self, outputs, save_dir, count):
        images = outputs.images
        # save images
        for image in images:
            fpath = ospj(save_dir, f"{count}.png")
            image = image.resize((512, 512))
            image.save(fpath)
            count += 1
        return count


    def run_pipe(self, prompts):
        kwargs = {
            "prompt": prompts,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
        }
        outputs = self.pipe(**kwargs)
        return outputs


    def get_inputs_datadream(self, label, classname):
        # update lora weights
        _c = classname if self.method == "datadream-cls" else ""
        lora_weight_dir = ospj(self.sd_lora_dir, _c)
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(
            lora_weight_dir, 
            weight_name="pytorch_lora_weights.safetensors", 
        )

        # prompt
        templates = TEMPLATES_SMALL[:1]
        n_repeat = self.n_image_per_class // len(templates) + 1
        prompts = [
            template.format(self.dataset_name, classname)
            for _ in range(n_repeat)
            for template in templates
        ]
        prompts = prompts[: self.n_image_per_class]
        return prompts


    def get_inputs_loft(self, label, classname):
        # update lora weights
        self.pipe.unload_lora_weights()
        _dir = ospj(self.sd_lora_dir, classname)
        filenames = os.listdir(_dir)
        for filename in filenames:
            lora_weight_dir = ospj(_dir, filename)
            self.pipe.load_lora_weights(
                lora_weight_dir, 
                weight_name="pytorch_lora_weights.safetensors", 
                adapter_name=filename,
            )

        # prompt
        prompt = TEMPLATES_SMALL[0].format(self.dataset_name, classname)

        # list of LoRA (or LoRA combinations) and its weights
        if self.loft_interpolation_weight in (0, 1):
            lora_names_list = [[_f] for _f in filenames]
        else:
            liw_len = (
                2 if isinstance(self.loft_interpolation_weight, float) 
                else len(self.loft_interpolation_weight)
            )
            
            lora_names_list = list(itertools.combinations(filenames, liw_len))
            random.shuffle(lora_names_list)
        n_repeat = self.n_image_per_class // len(lora_names_list) + 1
        lora_names_list = lora_names_list * n_repeat
        lora_names_list = lora_names_list[: self.n_image_per_class]
        
        if self.loft_interpolation_weight in (0, 1):
            lora_interp_weights_list = [[1] for _ in range(len(lora_names_list))]
        else:
            p = self.loft_interpolation_weight
            if isinstance(p, float):
                lora_interp_weights_list = [[p, 1-p]] * len(lora_names_list)
            else:
                lora_interp_weights_list = [list(p)] * len(lora_names_list)

        return prompt, lora_names_list, lora_interp_weights_list


    def run_datadream(self, label, classname):
        # prompts for input to SD
        prompts = self.get_inputs_datadream(label, classname)

        # set save directory
        save_dir = self.set_save_dir(classname, prompts)

        # start from self.count_start
        count = self.count_start
        prompts = prompts[self.count_start :]

        # run iteration
        iter_length = len(list(batch_iteration(prompts, self.bs)))
        for gen_iter, prompts_batch in enumerate(batch_iteration(prompts, self.bs)):
            # generate images
            outputs = self.run_pipe(prompts_batch)

            # save
            count = self.save_data(outputs, save_dir, count)


    def run_loft(self, label, classname):
        # prompt for input to SD
        prompt, lora_names_list, lora_interp_weights_list = self.get_inputs_loft(label, classname)

        # set save directory
        metadata = [
            ", ".join(str(_x) for _x in (_count, *lora_names, *lora_interp_weights))
            for _count, (lora_names, lora_interp_weights)
            in enumerate(zip(lora_names_list, lora_interp_weights_list))
        ]
        meta_data = [f"Prompt: {prompt}"] + metadata
        save_dir = self.set_save_dir(classname, meta_data)

        # start from self.count_start
        count = self.count_start
        lora_names_list = lora_names_list[self.count_start:]
        lora_interp_weights_list = lora_interp_weights_list[self.count_start:]

        # run iteration
        for gen_iter, (lora_names, lora_interp_weights) in enumerate(
            zip(lora_names_list, lora_interp_weights_list)
        ):
            # update LoRA interpolation 
            self.pipe.set_adapters(
                lora_names, adapter_weights=lora_interp_weights)

            # generate images
            outputs = self.run_pipe(prompt)

            # save
            count = self.save_data(outputs, save_dir, count)


def main(
    seed=0,
    dataset="imagenet",
    method="loft",
    n_shot=16,
    fewshot_seed="seed0",  # best or seed{number}.
    guidance_scale=2.0,
    num_inference_steps=50,
    n_image_per_class=100,
    count_start=0,
    n_set_split=1,
    split_idx=0,
    bs=5,
    is_tqdm: bool = True,
    loft_interpolation_weight: Union[float, list[float]] = 0,
):
    if isinstance(n_set_split, str):
        n_set_split = int(n_set_split)
    if isinstance(split_idx, str):
        split_idx = int(split_idx)
    if isinstance(n_shot, str):
        n_shot = int(n_shot)

    # set GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set local arguments
    args_local = set_local()
    sd_lora_dir = args_local["sd_lora_dir"]
    model_dir = args_local["model_dir"]

    # lora directory
    mid_dir = f"shot{n_shot}_{fewshot_seed}"
    sd_lora_dir = ospj(sd_lora_dir, dataset, method, mid_dir)

    # save directory
    mid0 = method
    if method == "loft":
        liw = (
            loft_interpolation_weight 
            if isinstance(loft_interpolation_weight, (float, int)) 
            else "-".join([str(_l) for _l in loft_interpolation_weight])
        )
        mid0 += f"_{liw}"
    mid1 = f"shot{n_shot}_{fewshot_seed}"
    save_dir = ospj(args_local["save_dir"], dataset, mid0, mid1)
    print(save_dir)
    
    # load SD pipeline
    pipe = get_pipe(model_dir, device, is_tqdm)

    # load instance
    generate_image = GenerateImage(
        pipe=pipe,
        device=device,
        method=method,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        n_image_per_class=n_image_per_class,
        save_dir=save_dir,
        count_start=count_start,
        bs=bs,
        n_shot=n_shot,
        dataset=dataset,
        fewshot_seed=fewshot_seed,
        sd_lora_dir=sd_lora_dir,
        loft_interpolation_weight=loft_interpolation_weight,
    )

    # class names
    classnames = CLASSNAMES[dataset]
    labels = [i for i in range(len(classnames))]
    iters = list(zip(labels, classnames))

    # parallel computing
    step = len(iters) // n_set_split
    start_idx = split_idx * step
    end_idx = (split_idx + 1) * step if (split_idx + 1) != n_set_split else len(iters)
    print(
        f"SPLIT!! Out of {len(classnames)} pairs, we generate from idx {start_idx} to {end_idx}."
    )
    iters_partial = iters[start_idx:end_idx]

    # generate & save synthetic images
    for data in tqdm(iters_partial, total=len(iters_partial)):
        set_seed(seed)
        generate_image.run(*data)


if __name__ == "__main__":
    fire.Fire(main)
