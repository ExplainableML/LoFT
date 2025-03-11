import os
from os.path import join as ospj
import json
import numpy as np
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DistributedSampler
import torchvision as tv

from util_data import (
    SUBSET_NAMES,
    configure_metadata, get_image_ids, get_class_labels,
    RandomResizedCrop, GaussianBlur, Solarization
)


NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transforms(model_type, img_size=224):

    if model_type == "clip":
        norm_mean = CLIP_NORM_MEAN
        norm_std = CLIP_NORM_STD
    elif model_type == "resnet50":
        norm_mean = NORM_MEAN
        norm_std = NORM_STD
    aux_transform = tv.transforms.Compose([
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomApply(
            [
                tv.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=0.8,
        ),
        tv.transforms.RandomGrayscale(p=0.2),
        GaussianBlur(0.2),
        Solarization(0.2),
    ])

    train_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.RandAugment(2, 9),
        tv.transforms.RandomResizedCrop(
            img_size, 
            scale=(0.25, 1.0), 
            interpolation=tv.transforms.InterpolationMode.BICUBIC,
            antialias=None,
        ),
        aux_transform,
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = tv.transforms.Compose([
        tv.transforms.Lambda(lambda x: x.convert("RGB")),
        tv.transforms.Resize(
            img_size, 
            interpolation=tv.transforms.functional.InterpolationMode.BICUBIC
        ),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(norm_mean, norm_std)
    ])
        
    return train_transform, test_transform


class DatasetFromMetadata(Dataset):
    def __init__(
        self, data_root, metadata_root, transform,
        dataset="imagenet",
    ):
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform
        self.image_ids = get_image_ids(self.metadata, proxy=False)
        self.image_labels = get_class_labels(self.metadata)
        self.image_ids = list(self.image_labels.keys())

    def get_data(self, fpath):
        x = Image.open(fpath)
        x = x.convert('RGB')
        return x
            
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = self.get_data(ospj(self.data_root, image_id))
        image_label = self.image_labels[image_id]
        image = self.transform(image)
        return image, image_label

    def __len__(self):
        return len(self.image_ids)


class DatasetSynthImage(Dataset):
    def __init__(
        self, 
        synth_train_data_dir, 
        transform, 
        n_img_per_cls=None,
        dataset='imagenet', 
        **kwargs
    ):
        self.synth_train_data_dir = synth_train_data_dir
        self.transform = transform
        self.n_img_per_cls = n_img_per_cls
        self.dataset = dataset
        
        self.image_paths, self.image_labels = self.get_data()

    def get_data(self):
        image_paths = []
        image_labels = []
        for label, class_name in enumerate(SUBSET_NAMES[self.dataset]):
            class_dir = ospj(self.synth_train_data_dir, class_name)
            count = 0
            for fname in os.listdir(class_dir):
                if not fname.endswith(".png"):
                    continue

                fpath = ospj(class_dir, fname)
                if os.path.getsize(fpath) / 1024 < 10:
                    # cannot identify image file since the size is 0kb
                    print(f"Cannot identify image file: {fpath}")
                    continue

                if self.n_img_per_cls is not None:
                    if count < self.n_img_per_cls:
                        count += 1
                    else:
                        break

                image_paths.append(fpath)
                image_labels.append(label)

        return image_paths, image_labels
                
    def __getitem__(self, idx):
        image_label = self.image_labels[idx]
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, image_label

    def __len__(self):
        return len(self.image_paths)


def split_eurosat(real_test_data_dir, transform, split):
    file_path = os.path.join(real_test_data_dir, 'real')
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    dataset = tv.datasets.EuroSAT(
        root=file_path,
        transform=transform,
        download=True,
    )

    split_file_path = os.path.join(file_path, 'split_zhou_EuroSAT.json')
    if not os.path.exists(split_file_path):
        # split taken from https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#eurosat
        raise ValueError(
            "Please download or copy split_zhou_EuroSAT.json "
            "from https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#eurosat "
            "into the dataset directory."
        )
    f = open(split_file_path)
    split_files = json.load(f)
    data = [os.path.join(file_path, 'eurosat', '2750', path[0]) for path in split_files[split]]
    dataset.samples = [sample for sample in dataset.samples if sample[0] in data]
    dataset.labels = [s[1] for s in dataset.samples]
    return dataset


def split_sun(real_test_data_dir, transform, split):
    file_path = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.SUN397(
        root=file_path,
        transform=transform,
        download=True,
    )

    import csv
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/sun397/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/sun397/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(file_path, 'SUN397') + '/'
    ind_to_keep = [i for i, file in enumerate(dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    dataset._image_files = [l for i, l in enumerate(dataset._image_files) if i in ind_to_keep]
    dataset._labels = [l for i, l in enumerate(dataset._labels) if i in ind_to_keep]
    return dataset


def split_caltech(real_test_data_dir, transform, split):
    file_path = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.Caltech101(
        root=file_path,
        transform=transform,
        download=True,
    )

    import csv
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    ind_to_keep = [i for i in range(len(dataset.index)) if
                   os.path.join(dataset.categories[dataset.y[i]],
                                'image_' + '{:04d}'.format(dataset.index[i]) +
                                '.jpg') in split_files]
    dataset.index = [dataset.index[i] for i in ind_to_keep]
    dataset.y = [dataset.y[i] for i in ind_to_keep]
    # shift everything from 2 up down by one, because faces_easy at idx=1 not used
    dataset.y = [i if i < 1 else i - 1 for i in dataset.y]
    # remove Faces_easy
    dataset.categories.remove("Faces_easy")
    dataset.annotation_categories.remove("Faces_3")
    return dataset


def split_aircraft(real_test_data_dir, transform, split):
    aircraft_path_train = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.FGVCAircraft(
        root=aircraft_path_train,
        split=split,
        annotation_level='variant',
        transform=transform,
        download=True,
    )
    return dataset


def split_cars(real_test_data_dir, transform, split):
    import csv
    file_path = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.StanfordCars(
        root=file_path,
        split='test' if split == 'test' else 'train',
        transform=transform,
        download=False,
    )
    split_file_path = os.path.join(file_path, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/stanford_cars/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/stanford_cars/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    tmp_dir = ospj(file_path, 'stanford_cars/cars_')
    tmp_dict = {k.replace(tmp_dir, ''): v for k,v in dataset._samples}
    dataset._samples = [(tmp_dir + k, v) for k,v in tmp_dict.items() if k in split_files]
    return dataset



def split_dtd(real_test_data_dir, transform, split):
    import csv
    dtd_path_train = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.DTD(
        root=dtd_path_train,
        split='train',
        transform=transform,
        download=True,
    )
    val_dataset = tv.datasets.DTD(
        root=dtd_path_train,
        split='val',
        transform=transform,
        download=True,
    )
    test_dataset = tv.datasets.DTD(
        root=dtd_path_train,
        split='test',
        transform=transform,
        download=True,
    )
    dataset._image_files = dataset._image_files + val_dataset._image_files + test_dataset._image_files
    dataset._labels = dataset._labels + val_dataset._labels + test_dataset._labels

    split_file_path = os.path.join(dtd_path_train, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/dtd/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/dtd/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(dtd_path_train, 'dtd', 'dtd', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    dataset._image_files = [l for i, l in enumerate(dataset._image_files) if i in ind_to_keep]
    dataset._labels = [l for i, l in enumerate(dataset._labels) if i in ind_to_keep]
    return dataset


def split_flowers(real_test_data_dir, transform, split):
    import csv
    flowers_path_train = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.Flowers102(
        root=flowers_path_train,
        split='train',
        transform=transform,
        download=True,
    )
    val_dataset = tv.datasets.Flowers102(
        root=flowers_path_train,
        split='val',
        transform=transform,
        download=True,
    )
    test_dataset = tv.datasets.Flowers102(
        root=flowers_path_train,
        split='test',
        transform=transform,
        download=True,
    )
    dataset._image_files = dataset._image_files + val_dataset._image_files + test_dataset._image_files
    dataset._labels = dataset._labels + val_dataset._labels + test_dataset._labels

    split_file_path = os.path.join(flowers_path_train, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/flowers102/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/flowers102/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(flowers_path_train, 'flowers-102', 'jpg') + '/'
    ind_to_keep = [i for i, file in enumerate(dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    dataset._image_files = [l for i, l in enumerate(dataset._image_files) if i in ind_to_keep]
    dataset._labels = [l for i, l in enumerate(dataset._labels) if i in ind_to_keep]
    return dataset


def split_food(real_test_data_dir, transform, split):
    import csv
    food_path_train = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.Food101(
        root=food_path_train,
        split='train',
        transform=transform,
        download=True,
    )
    test_dataset = tv.datasets.Food101(
        root=food_path_train,
        split='test',
        transform=transform,
        download=True,
    )
    dataset._image_files = dataset._image_files + test_dataset._image_files
    dataset._labels = dataset._labels + test_dataset._labels

    split_file_path = os.path.join(food_path_train, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/food101/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/food101/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'])
    file_path_full = os.path.join(food_path_train, 'food-101', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(dataset._image_files)
                   if str(file).replace(file_path_full, '') in split_files]
    dataset._image_files = [l for i, l in enumerate(dataset._image_files) if i in ind_to_keep]
    dataset._labels = [l for i, l in enumerate(dataset._labels) if i in ind_to_keep]
    return dataset


def split_pets(real_test_data_dir, transform, split):
    import csv
    pets_path_train = os.path.join(real_test_data_dir, 'real')
    dataset = tv.datasets.OxfordIIITPet(
        root=pets_path_train,
        split='trainval',
        target_types='category',
        download=True,
        transform=transform,
    )
    test_dataset = tv.datasets.OxfordIIITPet(
        root=pets_path_train,
        split='test',
        target_types='category',
        download=True,
        transform=transform,
    )
    dataset._images = dataset._images + test_dataset._images
    dataset._labels = dataset._labels + test_dataset._labels

    split_file_path = os.path.join(pets_path_train, 'split_coop.csv')
    if not os.path.exists(split_file_path):
        # split taken from DISEF paper
        # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/oxford_pets/split_coop.csv
        raise ValueError(
            "Please download or copy split_coop.json "
            "from https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/oxford_pets/split_coop.csv "
            "into the dataset directory."
        )
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'].split('/')[-1])
    file_path_full = os.path.join(pets_path_train, 'oxford-iiit-pet', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(dataset._images)
                   if str(file).replace(file_path_full, '') in split_files]
    dataset._images = [l for i, l in enumerate(dataset._images) if i in ind_to_keep]
    dataset._labels = [l for i, l in enumerate(dataset._labels) if i in ind_to_keep]
    return dataset


def get_data_loader(
    real_test_data_dir="",
    metadata_dir="metadata",
    dataset="imagenet", 
    eval_bs=32,
    model_type=None,
    is_validation=False,
):

    _, test_transform = get_transforms(model_type)

    """ Test & Val dataset/dataloader """

    def get_val_or_test_loader(phase):
        if dataset == 'imagenet':
            _dataset = DatasetFromMetadata(
                data_root=real_test_data_dir,
                metadata_root=ospj(metadata_dir, 'test'),
                transform=test_transform,
                dataset=dataset,
            )
        else:
            _dataset = {
                'pets': split_pets,
                'food101': split_food,
                'fgvc_aircraft': split_aircraft,
                'eurosat': split_eurosat,
                'cars': split_cars,
                'dtd': split_dtd,
                'flowers102': split_flowers,
                'sun397': split_sun,
                'caltech101': split_caltech,
            }[dataset](real_test_data_dir, test_transform, phase)

        _loader = torch.utils.data.DataLoader(
            _dataset, batch_size=eval_bs, shuffle=False, 
            num_workers=12, pin_memory=True)
            
        return _loader

    if is_validation:
        val_loader = get_val_or_test_loader(phase='val')
        test_loader = get_val_or_test_loader(phase='test')
        return val_loader, test_loader
    else:
        test_loader = get_val_or_test_loader(phase='test')
        return test_loader


def get_synth_train_data_loader(
    synth_train_data_dir,
    bs=32, 
    n_img_per_cls=50,
    dataset='imagenet',
    model_type='clip',
    is_distributed=False,
):
    train_transform, _ = get_transforms(model_type)

    train_dataset = DatasetSynthImage(
        synth_train_data_dir=synth_train_data_dir, 
        transform=train_transform,
        n_img_per_cls=n_img_per_cls,
        dataset=dataset,
    ) 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, 
        sampler=(
            DistributedSampler(train_dataset, shuffle=True) 
            if is_distributed else None
        ),
        shuffle=None if is_distributed else True,
        num_workers=12, pin_memory=True,
    )

    return train_loader



