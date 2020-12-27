import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional as TF
import numpy as np

from .MultiObjectDataset import MultiObjectDataset
from .DVRDataset import DVRDataset
from .SRNDataset import SRNDataset


class ColorJitterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dset,
        hue_range=0.1,
        saturation_range=0.1,
        brightness_range=0.1,
        contrast_range=0.1,
    ):
        self.hue_range = [-hue_range, hue_range]
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]

        self.base_dset = base_dset
        self.z_near = self.base_dset.z_near
        self.z_far = self.base_dset.z_far
        self.lindisp = self.base_dset.lindisp
        self.base_path = self.base_dset.base_path
        self.image_to_tensor = self.base_dset.image_to_tensor

    def apply_color_jitter(self, images):
        # apply the same color jitter over batch of images
        hue_factor = np.random.uniform(*self.hue_range)
        saturation_factor = np.random.uniform(*self.saturation_range)
        brightness_factor = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)
        for i in range(len(images)):
            tmp = (images[i] + 1.0) * 0.5
            tmp = F_t.adjust_saturation(tmp, saturation_factor)
            tmp = F_t.adjust_hue(tmp, hue_factor)
            tmp = F_t.adjust_contrast(tmp, contrast_factor)
            tmp = F_t.adjust_brightness(tmp, brightness_factor)
            images[i] = tmp * 2.0 - 1.0
        return images

    def __len__(self):
        return len(self.base_dset)

    def __getitem__(self, idx):
        data = self.base_dset[idx]
        data["images"] = self.apply_color_jitter(data["images"])
        return data


def get_split_dataset(dataset_type, datadir, split="all", **kwargs):
    # dataset_dict = {
    #     'srn': SRNDataset,
    #     'multi_obj': MultiObjectDataset,
    #     'dvr_gen': DVRDataset
    # }
    assert type(split) in (list, tuple)
    # assert split in ('train','test','val','train_test','train_val', 'all')
    # dataset = dataset_dict[dataset_type] # TODO
    if dataset_type == "srn":
        # For ShapeNet single-category (from SRN)
        if split == "all":
            train_set = SRNDataset(datadir, stage="train", **kwargs)
            val_set = SRNDataset(datadir, stage="val", **kwargs)
            test_set = SRNDataset(datadir, stage="test", **kwargs)
            dataset = [train_set, val_set, test_set]
        else:
            dataset = [SRNDataset(datadir, stage=s, **kwargs) for s in split]
    elif dataset_type == "multi_obj":
        # For multiple-object
        train_set = MultiObjectDataset(datadir, stage="train")
        test_set = MultiObjectDataset(datadir, stage="test")
        val_set = MultiObjectDataset(datadir, stage="val")
    elif dataset_type == "dvr":
        # For ShapeNet category agnostic training
        train_set = DVRDataset(datadir, stage="train", **kwargs)
        val_set = DVRDataset(datadir, stage="val", **kwargs)
        test_set = DVRDataset(datadir, stage="test", **kwargs)
    elif dataset_type == "dvr_gen":
        # For generalization training (train some categories, eval on others)
        train_set = DVRDataset(datadir, stage="train", list_prefix="gen_", **kwargs)
        val_set = DVRDataset(datadir, stage="val", list_prefix="gen_", **kwargs)
        test_set = DVRDataset(datadir, stage="test", list_prefix="gen_", **kwargs)
    elif dataset_type == "dvr_dtu":
        # DTU dataset
        list_prefix = "new_"
        train_set = DVRDataset(
            datadir,
            stage="train",
            list_prefix=list_prefix,
            max_imgs=49,
            sub_format="dtu",
            z_near=0.1,
            z_far=5.0,
            **kwargs
        )
        val_set = DVRDataset(
            datadir,
            stage="val",
            list_prefix=list_prefix,
            max_imgs=49,
            sub_format="dtu",
            z_near=0.1,
            z_far=5.0,
            **kwargs
        )
        test_set = DVRDataset(
            datadir,
            stage="test",
            list_prefix=list_prefix,
            max_imgs=49,
            sub_format="dtu",
            z_near=0.1,
            z_far=5.0,
            **kwargs
        )
        train_set = ColorJitterDataset(train_set)
    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)
    return dataset

    # if split == "train":
    #     return train_set
    # elif split == "val":
    #     return val_set
    # elif split == "test":
    #     return test_set
    # return train_set, val_set, test_set
