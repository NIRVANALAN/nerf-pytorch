import glob
import os
import warnings
from pathlib import Path

import ipdb
import mmcv
import numpy as np
import torch
from mmcls.datasets import build_dataloader, build_dataset
from mmcv import DictAction
from torch.utils.data import DataLoader, Dataset, TensorDataset
from util import image_to_normalized_tensor, todevice
from util.run_nerf_helpers import get_rays

from .data_util import get_split_dataset  # ColorJitterDataset
from .load_blender import *
from .load_deepvoxels import *
from .load_llff import *


class Incremental_dataset(Dataset):
    # this class serves for dynamic training_data update during co-training
    # dynamically add from inversion past, and set incremental_img_flag
    def __init__(self,
                 train_imgs,
                 train_poses,
                 test_instance,
                 intrinsics,
                 verbose=True,
                 device='cuda') -> None:
        # init with NeRF GT train_imgs, test_imgs
        self.test_instance = test_instance  # TODO
        self.verbose = verbose
        self.imgs = train_imgs.to(device)
        self.poses = train_poses.to(device)
        self.h, self.w, self.focal, self.c = intrinsics
        self.rays_rgb = self.build_rays_rgb(self.imgs, self.poses)
        self.runtime_rays_rgb = self.rays_rgb.clone(
        )  # update this attribute during training
        self.flags = torch.zeros(self.rays_rgb.shape[:-1]).cpu().long(
        )  # 0 denotes from original NeRF dataset
        self.device = device
        # import ipdb
        # ipdb.set_trace()

    # @ staticmethod
    def build_rays_rgb(self, imgs, poses, shuffle_init=True):
        # ugly code
        rays = torch.stack(  # TODO
            [
                get_rays(self.h, self.w, self.focal, p, self.c)
                for p in poses[:, :3, :4]
            ], 0)  # [N, ro+rd, H, W, 3] #?

        rays_rgb = torch.cat([rays, imgs[:, None]],
                             1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)  # [N, H, W, ro+rd+rgb, 3]
        # rays_rgb = rays_rgb.view(-1, 3, 3) # [NHW, ro+rd+rgb, 3]
        rays_rgb = torch.reshape(rays_rgb, (-1, 3, 3))  # [NHW, ro+rd+rgb, 3]
        if shuffle_init:
            rays_rgb = self.force_shuffle(rays_rgb)

        return rays_rgb

    def __len__(self):
        return self.rays_rgb.shape[0]

    def __getitem__(self, idx):
        sample = self.runtime_rays_rgb[idx]  # [2+1, 3*?]
        flag = self.flags[idx]
        return sample, flag
        # rays, target = sample[:2], sample[2]
        # return rays, target, flag

    @staticmethod
    def force_shuffle(rays_rgb, return_idx=False):

        rand_idx = torch.randperm(rays_rgb.shape[0])
        rays_rgb = rays_rgb[rand_idx]
        if return_idx:
            return rays_rgb, rand_idx
        return rays_rgb
    # dynamically update data, add data(img, pose) from inversion
    def incremental_update_data(self, incremental_path, update_runtime=True):
        # search for incremental data to be added
        incremental_imgs_path = sorted(
            glob.glob(str(incremental_path / '*.png')))
        incremental_ids = np.array([
            int(rgb_path.split('/')[-1].split('.')[0])
            for rgb_path in incremental_imgs_path
        ])
        if self.verbose:
            print('add incremental imgs id: {}'.format(incremental_ids))

        incremental_imgs = torch.stack([
            torch.from_numpy(imageio.imread(rgb_path)[..., :3] / 255)
            for rgb_path in incremental_imgs_path
        ]).to(self.device)
        incremental_poses = self.test_instance["poses"][incremental_ids,
                                                        ...].to(self.device)

        incremental_rays_rgb = self.build_rays_rgb(incremental_imgs,
                                                   incremental_poses)
        if update_runtime:
            self.runtime_rays_rgb = torch.cat(
                [self.runtime_rays_rgb, incremental_rays_rgb])

        return incremental_rays_rgb

        # needs ablation study

        # self.imgs = torch.cat([self.imgs, incremental_imgs], 0)
        # self.poses = torch.cat([self.poses, incremental_poses])

        # concat new incremental-imgs

        # new_flags = torch.ones((incremental_imgs.shape[0],
        #                         *self.rays_rgb.shape[1:-1]))  # maintain shape
        # self.flags = torch.cat([self.flags, new_flags])


def create_dataset(args):
    c, data, normalized_img_tensor, i_fixed, i_fixed_test, incremental_flags = None, None, None, None, None, None
    # decoder_images, decoder_images_normalized = None, None
    decoder_dataloader = None

    if args.dataset_type not in ("llff", "blender", "deepvoxels"):
        dataset = get_split_dataset(args.dataset_type,
                                    args.datadir,
                                    split=["train", "train_test"])

        instance = [s[args.srn_object_id] for s in dataset]
        views_avai = instance[0]["images"].shape[0]
        test_views_avai = instance[-1]["images"].shape[0]

        if len(args.srn_input_views_id.split()) > 1:
            input_views_ids = np.array(
                list(map(int, args.srn_input_views_id.split())))
        else:
            if args.input_view_sampling == "arange":
                input_views_ids = np.arange(0, args.srn_input_views)
            elif args.input_view_sampling == "linspace":
                input_views_ids = np.linspace(0,
                                              views_avai,
                                              args.srn_input_views,
                                              False,
                                              dtype=int)
            else:
                raise NotImplementedError(args.input_view_sampling)

        if args.add_decoder:
            if args.external_sampling == "instance":
                if args.decoder_dataset not in args.datadir:
                    bank_dir = args.datadir.replace(
                        args.datadir.split("/")[-1], args.decoder_dataset)
                    dataset_decoder = get_split_dataset(args.dataset_type,
                                                        bank_dir,
                                                        split=["train"])
                else:
                    dataset_decoder = dataset
                decoder_instances = [
                    dataset_decoder[0][i]
                    for i in range(args.decoder_train_objs)
                ]  # TODO
                decoder_images = torch.cat(
                    [i["images"][:] for i in decoder_instances], 0).float()

                decoder_images_normalized = image_to_normalized_tensor(  # TODO
                    decoder_images.permute(0, 3, 1,
                                           2)).float()  # 250 * 3 * 128 * 128
                decoder_dataloader = DataLoader(
                    TensorDataset(decoder_images_normalized),
                    batch_size=args.decoder_bs,
                    shuffle=True,
                    num_workers=4,
                )
            elif args.external_sampling == "uniform":

                bank_dir = (
                    Path(args.datadir).parent /
                    f"{args.decoder_dataset}_pool_sub" / (
                        str(args.decoder_train_objs)
                        # + ("_sim" if args.external_sampling == "importance" else "")
                    ) / "imgs.pt")

                print("decoder_dir: {}".format(bank_dir))

                external_imgs_tensor = torch.load(bank_dir)  # TODO memory bank

                print("size of AE dataset: {}".format(
                    external_imgs_tensor.size(0)))

                decoder_dataloader = DataLoader(
                    TensorDataset(external_imgs_tensor),
                    batch_size=args.decoder_bs,
                    shuffle=True,
                    num_workers=4,
                )
            elif args.external_sampling == "importance":
                train_set_feats = torch.load(
                    Path(instance[0]["path"]) / "feats.pt")

                bank_dir = (
                    Path(args.datadir).parent /
                    f"{args.decoder_dataset}_pool_sub" / ("10000")  # TODO
                )

                mbank = torch.load(bank_dir / "memory_bank.pt")

                ensemble_feat = train_set_feats[input_views_ids].mean(0)
                sim = mbank @ ensemble_feat
                ordered_sim = torch.argsort(
                    sim, descending=True)  # top semantically similar objects

                external_imgs_tensor = torch.load(
                    bank_dir /
                    "imgs.pt")[ordered_sim[:args.decoder_train_objs]]

                print("size of AE dataset: {}".format(
                    external_imgs_tensor.size(0)))

                decoder_dataloader = DataLoader(
                    TensorDataset(external_imgs_tensor),
                    batch_size=args.decoder_bs,
                    shuffle=True,
                    num_workers=4,
                )
            elif args.external_sampling == "imagenet":  # TODO

                cfg = mmcv.Config.fromfile(args.mmcv_config)

                # build the dataloader
                dataset_mmcv = build_dataset(cfg.data.test)
                decoder_dataloader = build_dataloader(
                    dataset_mmcv,
                    samples_per_gpu=cfg.data.samples_per_gpu,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False,
                    round_up=False,
                )

            else:
                raise NotImplementedError("no {} sampling stategy".format(
                    args.external_sampling))

        test_views = np.linspace(1,
                                 test_views_avai,
                                 test_views_avai,
                                 False,
                                 dtype=int)
        # np.random.choice(np.arange(views_avai), size=args.srn_input_views)
        # input_views_id = (0, 38)  # try input views id = [0, 38]

        train_poses = instance[0]["poses"][input_views_ids, ...][:, :3, :4]
        train_imgs = instance[0]["images"][input_views_ids, ...]
        test_imgs = instance[-1]["images"][test_views, ...]

        images = torch.cat([train_imgs, test_imgs], dim=0).float()

        normalized_img_tensor = image_to_normalized_tensor(  # TODO
            images.permute(0, 3, 1, 2)).float()  # 250 * 3 * 128 * 128

        # poses = torch.cat([
        # train_poses,
        test_poses = instance[-1]["poses"][test_views, ...][:, :3, :4]
        # ], )

        poses = torch.cat([
            train_poses,
            test_poses,
        ], )

        # poses = poses[:, :3, :4]  #!

        i_train = np.arange(0, train_imgs.shape[0]).tolist()
        i_test = np.arange(train_imgs.shape[0], images.size(0)).tolist()
        i_val = None

        # create fixed poses
        i_fixed = input_views_ids[list(
            map(int, args.srn_encode_views_id.split()))]  # todo

        # encode test views if set
        if not args.not_encode_test_views:
            i_fixed_test = list(map(int,
                                    args.srn_encode_test_views_id.split()))
            if i_fixed_test[0] == -1:
                i_fixed_test = (
                    i_train  # same idx of train split to test views for encoding
                )
            print("i_fixed: {}".format(i_fixed_test))
            i_fixed_test = [i + len(input_views_ids) for i in i_fixed_test]

        render_poses = instance[-1]["poses"][test_views, ...]
        # instance[1]["poses"][::test_idx_interv]
        # render_poses = render_poses[:, :3, :4]  #!

        near, far = dataset[0].z_near, dataset[0].z_far
        hwf = [*train_imgs.shape[1:3], instance[0]["focal"]]
        c = instance[0]["c"]
        data = instance[0]

        # init incremental dataset for co-training
        incremental_dataset = Incremental_dataset(train_imgs,
                                                  train_poses,
                                                  instance[-1],
                                                  intrinsics=[*hwf, c])

        print("object id: {} | input view id: {}".format(
            data["path"], input_views_ids))

    elif args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf,
              args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([
            i for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ])

        print("DEFINING BOUNDS")
        if args.no_ndc:
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0

        else:
            near = 0.0
            far = 1.0

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print("Loaded blender", images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 -
                                                           images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":

        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip)

        print("Loaded deepvoxels", images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.0
        far = hemi_R + 1.0

    else:
        raise NotImplementedError("Unknown dataset type", args.dataset_type,
                                  "exiting")
    print("NEAR FAR", near, far)

    return (images, poses, render_poses, hwf, i_train, i_val, i_test, near,
            far, c, data, normalized_img_tensor, i_fixed, i_fixed_test, [
                train_imgs, test_imgs, train_poses, test_poses,
                incremental_dataset
            ])
