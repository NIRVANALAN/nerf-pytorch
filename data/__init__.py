import ipdb
from .load_blender import *
from .load_deepvoxels import *
from .load_llff import *
import os

from .data_util import get_split_dataset, # ColorJitterDataset
from util import image_to_normalized_tensor

# __all__=[*]

def create_dataset(args):
    c, data, normalized_img_tensor, i_fixed, i_fixed_test = None, None, None, None, None
    decoder_images, decoder_images_normalized = None, None
    if args.dataset_type not in ("llff", "blender", "deepvoxels"):
        dataset = get_split_dataset(
            args.dataset_type, args.datadir, split=["train", "train_test"]
        )

        if args.add_decoder:
            if args.decoder_dataset not in args.datadir:
                decoder_dir = args.datadir.replace(
                    args.datadir.split("/")[-1], args.decoder_dataset
                )
                dataset_decoder = get_split_dataset(
                    args.dataset_type, decoder_dir, split=["train"]
                )
            else:
                dataset_decoder = dataset
            decoder_instances = [
                dataset_decoder[0][i] for i in range(args.decoder_train_objs)
            ]  # TODO
            decoder_images = torch.cat(
                [i["images"][:] for i in decoder_instances], 0
            ).float()

            decoder_images_normalized = image_to_normalized_tensor(  # TODO
                decoder_images.permute(0, 3, 1, 2)
            ).float()  # 250 * 3 * 128 * 128
            pass

        # dataset = get_split_dataset(
        #     args.dataset_type, args.datadir, split=["train_test"]
        # )

        instance = [s[args.srn_object_id] for s in dataset]
        views_avai = instance[0]["images"].shape[0]
        test_views_avai = instance[-1]["images"].shape[0]

        if len(args.srn_input_views_id.split()) > 1:
            # if len(list(map(int, args.srn_input_views_id.split())))
            # if args.srn_input_views <= 2:
            input_views_ids = np.array(list(map(int, args.srn_input_views_id.split())))
        else:
            input_views_ids = np.linspace(
                0, views_avai, args.srn_input_views, False, dtype=int
            )
        # np.array(range(views_avai)) [ :: views_avai // args.srn_input_views ]

        # extend fixed encode views
        # if args.enc_type != 'none' :
        #     encode_view_ids = list(map(int(args.srn_encode_views_id)))
        #     for enc_id in encode_view_ids:
        #         if enc_id not in input_views_ids:
        #             input_views_ids.extend(encode_view_ids)
        # input_views_ids = np.array(input_views_ids)

        test_views = np.linspace(1, test_views_avai, test_views_avai, False, dtype=int)
        # np.random.choice(np.arange(views_avai), size=args.srn_input_views)
        # input_views_id = (0, 38)  # try input views id = [0, 38]

        input_imgs = instance[0]["images"][input_views_ids, ...]
        test_imgs = instance[-1]["images"][test_views, ...]
        images = torch.cat([input_imgs, test_imgs], dim=0).float()

        normalized_img_tensor = image_to_normalized_tensor(  # TODO
            images.permute(0, 3, 1, 2)
        ).float()  # 250 * 3 * 128 * 128

        i_test = np.arange(len(input_views_ids), images.size(0)).tolist()
        i_train = np.arange(0, len(input_views_ids)).tolist()
        i_val = None

        poses = torch.cat(
            # poses = np.concatenate(  # TODO
            [
                instance[0]["poses"][input_views_ids, ...],
                # instance[1]["poses"][::test_idx_interv],
                instance[-1]["poses"][test_views, ...],
            ],
            # dim=0,
        )

        poses = poses[:, :3, :4]  #!

        # create fixed poses

        i_fixed = input_views_ids[
            list(map(int, args.srn_encode_views_id.split()))
        ]  # todo
        print("i_fixed: {}".format(i_fixed))

        # encode test views if set
        if args.encode_test_views:
            i_fixed_test = list(map(int, args.srn_encode_test_views_id.split()))
            if i_fixed_test[0] == -1:
                i_fixed_test = (
                    i_train  # same idx of train split to test views for encoding
                )

        render_poses = instance[-1]["poses"][test_views, ...]
        # instance[1]["poses"][::test_idx_interv]
        # render_poses = render_poses[:, :3, :4]  #!

        near, far = dataset[0].z_near, dataset[0].z_far
        hwf = [*input_imgs.shape[1:3], instance[0]["focal"]]
        c = instance[0]["c"]
        data = instance[0]
        print("object id: {} | input view id: {}".format(data["path"], input_views_ids))

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
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print("Auto LLFF holdout,", args.llffhold)
            i_test = np.arange(images.shape[0])[:: args.llffhold]

        i_val = i_test
        i_train = np.array(
            [
                i
                for i in np.arange(int(images.shape[0]))
                if (i not in i_test and i not in i_val)
            ]
        )

        print("DEFINING BOUNDS")
        if args.no_ndc:
            near = np.ndarray.min(bds) * 0.9
            far = np.ndarray.max(bds) * 1.0

        else:
            near = 0.0
            far = 1.0

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip
        )
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.0
        far = 6.0

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":

        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip
        )

        print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.0
        far = hemi_R + 1.0

    else:
        raise NotImplementedError("Unknown dataset type", args.dataset_type, "exiting")
    print("NEAR FAR", near, far)

    return (
        images,
        poses,
        render_poses,
        hwf,
        i_train,
        i_val,
        i_test,
        near,
        far,
        c,
        data,
        normalized_img_tensor,
        i_fixed,
        i_fixed_test,
        decoder_images,
        decoder_images_normalized,
    )
