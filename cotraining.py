import os
from datetime import datetime
import ipdb
from options.opts import config_parser, add_proj_parser, prepare_proj_parser
import numpy as np
import imageio
import time
import torch
from torchvision.utils import save_image
from tqdm import tqdm, trange
import trimesh
from pathlib import Path
import mcubes

import matplotlib.pyplot as plt

from util import *

from models import *
from data import create_dataset

from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type("torch.cuda.FloatTensor")

now = datetime.now()
# set seed

# print = lambda x: print(*x, flush=True)


def main():
    parser = config_parser()
    parser = prepare_proj_parser(parser)
    parser = add_proj_parser(parser)

    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(args.torch_seed)

    # Load data

    (images, poses, render_poses, hwf, i_train, i_val, i_test,
     incremental_flags, near, far, c, data, img_Tensor, i_fixed, i_fixed_test,
     decoder_dataloader
     # decoder_imgs,
     # decoder_imgs_normalized,
     ) = create_dataset(args)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    args.chunk //= len(args.srn_encode_views_id.split())

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file

    args.expname = f"{len(i_train)}_{'viewdirs' if args.viewdirs_res else 'viewinput'}_{args.basedir.split('/')[-1]}id{args.srn_object_id}_{args.external_sampling}_{args.decoder_train_objs}_{args.expname}_{now.strftime('%b_%d_%H_%M')}"
    print("expname: {}".format(args.expname))

    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    if args.add_decoder:
        os.makedirs(os.path.join(basedir, expname, "AE"), exist_ok=True)
    # output_f = os.path.join(basedir, expname, "log.txt")
    # with open(output_f, "a") as train_log:
    #     train_log.write("training log")

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, "summaries", expname))

    f = os.path.join(basedir, expname, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, "config.txt")
        with open(f, "w") as file:
            file.write(open(args.config, "r").read())

    # Create nerf model
    network = NetworkSystem(args, list(images.shape[1:3]), device=device)
    render_kwargs_train, render_kwargs_test, optimizer, start = (
        network.render_kwargs_train,
        network.render_kwargs_test,
        network.optimizer,
        network.start,
    )

    global_step = start
    #

    bds_dict = {
        "near": near,
        "far": far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = todevice(render_poses)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    # Move training data to GPU
    images = todevice(images)
    poses = todevice(poses)

    if use_batching:
        # For random ray batching
        # print("get rays")
        rays = torch.stack(  # TODO
            [get_rays(H, W, focal, p, c) for p in poses[:, :3, :4]],
            0)  # [N, ro+rd, H, W, 3] #?

        rays_rgb = torch.cat([rays, images[:, None]],
                             1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = torch.stack([rays_rgb[i] for i in i_train],
                               0)  # train images only
        incremental_flags_indices = torch.nonzero(
            incremental_flags).squeeze()  # get nonzero indices
        incremental_flags = torch.zeros(rays_rgb.shape[:-1]).long()
        incremental_flags[incremental_flags_indices,
                          ...] = 1  # bool mask matrix

        rays_rgb = rays_rgb.view(-1, 3, 3)
        # rays_rgb = torch.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]

        rays_rgb = rays_rgb[torch.randperm(
            rays_rgb.size(0))]  # * shuffle along the first axis
        rays_rgb = todevice(rays_rgb)
        # print("done")
        i_batch = 0

    print("TEST views number: {}".format(len(i_test)))
    psnr_savedir = os.path.join(basedir, expname, 'psnr')

    os.makedirs(psnr_savedir, exist_ok=True)

    # coord_ij for later use
    coord_cartesian = {
        "regular":
        cartesian_coord(H, W, reshape=True),
        "precrop":
        cartesian_coord(H, W, precrop_frac=args.precrop_frac, reshape=True),
    }

    if start != args.epoch - 1:  # ease of use in test
        start += 1

    # print
    print("i_fixed: {}".format(i_fixed))
    print("i_fixed_test: {}".format(i_fixed_test))

    # TOOD, init encoder imgs
    selected_encode_idx = i_fixed

    for i in trange(start, args.epoch):

        # Sample random ray batch
        if use_batching:  # by default
            # Random over all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_incremental_flag = incremental_flags[i_batch:i_batch +
                                                       N_rand]
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]  # 138*400*40*3
            pose = poses[img_i, :3, :4]  # c2w matrices

            if N_rand is not None:  # 1024
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose),
                                          c)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:  # 500
                    coords_ij = coord_cartesian["precrop"]
                    if i == start:
                        print(
                            f"[Config] Center cropping is enabled until iter {args.precrop_iters}"
                        )
                else:
                    coords_ij = coord_cartesian["regular"]

                select_inds = np.random.choice(
                    coords_ij.shape[0],
                    size=[N_rand],
                    replace=False,  # select 1024 by default
                )  # (N_rand,)
                select_coords = coords_ij[select_inds].long(
                )  # (N_rand, 2) 1024*2 by default

                rays_o = rays_o[
                    select_coords[:, 0],
                    select_coords[:,
                                  1]]  # (N_rand, 3). #* the same for all dirs beloning to the same image
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[
                    select_coords[:, 0],
                    select_coords[:,
                                  1]]  # (N_rand, 3) #* color of pixels in original image

        #### TEST ####
        if i > 1 and (i != start and
                      (i % args.i_testset == 0 or i + 1 == args.epoch)):
            testsavedir = os.path.join(basedir, expname,
                                       "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)

            # encode on test_set

            if args.enc_type != "none" and not args.not_encode_test_views:
                network.encode(img_Tensor[i_fixed_test], poses[i_fixed_test],
                               focal)  # TODO

            rgbs, disps = render_path(
                # torch.Tensor(poses[i_test]).to(device) if type(poses)
                poses[i_test],
                hwf,
                args.chunk,
                render_kwargs_test,
                # gt_imgs=images[i_test],
                savedir=testsavedir,
            )
            # print("Saved test set")

            img_loss = img2mse(rgbs, images[i_test], keepdims=True)
            # trans = extras["raw"][..., -1]
            # calculate view-specific psnr
            psnr = mse2psnr(img_loss).cpu().numpy()

            np.save(
                os.path.join(basedir, expname,
                             "test_psnr_epoch_{:06d}".format(i)),
                psnr,
            )
            # TODO
            psnr_baseline = np.load(
                'logs/PIXNERF/9views_viewdirs_raw__id0_instance_1_srn_car_traintest_resnet_6_0_viewasinput/test_psnr_epoch_010001.npy'
            )
            plt.plot(list(range(psnr_baseline.shape[0])), psnr_baseline)[0]
            plt.plot(list(range(psnr.shape[0])), psnr)[0]
            plt.savefig(os.path.join(psnr_savedir, f"iter_{i}.png"))
            plt.clf()

            log = f"[TEST] Iter: {i} Loss: {img_loss.mean().item()}  PSNR: {psnr.mean()}\n"
            print(log)

            writer.add_scalar("Loss/Test", img_loss.mean().item(), i)
            writer.add_scalar("PSNR/Test", psnr.mean().item(), i)
            writer.add_images("test_images", rgbs[:5].permute(0, 3, 1, 2), i)

            # if i % args.i_testset == 0 and i > 0:
            #     # Turn on testing mode
            #     with torch.no_grad():
            #         rgbs, disps = render_path(
            #             render_poses, hwf, args.chunk, render_kwargs_test
            #         )
            rgbs, disps = map(lambda x: x.cpu().numpy(), (rgbs, disps))
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname,
                                     "{}_spiral_{:06d}_".format(expname, i))
            imageio.mimwrite(moviebase + "rgb.mp4",
                             to8b(rgbs),
                             fps=10,
                             quality=8)
            imageio.mimwrite(moviebase + "disp.mp4",
                             to8b(disps / np.max(disps)),
                             fps=30,
                             quality=8)

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

        # save ckpt
        if i > 1 and (i % args.i_weights == 0 or i + 1 == args.epoch):
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step":
                    global_step,
                    "network_fn_state_dict":
                    render_kwargs_train["network_fn"].state_dict(),
                    "network_fine_state_dict":
                    render_kwargs_train["network_fine"].state_dict(),
                    "optimizer_state_dict":
                    optimizer.state_dict(),
                    "encoder":
                    network.encoder.state_dict()
                    if network.encoder is not None else None,
                },
                path,
            )
            print("Saved checkpoints at", path)

        if i + 1 == args.epoch:
            break

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(
            H,
            W,
            focal,
            chunk=args.chunk,
            rays=batch_rays,
            verbose=i < 10,
            retraw=True,
            **render_kwargs_train,
        )

        optimizer.zero_grad()

        # img_loss = img2mse(
        #     rgb,
        #     target_s,
        # )

        if args.incremental_path != None:
            img_loss = img2mse(rgb, target_s, keepdim=True)
            img_loss = (img_loss * batch_incremental_flag).mean() + (
                img_loss * (1 - batch_incremental_flag)).mean() * args.w_gt
        else:
            img_loss = img2mse(rgb, target_s)

        psnr = mse2psnr(img_loss.mean().detach())

        # trans = extras["raw"][..., -1]
        loss = img_loss

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            # psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE Writing log to tensorboard
        writer.add_scalar("Loss/Train", img_loss.item(), i)
        writer.add_scalar("PSNR/Train", psnr.item(), i)

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate**(global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        global_step += 1

    writer.close()


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    main()