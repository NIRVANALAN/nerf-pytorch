import os
import ipdb
from opts import config_parser
import numpy as np
import imageio
import time
import torch
from tqdm import tqdm, trange
import trimesh
import mcubes

import matplotlib.pyplot as plt

from util import *

from models import *
from data import create_dataset
import gc

torch.set_default_tensor_type("torch.cuda.FloatTensor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

# print = lambda x: print(*x, flush=True)

todevice = (
    lambda x: x.to(device) if type(x) is torch.Tensor else torch.Tensor(x).to(device)
)  # for compatibility. torch.Tensor(tensor) will fail if tensor already on cuda


def extract_mesh(render_kwargs, mesh_grid_size=80, threshold=50):
    network_query_fn, network = (
        render_kwargs["network_query_fn"],
        render_kwargs["network_fine"],
    )
    device = next(network.parameters()).device

    with torch.no_grad():
        points = np.linspace(-1, 1, mesh_grid_size)
        query_pts = (
            torch.tensor(
                np.stack(np.meshgrid(points, points, points), -1).astype(np.float32)
            )
            .reshape(-1, 1, 3)
            .to(device)
        )
        viewdirs = torch.zeros(query_pts.shape[0], 3).to(device)

        output = network_query_fn(query_pts, viewdirs, network)  # TODO

        grid = output[..., -1].reshape(mesh_grid_size, mesh_grid_size, mesh_grid_size)

        print("fraction occupied:", (grid > threshold).float().mean())

        vertices, triangles = mcubes.marching_cubes(
            grid.detach().cpu().numpy(), threshold
        )
        mesh = trimesh.Trimesh(vertices, triangles)

    return mesh


def main():
    parser = config_parser()
    args = parser.parse_args()
    # Load data

    (
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
        img_Tensor,
        i_fixed,
    ) = create_dataset(args)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    args.chunk //= len(args.srn_encode_views_id.split())

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    output_f = os.path.join(basedir, expname, "log.txt")
    with open(output_f, "a") as train_log:
        train_log.write("training log")

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

    basedir = args.basedir
    expname = args.expname

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

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print("RENDER ONLY")
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(
                basedir,
                expname,
                "renderonly_{}_{:06d}".format(
                    "test" if args.render_test else "path", start
                ),
            )
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", render_poses.shape)

            rgbs, _ = render_path(
                render_poses,
                hwf,
                args.chunk,
                render_kwargs_test,
                gt_imgs=images,
                savedir=testsavedir,
                render_factor=args.render_factor,
            )
            print("Done rendering", testsavedir)
            imageio.mimwrite(
                os.path.join(testsavedir, "video.mp4"), to8b(rgbs), fps=30, quality=8
            )

            return

    # Short circuit if only extracting mesh from trained model
    if args.mesh_only:
        mesh = extract_mesh(
            render_kwargs_test, mesh_grid_size=args.mesh_grid_size, threshold=50
        )

        testsavedir = os.path.join(
            basedir,
            expname,
            "renderonly_{}_{:06d}".format(
                "test" if args.render_test else "path", start
            ),
        )
        os.makedirs(testsavedir, exist_ok=True)
        path = os.path.join(testsavedir, "mesh.obj")

        print("saving mesh to ", path)

        mesh.export(path)

        return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    # Move training data to GPU
    images = todevice(images)
    poses = todevice(poses)

    if use_batching:
        # For random ray batching

        print("get rays")
        # rays = np.stack(
        #     [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]], 0
        # )  # [N, ro+rd, H, W, 3] #?
        # print("done, concats")
        # rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        # rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        # rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        # rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        # rays_rgb = rays_rgb.astype(np.float32)
        # print("shuffle rays")
        # np.random.shuffle(rays_rgb)

        # torch version

        rays = torch.stack(  # TODO
            [get_rays(H, W, focal, p, c) for p in poses[:, :3, :4]], 0
        )  # [N, ro+rd, H, W, 3] #?

        print("done, concats")
        rays_rgb = torch.cat([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = torch.stack([rays_rgb[i] for i in i_train], 0)  # train images only

        rays_rgb = rays_rgb.view(-1, 3, 3)
        # rays_rgb = torch.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]

        print("shuffle rays")
        # rays_rgb = rays_rgb.astype(np.float32)
        # np.random.shuffle(rays_rgb) # * shuffle along the first axis
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.size(0))]
        # # to cuda

        rays_rgb = todevice(rays_rgb)

        print("done")
        i_batch = 0

    print("Begin")
    print("TRAIN views are {}".format(i_train))
    print("TEST views {}".format(i_test))
    # print("VAL views are", i_val)  # TODO

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    if start != args.epoch - 1:  # ease of use in test
        start += 1
    for i in trange(start, args.epoch):

        #! encoder
        if args.enc_type != "none" and i == start or args.enable_encoder_grad:
            assert img_Tensor != None
            network.encode(img_Tensor[i_fixed], poses[i_fixed], focal)  # TODO

        # Sample random ray batch

        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch : i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]  # 138*400*40*3
            pose = poses[img_i, :3, :4]  # c2w matrices

            if N_rand is not None:  # 1024
                rays_o, rays_d = get_rays(
                    H, W, focal, torch.Tensor(pose), c
                )  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:  # 500
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords_ij = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                        ),
                        -1,
                    )
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}"
                        )
                else:
                    coords_ij = torch.stack(
                        torch.meshgrid(
                            torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)
                        ),
                        -1,
                    )  # stack in 'ij' indexing order. -> (H, W, 2)

                # coords_ij[i,j] = [i,j]
                coords_ij = torch.reshape(coords_ij, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(
                    coords_ij.shape[0],
                    size=[N_rand],
                    replace=False,  # select 1024 by default
                )  # (N_rand,)
                select_coords = coords_ij[
                    select_inds
                ].long()  # (N_rand, 2) 1024*2 by default

                rays_o = rays_o[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3). #* the same for all dirs beloning to the same image
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[
                    select_coords[:, 0], select_coords[:, 1]
                ]  # (N_rand, 3) #* color of pixels in original image

        #### TEST ####
        if i > 1 and (i % args.i_testset == 1 or i + 1 == args.epoch):
            testsavedir = os.path.join(basedir, expname, "testset_{:06d}".format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print("test poses shape", poses[i_test].shape)

            rgbs, disps = render_path(
                # torch.Tensor(poses[i_test]).to(device) if type(poses)
                poses[i_test],
                hwf,
                args.chunk,
                render_kwargs_test,
                # gt_imgs=images[i_test],
                savedir=testsavedir,
            )
            print("Saved test set")

            img_loss = img2mse(rgbs, images[i_test], keepdims=True)
            # trans = extras["raw"][..., -1]
            psnr = mse2psnr(img_loss)
            np.save(
                os.path.join(basedir, expname, "test_psnr_epoch_{:06d}".format(i)),
                psnr.cpu().numpy(),
            )
            log = f"[TEST] Iter: {i} Loss: {img_loss.mean().item()}  PSNR: {psnr.mean().item()}\n"
            print(log)
            with open(output_f, "a") as f:
                f.write(log)

            # if i % args.i_testset == 0 and i > 0:
            #     # Turn on testing mode
            #     with torch.no_grad():
            #         rgbs, disps = render_path(
            #             render_poses, hwf, args.chunk, render_kwargs_test
            #         )
            rgbs, disps = map(lambda x: x.cpu().numpy(), (rgbs, disps))
            print("Done, saving", rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, "{}_spiral_{:06d}_".format(expname, i)
            )
            imageio.mimwrite(moviebase + "rgb.mp4", to8b(rgbs), fps=10, quality=8)
            imageio.mimwrite(
                moviebase + "disp.mp4", to8b(disps / np.max(disps)), fps=30, quality=8
            )
            if i == args.epoch:
                break

            # if args.use_viewdirs:
            #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
            #     with torch.no_grad():
            #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
            #     render_kwargs_test['c2w_staticcam'] = None
            #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)
        #####  Core optimization loop  #####
        with profiler.record_function("render"):
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
        img_loss = img2mse(rgb, target_s)
        # trans = extras["raw"][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb0" in extras:
            img_loss0 = img2mse(extras["rgb0"], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lrate
        ################################

        # Rest is logging
        if i > 1 and (i % args.i_weights == 1 or i == args.epoch - 1):
            path = os.path.join(basedir, expname, "{:06d}.tar".format(i))
            torch.save(
                {
                    "global_step": global_step,
                    "network_fn_state_dict": render_kwargs_train[
                        "network_fn"
                    ].state_dict(),
                    "network_fine_state_dict": render_kwargs_train[
                        "network_fine"
                    ].state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoints at", path)

        gc.collect()
        if i % args.i_print == 0:
            # TODO
            log = f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}\n"
            tqdm.write(log)
            with open(output_f, "a") as f:
                f.write(log)
        global_step += 1


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # with profiler.profile(profile_memory=True, use_cuda=True) as prof: # memory leak here
    main()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
