import os
import sys
import time
from datetime import datetime
from pathlib import Path

import imageio
import ipdb
import matplotlib.pyplot as plt
import mcubes
import numpy as np
import torch
import imageio
import trimesh
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm, trange

from data import create_dataset
from models import *
from options.opts import add_proj_parser, config_parser, prepare_proj_parser
from stylegan2.projector import project
from util import *
from torch.utils.data import DataLoader

torch.set_default_tensor_type("torch.cuda.FloatTensor")

now = datetime.now()


def main():
    parser = config_parser()
    parser = prepare_proj_parser(parser)
    parser = add_proj_parser(parser)

    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(args.torch_seed)

    # Load data
    (images, poses, render_poses, hwf, i_train, i_val, i_test, near, far, c,
     data, img_Tensor, i_fixed, i_fixed_test, meta_data) = create_dataset(args)
    (train_imgs, test_imgs, train_poses, test_poses,
     incremental_dataset) = meta_data

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    args.chunk //= len(args.srn_encode_views_id.split())

    if args.render_test:
        render_poses = test_poses

    # Create log dir and copy the config file
    args.expname = f"{train_imgs.shape[0]}_{'viewdirs' if args.viewdirs_res else 'viewinput'}_{args.basedir.split('/')[-1]}id{args.srn_object_id}_{args.external_sampling}_{args.decoder_train_objs}_{args.expname}"
    # _{now.strftime('%b_%d_%H_%M')}
    print("expname: {}".format(args.expname))

    basedir = args.basedir
    expname = args.expname

    exp_basedir = Path(basedir) / expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # copy train_imgs for first projection
    train_img_dir = exp_basedir / 'stage0'
    train_img_dir.mkdir(exist_ok=True)
    for i in range(train_imgs.shape[0]):
        img = train_imgs[i].cpu().numpy()
        imageio.imwrite(train_img_dir / f'{i:06}.png', img)

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
    network = NetworkSystem(args, list(train_imgs.shape[1:3]), device=device)
    render_kwargs_train, render_kwargs_test, optimizer, start = (
        network.render_kwargs_train,
        network.render_kwargs_test,
        network.optimizer,
        network.start,
    )

    global_step = start

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

    # prepare coord_ij for later use
    coord_cartesian = {
        "regular":
        cartesian_coord(H, W, reshape=True),
        "precrop":
        cartesian_coord(H, W, precrop_frac=args.precrop_frac, reshape=True),
    }

    # Move training data to GPU
    train_imgs, train_poses, test_imgs, test_poses = list(
        map(todevice, [train_imgs, train_poses, test_imgs, test_poses]))

    if use_batching:
        #     # For random ray batching
        #     # print("get rays")
        rays = torch.stack(  # TODO
            [get_rays(H, W, focal, p, c) for p in train_poses[:, :3, :4]],
            0)  # [N, ro+rd, H, W, 3] #?

        rays_rgb = torch.cat([rays, train_imgs[:, None]],
                             1)  # [N, ro+rd+rgb, H, W, 3]

        rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)  # [N, H, W, ro+rd+rgb, 3]
        # rays_rgb = torch.stack([rays_rgb[i] for i in i_train],
        #                        0)  # train images only

        # rays_rgb = rays_rgb.view(-1, 3, 3)
        rays_rgb = torch.reshape(rays_rgb,
                                 [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        incremental_flags = torch.zeros(rays_rgb.shape[:-1]).long()

        # shuffle
        perm_randidx = torch.randperm(rays_rgb.size(0))
        rays_rgb = rays_rgb[perm_randidx]
        incremental_flags = incremental_flags[perm_randidx]

        i_batch = 0

    print("TEST views number: {}".format(test_imgs.shape[0]))
    psnr_savedir = os.path.join(basedir, expname, 'psnr_vis')

    os.makedirs(psnr_savedir, exist_ok=True)

    start_stage = (start+1) // args.cotraining_epoch

    # cotraining start
    # stage 0 -> initialize NeRF, Inversion Code 
    # stage>0: incrementally add data
    for stage in range(start_stage, args.cotraining_stage + 1):

        # inversion 
        stage_exp_dir = project(args, exp_basedir, stage)

        if stage > 0:
            # add incremental samples
            incremental_dataset.incremental_update_data(stage_exp_dir)

        # rebuild dataloader
        print('stage: {}, incremental_dataset size: {}'.format(stage, len(incremental_dataset)))
        train_dataloader = DataLoader(incremental_dataset,
                                    batch_size=N_rand,
                                    shuffle=True,
                                    num_workers=0,
                                    drop_last=True)
        # NeRF training
        for i in trange(0, args.cotraining_epoch):

            # Sample random ray batch
            if use_batching:  # by default
                batch = next(iter(train_dataloader))

                batch, batch_flags = map(todevice, batch)
                batch = torch.transpose(batch, 0, 1)  # 2 * BS * 3
                batch_rays, target_s = batch[:2], batch[2]

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

            # higher weight for original GT samples
            if args.w_gt != 1:
                img_loss = img2mse(rgb, target_s, keepdim=True)
                img_loss = (img_loss * batch_flags).mean() + (
                    img_loss * (1 - batch_flags)).mean() * args.w_gt
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

        #### TEST ####
        testsavedir = stage_exp_dir / "testset".format(stage)
        os.makedirs(testsavedir, exist_ok=True)
        print("test poses shape", test_poses.shape)
        # encode on test_set

        rgbs, disps = render_path(
            test_poses[::args.test_interv],
            hwf,
            args.chunk,
            render_kwargs_test,
            savedir=testsavedir,
        )

        img_loss = img2mse(rgbs, test_imgs[::args.test_interv], test=True)
        psnr = mse2psnr(img_loss).cpu().numpy()
        np.save(stage_exp_dir / 'test_psnr.npy', psnr)

        # TODO
        if stage>0:
            psnr_baseline = np.load(exp_basedir / 'stage0' /  f"test_psnr.npy")
            plt.plot(list(range(psnr_baseline.shape[0])), psnr_baseline)[0]
        plt.plot(list(range(psnr.shape[0])), psnr)[0]
        plt.savefig(os.path.join(psnr_savedir, f"stage_{stage}.png"))
        plt.clf()

        log = f"[TEST] Iter: {i} Loss: {img_loss.mean().item()}  PSNR: {psnr.mean()}\n"
        print(log)

        writer.add_scalar("Loss/Test", img_loss.mean().item(), i)
        writer.add_scalar("PSNR/Test", psnr.mean().item(), i)
        writer.add_images("test_images", rgbs[:5].permute(0, 3, 1, 2), i)

        # save also to log file

        f = os.path.join(basedir, expname, "log.txt")
        with open(f, "w+") as file:
            file.write(log)
        # save ckpt
        path = os.path.join(basedir, expname, "stage_{:02d}.tar".format(stage))
        torch.save(
            {
                'stage':
                stage,
                "global_step":
                global_step,
                "network_fn_state_dict":
                render_kwargs_train["network_fn"].state_dict(),
                "network_fine_state_dict":
                render_kwargs_train["network_fine"].state_dict(),
                "optimizer_state_dict":
                optimizer.state_dict(),
            },
            path,
        )
        print("Saved checkpoints at", path)

        # save video
        rgbs, disps = map(lambda x: x.cpu().numpy(), (rgbs, disps))
        print("Done, saving", rgbs.shape, disps.shape)
        moviebase = stage_exp_dir / "spiral_{:06d}_".format(global_step)
        imageio.mimwrite(str(moviebase) + "rgb.mp4",
                         to8b(rgbs),
                         fps=10,
                         quality=8)

    writer.close()


if __name__ == "__main__":
    main()
