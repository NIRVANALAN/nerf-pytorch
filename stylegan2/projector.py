import argparse
import glob
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import shutil
import torch
from PIL import Image
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from . import lpips
from .model import Discriminator, Generator

now = datetime.now()


def load_nerf_psnr(
    psnr_path,
    nerf_pred_path,
    sampling_range=50,
    sampling_strategy=None,
    sampling_numbers=5,
    verbose=True,
):

    # nerf_pred_path = Path(psnr_path).parent.parent / ('testset_{}'.format(
    #     psnr_path.name.split('_')[-1]))
    
    # stage>1
    if psnr_path != None:
        assert psnr_path.exists()
        psnr = np.load(psnr_path)

        ids = np.argsort(psnr)
        if 'last' in sampling_strategy:  # sample from good to worse
            assert 'uniform' not in sampling_strategy
            ids = ids[::-1]

        if 'both' in sampling_strategy:
            ids = np.concatenate(
                (ids[:sampling_range / 2], ids[-sampling_range / 2:]), 0)
        else:
            ids = ids[:sampling_range]

        if 'uniform' in sampling_strategy:
            ids_to_proj = ids[::int(sampling_range / sampling_numbers)]

        elif 'both' in sampling_strategy:
            ids_to_proj = np.concatenate(
                (ids[:sampling_numbers], ids[-sampling_numbers:]), 0)
        else:
            ids_to_proj = ids[:sampling_numbers]

        if verbose:
            print('projection on {} indices: {}'.format(
                ids_to_proj.shape[0], ids_to_proj.tolist()),
                  flush=True)
        files_to_proj = [
            Path(nerf_pred_path) / '{:03}.png'.format(idx)
            for idx in ids_to_proj
        ]
    # stage 0
    else:
        if verbose:
            print('projection on train_imgs')

        files_to_proj = glob.glob(
            str(Path(nerf_pred_path).parent / 'stage0/*'))

    return files_to_proj


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss +
                (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) +
                (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latents: list, strength):
    latents_n = []

    # only add noise to optim_vars
    for latent in latents:
        if latent.requires_grad:
            noise = torch.randn_like(latent) * strength
            latents_n.append(latent + noise)
        else:
            latents_n.append(latent)

    # TODO
    if latents_n[1].shape != latents_n[0].shape:
        latents_n[1] = latents_n[1].expand_as(latents_n[0])  # expand

    return latents_n


def make_image(tensor):
    return (tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(
        torch.uint8).permute(0, 2, 3, 1).to("cpu").numpy())


device = "cuda"

#class Projector():
#    def __init__():
#        pass


def project(args, exp_basedir, stage=0):
    '''
    do inversion in stage_{n-1}, save in stage_{n}
    '''
    n_mean_latent = 10000

    resize = min(args.size, 256)

    output_dir = Path(exp_basedir) / f'stage{stage}'

    if stage==1 and args.stage0_path != None:
        inversion_dir = Path(args.stage0_path)
        # directly copy inversion results from stage1
        if not output_dir.exists():
            shutil.copytree(inversion_dir.parent / 'stage1', output_dir)
    else:
        inversion_dir = Path(exp_basedir) / f'stage{max(0, stage-1)}'


    if stage == 0:
        step = 1000
        test_psnr_path = None
        test_pred_path=inversion_dir
    else:
        step = args.step
        test_psnr_path = inversion_dir/ f'test_psnr.npy'
        test_pred_path = inversion_dir / 'testset'

    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    imgs = []

    if not output_dir.exists():
        output_dir.mkdir()


    files = load_nerf_psnr(test_psnr_path, test_pred_path, args.sampling_range,
                           args.sampling_strategy, args.sampling_numbers)

    for imgfile in files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() /
                      n_mean_latent)**0.5

    if args.d_loss:
        D = Discriminator(args.size).cuda()
        ckpt_d = torch.load(args.ckpt,
                            map_location=lambda storage, loc: storage)['d']
        D.load_state_dict(ckpt_d)
        percept = lpips.DiscriminatorLoss(D, 4, False)
    else:
        percept = lpips.PerceptualLoss(model="net-lin",
                                       net="vgg",
                                       use_gpu=device.startswith("cuda"))

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    # prepare for latent code(s)
    # shared id code + independent pose code
    proj_latent_filepath = output_dir / 'proj_latent.pt'

    if Path(proj_latent_filepath).exists():
        return output_dir

    if stage > 0:
        # id_aware by default
        #TODO
        if args.stage0_path != None:
            proj_latent_stage0 = Path(args.stage0_path) / 'proj_latent.pt'
        else:
            proj_latent_stage0 = output_dir.parent / 'stage0' / 'proj_latent.pt'

        saved_result = torch.load(proj_latent_stage0)

        img_keys = list(saved_result.keys())[:-1]

        latent_poses = [
            saved_result[img_name]['latent_pose'] for img_name in img_keys
        ]
        latent_id = saved_result['latent_id'].squeeze().repeat(
            1, 1).detach().clone()
        # select Nearest code?
        latent_pose = torch.stack(latent_poses).mean(0).detach().clone()
        latent_pose = latent_pose.repeat(imgs.shape[0], 1)
    else: # original inversion
        latent_in = latent_mean.detach().clone().unsqueeze(0)
        latent_pose = latent_in.repeat(imgs.shape[0],
                                       1)  # independent pose code
        latent_id = latent_in.repeat(1, 1)  # shared identity code

        latent_id.requires_grad = True

    latent_pose.requires_grad = True
    latents = [latent_pose, latent_id]

    # also search noise
    for noise in noises:
        noise.requires_grad = True

    # optim
    optimizer = optim.Adam(latents + noises, lr=args.lr)

    # core loop
    pbar = tqdm(range(step))
    latent_path = []
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(
            0, 1 - t / args.noise_ramp)**2

        # add stochastic noise to latent_in
        latents_n = latent_noise(latents, noise_strength.item())
        # latent_n = g_ema.style_forward(latent_n, skip=9 - args.f1_d)

        # inference GAN
        assert isinstance(latents_n, (list))
        img_gen, _ = g_ema(latents_n,
                           input_is_latent=True,
                           noise=noises,
                           inject_index=args.inject_index)

        # loss funcs
        batch, channel, height, width = img_gen.shape
        if height > 256:  # resize to 256 for VGG loss computation
            factor = height // 256

            img_gen = img_gen.reshape(batch, channel, height // factor, factor,
                                      width // factor, factor)
            img_gen = img_gen.mean([3, 5])

        if args.d_loss:
            p_loss = 0
            for idx in range(0, len(img_gen), 4):  # d_loss bs divisible by 4
                p_loss += percept(img_gen[idx:idx + 4], imgs[idx:idx + 4])
        else:
            p_loss = percept(img_gen, imgs).sum()
        if args.normalize_vgg_loss:
            p_loss /= imgs.shape[0]
        n_loss = noise_regularize(noises)
        l2_loss = F.mse_loss(img_gen, imgs)

        l1_loss = F.l1_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.w_l1 * l1_loss + args.w_l2 * l2_loss

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        # log
        if (i + 1) % 10 == 0:
            latent_path.append(
                [latent.detach().clone() for latent in latents_n])
        pbar.set_description((
            f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
            f" mse: {l2_loss.item():.4f};  l1: {l1_loss.item():.4f}; lr: {lr:.4f}"
        ))

    # get last result
    img_gen, _ = g_ema(latent_path[-1],
                       input_is_latent=True,
                       noise=noises,
                       inject_index=args.inject_index)

    # prepare save path
    # path_base = stage_expdir

    # path_base = path_base / (
    #     now.strftime('%b%d_%H_%M') + '_{}_{}_{}_noise{}'.format(
    #         args.step, args.sampling_strategy, args.sampling_numbers,
    #         args.noise))  # timestamp + desc

    # print('saving to path_base: {}'.format(path_base))
    # if not path_base.exists():
    #     path_base.mkdir(parents=True)

    # save log
    with open(output_dir / 'loss.log', 'w') as f:
        f.write(
            f"perceptual: {p_loss.item():.4f};\n noise regularize: {n_loss.item():.4f};\n"
            f" mse: {l2_loss.item():.4f};\n  l1: {l1_loss.item():.4f}; lr: {lr:.4f}\n"
        )
    # save results
    img_ar = make_image(img_gen)
    result_file = {}

    for i, input_name in enumerate(files):
        filename = '{}'.format(os.path.splitext(os.path.basename(files[i]))[0])
        # args.inject_index,
        # len(files))

        noise_single = []
        for noise in noises:
            noise_single.append(noise[i:i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "noise": noise_single,
        }

        result_file[input_name].update({"latent_pose": latent_pose[i]})

        img_name = output_dir / '{}.png'.format(filename)
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

    result_file.update({
        "latent_id": latent_id[0],
    })

    torch.save(result_file, proj_latent_filepath)
    return output_dir
