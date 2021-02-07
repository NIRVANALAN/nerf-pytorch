import os
import ipdb
from options.opts import config_parser
import numpy as np
import imageio
import time
import torch
from torchvision.utils import save_image
from tqdm import tqdm, trange
import trimesh
import mcubes

import matplotlib.pyplot as plt

from util import *

from models import *
from data import create_dataset

from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type("torch.cuda.FloatTensor")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set seed


# print = lambda x: print(*x, flush=True)

todevice = (
    lambda x: x.to(device) if type(x) is torch.Tensor else torch.Tensor(x).to(device)
)  # for compatibility. torch.Tensor(tensor) will fail if tensor already on cuda


def main():
    parser = config_parser()
    args = parser.parse_args()
    # torch.set_deterministic(True)
    np.random.seed(0)
    if args.torch_seed != -1:
        torch.manual_seed(args.torch_seed)
        print("torch random seed: {}".format(args.torch_seed))
    else:
        print("ignore torch.manul_seed")

    #! AE
    assert args.add_decoder

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
        i_fixed_test,
        decoder_dataloader
        # decoder_imgs,
        # decoder_imgs_normalized,
    ) = create_dataset(args)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # print
    print("i_fixed: {}".format(i_fixed))
    print("i_fixed_test: {}".format(i_fixed_test))

    args.chunk //= len(args.srn_encode_views_id.split())

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file

    args.expname = f"{args.srn_input_views}views_{'raw' if not args.add_decoder else 'Y' if args.mlp_render else 'X'}_{args.basedir.split('/')[-1]}_id{args.srn_object_id}_{args.external_sampling}_{args.expname}"
    print("expname: {}".format(args.expname))

    basedir = args.basedir
    expname = args.expname

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    os.makedirs(os.path.join(basedir, expname, "AE"), exist_ok=True)

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

    if start != args.epoch - 1:  # ease of use in test
        start += 1

    criterion_ae = nn.MSELoss()

    for i in range(start, args.epoch):

        for idx, data in enumerate(tqdm(decoder_dataloader)):

            img_ae = data[0]

            img_ae = img_ae.cuda()

            # ===================forward=====================
            output_ae = network.encoder(img_ae, encode=False)[..., :3].permute(
                0, 3, 1, 2
            )
            ae_loss = criterion_ae(output_ae, img_ae)

            # tqdm.set_description("ae loss: {}".format(ae_loss.item()))
            # tqdm.refresh()  # to show immediately the update

            if global_step % 100 == 0:
                print(ae_loss.item())

            optimizer.zero_grad()

            loss = ae_loss

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!

            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay

            new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate
            ################################

            global_step += 1

        path = os.path.join(basedir, expname, "{:03d}.tar".format(i))
        torch.save(
            {
                "encoder": network.encoder.state_dict(),
                "iter": i,
                "global_step": global_step,
            },
            path,
        )
        print("Saved checkpoints at", path)


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    main()
