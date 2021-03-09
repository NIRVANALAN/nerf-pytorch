__author__ = "yslan"

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm, trange

import numpy as np
from models.networks import *
from data.data_util import get_split_dataset

from util.util import image_to_normalized_tensor, to_img

dir_prefix = "debug"

if not os.path.exists(f"{dir_prefix}/dc_img"):
    os.mkdir(f"{dir_prefix}/dc_img")

import argparse

parser = argparse.ArgumentParser("AE Parser")
parser.add_argument("-pt", action="store_true")
parser.add_argument("-objs", type=int, default=3, help="train on how many instaces")
parser.add_argument("-mlp_render", action="store_true")

args = parser.parse_args()
print(args.pt)

model_suffix = "pt" if args.pt else "no_pt"


num_epochs = 1000
batch_size = 64
learning_rate = 5e-4

# img_transform = transforms.Compose([
#    transforms.ToTensor(),
#    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# img_transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
# )

# dataset = MNIST(f"{dir_prefix}/data", transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# dataset

dataset_dir = "/mnt/lustre/yslan/Repo/NVS/Projects/volume_rendering/srn_dataset/cars"
dataset = get_split_dataset(
    "srn",
    dataset_dir,
    split=["train"]
    # "train_test"]
)[0]


instances = [dataset[i] for i in range(args.objs)]

images = torch.cat([i["images"][:] for i in instances], 0)


normalized_img_tensor = image_to_normalized_tensor(  # TODO
    images.permute(0, 3, 1, 2)
).float()  # 250 * 3 * 128 * 128

dataset_size = images.shape[0]


input_nc = images.shape[-1]
output_nc = input_nc
ngf = 64
norm = "batch"
use_dropout = True

norm_layer = get_norm_layer(norm)

# Net = ResnetGenerator(
#     input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6
# )

from models.encoder import SpatialEncoder

Net = SpatialEncoder(
    pretrained=args.pt, add_decoder=True, mlp_render=args.mlp_render
)  # default params

model_dir = f"{Net.__class__.__name__}/{model_suffix}_{args.objs}_mlp-{args.mlp_render}"
print(model_dir)


os.makedirs(f"{dir_prefix}/{model_dir}/dc_img", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
model = Net.to(device)

summary(model, (3, 128, 128))

# model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999)
)


for epoch in range(num_epochs):
    # for _, data in enumerate(tqdm(dataloader)):
    for iter in range(0, dataset_size, batch_size):
        img_i = np.random.choice(dataset_size, size=batch_size)
        img = normalized_img_tensor[img_i]
        # img, _ = data
        img = img.cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print("epoch [{}/{}], loss:{:.5f}".format(epoch + 1, num_epochs, loss.item()))
    if epoch % 50 == 0:
        pic = to_img(output.detach().cpu(), img_size=img.shape[2], channel=img.shape[1])
        save_image(
            pic,
            "./{}/{}/dc_img/image_{}.png".format(dir_prefix, model_dir, epoch),
        )

torch.save(model.state_dict(), f"./{dir_prefix}/{model_dir}.pth")