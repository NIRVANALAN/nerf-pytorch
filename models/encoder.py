"""
Implements image encoders
"""
from models.networks import ResnetBlock
import torch
from torch import nn
from torch._C import import_ir_module_from_buffer
import torch.nn.functional as F
import torchvision
import torch.autograd.profiler as profiler

from models import resnetfc


class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
        add_decoder=False,
        mlp_render_net=None,
        decoder_activation_layer=nn.Sigmoid(),
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        # norm_layer = util.get_norm_layer(norm_type)

        print(
            "Using torchvision",
            backbone,
            "encoder",
            "Pretrained: {}".format(pretrained),
        )
        self.model = getattr(torchvision.models, backbone)(
            pretrained=pretrained,
            # norm_layer=norm_layer
        )
        # Following 2 lines need to be uncommented for older configs
        self.model.fc = nn.Sequential()
        self.model.avgpool = nn.Sequential()
        self.mlp_render = mlp_render_net is not None

        # remove unused layers #TODO
        # self.model.layer

        # layer1-4

        self.latent_size = [0, 64, 128, 256, 512, 1024][
            num_layers
        ]  #! 64 + 64 + 128 + 256

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False
        )
        # self.latent (B, L, H, W)

        if add_decoder:
            decoder_model = []
            ngf = 64

            if self.mlp_render:
                filters = [256, 256, 256, 256, 256]
            else:
                filters = [256, 128, 64, 64, 64]
            use_bias = norm_type == "instance"
            n_downsampling = 4

            for i in range(n_downsampling):  # add upsampling layers
                # mult = 2 ** (n_downsampling - i)
                decoder_model += [
                    nn.ConvTranspose2d(
                        filters[i],
                        # ngf * mult,
                        # int(ngf * mult / 2),
                        filters[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    nn.BatchNorm2d(
                        filters[i + 1], affine=True, track_running_stats=True
                    ),
                    # norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            decoder_model = [nn.Sequential(*decoder_model)]  # conv decoder

            if mlp_render_net is not None:
                if mlp_render_net == "new_blk":
                    decoder_model += [  # DEBUG
                        resnetfc.ResnetBlockFC(ngf, 3, spatial_output=True)
                    ]
                else:
                    decoder_model += mlp_render_net
            else:  # default output net
                decoder_model += [
                    nn.Sequential(
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(ngf, 3, kernel_size=7, padding=0),
                    )
                ]

            decoder_model += [decoder_activation_layer]  # TODO
            self.decoder = nn.ModuleList(decoder_model)
        else:
            self.decoder = None

    def forward(self, x, encode=True):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            if encode:
                latents = [x]
                if self.num_layers > 1:
                    if self.use_first_pool:
                        x = self.model.maxpool(x)
                    x = self.model.layer1(x)
                    latents.append(x)
                if self.num_layers > 2:
                    x = self.model.layer2(x)
                    latents.append(x)
                if self.num_layers > 3:
                    x = self.model.layer3(x)
                    latents.append(x)
                if self.num_layers > 4:  # layer 5
                    x = self.model.layer4(x)
                    latents.append(x)

                self.latents = latents
                align_corners = None if self.index_interp == "nearest " else True
                latent_sz = latents[0].shape[-2:]
                for i in range(len(latents)):
                    latents[i] = F.interpolate(
                        latents[i],
                        latent_sz,
                        mode=self.upsample_interp,
                        align_corners=align_corners,
                    )
                self.latent = torch.cat(latents, dim=1)
                self.latent_scaling[0] = self.latent.shape[-1]
                self.latent_scaling[1] = self.latent.shape[-2]
                self.latent_scaling = (
                    self.latent_scaling / (self.latent_scaling - 1) * 2.0
                )
                # return self.latent  # 512 * w/2 * w/2
            else:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                # x = self.model.layer4(x)

        if self.decoder != None and not encode:
            x = self.decoder[0](x)  # deconv
            if self.mlp_render:  # Y
                x = self.decoder[1](x)  # mlp renderer RESBLKS
                B, C, H, W = x.shape[:]
                x = x.permute(0, 2, 3, 1)
                # x = x.view(B*H*W, C)  # flatten
                x = torch.reshape(x, (B * H * W, C))
                x = self.decoder[2](x)  # lin_out mlp render
                x = x.reshape(B, H, W, 5)  # TODO

            else:  # X
                x = self.decoder[1](x)  # 2D conv render
                x = x.permute(0, 2, 3, 1)

            outputs = self.decoder[-1](x)  # sigmoid activation

            return outputs

    def index(self, uv, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)

                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0  # (uv/image_size-0.5)*2

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )

            return samples[:, :, :, 0]  # (B, C, N)

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )
