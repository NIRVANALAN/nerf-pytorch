import torch
from torch import nn
from torch._C import import_ir_module_from_buffer
import torch.nn.functional as F

import ipdb

#  import torch_scatter
import torch.autograd.profiler as profiler


def init_model(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.constant_(m.bias, 0.0)
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class ResnetFC(nn.Module):
    def __init__(self,
                 D=5,
                 W=128,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 d_latent=0,
                 beta=0.0,
                 combine_layer=3,
                 use_spade=False,
                 use_viewdirs=True,
                 viewdirs_res=False,
                 input_views=1,
                 agg_type="cat",
                 *args,
                 **kwargs):
        """
        Resnet FC  backbone without multi-view input.
        Improved version of original nerf net
        :param D input size
        :param output_ch output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()

        self.D = D
        self.agg_type = agg_type
        self.d_latent = d_latent
        self.use_spade = use_spade
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.W = W
        self.use_viewdirs = use_viewdirs
        self.viewdirs_res = viewdirs_res
        print("using spade: {}".format(self.use_spade))

        if self.agg_type == "cat":
            self.d_latent *= input_views
        # if use_viewdirs: # * True by default
        if viewdirs_res:
            self.lin_in = nn.Linear(input_ch, W)
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList(
                [nn.Linear(input_ch_views + W, W // 2)])
        else:
            self.lin_in = nn.Linear(input_ch + input_ch_views, W)
            self.lin_out = nn.Linear(W, output_ch)
            nn.init.constant_(self.lin_out.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.combine_layer = combine_layer
        # self.use_spade = use_spade

        self.encoder_blks = nn.ModuleList(
            [ResnetBlockFC(W, beta=beta) for _ in range(combine_layer)])
        self.decoder_blks = nn.ModuleList([
            nn.Sequential(*[
                ResnetBlockFC(W, beta=beta) for _ in range(D - combine_layer)
            ])
        ])
        if not viewdirs_res:
            self.decoder_blks.append(self.lin_out)
        else:
            self.decoder_blks.append(
                nn.ModuleList(
                    [self.feature_linear, self.views_linears,
                     self.rgb_linear]))

        if self.d_latent != 0:
            n_lin_z = min(combine_layer, D)
            self.lin_z = nn.ModuleList(
                [nn.Linear(self.d_latent, W) for i in range(n_lin_z)])
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)
                nn.init.kaiming_normal_(self.lin_z[i].weight,
                                        a=0,
                                        mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, W) for _ in range(n_lin_z)])
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)
                    nn.init.kaiming_normal_(self.scale_z[i].weight,
                                            a=0,
                                            mode="fan_in")

        # if beta > 0:
        #     self.activation = nn.Softplus(beta=beta)
        # else:
        self.activation = nn.ReLU()

    def forward(
        self,
        xz,
    ):
        """
        :param zx (..., d_latent + d_in)
        """
        # with profiler.record_function("resnetfc_infer"):
        if self.d_latent > 0:
            z = xz[..., :self.d_latent]
            h = xz[..., self.d_latent:]
        else:
            h = xz

        if self.viewdirs_res:
            h, input_views = torch.split(h,
                                         [self.input_ch, self.input_ch_views],
                                         dim=-1)
            # ipdb.set_trace()

        # if self.input_ch > 0:
        h = self.lin_in(h)  #  -> 512
        # else:
        #     h = torch.zeros(self.W, device=x.device)  # ? Generative model?

        for i, l in enumerate(self.encoder_blks):
            if self.d_latent > 0 and i < self.combine_layer:
                if self.use_spade:
                    h = self.scale_z[i](z) * h + self.lin_z[i](z)
                else:
                    h += self.lin_z[i](z)
            elif i == self.combine_layer:
                h = combine_interleaved(h, agg_type="average")  # h = 2*B*256

            h = self.encoder_blks[i](h)

        # combine
        h = combine_interleaved(h, agg_type="average")  # h = 2*B*256

        # treat the remaining REBLKS as as decoder
        for i, l in enumerate(self.decoder_blks[:-1]):  # remove lin_out
            h = self.decoder_blks[i](h)

        if self.viewdirs_res:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.decoder_blks[-1](h)  # self.lin_out
            # if spatial:
            #     outputs = outputs.reshape(B, self.output_ch, H, W)
        return outputs

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 3),
            combine_type=conf.get_string("combine_type",
                                         "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs)

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 3),
            combine_type=conf.get_string("combine_type",
                                         "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs)


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """
    def __init__(self,
                 size_in,
                 size_out=None,
                 size_h=None,
                 beta=0.0,
                 spatial_output=False):
        super().__init__()
        # Attributes
        self.spatial_output = spatial_output  # transform output 2D
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)

        # Init
        nn.init.constant_(self.fc_0.bias, 0.0)
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        spatial_output = x.dim() == 4
        if spatial_output:
            B, C, H, W = x.shape[:]
            x = x.permute(0, 2, 3, 1)
            # x = x.view(B*H*W, C)  # flatten
            x = torch.reshape(x, (B * H * W, C))

        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        if spatial_output:
            return ((x_s + dx).reshape(B, H, W,
                                       self.size_out).permute(0, 3, 1,
                                                              2))  # TODO
        return x_s + dx


def repeat_interleave(input, repeats, dim):
    """
    Repeat interleave, does same thing as torch.repeat_interleave but faster.
    torch.repeat_interleave is currently very slow
    https://github.com/pytorch/pytorch/issues/31980
    """
    if dim >= 0:
        dim += 1
    output = input.unsqueeze(dim).expand(-1, repeats, *input.shape[1:])
    return output.reshape(-1, *input.shape[1:])


def combine_interleaved(t, agg_type="average"):
    # if len(inner_dims) == 1 and inner_dims[0] == 1:
    #     return t
    # t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if t.shape[0] == 1 or t.dim() == 2:  # single view input
        return t
    if agg_type == "average":
        t = torch.mean(t, dim=0)
    elif agg_type == "max":
        t = torch.max(t, dim=0)[0]  # remove indices
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t


class ResnetFC_GRL(nn.Module):
    def __init__(self,
                 D=5,
                 W=512,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 d_latent=0,
                 beta=0.0,
                 agg_layer=4,
                 use_viewdirs=True,
                 viewdirs_res=False,
                 agg_type="att",
                 input_views=1,
                 **kwargs):
        """
        Resnet FC  backbone without multi-view input.
        Improved version of original nerf net
        :param D input size
        :param output_ch output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()

        self.D = D
        self.d_latent = d_latent * input_views
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.W = W
        self.use_viewdirs = use_viewdirs
        self.viewdirs_res = viewdirs_res

        if D == agg_layer:
            dim_out = W + self.d_latent
        else:
            dim_out = W

        # if use_viewdirs: # * True by default
        if viewdirs_res:  # TODO
            self.lin_in = nn.Linear(input_ch, W)
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList(
                [nn.Linear(input_ch_views + W, W // 2)])
        else:
            self.lin_in = nn.Linear(input_ch + input_ch_views, W)
            self.lin_out = nn.Linear(dim_out, output_ch)
            nn.init.constant_(self.lin_out.bias, 0.0)
            nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.agg_layer = agg_layer
        # self.use_spade = use_spade

        # TODO self-att module
        self.agg_module = None

        self.blocks = [
            ResnetBlockFC(W, beta=beta) for _ in range(agg_layer - 1)
        ]
        if self.agg_layer < D:
            self.blocks.insert(
                self.agg_layer,
                ResnetBlockFC(size_in=W + self.d_latent, size_out=W,
                              beta=beta),
            )
        else:
            self.blocks.insert(-1,
                               ResnetBlockFC(size_in=W, size_out=W, beta=beta))

        self.blocks = nn.ModuleList(self.blocks)

        self.activation = nn.ReLU()

    def forward(
        self,
        xz,
    ):
        """
        :param zx (..., d_latent + d_in)
        """
        # with profiler.record_function("resnetfc_infer"):
        if self.d_latent > 0:
            z = xz[..., :self.d_latent]
            h = xz[..., self.d_latent:]
        else:
            h = xz

        if self.viewdirs_res:
            h, input_views = torch.split(h,
                                         [self.input_ch, self.input_ch_views],
                                         dim=-1)

        # if self.input_ch > 0:
        h = self.lin_in(h)  #  -> 512
        # else:
        #     h = torch.zeros(self.W, device=x.device)  # ? Generative model?

        for i, l in enumerate(self.blocks):
            if i == self.agg_layer:
                # h = self.agg_module(h)  # h = 2*B*256
                h = self.blocks[i](torch.cat([h, z], dim=-1))
            else:
                h = self.blocks[i](h)

        if self.viewdirs_res:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            if self.agg_layer == self.D:
                outputs = self.lin_out(torch.cat([h, z], -1))
            else:
                outputs = self.lin_out(h)
        return outputs

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 3),
            combine_type=conf.get_string("combine_type",
                                         "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs)
