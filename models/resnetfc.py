import torch
from torch import nn
import torch.nn.functional as F
import pdb

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
    def __init__(
        self,
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
        viewdirs_res=True,
        *args,
        **kwargs
    ):
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
        self.d_latent = d_latent
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.W = W
        self.use_viewdirs = use_viewdirs
        self.viewdirs_res = viewdirs_res

        # if use_viewdirs: # * True by default
        if viewdirs_res:
            self.lin_in = nn.Linear(input_ch, W)
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        else:
            self.lin_in = nn.Linear(input_ch + input_ch_views, W)
            self.output_linear = nn.Linear(W, output_ch)

        self.lin_out = nn.Linear(W, output_ch)
        nn.init.constant_(self.lin_out.bias, 0.0)
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.combine_layer = combine_layer
        # self.use_spade = use_spade

        self.pts_linears = nn.ModuleList(
            [ResnetBlockFC(W, beta=beta) for i in range(D)]
        )

        if d_latent != 0:
            n_lin_z = min(combine_layer, D)
            self.lin_z = nn.ModuleList([nn.Linear(d_latent, W) for i in range(n_lin_z)])
            # for i in range(n_lin_z):
            #     nn.init.constant_(self.lin_z[i].bias, 0.0)
            #     nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            # if self.use_spade:
            #     self.scale_z = nn.ModuleList(
            #         [nn.Linear(d_latent, W) for _ in range(n_lin_z)]
            #     )
            # for i in range(n_lin_z):
            #     nn.init.constant_(self.scale_z[i].bias, 0.0)
            #     nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        # if beta > 0:
        #     self.activation = nn.Softplus(beta=beta)
        # else:
        self.activation = nn.ReLU()

    def forward(
        self,
        x,
    ):
        """
        :param zx (..., d_latent + d_in)
        """
        with profiler.record_function("resnetfc_infer"):
            if self.d_latent > 0:
                z = x[..., : self.d_latent]
                x = x[..., self.d_latent :]

            if self.viewdirs_res:
                input_pts, input_views = torch.split(
                    x, [self.input_ch, self.input_ch_views], dim=-1
                )
                h = input_pts
            else:
                h = x

            # if self.input_ch > 0:
            h = self.lin_in(h)  #  -> 512
            # else:
            #     h = torch.zeros(self.W, device=x.device)  # ? Generative model?

            for i, l in enumerate(self.pts_linears):
                if self.d_latent > 0 and i < self.combine_layer:
                    tz = self.lin_z[i](h)
                    if self.use_spade:
                        sz = self.scale_z[i](h)
                        x = sz * x + tz
                    else:
                        x = x + tz

                h = self.pts_linears[i](h)

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
                outputs = self.output_linear(h)
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
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs
        )


class ResnetCombineFC(nn.Module):
    def __init__(
        self,
        D,
        output_ch=4,
        n_blocks=5,
        d_latent=0,
        W=128,
        beta=0.0,
        combine_layer=3,
        combine_type="average",
        use_spade=False,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        if D > 0:
            self.lin_in = nn.Linear(D, W)
            # nn.init.constant_(self.lin_in.bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")

        self.lin_out = nn.Linear(W, output_ch)
        # nn.init.constant_(self.lin_out.bias, 0.0)
        # nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = D
        self.d_out = output_ch
        self.d_hidden = W

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(W, beta=beta) for i in range(n_blocks)]
        )

        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)
            self.lin_z = nn.ModuleList([nn.Linear(d_latent, W) for i in range(n_lin_z)])
            # for i in range(n_lin_z):
            # nn.init.constant_(self.lin_z[i].bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, W) for _ in range(n_lin_z)]
                )
                # for i in range(n_lin_z):
                #     nn.init.constant_(self.scale_z[i].bias, 0.0)
                #     nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        init_model(self)

    def forward(
        self,
        zx,
        combine_inner_dims=(1,),
    ):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        with profiler.record_function("resnetfc_infer"):
            assert zx.size(-1) == self.d_latent + self.d_in
            if self.d_latent > 0:
                z = zx[..., : self.d_latent]
                x = zx[..., self.d_latent :]
            else:
                x = zx
            if self.d_in > 0:
                x = self.lin_in(x)
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)

            for blkid in range(self.n_blocks):
                if blkid == self.combine_layer:
                    # The following implements camera frustum culling, requires torch_scatter
                    #  if combine_index is not None:
                    #      combine_type = (
                    #          "mean"
                    #          if self.combine_type == "average"
                    #          else self.combine_type
                    #      )
                    #      if dim_size is not None:
                    #          assert isinstance(dim_size, int)
                    #      x = torch_scatter.scatter(
                    #          x,
                    #          combine_index,
                    #          dim=0,
                    #          dim_size=dim_size,
                    #          reduce=combine_type,
                    #      )
                    #  else:
                    x = util.combine_interleaved(
                        x, combine_inner_dims, self.combine_type
                    )

                if self.d_latent > 0 and blkid < self.combine_layer:
                    tz = self.lin_z[blkid](z)
                    if self.use_spade:
                        sz = self.scale_z[blkid](z)
                        x = sz * x + tz
                    else:
                        x = x + tz

                x = self.blocks[blkid](x)
            out = self.lin_out(self.activation(x))
            return out

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int("n_blocks", 5),
            d_hidden=conf.get_int("d_hidden", 128),
            beta=conf.get_float("beta", 0.0),
            combine_layer=conf.get_int("combine_layer", 3),
            combine_type=conf.get_string("combine_type", "average"),  # average | max
            use_spade=conf.get_bool("use_spade", False),
            **kwargs
        )


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()
        # Attributes
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
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        with profiler.record_function("resblock"):
            net = self.fc_0(self.activation(x))
            dx = self.fc_1(self.activation(net))

            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x
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
