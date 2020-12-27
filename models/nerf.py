import pdb
import torch

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd.profiler as profiler

# Model
class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        viewdirs_res=True,
        d_latent=0,
        *args,
        **kwargs
    ):
        """"""
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.viewdirs_res = viewdirs_res

        if viewdirs_res:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
            self.lin_in = nn.Linear(input_ch, W)

            ### Implementation according to the paper
            # self.views_linears = nn.ModuleList(
            #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        else:
            self.input_ch = input_ch + input_ch_views
            self.lin_in = nn.Linear(self.input_ch, W)
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears = nn.ModuleList(
            [self.lin_in]
            + [
                nn.Linear(W, W)
                if i not in self.skips
                else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)
            ],
        )

    def forward(self, x):
        with profiler.record_function("nerf_mlp inference"):
            if self.viewdirs_res:
                input_pts, input_views = torch.split(
                    x, [self.input_ch, self.input_ch_views], dim=-1
                )
                h = input_pts
            else:
                h = x
                input_pts = x

            for i, l in enumerate(self.pts_linears):
                h = self.pts_linears[i](h)
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

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

    def load_weights_from_keras(self, weights):
        assert self.viewdirs_res, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears])
            )
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears + 1])
            )

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear])
        )
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear + 1])
        )

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears])
        )
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears + 1])
        )

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear])
        )
        self.rgb_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_rbg_linear + 1])
        )

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear])
        )
        self.alpha_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_alpha_linear + 1])
        )