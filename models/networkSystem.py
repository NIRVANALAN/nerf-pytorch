from numpy.lib.utils import deprecate
import torch
from torch import nn
import torch.nn.functional as F
import os
from .nerf import NeRF
from .resnetfc import ResnetFC, ResnetFC_GRL
from .encoder import SpatialEncoder, ImageEncoder
from util import get_embedder, batchify

model_dict = {"nerf": NeRF, "resnetFC": ResnetFC, "resnetGRL": ResnetFC_GRL}
encoder_dict = {
    "spatial": SpatialEncoder,
    "global": ImageEncoder,
}


class NetworkSystem:
    def __init__(self, args, img_shape, device) -> None:
        super().__init__()
        """Instantiate NeRF's MLP model. Original create_model()"""
        if args.arch not in model_dict:
            raise NotImplementedError(
                "no {} architecture found. Please select in {}".format(
                    args.arch, model_dict.keys()
                )
            )

        self.start = 0
        # Note: this is world -> camera, and bottom row is omitted
        # self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        # self.register_buffer("image_shape", torch.empty(2), persistent=False)
        self.image_shape = torch.Tensor(img_shape)
        self.poses = torch.empty(1, 3, 4)
        self.transform_into_viewspace = args.transform_view_spaces
        nerf_net = model_dict[args.arch]
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(
                args.multires_views, args.i_embed
            )
        output_ch = 5 if args.N_importance > 0 else 4
        # * create encoder

        self.d_latent = 512 if args.enc_type != "none" else 0

        # self.encoder = make_encoder(conf["encoder"])
        skips = [4]
        self.model = nerf_net(
            D=args.netdepth,
            W=args.netwidth,
            input_ch=input_ch,
            output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            viewdirs_res=args.viewdirs_res,
            d_latent=self.d_latent,
            input_views=len(args.srn_encode_views_id.split()),
            agg_layer=args.agg_layer,
            agg_type=args.agg_type,
            use_spade=args.use_spade,
        ).to(device)

        # print(self.model)

        grad_vars = list(self.model.parameters())

        self.model_fine = None
        if args.N_importance > 0:
            self.model_fine = nerf_net(
                D=args.netdepth_fine,
                W=args.netwidth_fine,
                input_ch=input_ch,
                output_ch=output_ch,
                skips=skips,
                input_ch_views=input_ch_views,
                use_viewdirs=args.use_viewdirs,
                viewdirs_res=args.viewdirs_res,
                d_latent=self.d_latent,
                input_views=len(args.srn_encode_views_id.split()),
                agg_layer=args.agg_layer,
                agg_type=args.agg_type,
                use_spade=args.use_spade,
            ).to(device)
            grad_vars += list(self.model_fine.parameters())

            print(self.model_fine)

        # define encoder
        if args.enc_type != "none":
            assert args.enc_type in encoder_dict.keys()
            self.encoder = encoder_dict[args.enc_type](
                index_padding=args.index_padding,
                pretrained=not args.encoder_no_pretrain,
                add_decoder=args.add_decoder,
                mlp_render_net=self.model_fine.decoder_blks
                if args.mlp_render
                else None,  # TODO
            )  # default params
            print("encoder index padding mode: {}".format(args.index_padding))
            self.enable_encoder_grad = (
                args.enable_encoder_grad
            )  # update ConvNet gradient (not freeze weights)
            # self.d_latent = self.encoder.latent_size

            # add parameters of Encoder if needed
            if args.add_decoder:  # TODO
                grad_vars += list(self.encoder.parameters())

        else:
            self.encoder = None

        # Create optimizer, load model
        self.optimizer = torch.optim.Adam(
            params=grad_vars, lr=args.lrate, betas=(0.9, 0.999)
        )

        # load ckpt
        if args.ft_path is not None and args.ft_path != "None":
            ckpts = [args.ft_path]
        else:
            ckpts = [
                os.path.join(args.basedir, args.expname, f)
                for f in sorted(os.listdir(os.path.join(args.basedir, args.expname)))
                if "tar" in f
            ]

        print("Found ckpts", ckpts)
        if len(ckpts) > 0 and not args.no_reload:
            ckpt_path = ckpts[-1]
            print("Reloading from", ckpt_path)
            ckpt = torch.load(ckpt_path)

            self.start = ckpt["global_step"]
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            # Load model
            self.model.load_state_dict(ckpt["network_fn_state_dict"])
            if self.model_fine is not None:
                self.model_fine.load_state_dict(ckpt["network_fine_state_dict"])

        self.grad_vars = grad_vars

        network_query_fn = lambda inputs, viewdirs, network_fn: self.run_network(
            inputs,
            viewdirs,
            network_fn,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            netchunk=args.netchunk,
            # use_encoder=args.use_encoder,
        )  # ! save params using lambda

        render_kwargs_train = {
            "network_query_fn": network_query_fn,
            "perturb": args.perturb,
            "N_importance": args.N_importance,
            "network_fine": self.model_fine,
            "N_samples": args.N_samples,
            "network_fn": self.model,
            "use_viewdirs": args.use_viewdirs,
            "white_bkgd": args.white_bkgd,
            "raw_noise_std": args.raw_noise_std,
            # "encoder_net": encoder_net,
        }

        # NDC only good for LLFF-style forward facing data
        if args.dataset_type != "llff" or args.no_ndc:
            print("Not ndc!")
            render_kwargs_train["ndc"] = False
            render_kwargs_train["lindisp"] = args.lindisp

        render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
        render_kwargs_test["perturb"] = False
        render_kwargs_test["raw_noise_std"] = 0.0

        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.normalize_z = args.normalize_z
        self.agg_type = args.agg_type

        # return render_kwargs_train, render_kwargs_test, # start, optimizer

    # @staticmethod
    def transform_view_space(self, xyz, poses=None, return_rot=False):
        """transform cononical world coordinates into view spaces coords system

        Args:
            xyz (Tensor): cononical coord. 4096 * 3
        Output:
            N * 4096 * 3. N=number of input views
        """
        if poses == None:
            poses = self.poses
        xyz_rot = torch.matmul(
            poses[:, None, :3, :3], xyz.reshape(-1, xyz.shape[0], 3, 1)
        )[
            ..., 0
        ]  # N * BS * 3
        if return_rot:
            return xyz_rot
        return xyz_rot + poses[:, None, :3, 3]

    def run_network(
        self,
        inputs,  # 128*32*3
        viewdirs,
        fn,
        embed_fn,
        embeddirs_fn,
        netchunk=1024 * 64,
    ):
        """Prepares inputs and applies network 'fn'."""
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # 4096*3

        ###
        # view_space_inputs_flat = self.transform_view_space(inputs_flat)
        # homo_coord_viewspace = torch.cat(
        #     [view_space_inputs_flat, torch.ones(1, view_space_inputs_flat.shape[1], 1)],
        #     -1,
        # )
        # back_inputs = self.transform_view_space(view_space_inputs_flat, poses=self.c2w)
        ###

        if self.transform_into_viewspace:
            # Transform query points into the camera spaces of the input views
            if self.normalize_z:
                inputs_flat = self.transform_view_space(inputs_flat, return_rot=True)
            else:
                inputs_flat = self.transform_view_space(inputs_flat, return_rot=False)
            # inputs_flat = inputs_flat.unsqueeze(0)  # align with pix-nerf
        embedded = embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(
                input_dirs, [-1, input_dirs.shape[-1]]
            )  # 4096, 3

            if self.transform_into_viewspace:
                # transform viewdirs into view space
                input_dirs_flat = self.transform_view_space(input_dirs_flat)  # N*BS*3

            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        if self.encoder is not None:  # encode latent code
            # embedded = torch.repeat_interleave(embedded.unsqueeze(0), self.num_objs, 0)

            if not self.transform_into_viewspace:
                inputs_flat = self.transform_view_space(inputs_flat)
            uv = -inputs_flat[..., :2] / inputs_flat[..., 2:]  # * [X, -Y, -Z]

            if uv.shape[0] == 1 and not self.transform_view_space:
                uv = uv.unsqueeze(0)

            uv *= self.focal
            uv += self.c

            ### IMPORTANT: transform uv from Cartisian indexing into 'ij' indexing ###
            uv = torch.flip(uv, [-1])
            ### IMPORTANT ###

            latent = self.encoder.index(
                uv, self.image_shape
            ).detach()  # ([1, 512, 4096])

            latent = latent.permute(0, 2, 1)  # 2 4096 512

            # latent = torch.reshape(
            #     latent, [-1, latent.shape[-1]]
            # )  # 65536*512. Cannot use .view() due to not contiguous

            if embedded.dim() != latent.dim():
                embedded = embedded.unsqueeze(0)

            if latent.size(0) != 1:
                if self.agg_type == "cat":
                    latent = latent.permute(1, 2, 0).reshape(
                        (1, latent.shape[1], -1)
                    )  # merge the two feature dimension
                elif self.agg_type == "avg":
                    latent = latent.mean(dim=0, keepdim=True)
                else:  # default
                    pass

            embedded = torch.cat((latent, embedded), dim=-1)

        outputs_flat = batchify(fn, netchunk)(embedded)
        # outputs_flat = fn(embedded)  # TODO, memory peak
        outputs = torch.reshape(
            outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]
        )
        return outputs

    #
    def encode(self, images, poses, focal, c=None):
        """
        :param images (NS, 3, H, W)
        NS is number of input (aka source or reference) views
        :param poses (NS, 4, 4)
        :param focal focal length () or (2) or (NS) or (NS, 2) [fx, fy]
        :param z_bounds ignored argument (used in the past)
        :param c principal point None or () or (2) or (NS) or (NS, 2) [cx, cy],
        default is center of image
        """
        # with profiler.record_function("encoder"):
        self.num_objs = images.size(0)
        if len(images.shape) == 5:  # multi object input
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(
                1
            )  # Be consistent with NS = num input views
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        if self.enable_encoder_grad:
            self.encoder(images)  # memery bank
        else:
            with torch.no_grad():  # todo
                self.encoder(images)  # memery bank

        # * TEST TIME MULTIVIEW INPUT
        # if not self.model.training:
        self.c2w = poses
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat(
            (rot, trans), dim=-1
        )  # (B, 3, 4) #* inverse of original pose. w2c matrix of primary pose(s)

        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        # Handle various focal length/principal point formats
        if len(focal.shape) == 0:
            # Scalar: fx = fy = value for all views
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            # Vector f: fx = fy = f_i *for view i*
            # Length should match NS (or 1 for broadcast)
            focal = focal.unsqueeze(-1).repeat((1, 2))
        # Otherwise, can specify as
        focal[..., 1] *= -1.0
        self.focal = focal.float()

        if c is None:
            # Default principal point is center of image
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            # Scalar: cx = cy = value for all views
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            # Vector c: cx = cy = c_i *for view i*
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c