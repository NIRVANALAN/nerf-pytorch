import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config",
                        is_config_file=True,
                        help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")

    parser.add_argument("--basedir",
                        type=str,
                        default="./logs/",
                        help="where to store ckpts and logs")

    parser.add_argument("--datadir",
                        type=str,
                        default="./data/llff/fern",
                        help="input data directory")

    # training options
    parser.add_argument("--netdepth",
                        type=int,
                        default=8,
                        help="layers in network")
    parser.add_argument("--netwidth",
                        type=int,
                        default=256,
                        help="channels per layer")
    parser.add_argument("--netdepth_fine",
                        type=int,
                        default=8,
                        help="layers in fine network")
    parser.add_argument(
        "--netwidth_fine",
        type=int,
        default=256,
        help="channels per layer in fine network",
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 32 * 4,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument("--lrate",
                        type=float,
                        default=5e-4,
                        help="learning rate")
    parser.add_argument(
        "--lrate_decay",
        type=int,
        default=250,
        help="exponential learning rate decay (in 1000 steps)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1024 * 16,
        help=
        "number of rays processed in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--netchunk",
        type=int,
        default=1024 * 64,
        help=
        "number of pts sent through network in parallel, decrease if running out of memory",
    )
    parser.add_argument(
        "--no_batching",
        action="store_true",
        help="only take random rays from 1 image at a time",
    )
    parser.add_argument("--no_reload",
                        action="store_true",
                        help="do not reload weights from saved ckpt")
    parser.add_argument(
        "--ft_path",
        type=str,
        default=None,
        help="specific weights npy file to reload for coarse network",
    )

    # rendering options
    parser.add_argument("--N_samples",
                        type=int,
                        default=64,
                        help="number of coarse samples per ray")
    parser.add_argument(
        "--N_importance",
        type=int,
        default=0,
        help="number of additional fine samples per ray",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=1.0,
        help="set to 0. for no jitter, 1. for jitter",
    )
    parser.add_argument("--use_viewdirs",
                        action="store_true",
                        help="use full 5D input instead of 3D")
    parser.add_argument(
        "--i_embed",
        type=int,
        default=0,
        help="set 0 for default positional encoding, -1 for none",
    )
    parser.add_argument(
        "--multires",
        type=int,
        default=10,
        help="log2 of max freq for positional encoding (3D location)",
    )
    parser.add_argument(
        "--multires_views",
        type=int,
        default=4,
        help="log2 of max freq for positional encoding (2D direction)",
    )
    parser.add_argument(
        "--raw_noise_std",
        type=float,
        default=0.0,
        help=
        "std dev of noise added to regularize sigma_a output, 1e0 recommended",
    )

    parser.add_argument(
        "--render_only",
        action="store_true",
        help="do not optimize, reload weights and render out render_poses path",
    )
    parser.add_argument(
        "--render_test",
        action="store_true",
        help="render the test set instead of render_poses path",
    )
    parser.add_argument(
        "--render_factor",
        type=int,
        default=0,
        help=
        "downsampling factor to speed up rendering, set 4 or 8 for fast preview",
    )

    # mesh options
    parser.add_argument(
        "--mesh_only",
        action="store_true",
        help="do not optimize, reload weights and save mesh to a file",
    )
    parser.add_argument(
        "--mesh_grid_size",
        type=int,
        default=80,
        help=
        "number of grid points to sample in each dimension for marching cubes",
    )

    # training options
    parser.add_argument(
        "--precrop_iters",
        type=int,
        default=0,
        help="number of steps to train on central crops",
    )
    parser.add_argument(
        "--precrop_frac",
        type=float,
        default=0.5,
        help="fraction of img taken for central crops",
    )

    # dataset options
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="llff",
        help="options: llff / blender / deepvoxels",
    )
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help=
        "will load 1/N images from test/val sets, useful for large datasets like deepvoxels",
    )

    ## deepvoxels flags
    parser.add_argument(
        "--shape",
        type=str,
        default="greek",
        help="options : armchair / cube / greek / vase",
    )

    ## blender flags
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help=
        "set to render synthetic data on a white bkgd (always use for dvoxels)",
    )
    parser.add_argument(
        "--half_res",
        action="store_true",
        help="load blender synthetic data at 400x400 instead of 800x800",
    )

    ## llff flags
    parser.add_argument("--factor",
                        type=int,
                        default=8,
                        help="downsample factor for LLFF images")
    parser.add_argument(
        "--no_ndc",
        action="store_true",
        help=
        "do not use normalized device coordinates (set for non-forward facing scenes)",
    )
    parser.add_argument(
        "--lindisp",
        action="store_true",
        help="sampling linearly in disparity rather than depth",
    )
    parser.add_argument("--spherify",
                        action="store_true",
                        help="set for spherical 360 scenes")
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    # logging/saving options
    parser.add_argument(
        "--epoch",
        type=int,
        default=50000,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument(
        "--i_print",
        type=int,
        default=100,
        help="frequency of console printout and metric loggin",
    )
    parser.add_argument("--i_img",
                        type=int,
                        default=500,
                        help="frequency of tensorboard image logging")
    parser.add_argument("--i_weights",
                        type=int,
                        default=10000,
                        help="frequency of weight ckpt saving")
    parser.add_argument("--i_testset",
                        type=int,
                        default=5000,
                        help="frequency of testset saving")
    parser.add_argument(
        "--i_video",
        type=int,
        default=100000,
        help="frequency of render_poses video saving",
    )
    # model arch
    parser.add_argument(
        "--arch",
        type=str,
        default="nerf",
        help="specific network architecture. NeRF by default",
    )

    # semantic encoder
    parser.add_argument(
        "--viewdirs_res",
        action="store_true",  # !
        help="whether to send viewdirs as final residual layer",
    )
    parser.add_argument(
        "--enc_type",
        type=str,
        default="none",
        help="Whether to use uncoder & the type of encoder",
    )
    parser.add_argument(
        "--use_spade",
        action="store_true",
        help="Whether to use spade-like spatial feature transform",
    )
    # srn dataset
    parser.add_argument("--srn_object_id",
                        type=int,
                        default=0,
                        help="srn object id")

    parser.add_argument("--srn_input_views",
                        type=int,
                        default=None,
                        help="srn input view number")
    parser.add_argument("--srn_input_views_id",
                        type=str,
                        default="0",
                        help="srn input views id")
    parser.add_argument(
        "--srn_encode_views_id",
        type=str,
        default="-1",
        help="fixed training views for encoding view spaces",
    )
    parser.add_argument("--srn_test_idx_interv",
                        type=int,
                        default=2,
                        help="test some ids")
    parser.add_argument("--agg_layer", type=int, default=3, help="agg_layer")
    parser.add_argument("--normalize_z",
                        action="store_true",
                        help="Whether to shift z into origin")
    parser.add_argument(
        "--transform_view_spaces",
        action="store_true",
        help="Whether to transform coords into view space",
    )

    parser.add_argument("--index_padding",
                        type=str,
                        default="border",
                        help="grid_sample() padding mode")
    parser.add_argument("--encoder_no_pretrain",
                        action="store_true",
                        help="random init encoder")
    parser.add_argument("--agg_type",
                        type=str,
                        default="",
                        help="how to agg the features")

    parser.add_argument("--add_decoder",
                        action="store_true",
                        help="append decoder")

    parser.add_argument(
        "--decoder_train_objs",
        type=int,
        default=1,
        help="how many objs to train the AE separately",
    )

    parser.add_argument("--ae_batch", type=int, default=32, help="BS of AE")
    parser.add_argument("--ae_lambda",
                        type=float,
                        default=0.5,
                        help="weight for ae loss")
    parser.add_argument(
        "--encoder_interv",
        type=int,
        default=1,
        help="how many objs to train the AE separately",
    )

    parser.add_argument("--mlp_render",
                        action="store_true",
                        help="nerf mlp module as AE decoder")

    parser.add_argument(
        "--not_encode_test_views",
        action="store_true",
        help="whether to encode test views during test",
    )

    parser.add_argument(
        "--srn_encode_test_views_id",
        type=str,
        default="0 40",
        help="fixed training views for encoding view spaces",
    )

    parser.add_argument(
        "--decoder_dataset",
        type=str,
        default="car",
        help="instances for decoder to train",
    )

    parser.add_argument(
        "--nerf_train_dataset",
        type=str,
        default="car",
        help="instances for decoder to train",
    )

    parser.add_argument("--torch_seed",
                        type=int,
                        default=0,
                        help="manual seed for torch")

    parser.add_argument(
        "--input_view_sampling",
        type=str,
        default="linspace",
        help="linspace or arange to sample input views",
    )

    parser.add_argument(
        "--external_sampling",
        type=str,
        default="instance",
        help="importance / instance / uniform sampling",
    )

    parser.add_argument(
        "--memory_bank",
        action="store_true",
        help="whether to load memory_bank",
    )

    parser.add_argument(
        "--decoder_bs",
        type=int,
        default=32,
        help="how many objs to train the AE separately",
    )

    parser.add_argument(
        "--encoder_only",
        action="store_true",
        help="only encoder params will be trained",
    )

    parser.add_argument(
        "--encoder_ft_path",
        type=str,
        default=None,
        help="pretrained encoder ckpt",
    )

    parser.add_argument(
        "--decoder_only",
        action="store_true",
        help="encoder only for decoder, don't condition on the features",
    )

    parser.add_argument(
        "--fix_decoder_params",
        action="store_true",
        help="fix parmas of decoder",
    )

    parser.add_argument(
        "--mmcv_config",
        type=str,
        default=None,
        help="mmcv config file path",
    )

    parser.add_argument(
        "--random_ord_encode",
        action="store_true",
        help="encode random images each batch",
    )

    parser.add_argument("--number_encode_view",
                        type=int,
                        default=1,
                        help="number of images for encode each batch")

    parser.add_argument(
        "--stop_encoder_grad",
        action="store_true",
        help="freeze grad of encoder",
    )

    # incremental training
    parser.add_argument(
        "--incremental_path",
        type=str,
        default=None,
        help="where to inversion output, comes from train_test by default")

    parser.add_argument(
        "--test_interv",
        type=int,
        default=1,
        help="render poses interval",
    )

    return parser


def prepare_proj_parser(parser):
    usage = 'Parser for all scripts.'
    # parser = argparse.ArgumentParser(
    #     description="Image projector to the generator latent spaces")
    parser.add_argument("--ckpt",
                        type=str,
                        required=True,
                        help="path to the model checkpoint")
    parser.add_argument("--size",
                        type=int,
                        default=128,
                        help="output image sizes of the generator")
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--noise",
                        type=float,
                        default=0.05,
                        help="strength of the noise level")
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step",
                        type=int,
                        default=1000,
                        help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--w_l1",
                        type=float,
                        default=0,
                        help="weight of the l1 loss")
    parser.add_argument("--w_l2",
                        type=float,
                        default=0,
                        help="weight of the l1 loss")
    parser.add_argument(
        "--normalize_vgg_loss",
        action="store_true",
        help="normalize lpips by input image numbers",
    )

    return parser


def add_proj_parser(parser):
    parser.add_argument(
        "--id_aware",
        action="store_true",
        help="shared id latent code for all input files",
    )
    parser.add_argument(
        "--sample_range",
        type=int,
        default=50,
        help="how many nerf samples to choose from",
    )
    parser.add_argument(
        "--inject_index",
        type=int,
        default=4,
        help="injection starts from which layer in W",
    )
    parser.add_argument("--proj_latent",
                        type=str,
                        default=None,
                        help="path to the projected latent code")
    parser.add_argument("--nerf_pred_path",
                        type=str,
                        default=None,
                        help="path to the nerf prediction")
    parser.add_argument(
        "--nerf_psnr_path",
        type=str,
        default=None,
        help="path to the nerf model psnr.npy, load for degraded img projection"
    )
    parser.add_argument("--sampling_range",
                        type=int,
                        default=50,
                        help="sample range")
    parser.add_argument("--sampling_numbers",
                        type=int,
                        default=10,
                        help="sample numbers")
    parser.add_argument("--sampling_strategy",
                        type=str,
                        default=None,
                        help="how to sample?")
    parser.add_argument(
        "--d_loss",
        action="store_true",
        help="use discriminator_perceptual_loss",
    )
    parser.add_argument("--cotraining_stage",
                        type=int,
                        default=5,
                        help="how many stages to perform gan-nerf cotraining")
    parser.add_argument(
        "--cotraining_epoch",
        type=int,
        default=10000,
        help="frequency of console printout and metric loggin",
    )
    return parser
