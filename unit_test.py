import numpy as np
import torch
import imageio
from data import create_dataset
from options.opts import add_proj_parser, config_parser, prepare_proj_parser

if __name__ == '__main__':
    parser = config_parser()
    parser = prepare_proj_parser(parser)
    parser = add_proj_parser(parser)

    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(args.torch_seed)    
    (images, poses, render_poses, hwf, i_train, i_val, i_test,
    near, far, c, data, img_Tensor, i_fixed, i_fixed_test, meta_data) = create_dataset(args)
    (train_imgs, test_imgs, train_poses, test_poses,
     incremental_dataset) = meta_data

    import ipdb
    ipdb.set_trace()

    print(len(incremental_dataset))
