import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import image_to_normalized_tensor, get_mask_to_tensor


class SRNDataset(torch.utils.data.Dataset):
    """
    Dataset from SRN (V. Sitzmann et al. 2020)
    """

    def __init__(
        self,
        path,
        stage="train",
        image_size=(128, 128),
        world_scale=1.0,
    ):
        """
        :param stage train | val | test
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.base_path = path + "_" + stage
        self.dataset_name = os.path.basename(path)

        print("Loading SRN dataset", self.base_path, "name:", self.dataset_name)
        self.stage = stage
        assert os.path.exists(self.base_path)

        is_chair = "chair" in self.dataset_name
        if is_chair and stage == "train":
            # Ugly thing from SRN's public dataset
            tmp = os.path.join(self.base_path, "chairs_2.0_train")
            if os.path.exists(tmp):
                self.base_path = tmp

        self.intrins = sorted(
            glob.glob(os.path.join(self.base_path, "*", "intrinsics.txt"))
        )
        # self.image_to_tensor = image_to_normalized_tensor()
        self.mask_to_tensor = get_mask_to_tensor()

        self.image_size = image_size
        self.world_scale = world_scale
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)).cpu()

        if is_chair:
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8
        self.lindisp = False

    def __len__(self):
        return len(self.intrins)

    def __getitem__(self, index):
        intrin_path = self.intrins[index]
        dir_path = os.path.dirname(intrin_path)
        rgb_paths = sorted(glob.glob(os.path.join(dir_path, "rgb", "*.png")))
        pose_paths = sorted(glob.glob(os.path.join(dir_path, "pose", "*.txt")))

        assert len(rgb_paths) == len(pose_paths)

        with open(intrin_path, "r") as intrinfile:
            lines = intrinfile.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            height, width = map(int, lines[-1].split())

            if height != self.image_size[0]:
                scale = self.image_size[0] / height
                focal *= scale
                cx *= scale
                cy *= scale

        all_imgs = []
        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            # normalized_img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(
                np.loadtxt(pose_path, dtype=np.float32).reshape(
                    4, 4))  #.to(self._coord_trans.device)
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError(
                    "ERROR: Bad image at", rgb_path, "please investigate!"
                )
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(torch.from_numpy(img / 255))  # TODO
            # all_imgs_tensor.append(normalized_img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        # all_imgs_tensor = torch.stack(all_imgs_tensor)
        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[1:3] != self.image_size:
            if scale == 1:  # hasn't been modified yet
                scale = self.image_size[0] / all_imgs.shape[-2]
                focal *= scale
                cx *= scale
                cy *= scale
                all_bboxes *= scale

            all_imgs = F.interpolate(
                all_imgs.permute(0, 3, 1, 2), size=self.image_size, mode="area"
            ).permute(0, 2, 3, 1)
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale

        # TODO
        focal = torch.tensor(focal, dtype=torch.float32)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "c": torch.tensor([cx, cy], dtype=torch.float32),
            "images": all_imgs,  # 250  3 * 128 * 128
            # "images_tensor": all_imgs_tensor.permute(
            #     0, 2, 3, 1
            # ),  # 250  * 128 * 128 * 3
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }

        # np version
        # result = {
        #     "path": dir_path,
        #     "img_id": index,
        #     "focal": focal,
        #     "c": np.array([cx, cy], dtype=np.float32),
        #     "images": all_imgs.permute(0, 2, 3, 1)
        #     .cpu()
        #     .numpy(),  # 250  * 128 * 128 * 3
        #     "masks": all_masks.cpu().numpy(),
        #     "bbox": all_bboxes.cpu().numpy(),
        #     "poses": all_poses.cpu().numpy(),
        # }

        return result


class SceneInstanceDataset:
    """This creates a dataset class for a single object instance (such as a single car)."""

    def __init__(
        self,
        instance_idx,
        instance_dir,
        specific_observation_idcs=None,  # For few-shot case: Can pick specific observations only
        img_sidelength=None,
        num_images=-1,
    ):
        self.instance_idx = instance_idx
        self.img_sidelength = img_sidelength
        self.instance_dir = instance_dir
        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        color_dir = os.path.join(instance_dir, "rgb")
        pose_dir = os.path.join(instance_dir, "pose")
        param_dir = os.path.join(instance_dir, "params")

        if not os.path.isdir(color_dir):
            print("Error! root dir %s is wrong" % instance_dir)
            return

        self.has_params = os.path.isdir(param_dir)

        self.color_paths = sorted(glob.glob(os.path.join(color_dir, "rgb", "*.png")))
        self.pose_paths = sorted(glob.glob(os.path.join(pose_dir, "pose", "*.txt")))
        # self.color_paths = sorted(data_util.glob_imgs(color_dir))
        # self.pose_paths = sorted(glob(os.path.join(pose_dir, "*.txt")))

        # if self.has_params:
        #     self.param_paths = sorted(glob(os.path.join(param_dir, "*.txt")))
        # else:
        #     self.param_paths = []

        if specific_observation_idcs is not None:
            self.color_paths = pick(self.color_paths, specific_observation_idcs)
            self.pose_paths = pick(self.pose_paths, specific_observation_idcs)
            self.param_paths = pick(self.param_paths, specific_observation_idcs)
        elif num_images != -1:
            idcs = np.linspace(
                0, stop=len(self.color_paths), num=num_images, endpoint=False, dtype=int
            )
            self.color_paths = pick(self.color_paths, idcs)
            self.pose_paths = pick(self.pose_paths, idcs)
            self.param_paths = pick(self.param_paths, idcs)

    def set_img_sidelength(self, new_img_sidelength):
        """For multi-resolution training: Updates the image sidelength with whichimages are loaded."""
        self.img_sidelength = new_img_sidelength

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):

        intrinsics, _, _, _ = util.parse_intrinsics(
            os.path.join(self.instance_dir, "intrinsics.txt"),
            trgt_sidelength=self.img_sidelength,
        )
        intrinsics = torch.Tensor(intrinsics).float()

        rgb = data_util.load_rgb(self.color_paths[idx], sidelength=self.img_sidelength)
        rgb = rgb.reshape(3, -1).transpose(1, 0)

        pose = torch.from_numpy(
            np.loadtxt(self.pose_paths[idx], dtype=np.float32).reshape(
                4, 4))  #.to(self._coord_trans.device)
        pose = pose @ self._coord_trans

        data_util.load_pose(self.pose_paths[idx])

        if self.has_params:
            params = load_params(self.param_paths[idx])
        else:
            params = np.array([0])

        uv = np.mgrid[0 : self.img_sidelength, 0 : self.img_sidelength].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).long()  # UV
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "instance_idx": torch.Tensor([self.instance_idx]).squeeze(),
            "rgb": torch.from_numpy(rgb).float(),
            "pose": torch.from_numpy(pose).float(),
            "uv": uv,
            "param": torch.from_numpy(params).float(),
            "intrinsics": intrinsics,
        }
        return sample


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def load_params(filename):
    lines = open(filename).read().splitlines()

    params = np.array([float(x) for x in lines[0].split()]).astype(np.float32).squeeze()
    return params