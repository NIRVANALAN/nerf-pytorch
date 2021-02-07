import torch
from torchvision import datasets, transforms

dst_path_v0 = (
    "/mnt/lustre/yslan/Repo/NVS/Projects/volume_rendering/srn_dataset/chairs_pool/"
)


data_transform = transforms.Compose(
    [
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.9084, 0.9045, 0.9019], std=[0.2319, 0.2415, 0.2483]
        ),
    ]
)

external_dataset = datasets.ImageFolder(root=dst_path_v0, transform=data_transform)

dataset_loader = torch.utils.data.Dataloader(
    external_dataset, batch_size=32, shuffle=True, num_workers=4
)
