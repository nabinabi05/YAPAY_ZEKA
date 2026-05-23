import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']


def is_image_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def get_image_paths(dir_path: str) -> list:
    assert os.path.isdir(dir_path), \
        f"Directory '{dir_path}' does not exist or was not mounted successfully."
    images = []
    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                images.append(os.path.join(root, fname))
    return images


class ThermalVisibleDataset(Dataset):
    def __init__(self, thermal_dir, visible_dir, mode="paired", is_train=True,
                 img_size=(256, 256), split_ratio=0.8):
        super().__init__()
        assert mode in ("paired", "unpaired")
        all_thermal = get_image_paths(thermal_dir)
        all_visible = get_image_paths(visible_dir)
        if mode == "paired":
            assert len(all_thermal) == len(all_visible), \
                f"Paired mode: {len(all_thermal)} thermal vs {len(all_visible)} visible"
        split = int(len(all_thermal) * split_ratio)
        split_v = int(len(all_visible) * split_ratio)
        self.thermal_paths = all_thermal[:split] if is_train else all_thermal[split:]
        self.visible_paths = all_visible[:split_v] if is_train else all_visible[split_v:]
        self.mode = mode
        self.is_train = is_train
        self.img_size = img_size
        self.scale_size = (int(img_size[0] * 1.12), int(img_size[1] * 1.12))
        self.thermal_size = len(self.thermal_paths)
        self.visible_size = len(self.visible_paths)
        self.to_tensor_1ch = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.to_tensor_3ch = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    def __getitem__(self, index):
        thermal_path = self.thermal_paths[index % self.thermal_size]
        if self.mode == "paired":
            visible_path = self.visible_paths[index % self.visible_size]
        else:
            visible_path = self.visible_paths[random.randint(0, self.visible_size - 1)]
        thermal_img = Image.open(thermal_path).convert("L")
        visible_img = Image.open(visible_path).convert("RGB")

        if self.is_train:
            thermal_img = TF.resize(thermal_img, self.scale_size, TF.InterpolationMode.BICUBIC)
            visible_img = TF.resize(visible_img, self.scale_size, TF.InterpolationMode.BICUBIC)
            i, j, h, w = transforms.RandomCrop.get_params(thermal_img, self.img_size)
            thermal_img = TF.crop(thermal_img, i, j, h, w)
            visible_img = TF.crop(visible_img, i, j, h, w)
            if random.random() > 0.5:
                thermal_img = TF.hflip(thermal_img)
                visible_img = TF.hflip(visible_img)
        else:
            thermal_img = TF.resize(thermal_img, self.img_size, TF.InterpolationMode.BICUBIC)
            visible_img = TF.resize(visible_img, self.img_size, TF.InterpolationMode.BICUBIC)

        return {
            "thermal": self.to_tensor_1ch(thermal_img),
            "visible": self.to_tensor_3ch(visible_img),
            "thermal_path": thermal_path, "visible_path": visible_path,
        }

    def __len__(self):
        return max(self.thermal_size, self.visible_size)

def create_dataloader(thermal_dir, visible_dir, mode="paired", is_train=True,
                      batch_size=4, num_workers=2, img_size=(256,256), split_ratio=0.8):
    # num_workers=2 keeps Colab happy; 4+ can stall on the free runtime's CPU.
    dataset = ThermalVisibleDataset(thermal_dir, visible_dir, mode=mode,
                                    is_train=is_train, img_size=img_size, split_ratio=split_ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                      num_workers=num_workers, pin_memory=True, drop_last=is_train,
                      persistent_workers=(num_workers > 0))


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    import shutil
    print("Initiating ThermalVisibleDataset diagnostics...")

    mock_t = "mock_data/thermal"
    mock_v = "mock_data/visible"
    os.makedirs(mock_t, exist_ok=True)
    os.makedirs(mock_v, exist_ok=True)

    for i in range(4):
        Image.new('L',   (300, 300)).save(os.path.join(mock_t, f'img_{i:02d}.jpg'))
        Image.new('RGB', (300, 300)).save(os.path.join(mock_v, f'img_{i:02d}.jpg'))

    print("Testing paired loader...")
    loader = create_dataloader(mock_t, mock_v, mode="paired", batch_size=2, num_workers=0)
    batch  = next(iter(loader))
    print(f"  Thermal : {batch['thermal'].shape}  range [{batch['thermal'].min():.2f}, {batch['thermal'].max():.2f}]")
    print(f"  Visible : {batch['visible'].shape}  range [{batch['visible'].min():.2f}, {batch['visible'].max():.2f}]")

    print("Testing unpaired loader...")
    loader2 = create_dataloader(mock_t, mock_v, mode="unpaired", batch_size=2, num_workers=0)
    batch2  = next(iter(loader2))
    print(f"  Thermal : {batch2['thermal'].shape}")
    print(f"  Visible : {batch2['visible'].shape}")

    shutil.rmtree("mock_data")
    print("\nDataset pipeline verified successfully!")
