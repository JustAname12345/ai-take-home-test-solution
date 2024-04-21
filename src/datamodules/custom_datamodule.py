from typing import Optional
import os
from PIL import Image
import scipy.io
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.datasets import Flowers102
from pytorch_lightning import LightningDataModule

flower_names = {
        0: "Pink primrose",
        1: "Hard-leaved pocket orchid",
        2: "Canterbury bells",
        3: "Sweet pea",
        4: "English marigold",
        5: "Tiger lily",
        6: "Moon orchid",
        7: "Bird of paradise",
        8: "Monkshood",
        9: "Globe thistle",
        10: "Snapdragon",
        11: "Colt's foot",
        12: "King protea",
        13: "Spear thistle",
        14: "Yellow iris",
        15: "Globe-flower",
        16: "Purple coneflower",
        17: "Peruvian lily",
        18: "Balloon flower",
        19: "Giant white arum lily",
        20: "Fire lily",
        21: "Pincushion flower",
        22: "Fritillary",
        23: "Red ginger",
        24: "Grape hyacinth",
        25: "Corn poppy",
        26: "Prince of Wales feathers",
        27: "Stemless gentian",
        28: "Artichoke",
        29: "Sweet william",
        30: "Carnation",
        31: "Garden phlox",
        32: "Love in the mist",
        33: "Mexican aster",
        34: "Alpine sea holly",
        35: "Ruby-lipped cattleya",
        36: "Cape flower",
        37: "Great masterwort",
        38: "Siam tulip",
        39: "Lenten rose",
        40: "Barbeton daisy",
        41: "Daffodil",
        42: "Sword lily",
        43: "Poinsettia",
        44: "Bolero deep blue",
        45: "Wallflower",
        46: "Marigold",
        47: "Buttercup",
        48: "Oxeye daisy",
        49: "Common dandelion",
        50: "Petunia",
        51: "Wild pansy",
        52: "Primula",
        53: "Sunflower",
        54: "Pelargonium",
        55: "Bishop of Llandaff",
        56: "Gaura",
        57: "Geranium",
        58: "Orange dahlia",
        59: "Pink-yellow dahlia",
        60: "Cautleya spicata",
        61: "Japanese anemone",
        62: "Black-eyed susan",
        63: "Silverbush",
        64: "Californian poppy",
        65: "Osteospermum",
        66: "Spring crocus",
        67: "Bearded iris",
        68: "Windflower",
        69: "Tree poppy",
        70: "Gazania",
        71: "Azalea",
        72: "Water lily",
        73: "Rose",
        74: "Thorn apple",
        75: "Morning glory",
        76: "Passion flower",
        77: "Lotus",
        78: "Toad lily",
        79: "Anthurium",
        80: "Frangipani",
        81: "Clematis",
        82: "Hibiscus",
        83: "Columbine",
        84: "Desert-rose",
        85: "Tree mallow",
        86: "Magnolia",
        87: "Cyclamen",
        88: "Watercress",
        89: "Canna lily",
        90: "Hippeastrum",
        91: "Bee balm",
        92: "Pink quill",
        93: "Foxglove",
        94: "Bougainvillea",
        95: "Camellia",
        96: "Mallow",
        97: "Mexican petunia",
        98: "Bromelia",
        99: "Blanket flower",
        100: "Trumpet creeper",
        101: "Blackberry lily"
    }
def load_flower_names(index):

    return flower_names.get(index, "Unknown")

def generate_description(flower_name):
    """Generates a simple description for a given flower name."""
    return f"This is a beautiful image of a {flower_name}, known for its distinct characteristics and vibrant colors."


class Flowers102Dataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.data = Flowers102(root=root_dir, split=split, download=True, transform=transform)
        mat_path = root_dir+'/flowers-102/imagelabels.mat'
        self.labels_mat = self.load_labels(mat_path)

    def load_labels(self, mat_file_path):
        """Load labels from the specified MATLAB file."""
        mat = scipy.io.loadmat(mat_file_path)
        return mat['labels'][0] - 1  # Adjust index to zero-based if MATLAB file is one-based

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, _ = self.data[idx]  # Original dataset index is not used
        label = self.labels_mat[idx]  # Get the actual label using index
        flower_name = load_flower_names(label)  # Get the flower name using the label index
        description = generate_description(flower_name)

        return image, description

class CustomDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int=8, num_workers: int=4, pin_memory: bool = False, img_size: int=256, augmentations: bool=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.img_size = img_size
        self.augmentations = augmentations
        self.transforms = self._setup_transforms()

    def _setup_transforms(self):
        base_transforms = [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        if self.augmentations:
            augmented_transforms = [transforms.ColorJitter(), transforms.RandomHorizontalFlip()]
            return transforms.Compose(augmented_transforms + base_transforms)
        return transforms.Compose(base_transforms)

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = Flowers102Dataset(self.data_dir, 'train', transform=self.transforms)
        self.val_dataset = Flowers102Dataset(self.data_dir, 'val', transform=self.transforms)
        self.test_dataset = Flowers102Dataset(self.data_dir, 'test', transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)