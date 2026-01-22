import ast
import os

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

# ImageNet
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES

IMAGENET_LABEL_DICT = ast.literal_eval(open("models/bos_model/demo/resources/imagenet_class_labels.txt", "r").read())


class DataLoaderImageNet(Dataset):
    IMAGENET_CLASS_TO_IDX = {name: idx for idx, name in enumerate(IMAGENET2012_CLASSES)}
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        input_loc: str,
        model_id: str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        self.input_loc = input_loc.rstrip("/")
        self.batch_size = int(batch_size)
        self.model_id = model_id

        files = []
        for root, _, filenames in os.walk(self.input_loc):
            for fname in filenames:
                _, ext = os.path.splitext(fname)
                if ext.lower() in self.IMG_EXTS:
                    files.append(os.path.join(root, fname))

        files = sorted(set(files))
        if shuffle and len(files) > 1:
            g = torch.Generator()
            g.manual_seed(int(seed))
            perm = torch.randperm(len(files), generator=g).tolist()
            files = [files[i] for i in perm]

        self.files = files
        self.total_files = len(self.files)
        self.current_file_idx = 0

        try:
            self._preprocessor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True, local_files_only=True)
        except EnvironmentError as e:
            self._preprocessor = AutoImageProcessor.from_pretrained(self.model_id, use_fast=True)
        self._torch_loader = DataLoader(self, batch_size=self.batch_size, shuffle=False)

        logger.debug(
            f"[DataLoaderImageNet] Found {self.total_files} files under {self.input_loc} (shuffle={shuffle}, seed={seed})"
        )

    def _label_from_path(self, fpath):
        parent = os.path.basename(os.path.dirname(fpath))
        name, ext = os.path.splitext(os.path.basename(fpath))
        synth = os.path.join(os.path.dirname(os.path.dirname(fpath)), name + "_" + parent + ext)
        cls_name = synth.rsplit("/", 1)[-1].rsplit(".", 1)[0].rsplit("_", 1)[-1]
        return self.IMAGENET_CLASS_TO_IDX[cls_name]

    def reset(self):
        self.current_file_idx = 0

    def __len__(self):
        return self.total_files

    def __getitem__(self, idx):
        if self.current_file_idx >= self.total_files:
            self.reset()

        fpath = self.files[self.current_file_idx]
        self.current_file_idx += 1

        try:
            img = Image.open(fpath)
            label = self._label_from_path(fpath)

            if img.mode == "L":
                img = img.convert("RGB")

            px = self._preprocessor(img, return_tensors="pt")["pixel_values"][0]
            return {"pixel_values": px, "labels": torch.tensor(label, dtype=torch.long)}

        except Exception as e:
            logger.warning(f"[DataLoaderImageNet] Failed to load {fpath}: {e}")
            return self.__getitem__(idx)

    def __iter__(self):
        return iter(self._torch_loader)


class DataLoaderImageNetSample(DataLoaderImageNet):
    def _label_from_path(self, fpath):
        fname = os.path.basename(fpath)
        stem, _ = os.path.splitext(fname)
        parts = stem.split("_")
        cls_name = parts[-1]
        return self.IMAGENET_CLASS_TO_IDX[cls_name]


def get_data_loader(data_dir):
    if data_dir == "models/bos_model/demo/dataset/sample/":
        return DataLoaderImageNetSample
    else:
        return DataLoaderImageNet
