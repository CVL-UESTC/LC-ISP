# Copyright 2020 by Andrey Ignatov. All Rights Reserved.
"""Load LC-ISP samples from split IDs and metadata."""

from torch.utils.data import Dataset
import numpy as np
import torch
import json
import os
import random
import cv2


def trans4channels(raw):
    B  = raw[0::2, 0::2]
    GB = raw[0::2, 1::2]
    GR  = raw[1::2, 0::2]
    R = raw[1::2, 1::2]

    data = np.stack((B,GR,GB,R))
    return data


class UltraISPDataset_pro(Dataset):
    def __init__(self, dataset_dir, bl, wl, test=False, ret_path=False):
        self.test = test
        self.bl = bl
        self.wl = wl
        self.dataset_dir = dataset_dir
        self.ret_path = ret_path
        if wl <= bl:
            raise ValueError('RAW white level must be greater than black level.')
        split = 'test' if test else 'train'
        self.txt = os.path.join(dataset_dir, f'{split}_ids.txt')
        with open(self.txt, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]
        if not self.ids:
            raise ValueError(f'Empty dataset split: {self.txt}')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError(f'Duplicate sample IDs in {self.txt}')
        meta_path = os.path.join(dataset_dir, f'{split}_meta.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        if metadata.get('split') != split or metadata.get('complete') is not True:
            raise ValueError(f'{meta_path} must contain a complete {split} split.')
        self.data_root = os.path.join(dataset_dir, metadata['files_relative_to'])
        self.pairs = []
        for sample_id in self.ids:
            try:
                patch = metadata['patches'][sample_id]
                self.pairs.append(tuple(patch[field] for field in ('raw', 'proraw', 'target')))
            except KeyError as error:
                raise ValueError(f'{meta_path}: missing {error.args[0]!r} for sample {sample_id!r}.') from error

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_name, pro_name, ref_name = self.pairs[idx]

        raw_path = os.path.join(self.data_root, raw_name)
        pro_path = os.path.join(self.data_root, pro_name)
        ref_path = os.path.join(self.data_root, ref_name)
        raw_image = np.float32(np.fromfile(raw_path,dtype=np.uint16).reshape(448,448))
        raw_image = trans4channels(raw_image)
        raw_image = np.clip(((raw_image - self.bl) / (self.wl - self.bl)),0,1)
        pro_image = np.fromfile(pro_path, dtype=np.uint16).reshape(448, 448, 3)

        pro_image = pro_image.astype(np.float32) / 65535.0  # 归一化[1](@ref)
        pro_image = pro_image.transpose((2, 0, 1))  # HWC→CHW[1](@re

        dslr_image = cv2.imread((ref_path))[:,:,::-1]
        dslr_image = np.float32(dslr_image) / 255.0
        dslr_image = dslr_image.transpose((2, 0, 1))

        if self.test is False:
            raw_image, pro_image, dslr_image = self._augment(raw_image, pro_image, dslr_image)
        raw_image = torch.from_numpy(raw_image)
        pro_image = torch.from_numpy(pro_image)
        dslr_image = torch.from_numpy(dslr_image)
        if self.ret_path:
            return raw_image, pro_image, dslr_image, raw_path
        else:
            return raw_image, pro_image, dslr_image


    def _augment(self, *imgs):
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        def _augment_func(img, hflip, vflip, rot90):
            if hflip:   img = img[:, :, ::-1]
            if vflip:   img = img[:, ::-1, :]
            if rot90:   img = img.transpose(0, 2, 1) # CHW
            return np.ascontiguousarray(img)
        return (_augment_func(img, hflip, vflip, rot90) for img in imgs)
