import json
import os
from typing import List, Dict
from PIL import Image

import numpy as np
import torch
from torch import tensor, zeros, stack
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from VRD import config
from VRD.util import load_image, load_json


def make_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


class VRDDataset(Dataset):
    def __init__(self, json_file: json, image_path: str):
        self.json_file = json_file
        self.image_path = image_path

        vocab_path: str = config.path['objects_path']
        rel_path: str = config.path['predicate_path']
        self.vocab = load_json(vocab_path)
        self.rel = load_json(rel_path)

        self.image_id_list: List = []
        self.spo_list: List = []
        self.make_dataset()

        self.transform = make_transform()

    def __len__(self) -> int:
        return len(self.image_id_list)

    def __getitem__(self, idx):
        image_id = self.image_id_list[idx]
        n_spo = len(self.spo_list[idx])
        label: np.array = np.zeros((n_spo, 3))
        target: Tensor = torch.zeros(len(self.vocab))

        image = Image.open(os.path.join(self.image_path, image_id)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        for spo_idx, spo in enumerate(self.spo_list[idx]):

            label[spo_idx, 0] = spo[0]
            target[spo[0]] = 1
        label = Tensor(self.spo_list[idx])
        return image, label

    def make_dataset(self):
        for idx, (image_id, label) in enumerate(self.json_file.items()):
            spo_list = []
            n_spo = len(label)

            if n_spo == 0:
                continue
            self.image_id_list.append(image_id)
            for spo_idx, spo in enumerate(label):
                subject_id = spo['subject']['category']
                subject_bbox = spo['subject']['bbox']
                object_id = spo['object']['category']
                object_bbox = spo['object']['bbox']

                spo_list.append([subject_id, subject_bbox])
                spo_list.append([object_id, object_bbox])
            self.spo_list.append(spo_list)


def scene_graph_collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, labels = zip(*data)
    images = stack(images, 0)
    batch_size = len(labels)

    length = [label.shape[0] for label in labels]
    print(max(length))
    targets = zeros(batch_size, max(length), 3).long()
    print(targets.shape)
    for batch_idx, label in enumerate(labels):
        for i, line in enumerate(label):
            targets[batch_idx, i] = line
    return images, targets, length


def get_dataloader(args):
    train_path = config.path['json_train_dataset_path']
    train_image_path = config.path['train_image_path']
    json_train: json = load_json(train_path)
    train_dataset = VRDDataset(json_file=json_train, image_path=train_image_path)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)

    test_image_path = config.path['val_image_path']
    test_path = config.path['json_test_dataset_path']
    json_test: json = load_json(test_path)
    test_dataset = VRDDataset(json_file=json_test, image_path=test_image_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    return train_data_loader, test_dataloader
