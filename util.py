import pickle
import json
from PIL import Image

import numpy as np
from torchvision import transforms

from VRD.config import path


def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 244], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image


def make_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def load_pkl(path: str):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl


def load_json(path: str) -> json:
    with open(path, 'rb') as file:
        json_file: json = json.load(file)
    return json_file


def load_tag_representation(vocab) -> np.array:
    with open(path['tag_representation_path'], 'rt', encoding='UTF-8') as txt_file:
        tag_representation = np.zeros((len(vocab), 300))
        for idx, line in enumerate(txt_file.readlines()):
            line = line.split()
            tag_representation[idx] = list(map(float, line[1:]))
    return tag_representation


def load_spo_list():
    spo_train_list, spo_test_list = [], []
    with open(path['spo_train_path'], 'rt', encoding='UTF-8') as txt_file:
        for line in txt_file.readlines():
            line = line.split()
            tag1 = ''.join(line[0])
            tag2 = ''.join(line[1])
            relation = line[2:]
            spo_train_list.append((tag1, tag2, relation))

    with open(path['spo_test_path'], 'rt', encoding='UTF-8') as txt_file:
        for line in txt_file.readlines():
            line = line.split()
            tag1 = ''.join(line[0])
            tag2 = ''.join(line[1])
            relation = line[2:]
            spo_test_list.append((tag1, tag2, relation))
    return spo_train_list, spo_test_list
