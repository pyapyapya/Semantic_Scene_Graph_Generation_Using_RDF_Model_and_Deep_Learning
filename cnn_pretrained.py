import argparse
import numpy as np
import os
import pickle
from PIL import Image
from typing import List

import torch
import matplotlib.pyplot as plt
from torch import nn, save
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from config import path
from model import CNN
from data_loader import get_dataloader
from util import load_json


def cnn_only_train(args, data_loader, vocab):
    device = torch.device('cuda')

    encoder = CNN(vocab).to(device)

    criterion = nn.CrossEntropyLoss()
    params = list(encoder.parameters())
    optim = Adam(params, lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8)
    lr_schedular = StepLR(optim, step_size=200, gamma=0.1)

    epoch_list = []
    loss_lst = []
    for epoch in range(args.epochs):
        print('Epoch: ', epoch+1, '/', args.epochs)

        loss_value = 0
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            label = label.to(device)
            outputs = encoder(image)
            optim.zero_grad()
            loss = criterion(outputs.float(), label.float()).to(device)
            loss.backward()
            optim.step()

            loss_value = loss.item()
        lr_schedular.step()
        epoch_list.append(epoch)
        loss_lst.append(loss_value)
        print('Loss: ', loss_value)

    plt.plot(epoch_list, loss_lst)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.loss_path)
    plt.clf()
    save(encoder.state_dict(), args.encoder_path)


def main(args):
    vocab: List = load_json(path['objects_path'])

    train_dataloader, test_dataloader = get_dataloader(args)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    cnn_only_train(args=args, vocab=vocab, data_loader=train_dataloader)


if __name__ == '__main__':
    n_train = '0'
    parser = argparse.ArgumentParser()

    # Load Path
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained model')

    # Save path
    parser.add_argument('--encoder_path', type=str, default='models/encoder' + n_train + '.pth')
    parser.add_argument('--decoder_path', type=str, default='models/decoder' + n_train + '.pth')
    parser.add_argument('--loss_path', type=str, default='models/cnn_loss_graph' + n_train + '.png')
    parser.add_argument('--precision_path', type=str, default='models/cnn_precision_graph' + n_train + '.png')
    parser.add_argument('--recall_path', type=str, default='models/cnn_recall_graph' + n_train + '.png')
    parser.add_argument('--f1_score_path', type=str, default='models/cnn_f1_score_graph' + n_train + '.png')

    # Training HyperParameters
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
