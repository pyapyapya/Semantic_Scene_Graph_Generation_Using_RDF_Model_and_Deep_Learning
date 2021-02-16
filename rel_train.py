import argparse
import numpy as np
import os
from typing import List

import torch
import matplotlib.pyplot as plt
from torch import nn, save, Tensor, LongTensor, FloatTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from VRD.config import path
from VRD.vrd_preprocess import TwoTag
from VRD.model import RNNRelationModel
from VRD.data_loader import get_dataloader
from VRD.util import load_json, load_pkl, load_tag_representation, load_spo_list


def get_tag_representation(spo_list, rel, tag_representation, tag1, tag2, idx):
    if idx is None:
        relation: np.array = np.zeros(len(rel))
    else:
        relation: np.array = np.array(list(map(int, spo_list[idx][2])))
    wi: Tensor = Tensor(tag_representation[tag1])
    wj: Tensor = Tensor(tag_representation[tag2])

    return wi, relation, wj


def post_processing(tag1_name, tag2_name, outputs, rel):
    animal_object = ['person', 'dog', 'horse', 'mouse', 'bear', 'giraffe', 'cat', 'elephant']
    action_able_object = ['person', 'truck', 'bus', 'car', 'motorcycle', 'bike', 'van', 'hand', 'elephant', 'dog',
                          'horse', 'plane', 'cart', 'mouse', 'ball', 'bear', 'giraffe', 'suitcase', 'cat']
    wear_able_object = ['shirt', 'glasses', 'hat', 'jacket', 'bag', 'shoe', 'shoes', 'helmet', 'coat', 'skateboard',
                        'shorts', 'jeans', 'sunglasses', 'tie', 'snowboard', 'suitcase']

    action_predicate = ['touch', 'wear', 'carry', 'ride', 'watch', 'eat', 'sleep on', 'rest on', 'hit', 'feed', 'kick',
                        'hold', 'look', 'use', 'stand on', 'contain', 'fly']
    drive_predicate = ['drive on', 'drive']

    predicate = action_predicate + drive_predicate

    check_predicate_index = []

    for relation in predicate:
        rel_idx = rel.index(relation)
        check_predicate_index.append(rel_idx)

    # delete 'wear' predicate in animal objects
    if tag1_name in animal_object and tag2_name in animal_object:
        rel_idx = rel.index('wear')
        outputs[rel_idx] = 0

    # delete predicate in objects
    elif (tag1_name not in action_able_object) and (tag2_name in action_able_object):
        print('before outputs', outputs)
        outputs[check_predicate_index] = 0
        print('after outputs', outputs)

    return outputs


def tag_relation_model_train(args, data_loader, spo_list, tag_representation, vocab, rel, two_tag):
    device = torch.device('cuda')
    relation_model = RNNRelationModel(embed_size=args.embed_size, hidden_size=args.hidden_size,
                                      tag_representation=tag_representation, vocab_size=len(vocab),
                                      rel_size=len(rel), num_layer=args.num_layer).to(device)
    relation_model = relation_model.train()

    criterion = nn.BCEWithLogitsLoss()
    params = list(relation_model.parameters())
    optim = Adam(params, lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8)
    lr_schedular = StepLR(optim, step_size=100, gamma=0.1)

    epoch_list = []
    loss_lst = []
    loss_value = 0
    for epoch in range(args.epochs):
        print('Epoch: ', epoch + 1, '/', args.epochs)
        cnt_tags_relationships = set()
        cnt = 0
        for i, (image, labels) in enumerate(data_loader):
            loss = 0
            optim.zero_grad()
            for idx, spo in enumerate(labels):
                tag1_idx = spo[idx][0].long()
                tag2_idx = spo[idx][1].long()
                predicate_id = spo[idx][2].long()
                tag1_name = vocab[tag1_idx]
                tag2_name = vocab[tag2_idx]

                two_tag_idx = two_tag(tag1_name, tag2_name)
                if two_tag_idx is None or predicate_id.item() == 0:
                    continue
                cnt_tags_relationships.add(two_tag_idx)
                wi, relation, wj = get_tag_representation(spo_list, rel, tag_representation,
                                                          tag1_idx, tag2_idx, two_tag_idx)
                targets = FloatTensor([relation]).to(device)
                outputs = relation_model(tag1_idx, tag2_idx).cuda()
                loss = criterion(outputs, targets.float()).cuda()
                loss.backward()
                optim.step()
                loss_value = loss.item()
                print('loss_value', loss_value)
        lr_schedular.step()
        epoch_list.append(epoch)
        loss_lst.append(loss_value)

    plt.plot(epoch_list, loss_lst)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.loss_path)
    plt.clf()
    save(relation_model.state_dict(), args.decoder_path)


def test_rdf(args, data_loader, spo_list, tag_representation, vocab, rel, two_tag):
    device = torch.device('cuda')
    relation_model = RNNRelationModel(embed_size=args.embed_size, hidden_size=args.hidden_size,
                                      tag_representation=tag_representation,
                                      vocab_size=len(vocab), rel_size=len(rel), num_layer=args.num_layer).eval()

    relation_model = relation_model.eval().to(device)

    relation_model.load_state_dict(torch.load(args.decoder_path))

    output_list: List = []
    target_list: List = []
    cnt_tags_relationships = set()

    for i, (image, label) in enumerate(data_loader):
        output_class = np.zeros(len(rel))
        target_class = np.zeros(len(rel))

        for idx, spo in enumerate(label):
            tag1_idx = spo[idx][0].long()
            tag2_idx = spo[idx][1].long()
            predicate_id = spo[idx][2].long()

            tag1_name = vocab[tag1_idx]
            tag2_name = vocab[tag2_idx]

            two_tag_idx = two_tag(tag1_name, tag2_name)
            if (two_tag_idx is None) or (two_tag_idx in cnt_tags_relationships):
                continue

            cnt_tags_relationships.add(two_tag_idx)
            wi, relation, wj = get_tag_representation(spo_list, rel, tag_representation,
                                                      tag1_idx, tag2_idx, two_tag_idx)
            outputs = relation_model(tag1_idx, tag2_idx)
            outputs = torch.sigmoid(outputs).data > 0.5
            _, sample_ids = outputs.nonzero(as_tuple=True)
            sample_ids = sample_ids.cpu().numpy()
            output_class[sample_ids] = 1

            # print('outputs_class', output_class)
            # print('target', relation)
            # target_class[relation] = 1
            # output_class = post_processing(tag1_name, tag2_name, output_class, rel)
            # print(output_class)
            output_list.append(output_class)
            target_list.append(relation)

    print(len(cnt_tags_relationships))
    output_list: np.array = np.array(output_list)
    target_list: np.array = np.array(target_list)
    target_name = rel
    precision = precision_score(y_pred=output_list, y_true=target_list, average='samples')
    recall = recall_score(y_pred=output_list, y_true=target_list, average='samples')
    f1 = f1_score(y_pred=output_list, y_true=target_list, average='samples')
    report = classification_report(y_pred=output_list, y_true=target_list, target_names=target_name)

    print(precision)
    print(recall)
    print(f1)
    print(report)


def main(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    spo_train_list, spo_test_list = load_spo_list()
    vocab: List = load_json(path['objects_path'])
    rel: List = load_json(path['predicate_path'])
    train_two_tag = load_pkl(path['train_two_tag_path'])
    test_two_tag = load_pkl(path['train_two_tag_path'])
    tag_representation = load_tag_representation(vocab)
    train_dataloader, test_dataloader = get_dataloader(args=args)

    print(len(rel))

    tag_relation_model_train(args=args, data_loader=train_dataloader, spo_list=spo_train_list,
                             tag_representation=tag_representation, vocab=vocab, rel=rel, two_tag=train_two_tag)
    test_rdf(args=args, data_loader=test_dataloader, spo_list=spo_train_list,
             tag_representation=tag_representation, vocab=vocab, rel=rel, two_tag=train_two_tag)


if __name__ == '__main__':
    n_train = '10'
    parser = argparse.ArgumentParser()

    # Load Path
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained model')

    # Save path
    parser.add_argument('--encoder_path', type=str, default='E:\\untitled3\VRD\models\encoder' + '9' + '.pth')
    parser.add_argument('--decoder_path', type=str, default='models/decoder' + n_train + '.pth')
    parser.add_argument('--loss_path', type=str, default='models/loss_graph' + n_train + '.png')
    parser.add_argument('--precision_path', type=str, default='models/precision_graph' + n_train + '.png')
    parser.add_argument('--recall_path', type=str, default='models/recall_graph' + n_train + '.png')
    parser.add_argument('--f1_score_path', type=str, default='models/f1_score_graph' + n_train + '.png')

    # RNN Model HyperParameters
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of rnn hidden states')
    parser.add_argument('--num_layer', type=int, default=1, help='number of layers in rnn')

    # Training HyperParameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    args = parser.parse_args()
    print(args)
    main(args)
