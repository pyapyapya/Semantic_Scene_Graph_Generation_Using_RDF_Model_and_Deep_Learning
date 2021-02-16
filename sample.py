import argparse
import numpy as np
import os
import pickle
from typing import List
from PIL import Image

import torch
import matplotlib.pyplot as plt
import rdflib
import networkx as nx

from torch import nn, save, Tensor, LongTensor, FloatTensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph

from VRD.config import path
from VRD.vrd_preprocess import TwoTag
from VRD.model import CNN, RNNRelationModel
from VRD.data_loader import get_dataloader
from VRD.util import load_image, load_json, load_pkl, load_tag_representation, load_spo_list, make_transform


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
                        'hold', 'look', 'use', 'stand on', 'contain', 'fly', 'has']
    drive_predicate = ['drive on', 'drive']

    predicate = action_predicate + drive_predicate

    check_predicate_index = []

    for relation in predicate:
        rel_idx = rel.index(relation)
        check_predicate_index.append(rel_idx)

    # delete 'wear' predicate in animal objects
    if tag1_name in animal_object and tag2_name in animal_object:
        rel_idx = rel.index('wear')
        outputs[0][rel_idx] = 0

    # delete predicate in objects
    elif tag1_name not in action_able_object:
        # print('before outputs', outputs)
        outputs[0][check_predicate_index] = 0
        # print('after outputs', outputs)

    return outputs


def sample(args, spo_list, tag_representation, vocab, rel, two_tag):
    g = rdflib.Graph()
    transform = make_transform()

    device = torch.device('cuda')
    tag_extract_model = CNN(vocab).eval()
    tag_extract_model.to(device)

    relation_model = RNNRelationModel(embed_size=args.embed_size, hidden_size=args.hidden_size,
                                      tag_representation=tag_representation,
                                      vocab_size=len(vocab), rel_size=len(rel), num_layer=args.num_layer).eval()

    relation_model = relation_model.eval().to(device)

    relation_model.load_state_dict(torch.load(args.decoder_path))

    encoder = CNN(vocab=vocab).eval()
    decoder = RNNRelationModel(embed_size=args.embed_size,
                               hidden_size=args.hidden_size, tag_representation=tag_representation,
                               vocab_size=len(vocab), rel_size=len(rel), num_layer=args.num_layer)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    feature = encoder(image_tensor)

    sampled_ids_prop = torch.sigmoid(feature)
    sampled_ids = torch.sigmoid(feature).data > 0.5
    _, sampled_ids = sampled_ids.nonzero(as_tuple=True)
    sampled_ids = sampled_ids.tolist()

    tag_list: List = []
    for tag_idx in sampled_ids:
        tag_name = vocab[tag_idx]
        tag_list.append(tag_name)
    print('tag_list: ', tag_list)

    for tag1 in sampled_ids:
        for tag2 in sampled_ids:
            if tag1 == tag2:
                continue
            tag1_idx = tag1
            tag2_idx = tag2

            tag1_name = vocab[tag1_idx]
            tag2_name = vocab[tag2_idx]

            two_tag_idx = two_tag(tag1_name, tag2_name)

            outputs = relation_model(tag1_idx, tag2_idx)

            outputs_data = torch.sigmoid(outputs).data
            outputs = torch.sigmoid(outputs).data > 0.5
            outputs = post_processing(tag1_name, tag2_name, outputs, rel)

            _, sample_ids = outputs.nonzero(as_tuple=True)
            sample_ids = sample_ids.cpu().numpy()

            for sample_id in sample_ids:
                rel_name = rel[sample_id]
                # g.add((tag1_name, rel_name, tag2_name))
                print(tag1_name+'-'+rel_name+'-'+tag2_name)

    """
    G = rdflib_to_networkx_multidigraph(g)

    # Plot Networkx instance of RDF Graph
    pos = nx.spring_layout(G, scale=2)
    edge_labels = nx.get_edge_attributes(G, 'r')
    nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)
    nx.draw(G, with_labels=True)
    """
    image = Image.open(args.image)
    plt.imshow(np.array(image))
    plt.show()


def main(args):
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    spo_list = load_spo_list()
    vocab: List = load_json(path['objects_path'])
    rel: List = load_json(path['predicate_path'])
    two_tag = load_pkl(path['test_two_tag_path'])
    tag_representation = load_tag_representation(vocab)

    sample(args, spo_list, tag_representation, vocab, rel, two_tag)


if __name__ == '__main__':
    n_train = '10'
    parser = argparse.ArgumentParser()

    # Load Path
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained model')
    parser.add_argument('--image', type=str, default=os.path.join(path['val_image_path'], '8357990030_24ec7cd711_b.jpg'))

    # Save path
    parser.add_argument('--encoder_path', type=str, default='models/encoder' + '5' + '.pth')
    parser.add_argument('--decoder_path', type=str, default='models/decoder' + n_train + '.pth')

    # RNN Model HyperParameters
    parser.add_argument('--embed_size', type=int, default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of rnn hidden states')
    parser.add_argument('--num_layer', type=int, default=1, help='number of layers in rnn')

    # Training HyperParameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    args = parser.parse_args()
    print(args)
    main(args)
