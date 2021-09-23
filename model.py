from torch import nn, cat, stack, sigmoid, Tensor, LongTensor, FloatTensor
from torchvision.models import resnet34, resnet50


class CNN(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        resnet = resnet34(pretrained=True)
        module = list(resnet.children())[:-1]
        num_fts = resnet.fc.in_features
        resnet.fc = nn.Linear(num_fts, len(vocab))
        self.linear = nn.Linear(resnet.fc.in_features, len(vocab))
        self.resnet = nn.Sequential(*module)
        self.bn = nn.BatchNorm1d(len(vocab), momentum=0.1)

    def forward(self, image):
        features = self.resnet(image)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class RNNRelationModel(nn.Module):
    def __init__(self, embed_size, hidden_size, tag_representation, vocab_size, rel_size, num_layer, max_seq_length=3):
        super().__init__()
        self.tag_representation = FloatTensor(tag_representation)
        self.embed = nn.Embedding(vocab_size, embed_size).from_pretrained(self.tag_representation)
        self.lstm = nn.LSTM(embed_size*2, hidden_size, num_layer, batch_first=True)
        # self.gru = nn.GRU(embed_size*2, hidden_size, num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, rel_size)
        self.max_seq_length = max_seq_length

    def forward(self, tag1_idx, tag2_idx):
        tag1 = LongTensor([tag1_idx]).cuda()
        tag2 = LongTensor([tag2_idx]).cuda()

        tag1 = self.embed(tag1).cuda()
        tag2 = self.embed(tag2).cuda()
        labels = cat((tag1, tag2)).requires_grad_(True).cuda()
        labels = labels.unsqueeze(0)
        labels = labels.reshape(1, 1, -1)
        hiddens, states = self.lstm(input=labels)
        # hiddens, states = self.gru(input=labels)
        outputs = self.linear(hiddens.squeeze(1)).requires_grad_(True)

        # if using Bi-LSTM model
        # outputs = outputs.squeeze(0).requires_grad_(True)
        # outputs = outputs[-1, :].requires_grad_(True)
        # outputs = outputs.unsqueeze(0).requires_grad_(True)

        return outputs

    def sample(self, tags, states=None):
        sampled_id = []
        inputs = self.embed(tags)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.unsqueeze(0)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            # hiddens, states = self.gru(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            print(predicted.shape)
            sampled_id.append(predicted)
            sampled_id = stack(sampled_id, 1)
        return sampled_id


class FCRelationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(19, 6)

    def forward(self, tag) -> Tensor:
        relation = self.linear(tag)
        return relation
