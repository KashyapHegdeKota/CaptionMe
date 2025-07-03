import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn =False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(train_cnn)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        captions = captions[:, :-1]  # Exclude the last word for input
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # Concatenate features with embeddings
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        caption_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            caption_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return caption_ids
    