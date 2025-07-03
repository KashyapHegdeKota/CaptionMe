import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import argparse

from src.dataset import FlickrDataset
from src.model import EncoderCNN, DecoderRNN

def train():
    parser = argparse.ArgumentParser(description='Train a neural image captioning model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embed_size', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=512, help='LSTM hidden state dimension')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--save_every', type=int, default=1, help='Save model checkpoint every N epochs')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists('models'):
        os.makedirs('models')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = FlickrDataset(
        root_dir= "data/flickr8k/images",
        captions_file="data/flickr8k/captions.txt",
        transform=transform,
        freq_threshold=5
    )

    def collate_fn(data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        images = torch.stock(images,0)
        lengths = [len(caption) for caption in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, caption in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = caption[:end]
        return images, targets, lengths
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    vocab_size = len(dataset.vocab)
    encoder= EncoderCNN(embed_size=args.embed_size, train_cnn=False).to(device)
    decoder= DecoderRNN(args.embed_size, args.hidden_size, vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    total_steps = len(dataloader)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total Batches per epoch: {total_steps}")

    for epoch in range(1, args.epochs + 1):
        for i, (images, captions, lengths) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()
            features = encoder(images)
            outputs = decoder(features, captions)

            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}')

        if epoch % args.save_every == 0:
            torch.save(decoder.state_dict(),os.path.join('models', f'decoder_epoch_{epoch}.pth'))
            torch.save(encoder.state_dict(), os.path.join('models', f'encoder_epoch_{epoch}.pth'))
            print(f'Model saved for epoch {epoch}')
if __name__ == "__main__":
    train()