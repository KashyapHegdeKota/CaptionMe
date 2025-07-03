# predict.py
import torch
from torchvision import transforms
from PIL import Image
import pickle
import argparse

# Import your model classes
from src.model import EncoderCNN, DecoderRNN
from src.dataset import Vocabulary # Need the class definition

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(encoder_path, decoder_path, vocab_path):
    """Loads the trained model and vocabulary."""
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Model parameters (must be the same as during training)
    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)
    
    # Initialize models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    # Load the trained model weights
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab

def preprocess_image(image_path, transform):
    """Loads and preprocesses a single image."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0) # Add batch dimension
    return image.to(device)

def generate_caption(image_tensor, encoder, decoder, vocab, max_len=20):
    """Generates a caption for a preprocessed image tensor."""
    with torch.no_grad(): # We don't need to compute gradients
        feature = encoder(image_tensor)
        
        # The .sample() method is complex, let's write a simple greedy search here
        # which is easier to understand and often what .sample() does.
        sampled_ids = []
        inputs = feature.unsqueeze(1) # Start with the image feature
        states = None # Initial hidden states for LSTM

        for _ in range(max_len):
            hiddens, states = decoder.lstm(inputs, states) # (batch, 1, hidden_size)
            outputs = decoder.linear(hiddens.squeeze(1))   # (batch, vocab_size)
            _, predicted_idx = outputs.max(1)              # Get the most likely word index
            
            sampled_ids.append(predicted_idx.item())
            
            # Stop if we generate the <end> token
            if predicted_idx.item() == vocab.stoi['<END>']:
                break
            
            # Prepare the input for the next time step
            inputs = decoder.embed(predicted_idx).unsqueeze(1)

    # Convert word IDs to words
    words = [vocab.itos[idx] for idx in sampled_ids]
    
    # Clean up the output
    caption = ' '.join(words[1:-1]) # Remove <start> and <end> tokens
    return caption.capitalize() + '.'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file.')
    parser.add_argument('--encoder_path', type=str, default='models/encoder_epoch_10.pth', help='Path to the trained encoder.')
    parser.add_argument('--decoder_path', type=str, default='models/decoder_epoch_10.pth', help='Path to the trained decoder.')
    parser.add_argument('--vocab_path', type=str, default='models/vocab.pkl', help='Path to the vocabulary file.')
    args = parser.parse_args()

    # Define the same image transformations as in training (without augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    encoder, decoder, vocab = load_model(args.encoder_path, args.decoder_path, args.vocab_path)
    image_tensor = preprocess_image(args.image_path, transform)
    caption = generate_caption(image_tensor, encoder, decoder, vocab)
    
    print("\n---")
    print(f"Generated Caption: {caption}")
    print("---\n")