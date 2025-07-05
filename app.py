import streamlit as st
from PIL import Image
import torch
import time
import torchvision.transforms as transforms
from src.model import EncoderCNN, DecoderRNN
from src.dataset import Vocabulary  
from predict import load_model, preprocess_image

st.set_page_config(
    page_title="CaptionMe| AI Image Captioner",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("CaptionMe: AI Image Captioner")
st.write("Upload an image to generate a caption using a pre-trained neural image captioning model.")
st.write("This model uses a Convolutional Neural Network (CNN) to extract image features and a Recurrent Neural Network (RNN) to generate captions.")
st.write("The model was trained on the Flickr8k dataset, which contains images and their corresponding captions.")
st.write("---")

@st.cache_resource
def load_models():
    """Load the pre-trained models and vocabulary."""
    encoder_path = 'models/encoder_epoch_10.pth'
    decoder_path = 'models/decoder_epoch_10.pth'
    vocab_path = 'models/vocab.pkl'
    
    encoder, decoder, vocab = load_model(encoder_path, decoder_path, vocab_path)
    return encoder, decoder, vocab

encoder, decoder, vocab = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
    with col2:
        caption_placeholder = st.empty()
        caption_placeholder.write("Generating caption...")

        with st.spinner("Processing..."):
            image_tensor = preprocess_image(uploaded_file, transform)
            with torch.no_grad():
                feature = encoder(image_tensor)
                sampled_ids = []
                inputs = feature.unsqueeze(1)  # Start with the image feature
                states = None
                max_len = 20

                for _ in range(max_len):
                    hiddens, states = decoder.lstm(inputs, states)
                    outputs = decoder.linear(hiddens.squeeze(1))
                    _, predicted_idx = outputs.max(1)

                    if predicted_idx.item() == vocab.stoi['<END>']:
                        break
                    sampled_ids.append(predicted_idx.item())
                    inputs = decoder.embed(predicted_idx).unsqueeze(1)
            words = [vocab.itos[idx] for idx in sampled_ids]
            caption = ' '.join(words).capitalize() + '.'
        caption_placeholder.write(f"**Generated Caption:** {caption}")
st.sidebar.title("About the Project")
st.sidebar.info(
    "**Author:** Kashyap Hegde Kota\n\n"
    "**Tech Stack:**\n"
    "- Python, PyTorch\n"
    "- Streamlit\n"
    "- ResNet-50 & LSTM\n"
    "- NLTK, Pandas, Pillow\n"
    "\n\n"
    "This project is a practical implementation of an Encoder-Decoder model for a multimodal task. "
    "Check out the code on [GitHub](https://github.com/KashyapHegdeKota/CaptionMe)!"
)
