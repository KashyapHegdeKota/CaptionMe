# 📸 Echo: AI Image Captioning System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

A complete, end-to-end deep learning project that generates descriptive, human-like captions for any given image. Built from the ground up, this system showcases a sophisticated Encoder-Decoder architecture combining state-of-the-art Computer Vision and NLP techniques.

---

## ✨ Live Demo

### ➡️ [captionme.your-domain.com](http://captionme.your-domain.com)

![Echo Demo GIF](assets/demo.gif)

---

## 📚 Table of Contents

- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Use](#-how-to-use)
- [Results & Examples](#-results--examples)
- [Future Improvements](#-future-improvements)

---

## 💡 Key Features

- **Encoder-Decoder Framework**: Implements a modern **CNN-RNN** architecture, a cornerstone of many sequence-to-sequence AI tasks.
- **Transfer Learning**: Utilizes a pre-trained **ResNet-50** model, leveraging knowledge from the massive ImageNet dataset for powerful image feature extraction.
- **Custom RNN Decoder**: The **LSTM (Long Short-Term Memory)** network was trained from scratch on the Flickr8k dataset to generate coherent and contextually relevant captions.
- **Robust Data Pipeline**: Features a complete data pipeline for preprocessing images and text, including building a custom vocabulary and a `DataLoader` with padding for variable-length sequences.
- **Interactive Web App**: Deployed as a polished and user-friendly web application using **Streamlit**, allowing anyone to get a caption in real-time.

---

## 🧠 Model Architecture

The system is composed of two main components: an Encoder and a Decoder, which work together to translate pixels into words.

<p align="center">
  <img src="https://i.imgur.com/kUMaA7r.png" width="700">
</p>

### 1. The Encoder (The "Eye")

The Encoder's job is to "see" the image and distill its visual information into a compact numerical representation (a feature vector).

- A pre-trained **ResNet-50** CNN acts as the backbone.
- The final classification layer is removed to extract the rich feature vector from the penultimate layer, encoding the image's content.

### 2. The Decoder (The "Mouth")

The Decoder's job is to take the image's feature vector and generate a text sequence.

- An **LSTM** network is used for its ability to handle sequential data and remember context.
- The image feature vector initializes the LSTM's hidden state, "priming" it with what to write about.
- At each step, the LSTM takes the previously generated word to predict the next, continuing until an `<end>` token is produced.

---

## 🛠️ Tech Stack

- **Core Libraries:** Python, PyTorch, Torchvision
- **Data Science:** Pandas, NumPy, NLTK
- **Web & Deployment:** Streamlit, Streamlit Cloud, Cloudflare
- **Tools:** Git, GitHub, Google Colab (for training)

---

## 📂 Project Structure

ai_image_captioning/
├── .streamlit/
│ └── config.toml
├── assets/
│ └── demo.gif
├── data/
│ └── flickr8k/
├── models/
│ ├── encoder-5.pth
│ ├── decoder-5.pth
│ └── vocab.pkl
├── notebooks/
│ └── 01_data_exploration.ipynb
├── src/
│ ├── dataset.py
│ └── model.py
├── app.py
├── predict.py
├── train.py
├── README.md
└── requirements.txt

---

## ⚙️ Setup & Installation

To run this project locally, follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/KashyapHegdeKota/CaptionMe.git
cd ai-image-captioning
```

**2. Create and activate a virtual environment:**

```python
python -m venv venv
source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Download NLTK data:**

```python
#Run this in a Python interpreter
import nltk
nltk.download('punkt')
```

**5. Download the dataset:**
The training script is configured for the Flickr8k dataset.
Download from Kaggle and place the contents in the data/flickr8k/ directory.

---

## 🚀 How to Use

**Run the web application**
The easiest way to use the model is through the interactive Streamlit app.

```bash
streamlit run app.py
```

Navigate to http://localhost:8501 in your browser

**Generate a Caption from the Command Line**
Use the predict.py script for single-image inference. The script defaults to using the epoch 10 models.

```bash
python predict.py --image-path /path/to/your/image.jpg
```

**Train the Model**
To train the model from scratch, run the training script train.py. Training is best performed on a GPU or Google Colab(which is what I used).

```bash
#Example command for training.
python train.py --epochs 10 --batch_size 64
```

## 📊 Results & Examples

The model was trained for 10 epochs on the Flickr8k dataset, achieving a final training perplexity of approximately 7.4. The model checkpoint from epoch 10 was selected for deployment to balance performance and prevent overfitting.

|                                                                Original Image                                                                | Generated Caption by Echo                            |
| :------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------- |
| <img src="https://unsplash.com/photos/a-brown-dog-standing-on-top-of-snow-covered-ground-M0UMCtHpxkQ?w=400" alt="A dog playing in the snow"> | "A black and white dog is running through the snow." |
