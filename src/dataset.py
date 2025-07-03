import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from collections import Counter
import nltk
import re  # <--- IMPORT THE REGEX MODULE

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        """
        This is the most robust __init__ method. It uses regular expressions
        to parse the captions file, making it immune to tab vs. space issues.
        """
        self.root_dir = root_dir
        self.transform = transform

        image_infos = []
        captions = []
        
        # --- THE ULTIMATE FIX: PARSE WITH REGEX ---
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Use re.split to split on one or more whitespace characters.
                # maxsplit=1 ensures we only split into two parts:
                # the image_info and the rest of the line (the caption).
                parts = re.split(r'\s+', line, maxsplit=1)
                
                if len(parts) == 2:
                    image_info, caption = parts
                    image_infos.append(image_info)
                    captions.append(caption)
        
        # --- ADDING A SANITY CHECK ---
        if not image_infos:
            raise ValueError(
                "The caption file was parsed, but no valid entries were found. "
                "Please check the format of your captions.txt file. "
                "Expected format: 'image_name.jpg#0   Some caption text...'"
            )
        # -----------------------------

        # Create the DataFrame from the manually parsed lists
        self.df = pd.DataFrame({
            'image_info': image_infos,
            'caption': captions
        })
        def clean_filename(info):
            filename=info.split('#')[0]
            jpg_pos = filename.find('.jpg')
            if jpg_pos != -1:
                return filename[:jpg_pos + 4]  # Include the '.jpg' extension
            else:
                return filename
        # The 'image_info' column is like 'image.jpg#0'. We extract just 'image.jpg'.
        self.df['image'] = self.df['image_info'].apply(clean_filename)
        self.df = self.df.drop(columns=['image_info'])

        self.df.dropna(subset=['caption'], inplace=True)
        self.df = self.df.reset_index(drop=True)

        # Assign the cleaned data to class attributes
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Build the vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
    

    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions.iloc[idx]
        img_id = self.imgs.iloc[idx]
        img_path = os.path.join(self.root_dir, str(img_id))
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)   
        
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<END>"])

        return img, torch.tensor(numericalized_caption)

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.lower() for tok in nltk.word_tokenize(str(text))]
    def build_vocab(self, sentence_list):
        frequencies= Counter()
        idx = 4

        for sentence in sentence_list:
            for words in self.tokenizer(sentence):
                frequencies[words] += 1
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    