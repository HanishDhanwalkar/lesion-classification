import pandas as pd
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

data_path = '../data/'
df = pd.read_csv(data_path + 'data.csv')


img = Image.open(data_path + 'images/' + df['filename'].iloc[0] + '.jpg') 
inputs = processor(text=["a photo of a cat", "a photo of a man"], images=img, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image

probs = logits_per_image.softmax(dim=1)
print(probs)