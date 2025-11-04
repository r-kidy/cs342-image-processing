import torch
import timm
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# get cwd
cwd = os.getcwd()
#check img_align_celeba directory exists
if not os.path.exists(os.path.join(cwd, "img_align_celeba")):
    raise FileNotFoundError("img_align_celeba directory not found in the current working directory.")

# 1. Load pretrained Vision Transformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
vit.eval() # disable dropout, etc.

# 2. Define preprocessing consistent with ImageNet training
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
),
])
# 3. Directory containing CelebA images (modify path)
image_dir = "img_align_celeba"
sample_subset_size = 20000
image_list = sorted(os.listdir(image_dir))[:sample_subset_size] # sample subset

# 4. Extract embeddings
all_features = []
with torch.no_grad():
    for fname in tqdm(image_list):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        # extracting feature vector from image
        # vit.forward_features(x) returns feature vector for all patches (regions of image) and CLS token
        # we take only the CLS token (first token) as the image representation
        features = vit.forward_features(x)[ :, 0, :] # shape: (1, 768)
        all_features.append(features.cpu().numpy())
        
all_features = np.concatenate(all_features, axis=0)
#print("Feature matrix shape:", all_features.shape) # e.g. (20000, 768)


# 5. Save for later use
np.save("celeba_vit_embeddings.npy", all_features)