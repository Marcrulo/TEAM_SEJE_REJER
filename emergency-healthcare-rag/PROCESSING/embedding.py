from sentence_transformers import SentenceTransformer
import faiss
import os
import torch
from transformers import AutoModel
import logging
import numpy as np
import sys


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load the model
# EMBEDDING-SIZE:   1.024
# SEQUENCE-LENGTH: 32.000    ~100.000 words
name = "Qwen/Qwen3-Embedding-4B"
model = SentenceTransformer(name).to(device)
torch.cuda.empty_cache()

# %%
dim_size = 2560
index = faiss.IndexFlatL2(dim_size)

folders = [f for f in os.listdir('data_txt_chunky/topics') if os.path.isdir(os.path.join('data_txt_chunky/topics', f))]
folders.sort()

for folder in folders:
    print(folder)
    folder_path = os.path.join('data_txt_chunky/topics', folder)
    
    # Sort txt files in the folder
    for txt_file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, txt_file)

        with open(file_path, 'r') as f:
            print("-", txt_file)
            content = f.read()
            try:
                index.add(model.encode([content]))
            except Exception as e:
                print("SKIPPED")
                print(e)
                index.add(model.encode(["EMPTY"]))


faiss.write_index(index, "embeddings_final_chunker.faiss")


