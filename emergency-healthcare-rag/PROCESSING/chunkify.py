import faiss
import os
import logging
import numpy as np
import sys

# Set up logging
logging.basicConfig(
    filename='embedding_log.txt',     # Name of your log file
    filemode='w',                     # Overwrite on each run; use 'a' to append
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO                # or DEBUG if you want more verbosity
)
logger = logging.getLogger()


# Chunk sizes
CHUNK_SIZE = int(sys.argv[1]) 
OVERLAP    = int(sys.argv[2])
print(f"Chunk size: {CHUNK_SIZE}, Overlap: {OVERLAP}")

# Input and output base paths
input_base  = 'data_txt/topics'
output_base = 'data_txt_chunky/topics'

# Get all folders in input path
folders = [f for f in os.listdir(input_base) if os.path.isdir(os.path.join(input_base, f))]

for folder in folders:
    input_folder_path = os.path.join(input_base, folder)
    output_folder_path = os.path.join(output_base, folder)
    os.makedirs(output_folder_path, exist_ok=True)  # Ensure output folder exists

    txt_files = [f for f in os.listdir(input_folder_path) if f.endswith('.txt')]

    for txt_file in txt_files:
        with open(os.path.join(input_folder_path, txt_file), 'r') as f:
            text = f.read()
            words = text.split()

        total_words = len(words)
        chunk_id = 0

        print(f"Processing file: {folder}/{txt_file} ({total_words} words)")

        for start in range(0, total_words, CHUNK_SIZE - OVERLAP):
            end = min(start + CHUNK_SIZE, total_words)
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            chunk_filename = txt_file.replace('.txt', f'_chunk{CHUNK_SIZE}-{OVERLAP}_{chunk_id}.txt')
            chunk_path = os.path.join(output_folder_path, chunk_filename)

            with open(chunk_path, 'w') as cf:
                cf.write(chunk_text)

            chunk_id += 1

