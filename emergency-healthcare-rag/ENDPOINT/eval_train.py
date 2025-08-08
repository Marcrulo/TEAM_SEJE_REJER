# Imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import os
import numpy as np
import json
import time
import faiss

from model import *

# Get list of all training statements
statements = []
for statement_file in sorted(os.listdir(os.path.join('data/train/statements'))):
    with open(os.path.join('data/train/statements', statement_file), 'r') as f:
        statements.append(f.read().strip())
statements = np.array(statements)

# Get lists of labels (binary and multiclass)
labels_binary = []
labels_topic = []
for label_file in sorted(os.listdir(os.path.join('data/train/answers'))):
    with open(os.path.join('data/train/answers', label_file), 'r') as f:
        json_load = json.load(f)
        labels_binary.append(json_load['statement_is_true'])
        labels_topic.append(json_load['statement_topic'])
labels_binary = np.array(labels_binary)
labels_topic = np.array(labels_topic)


predictions_binary = []
predictions_topic  = []
# N = len(statements)
N = 200
for i in range(N):
    print(f"{i}/{N} - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    prediction = predict(statements[i])
    print(prediction)
    predictions_binary.append(prediction[0])
    predictions_topic.append(prediction[1])
    print('---')
print('\n-------------')
print("Binary accuracy: ", np.mean(np.array(predictions_binary) == labels_binary[:N]))
print("Topic accuracy:  ", np.mean(np.array(predictions_topic) == labels_topic[:N]))

# If topic prediction is 1, check accuracy of binary prediction
topic_trues = 0
counter = 0
for i in range(N):
    if predictions_topic[i] == labels_topic[i]:
        topic_trues += 1
        if predictions_binary[i] == labels_binary[i]:
            counter += 1
print("Topic=true accuracy: ", counter / topic_trues if topic_trues > 0 else 0)
    