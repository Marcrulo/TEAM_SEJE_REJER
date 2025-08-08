import json
from typing import Tuple
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from huggingface_hub import login


# INIT
fname = os.path.join(os.getcwd())#, "emergency-healthcare-rag")
sys.path.append(fname)

# MULTICLASS PART
import faiss
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")


# Save folder and article names
DATAFOLDER = "data_txt_chunky"
folders = [f for f in os.listdir(os.path.join(DATAFOLDER,'topics')) if os.path.isdir(os.path.join(DATAFOLDER,'topics', f))]
folders.sort()

folder_names  = []
article_names = []
for folder in folders:
    for txt_file in sorted(os.listdir(os.path.join(DATAFOLDER, 'topics', folder))):
        folder_names.append(folder)
        article_names.append(os.path.join(folder,txt_file))

index = faiss.read_index(os.path.join(fname,"embeddings_final_chunker.faiss"))
with open(os.path.join(fname,'data','topics.json'), "r") as f:
    topics = json.load(f)

# BINARY CLASSIFICATION PART
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  
    bnb_4bit_compute_dtype="bfloat16",  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",       
)
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
    torch_dtype="auto",#torch.bfloat16,
)


def rag_search(statement: str) -> int:

    prompt_embedding = model.encode([statement])
    _, I = index.search(prompt_embedding[0,None], 1)
    with open(os.path.join(DATAFOLDER, 'topics', article_names[I[0][0]]), 'r') as f:
        reference = f.read()
    prediction = folder_names[I[0][0]]
    prediction_id = topics[prediction]

    return prediction_id, reference


def truth_predict(statement: str, reference: str) -> int:
    
    print(len(reference), len(reference.split(" ")))

    system_prompt = (
        "You are a helpful fact-checking assistant."
        "You will be given a reference document and a statement."
        "Respond only with 'TRUE' if the statement is supported by the reference, otherwise respond 'FALSE'."
        "Do not guess. You MAY use outside knowledge to determine the reliability of the statement."
    )
    #"Do not guess. Do not use outside knowledge."

    #user_prompt = statement
    user_prompt = (
        "Reference:"
        f"{reference}"
        "Statement:"
        f"{statement}"
        "Answer:"

    )
    
    
    # Define medical statement (user prompt)
    messages = [
        {"role": "system", "content": system_prompt},#+f'\n{big_text}'},
        {"role": "user", "content": user_prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Tokenize and move to device
    model_inputs = tokenizer([text], return_tensors="pt").to(model_llm.device)

    # Ensemble of <n_attempts> models
    temp_output = []
    n_attempts = 1
    threshold = n_attempts // 2
    for _ in range(n_attempts):
        if sum(temp_output) > threshold:
            break
        generated_ids = model_llm.generate(
            **model_inputs,
            max_new_tokens=5,
            temperature=0.2
        )
        output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]

        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        temp_output.append("true" in content.lower())

    output = (sum(temp_output) > threshold)
    return int(output)  


def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    
    statement_topic, reference = rag_search(statement)

    statement_is_true = truth_predict(statement, reference)
    
    return statement_is_true, statement_topic


