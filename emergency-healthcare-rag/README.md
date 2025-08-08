
## GUIDE: How to PROCESSING

1. Manually updated `data/` folder slightly to fragment big articles
2. Run `process_text.py` to remove start and end from articles in `data/` and put the results in `data_txt/`
3. Run `python chunkify.py <CHUNKSIZE> <OVERLAP>` to chunk the articles, which are put in the `data_txt_chunky/` folder. The chunks are:
    - CHUNKSIZE: 1000 - OVERLAP: 750
    - CHUNKSIZE:  500 - OVERLAP: 375
    - CHUNKSIZE:  375 - OVERLAP: 263
    - CHUNKSIZE: 200 - OVERLAP: 150
    - CHUNKSIZE: 150 - OVERLAP: 100
4. Run `embedding.py` to add the embedded version of all the chunked data into a "FAISS index" file, `embeddings_final_chunker.faiss`

## GUIDE: How to ENDPOINT (Inference)
1. Train locally: `eval_train.py`
2. Run endpoint: `api.py`
3. Both use the `model.py` file as the model