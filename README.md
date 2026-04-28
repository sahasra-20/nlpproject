# Agricultural QA Transformer with RAG

This project implements a from-scratch Transformer-based Question Answering (QA) model tailored for the agricultural domain, augmented with Retrieval-Augmented Generation (RAG) capabilities.

## Architecture & Parameters
The core model is a custom PyTorch Encoder-Decoder Transformer with the following configurations (defined in `config.py`):

- **Parameters**: ~51.8 Million
- **Vocab Size**: 50,000 (BPE Tokenizer)
- **Hidden Dimension**: 384
- **Attention Heads**: 6 (64 dims per head)
- **Encoder Layers**: 4
- **Decoder Layers**: 4
- **Feed Forward Dim**: 1024
- **Dropout**: 0.15
- **Sequence Length**: 80 (Source), 100 (Target), 320 (RAG Context)

## Pre-trained Embeddings
The model natively supports injecting pre-trained embedding weights into the bottom layer to bootstrap vocabulary understanding.
- **BERT Embeddings** (`models/embeddings_bert_static.npy`): Currently enabled in config.
- **Word2Vec Embeddings** (`models/embeddings_word2vec.npy`): A fallback/alternative option. 

## RAG (Retrieval-Augmented Generation) Pipeline
To prevent hallucinations and provide highly specific agricultural answers, a built-in RAG system retrieves relevant chunks of knowledge before generation.
- **Knowledge Base Build**: `rag_knowledge_base.py` 
- **Retrieval Engine**: `rag_retriever.py` (uses the QA model's own encoder to embed text chunks and FAISS/numpy for cosine similarity search).
- **Inference with Context**: `rag_inference.py` prepends retrieved knowledge to the question using `[CTX]` and `[Q]` separators.
- **Training Strategy**: 30% of training batches randomly include context blocks so the model learns both memory-based generation and reading comprehension.



## Data Sources

- [Kisan Call Centre (KCC)](https://www.data.gov.in/resource/kisan-call-centre-kcc-transcripts-farmers-queries-answers): Real farmer queries and expert answers.  
- [Farmers Call Query (Kaggle)](https://www.kaggle.com/datasets/daskoushik/farmers-call-query-data-qa): Agricultural QA dataset.  
- [AgroQA Dataset](https://github.com/JonaOmara/AgroQA-Dataset): Structured agriculture QA dataset.  
- [Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca): General conversational instruction–response data.
## Running the Pipeline

**1. Data Preparation**
Download `data.csv`from the Drive link and upload it to the main folder. 
**2. Full Orchestration**
You can run all components sequentially via the orchestrator:
```bash
python main.py
```

**3. Individual Modules**
- **Train**: `python train.py`
- **Inference (Pure Memory)**: `python inference.py`
- **Inference (RAG)**: `python rag_inference.py`
- **Chat UI**: `python chat.py`
- **Evaluation (ROUGE/BLEU)**: `python evaluate_model.py`

## Requirements
- `torch`
- `numpy`
- `pandas`
- `faiss-cpu` (Optional, defaults to numpy cosine search if unavailable)
- `evaluate` and `rouge_score` (For running `evaluate_model.py`)
- `seaborn` and `matplotlib` (For visualizations)

## Metrics

### BLEU Score
| Metric | Value |
|--------|-------|
| **BLEU Score** | **0.1295** |
| Brevity Penalty | 0.5385 |
| Length Ratio | 0.6177 |
| Translation Length | 13,309 |
| Reference Length | 21,546 |



### ROUGE Scores
| Metric | Score |
|--------|-------|
| **ROUGE-1** | 0.4457 |
| **ROUGE-2** | 0.3073 |
| **ROUGE-L** | 0.4285 |
| **ROUGE-Lsum** | 0.4295 |
