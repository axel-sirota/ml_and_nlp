# ML & NLP Course Outline

Generic delivery (Data Trainers LLC). Bringing ML practitioners into real-world NLP usage.
Through-line: every notebook builds toward a fine-tuned transformer in a simple Gradio chatbot.

---

## Part A - What NLP Can Do

### A1 - NLP From Scratch
- Why NLP is hard: ambiguity, tokenization, morphology.
- Tokenization and normalization (spaCy / TextBlob).
- Named Entity Recognition with spaCy displacy.
- Topic modelling (LDA) on a small corpus.
- Tools-first, no deep learning yet. Bridge: "watch one line do this next."
- Objectives: read raw text with classical tools; extract entities and topics with no modelling.

### A2 - Pipeline Tour
- The `pipeline()` mental model: task -> default model -> call it.
- Sentiment analysis, zero-shot classification, NER, question answering, summarization.
- Swapping models from the HuggingFace Hub.
- The WOW: solve real NLP tasks in three lines, no training. Bridge to Part B.
- Objectives: solve 5+ NLP tasks with pretrained transformers via pipeline().

### A3 - Capstone A: Customer Support Routing
- Twitter customer-support routing (TWCS).
- Derive labels not present in the file.
- Zero-shot classification for routing; optional NER for metadata.
- Objectives: ship a routing solution end-to-end with Part A tools, no training.

---

## Part B - How It Works (Build It)

### B4 - PyTorch Tensors
- Tensor creation, dtype, shape; arithmetic and matrix ops.
- Reshape, indexing, broadcasting.
- Autograd: requires_grad, backward, grad, no_grad.
- Objectives: build and differentiate a small tensor expression.

### B5 - Word2Vec and Sentence Embeddings
- Word2Vec via gensim, pretrained (no training): similarity, analogies, 2D projection.
- Why averaging word vectors is a weak sentence baseline.
- Sentence-transformers: encode sentences, cosine similarity, semantic search.
- Datasets: STS-B for sentence similarity.
- Objectives: do word analogies and sentence similarity with pretrained models.

### B6 - PyTorch NN Basics
- nn.Module, layers, activations, Sequential.
- Loss (CrossEntropyLoss), optimizer, the training loop.
- TensorDataset + DataLoader.
- Objectives: train a small MLP on toy data end-to-end.

### B7 - MLP on Word2Vec (THE STOPPER)
- Words -> word2vec vectors -> document vector -> MLP -> classify.
- The baseline bar; build and train the MLP; beat the baseline.
- The "embeddings as features" pattern. Bridge to transformers (attention) in Part C.
- Datasets: news titles / SST-2.
- Objectives: train an MLP that beats a baseline using word2vec features.

### B8 - Capstone B: Sentiment Classifier
- IMDB sentiment with the student's own MLP on word2vec features.
- Beat a baseline; optionally compare to a pipeline() zero-shot result.
- Datasets: IMDB.
- Objectives: independently train an MLP sentiment classifier.

---

## Part C - Transfer Learning

### C9 - DistilBERT Chatbot
- Why transformers: the MLP ceiling; attention intuition (plain English).
- AutoTokenizer; load distilbert-base-uncased with a classification head.
- Fine-tune with the HuggingFace Trainer; evaluate vs the MLP.
- Load the fine-tuned model into a simple Gradio chatbot.
- Datasets: SST-2 / IMDB.
- Objectives: fine-tune DistilBERT and chat with it through a Gradio UI.

### C10 - Encoder-Decoder Q&A Chatbot
- Encoder vs decoder vs encoder-decoder; when seq2seq is the right shape.
- Load a seq2seq model (t5-small / bart-base); seq2seq tokenization.
- Fine-tune on a Q&A dataset; generate answers with .generate().
- Load into a Gradio chatbot.
- Datasets: SQuAD.
- Objectives: fine-tune a seq2seq model and ask it questions through a chatbot.

### C11 - Capstone C: Ship a Chatbot
- Choose a path: encoder-only classification chatbot or encoder-decoder Q&A chatbot.
- Load a pretrained model, fine-tune on a chosen public dataset, evaluate.
- Ship a Gradio chatbot; write up what works and what fails.
- Objectives: deliver a working fine-tuned chatbot.

---

## Basics (archive / pre-work, not core flow)

- Classical Text Classification (TF-IDF + sklearn).
- PyTorch Tensor Exercises, PyTorch Exercises, Scikit-Learn Exercises.