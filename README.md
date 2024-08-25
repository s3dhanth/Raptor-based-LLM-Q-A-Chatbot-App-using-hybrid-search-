# Raptor-based-LLM-Q-A-Chatbot-App-using-hybrid-search-

This repository contains a Gradio application for an enhanced Raptor Q&A bot that leverages LLaMA 3.1 With hybrid Retrieval (custom embeddings and BM25) and Pinecone Vector Database. The app allows users to interact with the Model and get responses based on their queries.

## DataSet

- Data1: History of Tuscan-inspired sanctuary
- Data2: History of Giovanni
- Data3: Different Cuisine of Bella Vista

## Overview
- Sentence_Transformer.py = Converted text into vectors using Sentence Transformer Embedding (384 vectors)
- Main.py = Converted text into vectors using LLaMA3.1 Embedding (4096 vectors)
### Llama3.1 Model Embedding after dimension reduction
![image](https://github.com/user-attachments/assets/9d715cba-22b0-4e6c-a3da-01b88dac268a)
  
### Sentence Transformer (MiniLM) Model Embedding after dimension reduction
![image](https://github.com/user-attachments/assets/fb82b515-3bfc-47cb-96fb-21694d65dde1)

### Hybrid Search with sparse-dense vector

![image](https://github.com/user-attachments/assets/4ad8c914-746a-4072-b1c4-e93a57066a68)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/s3dhanth/Raptor-based-LLM-Q-A-Chatbot-App-using-hybrid-search-.git

sh
Copy code
pip install -r requirements.txt

run python main.py (llama3.1)
          or
run python Sentence_transformer.py (Transformer)

-to Ask custom question use (rag.invoke("question")) 

# Gradio Application
- # run python app.py (llama3.1 embeddings)
- # run python Sentence_transformers.py (MiniLM model embeddings)

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
```
## Model (Sentence Transformer MiniLM) 
- In the text box Enter our query and submit 
![minilm](https://github.com/user-attachments/assets/cd40599d-2576-43c2-b099-99b760715919)

## Notebooks overview
- Raptor_sbert_2dim -   Reduction to 10 Dimensions
- Raptor_sbert_2dim -   Reduction to 2 Dimensions
- MiniLMapp - Sentence Transformer model with gradio (jupyter)
- Sentence_Transformer = For Model testing (use it directly)
- Main.py = For LLama embeddings Model testing (use it directly)
  

