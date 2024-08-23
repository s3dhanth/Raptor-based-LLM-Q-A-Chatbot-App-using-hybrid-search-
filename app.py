from sentence_transformers import SentenceTransformer, util, InputExample, losses
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.chat_models import ChatOllama
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import tiktoken
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
import gradio as gr
from main import get_combined_text

combined_texts = get_combined_text



model1 = ChatOllama(temperature=0, model='llama3.1')
api_key = os.getenv("API_KEY")
pc = Pinecone(api_key= api_key)
index_name = 'sbert-50dim'
bm25encoder = BM25Encoder()
bm25encoder.fit(combined_texts)
def chatfunction(text_box,history):
    querry = text_box
    index =pc.Index(index_name)
    retriever = PineconeHybridSearchRetriever(
    embeddings= OllamaEmbeddings(model='llama3.1'), sparse_encoder=bm25encoder, index=index,top_k=  165)
    
    template = """
    Answer the question based only on the following context:
    {context}

    Answer the following question:
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model1
        | StrOutputParser())

    response_text = rag_chain.invoke(querry)

    return response_text




gr.ChatInterface(fn =chatfunction, textbox= gr.Textbox(placeholder= " enter message here"),
                 chatbot= gr.Chatbot()).launch()






















from main import combined_texts
api_key = os.getenv("API_KEY")

def chatfunction(text_box,history):
    os.
    pc = Pinecone(api_key= api_key)
    querry = text_box
    index =pc.Index(index_name)
    retriever = PineconeHybridSearchRetriever(
    embeddings= OllamaEmbeddings(model='llama3.1'), sparse_encoder=bm25encoder, index=index,top_k=  165)
    
    template = """
    Answer the question based only on the following context:
    {context}

    Answer the following question:
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model1
        | StrOutputParser())

    response_text = rag_chain.invoke(querry)

    return response_text

gr.ChatInterface(fn =chatfunction, textbox= gr.Textbox(placeholder= " enter message here"),
                 chatbot= gr.Chatbot()).launch()