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
from Sentence_Transformer import initializing_server
from langchain_huggingface import HuggingFaceEmbeddings
from Sentence_Transformer import get_retrieval
combined_texts = get_combined_text
model1 = ChatOllama(temperature=0, model='llama3.1')
load_dotenv()
def chatfunction(text_box,history):

    querry = text_box
    retrieval = get_retrieval()
    
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
        {"context": retrieval | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model1
        | StrOutputParser())

    response_text = rag_chain.invoke(querry)

    return response_text




gr.ChatInterface(fn =chatfunction, textbox= gr.Textbox(placeholder= " enter message here"),
                 chatbot= gr.Chatbot()).launch()
