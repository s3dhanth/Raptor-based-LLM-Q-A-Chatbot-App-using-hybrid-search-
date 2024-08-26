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
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import nltk
from nltk.corpus import wordnet

# Download necessary NLTK data (if not already downloaded)
nltk.download('wordnet')
nltk.download('omw-1.4')
# Load environment variables from .env file
load_dotenv()


loader = DirectoryLoader('data', glob="**/*.txt")
docs = loader.load()
def expand_query(query):
    words = query.split()
    expanded_words = []

    for word in words:
        synonyms = wordnet.synsets(word)
        lemmas = set()

        for syn in synonyms:
            for lemma in syn.lemmas():
                lemmas.add(lemma.name())

        if lemmas:
            expanded_words.append(f"({word} OR {' OR '.join(lemmas)})")
        else:
            expanded_words.append(word)

    return ' '.join(expanded_words)
def doc_text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,)
    docs = text_splitter.split_documents(docs)
    texts = [doc.page_content for doc in docs]
    return texts

def reduce_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 1234):
    max_clusters = min(max_clusters, len(embeddings))
    bics = [GaussianMixture(n_components=n, random_state=random_state).fit(embeddings).bic(embeddings)
            for n in range(1, max_clusters)]
    return np.argmin(bics) + 1

def gmm_clustering(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def format_cluster_texts(df):
    clustered_texts = {}
    for cluster in df['Cluster'].unique():
        cluster_texts = df[df['Cluster'] == cluster]['Text'].tolist()
        clustered_texts[cluster] = " --- ".join(cluster_texts)
    return clustered_texts

def summary_generation(model1):
    template = """You are an assistant to create a detailed summary of the text input prodived.
    Text:
    {text}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model1 | StrOutputParser()

    summaries = {}
    for cluster, text in tqdm(clustered_texts.items()):
        summary = chain.invoke({"text": text})
        summaries[cluster] = summary

    return summaries


def clustered_summaries(simple_labels,summaries):
    clustered_summaries = {}
    for i, label in enumerate(simple_labels):
        if label not in clustered_summaries:
            clustered_summaries[label] = []
    
        clustered_summaries[label].append(list(summaries.values())[i])
    return clustered_summaries


def final_summaries(clustered_summaries):
    final_summaries = {}
    for cluster, texts in clustered_summaries.items():
        combined_text = ' '.join(texts)
        summary = chain.invoke({"text": combined_text})
        final_summaries[cluster] = summary
    return final_summaries

def initializing_server(api_key,combined_texts):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    pc = Pinecone(api_key= api_key)
    index_name = 'sbert-50dim'
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name,dimension=384,metric="dotproduct",spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"))
        bm25encoder = BM25Encoder()
        bm25encoder.fit(combined_texts)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index =pc.Index(index_name)
        retriever = PineconeHybridSearchRetriever(
        embeddings= embeddings, sparse_encoder=bm25encoder, index=index,top_k= 60)
        retriever.add_texts(combined_texts)
    else:
        bm25encoder = BM25Encoder()
        bm25encoder.fit(combined_texts)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index =pc.Index(index_name)
        retriever = PineconeHybridSearchRetriever(
            embeddings= embeddings, sparse_encoder=bm25encoder, index=index,top_k= 60)
    
    return retriever

def get_combined_text():
    return combined_texts

texts = doc_text_splitter(docs)
model1 = ChatOllama(temperature=0, model='llama3.1')
model = SentenceTransformer('all-MiniLM-L6-v2')
global_embeddings = [model.encode(txt) for txt in texts]

dim = 10
template = """You are an assistant to create a detailed summary of the text input prodived.
Text:
{text}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model1 | StrOutputParser()
global_embeddings_reduced = reduce_cluster_embeddings(global_embeddings, dim)

labels, _ = gmm_clustering(global_embeddings_reduced, threshold=0.5)

simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

df = pd.DataFrame({
    'Text': texts,
    'Embedding': list(global_embeddings_reduced),
    'Cluster': simple_labels
})

clustered_texts = format_cluster_texts(df)



summaries = summary_generation(model1)

embedded_summaries = [model.encode(summary) for summary in summaries.values()]

embedded_summaries_np = np.array(embedded_summaries)

labels, _ = gmm_clustering(embedded_summaries_np, threshold=0.5)

simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]

clustered_summaries = clustered_summaries(simple_labels,summaries)

final_summaries = final_summaries(clustered_summaries)

texts_from_df = df['Text'].tolist()
texts_from_clustered_texts = list(clustered_texts.values())
texts_from_final_summaries = list(final_summaries.values())

combined_texts = texts_from_df + texts_from_clustered_texts + texts_from_final_summaries

# os.env

api_key = os.getenv("API_KEY")
#getembeddings = OllamaEmbeddings(model = 'llama3.1')
retriever = initializing_server(api_key,combined_texts)
final = retriever
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
    | StrOutputParser()
)
def get_retrieval():
    return final


question = input('Enter your query?')
question = expand_query(question)
print(rag_chain.invoke(question))