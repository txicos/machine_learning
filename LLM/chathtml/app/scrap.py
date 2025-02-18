#import requests
import logging
from bs4 import BeautifulSoup
from readabilipy import simple_json_from_html_string

from sentence_transformers import SentenceTransformer, util

from chonkie import SentenceChunker
from autotiktokenizer import AutoTikTokenizer
import faiss
import numpy as np

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import streamlit as st


from time import sleep

@st.cache_data(show_spinner=False)
def get_text_from_url(url):
    """
    Fetches and extracts text content from a given URL.
    """

    options = Options()
    options.add_argument("--headless")
    # add any other options you need here

    service = Service(executable_path=r'/usr/local/bin/geckodriver')
    driver = webdriver.Firefox(service=service, options=options)

    try:
  
        logging.info("Fetching")
        # Load the page
        driver.get(url)
        sleep(10)
        # Get the HTML source of the page
        page_source = driver.page_source
        

    except Exception as e:
        logging.error(f"Error fetching the URL: {e}")
        return ''
    finally:
        driver.quit()

    logging.info("Parsing")
    
    article = simple_json_from_html_string(page_source, use_readability=True)
    
    lines = []
    
    htmlParse = BeautifulSoup(article['plain_content'], 'html.parser') 
  
    # getting all the paragraphs 
    for para in htmlParse.find_all("p"): 
        lines.append(para.get_text())  

    text = '\n'.join(line for line in lines if line)
    
    return text

    
def chunk_size(tokenizer, sentence):
    
    tokens = tokenizer.encode(sentence)

    chk_size =  len(tokens)
    return chk_size

def extract_chunks(text, sentence):
    tokenizer = AutoTikTokenizer.from_pretrained("gpt2")

    chunk_s = chunk_size(tokenizer, sentence)
    
    chunker = SentenceChunker(
        tokenizer=tokenizer,
        chunk_size=chunk_s,
        chunk_overlap=chunk_s/2,
        min_sentences_per_chunk=1
    )
       
    
    chunks = chunker.chunk(text)
    
    sentences = [chunk.text for chunk in chunks]

    return sentences


def extract_relevant_context_with_faiss(text, reference_sentence, model_name='all-MiniLM-L6-v2'):
    """
    Extracts contextually relevant parts of the text based on a given reference sentence using Sentence Transformers.
    """
    try:
    
        sentences = extract_chunks(text, reference_sentence)
        
        # Load the Sentence Transformer model
        model = SentenceTransformer(model_name)
        
        # Encode the sentences and the reference sentence
        sentence_embeddings = model.encode(sentences)
        reference_embedding = model.encode([reference_sentence])
        
        
        #num_vectors = len(sentence_embeddings)
        dim = len(sentence_embeddings[0])
        faiss_index = faiss.IndexFlatIP(dim) 

        faiss_index.add(np.array(sentence_embeddings, dtype=np.float32))

        k = 5  # Number of nearest neighbors to retrieve
        _, indices = faiss_index.search(np.array(reference_embedding, dtype=np.float32), k)    
                
        # Select sentences that meet 
        relevant_sentences = [
            sentences[i] for i in indices[0]
        ]
        
        return relevant_sentences
    except Exception as e:
      logging.error(f"Error during context extraction: {e}")
      return None
    
def extract_relevant_paragraphs(text, relevant_sentences):
  sentences = text.split('\n')
  
  paragraphs = []
  
  for r in relevant_sentences:
    for p in sentences:
      if r in p:
        paragraphs.append(p)
        
  return paragraphs

def perform_rag(text, query):
   chunked_context = extract_relevant_context_with_faiss(text, query)
   context = extract_relevant_paragraphs(text, chunked_context)
   return "\n".join(context)

# # Example usage
# long_text = get_text_from_url("https://apublica.org/2024/11/eleicao-de-trump-e-desastre-para-o-clima-e-vai-beneficiar-ele-proprio-diz-jose-cheibub/")
# reference = "o que Trump pretende fazer a respeito dos projetos aprovados pelo governo Biden?"
# context = extract_relevant_context_with_faiss(long_text, reference)
# print(extract_relevant_paragraphs(long_text, context))