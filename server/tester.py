"""
Testing the embeddings generated by generator_from_urls

Made by Fabien ROGER and Alexandre SAJUS
https://github.com/FabienRoger
https://github.com/AlexandreSajus
"""

# Imports
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import time
import requests
import random
import html2text
import nltk
from sentence_transformers import SentenceTransformer
import faiss
print("Importing libraries")
nltk.download('punkt')
print("")

# Enlarge print display
pd.set_option('display.max_colwidth', 500)

# Load the model
print("Loading the model")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("")

#Loading
folder = "News V1/"
urls = np.load(folder + "urls.npy")
ids = np.load(folder + "ids.npy")
quotes_df = pd.read_csv(folder + 'quotes.csv')
quotes = quotes_df['sentence']
embeds = np.load(folder + "embeds.npy")

# Generate the faiss index

embeds = embeds.astype("float32")
d = embeds.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeds)


def get_most_similars(s, n):
    # Returns the n most similars sentences to the query s

    # Calculate the embedding of the query
    e = model.encode(s)
    # Normalize it
    e = e/np.linalg.norm(e)

    # Use faiss to get the most similar sentences (according to the cosine similarity)
    query = v = np.array([e])
    _, indexes = index.search(query, n)
    indexes = list(indexes[0, :])

    # Return the n best ones (sentence and url)
    return [[quotes[indexes[i]], urls[ids[indexes[i]]]] for i in range(n)]


test = "#OPINION: With #Ennahda in power, Tunis saw 6,798 protests, 15,000 #coronavirus deaths, a 200% increase in infectioâ€¦ https://t.co/SIVHpZ3rPZ"
print("Testing: " + test)
print("")
print("Output: ")
results = get_most_similars(test, 3)

for i in range(len(results)):
    print("TEXT BEGIN")
    print(results[i][0])
    print(results[i][1])
    print("TEXT END \n")