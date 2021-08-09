"""
Generating embeddings for each sentence in trusted websites to fact-check an input sentence

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

# Input websites
sites = ["https://www.nytimes.com/international/",
        "https://www.economist.com/",
        "https://www.bbc.com/",
        "https://www.washingtonpost.com/"]

# Enlarge print display
pd.set_option('display.max_colwidth', 500)

# Load the model
print("Loading the model")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("")

# Returns the homepage url of a site


def get_ref(site):
    slash = 0
    for i in range(len(site)):
        if site[i] == "/":
            slash += 1
        if slash == 3:
            return site[:i]
    return None

# Extract all links from websites


def extract_all_links(sites):
    links = []
    for site in sites:
        try:
            ref = get_ref(site)
            html = requests.get(site, timeout=5).text
            soup = bs(html, 'html.parser').find_all('a')
            for link in soup:
                try:
                    url = link.get('href')
                    if url[0:5] == "https":
                        links.append(url)
                    elif url[0] == "/":
                        links.append(ref + url)
                except:
                    pass
        except:
            pass
    return links


print("Extracting urls")
urls = extract_all_links(sites)
print("")

print("URLs retrieved: " + str(len(urls)))
print(urls[:20])
print("...")
print("")

# ids refers to the index of the url in urls
ids = []
# quotes are the sentences extracted online
quotes = []
# embeds are the embeddings of the sentences
embeds = np.zeros((0, 384))

# List for Randomizing our request rate to avoid getting denied by the website
rate = [i/10 for i in range(10)]

# Iterating through the URLS
print("Generating embeddings")
for i in range(len(urls)):

    if i % 50 == 0:
        print("Iteration: " + str(i) + "/" + str(len(urls)))

    try:
        # Accessing the Webpage
        page = requests.get(urls[i])

        # Getting the webpage's content in pure html
        soup = bs(page.content, features="html.parser")

        # Convert it to text
        text = (html2text.html2text(str(soup)))

        # Convert it to sentences
        sentences = nltk.tokenize.sent_tokenize(text)

        if len(sentences) != 0:

            # Add the id of the url for each sentence
            ids += [i]*len(sentences)

            # Add the sentences to quotes
            quotes += sentences

            # Calculate the embedding of each sentence
            embeddeds = model.encode(sentences, show_progress_bar=False)

            # Normalize the embedding of each sentence
            embeddeds = embeddeds / \
                np.linalg.norm(embeddeds, axis=1, keepdims=True)

            # Add the embeddings to embeds
            embeds = np.concatenate((embeds, embeddeds))

    except:
        pass

    # Randomizing our request rate to avoid getting denied by the website
    time.sleep(random.choice(rate))
print("Finished")
print("")

# Remove duplicates
print("Removing duplicates")
new_ids = []
new_quotes = []
new_embeds = []

for i in range(len(quotes)):
    if i % 10000 == 0:
        print("Iteration: " + str(i) + "/" + str(len(quotes)))
    if quotes[i] not in new_quotes:
        new_ids.append(ids[i])
        new_quotes.append(quotes[i])
        new_embeds.append(embeds[i])

ids = new_ids
quotes = new_quotes
embeds = new_embeds
print("")

# Convert to arrays
print("Converting to arrays")
urls = np.array(urls)
ids = np.array(ids)
quotes_df = pd.DataFrame(quotes, columns=['sentence'])
embeds = np.array(embeds)
print("")

# Saving
print("Saving...")
np.save("urls", urls)
np.save("ids", ids)
quotes_df.to_csv('quotes.csv')
np.save("embeds", embeds)
print("")

"""
#Loading
urls = np.load("urls.npy")
ids = np.load("ids.npy")
quotes = np.load("quotes.npy")
embeds = np.load("embeds.npy")
"""

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


test = "Vaccines CANNOT prevent COVID transmission anymore - CDC’s Dr. Rochelle Walensky admits."
print("Testing: " + test)
print("")
print("Output: ")
print(get_most_similars(test, 5))
