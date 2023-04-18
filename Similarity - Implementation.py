#!/usr/bin/env python
# coding: utf-8

# In[49]:


from sentence_transformers import SentenceTransformer


# In[50]:


import gradio as gr


# In[51]:


import pandas as pd


# In[52]:


import numpy as np


# In[53]:


import faiss


# In[54]:


import snscrape.modules.twitter as sntwitter


# In[55]:


import re
import pickle


# In[56]:


# Load the Faiss index and embeddings dictionary
index = faiss.read_index("my_similarity_emoji_index.index")
with open('embeddings.pickle', 'rb') as f:
    embeddings = pickle.load(f)


# In[57]:


# Load the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# In[58]:


def get_tweets(username):
    if username[0] == '@':
        username = username[1:]
    try:
        tweets = []
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f"from:{username}").get_items()):
            if i >= 40:
                break
            tweets.append(tweet.content)
        return tweets
    except Exception as e:
        print(f"Error: {e}")


# In[59]:


def remove_special_chars(column):
    special_chars = ['~', ':', "'", '+', '[', '\\', '^', '@'
                      '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '’', '“', '”', "…"]
    for char in special_chars:
        column = column.str.replace(char, '')
    return column
# Define a function to remove Twitter handles from a string
def remove_handles(tweet):
    pattern = r'@[A-Za-z0-9_]+'
    return re.sub(pattern, '', tweet)


# In[60]:


def tweet_prep(tweets):
    tweets_df = pd.DataFrame(tweets, columns=["Tweets"])
    tweets_df['Tweets'] = tweets_df['Tweets'].astype(str)
    pattern = r'http\S+'
    # apply the regex pattern to the 'text' column
    tweets_df['Tweets'] = tweets_df['Tweets'].str.replace(pattern, '', regex=True)
    # removing handles
    tweets_df['Tweets'] = tweets_df['Tweets'].apply(lambda tweet: remove_handles(tweet))
    # removing special characters
    tweets_df['Tweets'] = remove_special_chars(tweets_df['Tweets'])
    return(tweets_df)


# In[61]:


def predict(user):
    # Load embeddings and user list
    with open('embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)
    users = list(embeddings.keys())
    # Load model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    try:
        tweets = get_tweets(user)
        # Do something with the tweets
    except Exception as e:
        return "Error: {}".format(str(e))
    #Clean tweets
    tweets_df = tweet_prep(tweets)
    #Embeddings
    tweet_embeddings = []
    for tweet in tweets_df['Tweets']:
        embedding = model.encode(tweet)
        tweet_embeddings.append(embedding)
    tweet_embeddings = np.array(tweet_embeddings).astype('float32')
    # Get average embedding
    avg_embedding = np.mean(tweet_embeddings, axis=0)
    # Reshape embedding for FAISS
    avg_embedding = avg_embedding.reshape(1, -1)
    # Create FAISS index
    index = faiss.read_index("my_similarity_emoji_index.index")
    # Search for similar user
    D, I = index.search(avg_embedding, 1)
    similar_user = users[I[0][0]]  
    return similar_user


# In[62]:


# Define Gradio interface
input_text = gr.inputs.Textbox(label="Enter a twitter handle: ")
output_text = gr.outputs.Textbox(label="Most similar user handle: ")

app_interface = gr.Interface(fn=predict, inputs=input_text, outputs=output_text)

# Launch the interface
app_interface.launch()


# In[ ]:




