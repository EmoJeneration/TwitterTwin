#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sentence_transformers import SentenceTransformer


# In[2]:


import faiss


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# In[6]:


df = pd.read_csv(r'/Users/olivia/TwitterTwin/similaritytweets.csv')


# In[24]:


print(df)


# In[29]:


import re
df['Tweets'] = df['Tweets'].astype(str)

pattern = r'http\S+'

# apply the regex pattern to the 'text' column
df['Tweets'] = df['Tweets'].str.replace(pattern, '', regex=True)

# define a list of special characters to remove
special_chars =  ['~', ':', "'", '+', '[', '\\', '^',
                      '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.', '_', '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '’', '“', '”', "…"]
# define a function to remove special characters
def remove_special_chars(column):
    for char in special_chars:
        column = column.str.replace(char, '')
    return column

# apply the function to the DataFrame column
df['Tweets'] = remove_special_chars(df['Tweets'])


# In[30]:




# Define a regex pattern to match Twitter handles
pattern = r'@[A-Za-z0-9_]+'

# Define a function to remove Twitter handles from a string
def remove_handles(tweet):
    return re.sub(pattern, '', tweet)

# Apply the function to the 'Tweets' column using apply() and a lambda function
df['Tweets'] = df['Tweets'].apply(lambda tweet: remove_handles(tweet))

# Print the first 5 rows of the DataFrame
print(df.head())
print(df)


# In[32]:


df


# In[33]:


# Create average embeddings for each user
embeddings = {}
for handle, group in df.groupby('User'):
    tweet_embeddings = []
    for tweet in group['Tweets']:
        embedding = model.encode(tweet)
        tweet_embeddings.append(embedding)
    tweet_embeddings = np.array(tweet_embeddings).astype('float32')
    average_embedding = np.mean(tweet_embeddings, axis=0)
    embeddings[handle] = average_embedding


# In[35]:


# Define dimension of embeddings
d = 384

# Create empty index
index = faiss.IndexFlatIP(d)


# In[263]:


# Storing the embeddings
import pickle
with open('embeddings.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[36]:


# Adding the embeddings to the FAISS index
x = np.empty((len(embeddings), d), dtype=np.float32)
for i, (key, value) in enumerate(embeddings.items()):
    x[i] = value
index = faiss.IndexFlatIP(d)
index.add(x)


# In[192]:


# Saving the Index locally
faiss.write_index(index, "my_similarity_emoji_index.index")


# In[ ]:





# In[ ]:




