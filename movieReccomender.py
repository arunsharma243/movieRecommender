#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[3]:


movies=pd.read_csv('data/tmdb_5000_movies.csv')
credits=pd.read_csv('data/tmdb_5000_credits.csv')


# In[4]:


movies.head(3)


# In[7]:


credits.head(3)


# In[9]:


movies.shape


# In[11]:


credits.shape


# In[13]:


movies=movies.merge(credits, on='title')


# In[15]:


movies.head(3)


# In[17]:


movies.shape


# In[19]:


movies['original_language'].value_counts()


# In[21]:


movies.columns


# In[23]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[25]:


movies.head(3)


# In[27]:


movies.shape


# In[29]:


movies.isnull().sum()


# In[31]:


movies.dropna(inplace=True)


# In[33]:


movies.shape


# In[35]:


movies.isnull().sum()


# In[37]:


movies.duplicated().sum()


# In[39]:


movies.iloc[0]['genres']


# In[41]:


import ast

def convert(text):
    l=[]
    for i in ast.literal_eval(text):
        l.append(i['name'])

    return l


# In[43]:


movies['genres']=movies['genres'].apply(convert)


# In[44]:


movies.head(3)


# In[47]:


movies.iloc[0]['keywords']


# In[49]:


movies['keywords']=movies['keywords'].apply(convert)


# In[50]:


movies.head(3)


# In[51]:


movies.iloc[0]['cast']


# In[53]:


def convert_cast(text):
    l=[]
    counter=0
    for i in ast.literal_eval(text):
        if counter<3:
           l.append(i['name'])
        counter+=1

    return l


# In[57]:


movies['cast']=movies['cast'].apply(convert_cast)


# In[58]:


movies.head(3)


# In[59]:


movies.iloc[0]['crew']


# In[60]:


def fetch_director(text):
    l=[]
    for i in ast.literal_eval(text):
        if i['job']=='Director':
            l.append(i['name'])
            break

    return l


# In[61]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[62]:


movies.head(3)


# In[63]:


movies.iloc[0]['overview']


# In[71]:


movies['overview']=movies['overview'].apply(lambda x:x.split())
movies.head(3)


# In[73]:


def remove_space(word):
    l=[]
    for i in word:
        l.append(i.replace(" ",""))
    return l


# In[75]:


movies['cast']=movies['cast'].apply(remove_space)
movies['crew']=movies['crew'].apply(remove_space)
movies['genres']=movies['genres'].apply(remove_space)
movies['keywords']=movies['keywords'].apply(remove_space)


# In[77]:


movies.head(3)


# In[79]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[81]:


movies.head(3)


# In[83]:


movies.iloc[0]['tags']


# In[85]:


new_df=movies[['movie_id','title','tags']]


# In[87]:


new_df.head()


# In[89]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[91]:


new_df.head()


# In[93]:


new_df.iloc[0]['tags']


# In[95]:


new_df['tags']= new_df['tags'].apply(lambda x:x.lower())


# In[97]:


new_df.head()


# In[99]:


import nltk
from nltk.stem import PorterStemmer


# In[100]:


ps=PorterStemmer()


# In[103]:


def stems(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))

    return " ".join(l)


# In[105]:


new_df['tags']= new_df['tags'].apply(stems)


# In[106]:


new_df.iloc[0]['tags']


# In[109]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[111]:


vector=cv.fit_transform(new_df['tags']).toarray()


# In[113]:


vector


# In[115]:


vector.shape


# In[117]:


from sklearn.metrics.pairwise import cosine_similarity


# In[119]:


similary= cosine_similarity(vector)


# In[120]:


similary


# In[121]:


similary.shape


# In[122]:


new_df[new_df['title'] == 'Spider-Man'].index[0]


# In[123]:


def recommend(movie):
    index=new_df[new_df['title'] == movie].index[0]
    distances=sorted(list(enumerate(similary[index])),reverse=True,key=lambda x:x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)


# In[131]:


recommend("Avatar")


# In[133]:


import pickle

pickle.dump(new_df,open('artifacts/movie_list.pkl','wb'))
pickle.dump(new_df,open('artifacts/similarity.pkl','wb'))


# In[ ]:




