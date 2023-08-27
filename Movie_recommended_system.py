#!/usr/bin/env python
# coding: utf-8

# # Movie recommendation system Types 
# ## 1-Content based-ek aisa recommendation system hai jo content ke similarity ke basis par content recommand krta h like listen romantic song then recommnded romantic song ,watching horror movies recommand horror movies etc,tags created in this 
# ## 2-Colloborative filtering-user ke similarity ke basis pr recommndation deta h for ex two users have give same rating on a particular movie,then again a first user watching another movies so according to same similarity we recommanded user2 also that movies 
# ## 3-Hybrid-combination of both
# 
# ## In this we work on content based movie recommendation system

# In[2]:


import  numpy as np
import pandas as pd


# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')


# In[5]:


credits=pd.read_csv('tmdb_5000_credits.csv')


# In[14]:


movies.shape


# In[15]:


credits.shape


# In[11]:


movies.head(10)


# In[7]:


credits.head(10)


# In[8]:


credits.head(1)['cast']


# In[9]:


credits.head(1)['cast'].values


# In[16]:


#merge this two datasets based on title
movies.merge(credits,on='title').shape


# In[17]:


movies=movies.merge(credits,on='title')


# In[18]:


movies.head(1)


# In[19]:


movies.info()


# In[23]:


#our important columns
#genres,movies_id,keywords,title,overview,cast,crew
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[24]:


movies.head(1)


# In[25]:


#we merge overview,genres,keywords,cast,crew to make new column tags


# # Data preprocessing

# In[26]:


#check for missing data
movies.isnull().sum()


# In[28]:


movies.dropna(inplace=True)


# In[29]:


movies.isnull().sum()


# In[31]:


#check for duplicate data
movies.duplicated().sum()


# In[32]:


#now working on jason format columns
movies.iloc[0].genres


# In[33]:


# we have to convert it is in ['Action','Adventure','Fantasy','Science Fiction'] format


# In[40]:


def convert(obj):     # ast.literal_eval-CONVERT string containing list into python list
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[42]:


movies['genres'].apply(convert) 
#The apply() method allows you to apply a function along one of the axis of the DataFrame, default 0, which is the index (row) axis.


# In[43]:


movies['genres']=movies['genres'].apply(convert) 


# In[44]:


movies.head(1)


# In[45]:


movies['keywords']=movies['keywords'].apply(convert)


# In[46]:


movies.head(1)


# In[48]:


movies['cast'] [0]


# In[49]:


# we  want first three cast  details only
def convert3(obj):     
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[50]:


movies['cast'].apply(convert3)


# In[51]:


movies['cast']=movies['cast'].apply(convert3)


# In[52]:


movies.head(1)


# In[53]:


movies['crew'] [0]


# In[54]:


# we want only those whose job is director and we extract only those name 
def fetch_director(obj):     
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[55]:


movies['crew'].apply(fetch_director)


# In[56]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[57]:


movies.head(1)


# In[58]:


movies['overview'] [0]


# In[59]:


#convert it also in list
movies['overview'].apply(lambda x:x.split())


# In[60]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[61]:


movies.head(1)


# In[63]:


# here we remove spaces which is between a element of list for ex Sam Worthington into SamWorthington 
movies['genres']=movies['genres'].apply(lambda x : [i.replace(" ","") for i in x])


# In[64]:


movies['keywords']=movies['keywords'].apply(lambda x : [i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x : [i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x : [i.replace(" ","") for i in x])


# In[101]:


movies.head(5)


# In[102]:


# now we create a new column tags and  we concatenate our lists i.e. overview, genres, keywords,cast, crew in it
movies['tags']=movies['overview']+ movies['genres']+ movies['keywords']+movies['cast']+movies['crew']


# In[103]:


movies['tags']


# In[104]:


movies.head(1)


# In[105]:


new_df = movies[['movie_id','title','tags']] # we use movies[]-for selecting one column
                                              # we use movies[[]]- for selectin multiple columns


# In[106]:


new_df


# In[107]:


# now we convert tag column into string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[108]:


new_df


# In[109]:


new_df['tags'] [0]


# In[110]:


#convert all these string into lowercase it is recommanded
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[111]:


new_df['tags'] [0]


# In[112]:


new_df.head()


# In[81]:


# our challenges is to show similaity between two tags
#so here we use vectorization to convert our tags into vectors i.e text to vectors using technique bag of words
# using this technique we can,t use stop_words-and, the, are,is,in,a etc
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[84]:


cv.fit_transform(new_df['tags']).toarray()


# In[86]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[87]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[88]:


vectors


# In[89]:


vectors[0]


# In[90]:


#here our movies converted into vectors now


# In[91]:


#now we apply stemming so that same word converted into root words like
#[actor,actors,acting] after applying lemming convert into [actor,actor,actor]


# In[92]:


get_ipython().system('pip install nltk')


# In[117]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[118]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[119]:


new_df['tags']=new_df['tags'].apply(stem)


# In[120]:


# our challenges is to show similaity between two tags
#so here we use vectorization to convert our tags into vectors i.e text to vectors using technique bag of words
# using this technique we can,t use stop_words-and, the, are,is,in,a etc
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[121]:


cv.fit_transform(new_df['tags']).toarray()


# In[122]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[123]:


vectors


# In[124]:


vectors[0]


# In[126]:


cv.get_feature_names_out()


# In[127]:


'''now our each movie converted into vectors so our next challenge we calculated distance between two movies using cosine 
distance,more distance less similarity and vice versa'''


# In[128]:


#in cosine distance smaller the angle smaller the distance and vice versa


# In[131]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)


# In[132]:


similarity.shape


# In[133]:


similarity[0]


# In[145]:


list(enumerate(similarity[0])) 
#It returns an iterator of pairs, where each pair consists of an index and the value associated with that index in the sequence.


# In[146]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[165]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[166]:


recommend('Avatar')


# In[167]:


recommend('John Carter')


# In[169]:


recommend('Tangled')


# In[173]:


import pickle


# In[175]:


pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




