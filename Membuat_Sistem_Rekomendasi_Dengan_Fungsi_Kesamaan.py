#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import library yang dibutuhkan
import pandas as pd
import numpy as np


# In[2]:


#lakukan pembacaan dataset
movie_rating_df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/movie_rating_df.csv')


# In[3]:


#tampilkan 5 baris teratas dari movive_rating_df
print(movie_rating_df.head())


# In[4]:


#tampilkan info mengenai tipe data dari tiap kolom
print(movie_rating_df.info())


# In[5]:


#Simpan actor_name.csv pada variable name_df 
name_df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/actor_name.csv')


# In[6]:


#Tampilkan 5 baris teratas dari name_df
print(name_df.head())


# In[7]:


#Tampilkan informasi mengenai tipe data dari tiap kolom pada name_df
print(name_df.info())


# In[8]:


#Menyimpan dataset pada variabel director_writers
director_writers = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/directors_writers.csv')


# In[9]:


#Manampilkan 5 baris teratas
print(director_writers.head()) 


# In[10]:


#Menampilkan informasi tipe data
print(director_writers.info()) 


# In[11]:


#Convert into List
import pandas as pd
director_writers = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/directors_writers.csv')


# In[12]:


#Mengubah director_name menjadi list
director_writers['director_name'] = director_writers['director_name'].apply(lambda row: row.split(','))
director_writers['writer_name'] = director_writers['writer_name'].apply(lambda row: row.split(','))


# In[13]:


#Tampilkan 5 data teratas
print(director_writers.head())


# In[15]:


#update name_df
name_df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/actor_name.csv')


# In[16]:


#Kita hanya akan membutuhkan kolom nconst, primaryName, dan knownForTitles
name_df = name_df[['nconst','primaryName','knownForTitles']]


# In[17]:


#Tampilkan 5 baris teratas dari name_df
print(name_df.head())


# In[18]:


#Melakukan pengecekan variasi
print(name_df['knownForTitles'].apply(lambda x: len(x.split(','))).unique())


# In[19]:


#Mengubah knownForTitles menjadi list of list
name_df['knownForTitles'] = name_df['knownForTitles'].apply(lambda x: x.split(','))


# In[20]:


#Mencetak 5 baris teratas
print(name_df.head())


# In[21]:


import numpy as np
#menyiapkan bucket untuk dataframe
df_uni = []


# In[22]:


df_uni = []

for x in ['knownForTitles']:
    #mengulang index dari tiap baris sampai tiap elemen dari knownForTitles
    idx = name_df.index.repeat(name_df['knownForTitles'].str.len())
   
   #memecah values dari list di setiap baris dan menggabungkan nya dengan rows lain menjadi dataframe
    df1 = pd.DataFrame({
        x: np.concatenate(name_df[x].values)
    })
    
    #mengganti index dataframe tersebut dengan idx yang sudah kita define di awal
    df1.index = idx
    #untuk setiap dataframe yang terbentuk, kita append ke dataframe bucket
    df_uni.append(df1)
    
#menggabungkan semua dataframe menjadi satu
df_concat = pd.concat(df_uni, axis=1)

#left join dengan value dari dataframe yang awal
unnested_df = df_concat.join(name_df.drop(['knownForTitles'], 1), how='left')

#select kolom sesuai dengan dataframe awal
unnested_df = unnested_df[name_df.columns.tolist()]
print(unnested_df)


# In[23]:


unnested_drop = unnested_df.drop(['nconst'], axis=1)


# In[24]:


#menyiapkan bucket untuk dataframe
df_uni = []

for col in ['primaryName']:
    #agregasi kolom PrimaryName sesuai group_col yang sudah di define di atas
    dfi = unnested_drop.groupby(['knownForTitles'])[col].apply(list)
    #Lakukan append
    df_uni.append(dfi)
df_grouped = pd.concat(df_uni, axis=1).reset_index()
df_grouped.columns = ['knownForTitles','cast_name']
print(df_grouped)


# In[25]:


#join table
movie_rating_df = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/movie_rating_df.csv')


# In[26]:


#join antara movie table dan cast table 
base_df = pd.merge(df_grouped, movie_rating_df, left_on='knownForTitles', right_on='tconst', how='inner')


# In[27]:


#join antara base_df dengan director_writer table
base_df = pd.merge(base_df, director_writers, left_on='tconst', right_on='tconst', how='left')
print(base_df.head())


# In[28]:


#Melakukan drop terhadap kolom knownForTitles
base_drop = base_df.drop(['knownForTitles'], axis=1)
print(base_drop.info())


# In[29]:


#Mengganti nilai NULL pada kolom genres dengan 'Unknown'
base_drop['genres'] = base_drop['genres'].fillna('Unknown')


# In[30]:


#Melakukan perhitungan jumlah nilai NULL pada tiap kolom
print(base_drop.isnull().sum())


# In[31]:


#Mengganti nilai NULL pada kolom dorector_name dan writer_name dengan 'Unknown'
base_drop[['director_name','writer_name']] = base_drop[['director_name','writer_name']].fillna('unknown')


# In[32]:


#karena value kolom genres terdapat multiple values, jadi kita akan bungkus menjadi list of list
base_drop['genres'] = base_drop['genres'].apply(lambda x: x.split(','))


# In[33]:


#Drop kolom tconst, isAdult, endYear, originalTitle
base_drop2 = base_drop.drop(['tconst','isAdult','endYear','originalTitle'], axis=1)


# In[34]:


#Drop kolom tconst, isAdult, endYear, originalTitle #Reformat Table
base_drop2 = base_drop.drop(['tconst','isAdult','endYear','originalTitle'], axis=1)

base_drop2 = base_drop2[['primaryTitle','titleType','startYear','runtimeMinutes','genres','averageRating','numVotes','cast_name','director_name','writer_name']]

base_drop2.columns = ['title','type','start','duration','genres','rating','votes','cast_name','director_name','writer_name']
print(base_drop2.head())


# In[35]:


#Klasifikasi berdasar title, cast_name, genres, director_name, dan writer_name KLASIFIKASI_METADATA
feature_df = base_drop2[['title','cast_name','genres','director_name','writer_name']]


# In[36]:


#Tampilkan 5 baris teratas
print(feature_df.head())


# In[37]:


#Bagaimana cara membuat fungsi untuk strip spaces dari setiap row dan setiap elemennya
def sanitize(x):
    try:
        #kalau cell berisi list
        if  isinstance(x, list):
            return [i.replace(' ','').lower() for i in x]
        #kalau cell berisi string
        else:
            return [x.replace(' ','').lower()]
    except:
        print(x)
        
#Kolom : cast_name, genres, writer_name, director_name        
feature_cols = ['cast_name','genres','writer_name','director_name']

#Apply function sanitize 
for col in feature_cols:
    feature_df[col] = feature_df[col].apply(sanitize)


# In[38]:


# Bagaimana cara membuat fungsi untuk membuat metadata soup (menggabungkan semua feature menjadi 1 bagian kalimat) untuk setiap judulnya
#kolom yang digunakan : cast_name, genres, director_name, writer_name
def soup_feature(x):
    return ' '.join(x['cast_name']) + ' ' + ' '.join(x['genres']) + ' ' + ' '.join(x['director_name']) + ' ' + ' '.join(x['writer_name'])

#membuat soup menjadi 1 kolom 
feature_df['soup'] = feature_df.apply(soup_feature, axis=1)


# In[39]:


#Cara menyiapkan CountVectorizer (stop_words = english) dan fit dengan soup yang kita buat di atas
#import CountVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

#definisikan CountVectorizer dan mengubah soup tadi menjadi bentuk vector
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(feature_df['soup'])

print(count)
print(count_matrix.shape)


# In[40]:


#Cara membuat model similarity antara count matrix
#Import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
#Gunakan cosine_similarity antara count_matrix 
cosine_sim = cosine_similarity(count_matrix, count_matrix)
#print hasilnya
print(cosine_sim)


# In[42]:


#Cara membuat content based recommender system
indices = pd.Series(feature_df.index, index=feature_df['title']).drop_duplicates()

def content_recommender(title):
    #mendapatkan index dari judul film (title) yang disebutkan
    idx = indices[title]

    #menjadikan list dari array similarity cosine sim 
    #hint: cosine_sim[idx]
    sim_scores = list(enumerate(cosine_sim[idx]))

    #mengurutkan film dari similarity tertinggi ke terendah
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #untuk mendapatkan list judul dari item kedua sampe ke 11
    sim_scores = sim_scores[1:11]

    #mendapatkan index dari judul-judul yang muncul di sim_scores
    movie_indices = [i[0] for i in sim_scores]

    #dengan menggunakan iloc, kita bisa panggil balik berdasarkan index dari movie_indices
    return base_df.iloc[movie_indices]

#aplikasikan function di atas
print(content_recommender('The Lion King'))


# In[ ]:




