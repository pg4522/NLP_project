#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras_preprocessing.text import text_to_word_sequence, Tokenizer
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict


# ## File path

# In[2]: User Setting


file_path = 'C:/Users/SJH/OneDrive - korea.ac.kr/문서/MBTI 500.csv'
USER_SAMPLE = 1


# ## Date Load & check

# In[3]:


data = pd.read_csv(file_path, encoding = "UTF-8")


# In[4]:


if USER_SAMPLE :
    data = data.groupby('type').sample(frac = 0.2)
data.reset_index(drop = True, inplace = True)


# In[5]:


X, y = data.drop('type', axis = 1), data['type']


# ## Distribution of label and so on

# In[6]:


data.info()


# In[7]:


data.isnull().sum()


# In[8]:


print(X.nunique(), len(X)) # Same


# In[9]:


sns.countplot(x='type', data=data, order=y.value_counts().index)
plt.show()
count = data.groupby('type').count()

count.sort_values(by=['posts'], ascending=False)
print(count, y.value_counts(normalize=True), sep = '\n======================\n')


# ## Top 3000 words

# In[10]:


tokenizer_top_words = Tokenizer(oov_token="<OOV>", split=' ', num_words=3000)
tokenizer_top_words.fit_on_texts(X.iloc[:, 0])


# In[11]:


# only 3000 words encoding
# tmp = X.head(10)
X_tp_words = X.copy()
X_tp_words['tok_tw'] = X_tp_words.apply(lambda v : tokenizer_top_words.texts_to_sequences([v['posts']]), axis = 1)


# In[12]:


X_tp_words['tok_tw'] = X_tp_words.apply(lambda v: np.array(v['tok_tw']).reshape(-1, 1).tolist(), axis = 1)


# In[13]:


X_tp_words['tok_tw_bool']=X_tp_words.apply(lambda v: list(map(lambda t: int(t[0] > 1), v['tok_tw'])), axis = 1)


# ## Words used more than 1000 times

# In[14]:


word_dict = tokenizer_top_words.word_counts
word_dict = OrderedDict(sorted(word_dict.items(), key = lambda t : t[-1],reverse= True))


# In[15]:


word_dict_top = []
for i, (key, value) in enumerate(word_dict.items()) :
    if value >= 1000 :
        word_dict_top.append(key)
print(f"size is {len(word_dict_top)}")


# In[16]:


# Boolean encoding
X_freq_words = X.copy()
X_freq_words['tok_tw'] = X_freq_words.apply(lambda v : WordPunctTokenizer().tokenize(v['posts']), axis = 1)


# In[17]:


X_freq_words['tok_tw_bool'] = X_freq_words.apply(lambda row : [1 if x in row['tok_tw'] else 0 for x in word_dict_top], axis = 1)

# In[ ]:


X_freq_words['tok_tw_in'] = X_freq_words.apply(lambda row : [1 if x in word_dict_top else 0 for x in row['tok_tw']], axis = 1)


# ## Dataframe print

# In[ ]:


X_tp_words


# In[ ]:


X_freq_words

