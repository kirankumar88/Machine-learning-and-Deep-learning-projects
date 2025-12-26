#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install nltk
import warnings
warnings.filterwarnings('ignore')
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("amazonreviews.tsv",sep="\t")
df.head()


# In[3]:


df.loc[0]['review']


# In[4]:


df.info()


# # Inference :
# Dataset contains 10,000 reviews with 0 missing values, enabling full utilization without imputation risk.

# In[5]:


(df==0).sum()


# Inference : No zeros found

# # Inference :
# No null or zero values detected (100% data completeness), ensuring robust downstream modeling without data leakage

# In[6]:


df.describe()


# # Inference :
# Sentiment classes are near-balanced (Neg: 50.97%, Pos: 49.03%), eliminating the need for resampling or class weighting.

# # Inference :
# The dataset exhibits a near-balanced class distribution, with 5,097 negative reviews and 4,903 positive reviews. This balance minimizes classification bias and allows the use of accuracy as a reliable evaluation metric, supplemented by precision, recall, and F1-score for robustness.

# In[7]:


df.isna().sum()


# # Inference :
# Doesnt have any null values


# In[9]:


df["label"].value_counts()


# # Inference :
# The sentiment labels are nearly evenly distributed, with 5,097 negative and 4,903 positive reviews, ensuring unbiased model training and reliable performance evaluation without the need for class balancing techniques.

# In[10]:


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[11]:


# Download resources (run once)
nltk.download("stopwords")
nltk.download("wordnet")


# In[12]:


# Initialize tools
stop_words = set(stopwords.words("english"))
stop_words -= {"not", "no", "nor", "never"}   # preserve negation
lemmatizer = WordNetLemmatizer()


# In[13]:


# Defining text preprocessing function

def text_preprocess(text):
    text = text.lower() # convert to Lowercase
    text = re.sub(r"[^a-z\s]", "", text) # Remove punctuation & numbers
    words = text.split() # Tokenization
    words = [w for w in words if w not in stop_words]  # Stop word removal
    words = [lemmatizer.lemmatize(w) for w in words] # Lemmatization
    return " ".join(words)   # Reconstruct sentences
    


# In[14]:


# Apply and verify text preprocessing function to dataframe

df["clean_review"] = df["review"].apply(text_preprocess) #Text preprocessing
df[["review", "clean_review"]].head() # Compare and check results


# In[15]:


df.head()


# # World cloud

# In[16]:


#!pip install wordcloud


# In[17]:


from PIL import Image
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import numpy as np


# In[19]:


# Combine all reviews into one string
text = " ".join(df["clean_review"].dropna().astype(str))

wordcloud = WordCloud(
    width=800,
    height=600,
    background_color="black",
    stopwords=set(STOPWORDS)
)

wordcloud.generate(text)

# Save high-resolution image
wordcloud.to_file(
    "Amazon_reviews.png"
)

plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# # Label encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
df.head()


# # Feature extraction  and model building :  Pipeline creation

# In[ ]:


# Define model evaluation function :

def evaluate_clf(y,yhat):
    cm = confusion_matrix(y,yhat)
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True,fmt='d', cbar=False)
    plt.show()
    print(classification_report(y,yhat))


# In[ ]:


from sklearn.model_selection import train_test_split

X = df["clean_review"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# # Logistic regression

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


logr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('lreg', LogisticRegression(max_iter=400))
])


# In[ ]:


logr_pipeline.fit(X_train, y_train)


# In[ ]:


yhat_lr = logr_pipeline.predict(X_test)


# In[ ]:


evaluate_clf(y_test, yhat_lr)


# # Inference :
# 
# 1. Class 0 (Negative Reviews)
# Precision = 0.86, Recall = 0.86 indicates reliable identification of negative sentiment with low false-alarm and miss rates, supporting proactive issue detection.
# 
# 2. Class 1 (Positive Reviews)
# Precision = 0.85, Recall = 0.85 demonstrates consistent recognition of positive feedback without sentiment bias.

# # SVM model

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5
    )),
    ('svm', SVC(kernel='linear'))
])


# In[ ]:


svm_pipeline.fit(X_train,y_train)


# In[ ]:


yhat_svm = svm_pipeline.predict(X_test)
yhat_svm


# In[ ]:


evaluate_clf(y_test, yhat_svm)


# # Inference :
# 
# 1. Class 0 (Negative Reviews)
# Precision = 0.86, Recall = 0.85 indicates strong negative sentiment detection with slightly higher false-negative risk compared to Logistic Regression.
# 
# 2.Class 1 (Positive Reviews)
# Precision = 0.85, Recall = 0.86 shows effective capture of positive sentiment with marginally improved recall over Logistic Regression.

# | Metric                 | Logistic Regression | Linear SVM |
# | ---------------------- | ------------------- | ---------- |
# | Accuracy               | **0.86**            | 0.85       |
# | Precision              | 0.86                | 0.86       |
# | Recall                 | **0.86**            | 0.85       |
# | F1-score               | **0.86**            | 0.85       |
# | Speed                  | **Fast**            | Slower     |
# | Interpretability       | **High**            | Medium     |
# | Production Suitability | **High**            | Medium     |
# 
# # Inference :
# Logistic Regression performs slightly better overall and is faster and easier to deploy, making it the preferred production model.

# # Final Summary :

# 1. Analyzed 10,000 customer reviews using an automated sentiment analysis system to understand overall customer satisfaction beyond star ratings. The data was complete, balanced, and reliable, allowing accurate evaluation of customer opinions expressed in text reviews.
# 
# 2. The sentiment model correctly classified reviews with ~85–86% accuracy, demonstrating strong and consistent performance across both positive and negative feedback. This means the system can reliably detect dissatisfied customers as well as positive experiences from review text alone.
# 
# 3. From a business perspective, this solution enables real-time monitoring of negative sentiment trends, early identification of product or service issues, and faster response to customer complaints.
# 
# 4. The approach is scalable, cost-effective, and suitable for deployment in a production environment to support customer experience and product quality decisions.
