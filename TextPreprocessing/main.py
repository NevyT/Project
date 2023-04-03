import pickle
import statistics

import pandas as pd

import nltk
import unidecode as unidecode
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder

import re
import string

import spacy
from spacy import displacy

from xgboost import XGBClassifier
# for Lemmatization
# nltk.download('omw-1.4')

def clean_text(text):
    text = text.lower()
    # remove
    text = re.sub('\[.*?\]', '', text)
    # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove tags
    text = re.sub('<.*?>+', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove breaklines
    text = re.sub('\n', '', text)
    # remove numbers
    text = re.sub('\w*\d\w*', '', text)

    # Changing Text that's not in ascii into ascii (Ex: kožušček -> kozuscek; 30 \U0001d5c4\U0001d5c6/\U0001d5c1 -> 30 km/h)
    text = unidecode.unidecode(text)

    # transform text into token
    text_token = nltk.word_tokenize(text)

    # remove stopwords
    words = [w for w in text_token if w not in sw]

    return ' '.join(words)

lemmatizer = WordNetLemmatizer()

def lemmatize_sentence(text):
    # transform text into token
    text_token = nltk.word_tokenize(text)
    lemmatized_sentence = []
    for word in text_token:
        lemmatized_sentence.append(word)
    return " ".join(lemmatized_sentence)

nlp = spacy.load("en_core_web_sm")

sw = stopwords.words('english')

df = pd.read_csv('alldata_1_for_kaggle.csv', encoding="latin-1")
df = df.rename(columns={'Unnamed: 0': 'index', '0': 'Type', 'a': 'Description'})

df['Description'] = df['Description'].apply(clean_text)
df['Description'] = df['Description'].apply(lemmatize_sentence)

# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vector = TfidfVectorizer(max_df=0.8,min_df=0.2)
docs = vector.fit_transform(df['Description'].tolist())
with open('Vectorizer','wb') as f:
    pickle.dump(vector,f)
# docs = vector.transform(df['Description'])
features = vector.get_feature_names()

le = LabelEncoder()
df['Type'] = le.fit_transform(df['Type'])

mxgb = XGBClassifier(objective='multi:softmax',num_class=3)
classification_report_XGBoost = []

SKCV = model_selection.StratifiedKFold(n_splits=10)
for train_index,test_index in SKCV.split(docs,df['Type']):
    X_train, X_test = docs[train_index], docs[test_index]
    y_train,y_test = df['Type'][train_index], df['Type'][test_index]

    mxgb.fit(X_train,y_train)

    prediction = mxgb.predict(X_test)

    matrix = classification_report(y_test,prediction,output_dict=True)
    # print(matrix)
    classification_report_XGBoost.append(matrix)

for i in classification_report_XGBoost:
    print(i)

NewDict = newDict = {
    '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    '2': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
}

for i in classification_report_XGBoost:
    for key,value in i.items():
        if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg' :
            # print(key,value.keys())
            if list(value.keys())[0] == 'precision':
                NewDict[key]['precision'].append(value['precision'])
            else:
                break
            if list(value.keys())[1] == 'recall':
                NewDict[key]['recall'].append(value['recall'])
            else:
                break
            if list(value.keys())[2] == 'f1-score':
                NewDict[key]['f1-score'].append(value['f1-score'])
            else:
                break
            if list(value.keys())[3] == 'support':
                NewDict[key]['support'].append(value['support'])
            else:
                break

for key,value in NewDict.items():
    print('Class: ', key, ' | Avg Precision: ', round(statistics.fmean(value['precision']),2), ' | Avg Recall: ', round(statistics.fmean(value['recall']),2), ' | Avg F1-Score: ', round(statistics.fmean(value['f1-score']),2), ' | Support: ', value['support'])
mean_xgboost = []
for i in classification_report_XGBoost:
    mean_xgboost.append(i['accuracy'])
print("Average Accuracy for XGBoost: ", round(statistics.fmean(mean_xgboost),3))
# print(statistics.fmean(precision_0))
print("")

# print(classification_report_XGBoost)
# with open('modelXGBoost','wb') as f:
#     pickle.dump(mxgb,f)