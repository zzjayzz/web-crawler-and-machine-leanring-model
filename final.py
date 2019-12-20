
#########################   this code may takes 5-10mins to run(with 8GB RAM) #############################
#########################change the line46 to load the test file ########################################
########################################################################################################
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import time
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import random
from scipy.sparse import hstack
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv('df_train.csv')


def trans_to_df(filename):
    df_all1 = pd.DataFrame(columns=['title', 'author', 'institution', 'year', 'abstract', 'label'])
    f = open(filename)
    for line in f:
        if line != '\n':
            lines = line.split('\t')
            lines[5] = re.sub('\n', '', lines[5])
            df_all1 = df_all1.append(
                {'author': lines[0], 'abstract': lines[1], 'title': lines[2], 'institution': lines[3], 'year': lines[4],
                 'label': lines[5]}, ignore_index=True)
    return df_all1


df_all = trans_to_df('sample.txt')
df_all['number_author'] = 0
df_all['new_author_count'] = 0

# df_all=df_all.reset_index(drop=True)
df_all['abstract'] = df_all['abstract'] + df_all['title'] * 5 + df_all['author'] * 5 + df_all['institution'] * 5
def even(df1, label):
    df_label = df1.loc[df1['label'] == label].copy()
    return df_label

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+', ' ', total_text)
        total_text = re.sub(r"http\S+", "", total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        total_text = re.sub(r"n\'t", " not", total_text)
        total_text = re.sub(r"\'re", " are", total_text)
        total_text = re.sub(r"\'s", " is", total_text)
        total_text = re.sub(r"\'d", " would", total_text)
        total_text = re.sub(r"\'ll", " will", total_text)
        total_text = re.sub(r"\'t", " not", total_text)
        total_text = re.sub(r"\'ve", " have", total_text)
        total_text = re.sub(r"\'m", " am", total_text)
        #         for sub in set1:
        #                 total_text = re.sub(sub, " ", total_text)
        for word in total_text.split():
            # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "

        df_all.loc[index, column] = string


def dict1(df_even):
    i = 0
    list_author = []
    list_author1 = []
    list_author2 = []
    list_author3 = []
    list_author4 = []
    list_author5 = []
    list_author6 = []
    sum = 0
    list_author = list(df_even['author'])
    dict2 = {}
    for list_author1 in list_author:
        list_author1 = list_author1.split(':')
        for list_author2 in list_author1:
            list_author3.append(list_author2)
    #     print(list_author3)
    # list_author4=list(set(list_author3))[1:]
    list_author4 = list(set(list_author3))
    for list_unique_author in list_author4:
        for list_author in list_author3:
            if list_unique_author == list_author: i += 1
        dict2[list_unique_author] = i
        i = 0
    #     print(dict2)
    for index, row in df_even.iterrows():
        list_author5 = row['author']
        list_author5 = list_author5.split(':')
        #         print(list_author5)
        for list_author6 in list_author5:
            if not (list_author6 == ''):
                sum = dict2[list_author6] + sum
            else:
                list_author5.remove('')
        df_even.loc[index, 'number_author'] = len(list_author5)
        df_even.loc[index, 'new_author_count'] = sum
        sum = 0


# text processing stage.
stop_words = set(stopwords.words('english'))
for index, row in df_all.iterrows():
    if type(row['abstract']) is str:
        nlp_preprocessing(row['abstract'], index, 'abstract')
    else:
        print("there is no text description for id:", index)

list_datafrme = []
labels = df_all['label'].unique()
for label in labels:
    df_even = even(df_all, label)
    dict1(df_even)
    #     display(df_even)
    list_datafrme.append(df_even)

df_all = pd.concat(list_datafrme)
df_all.reset_index(drop=True)

list_year = []
list_year = list(df_all['year'])
# df_new_all['year'] = df_new_all['year'].astype(int)
list_year_new = []
for list_y in list_year:
    list_y = re.sub('[^0-9]', '', list_y)
    list_year_new.append(list_y)
df_all['year'] = list_year_new
df_all['year'] = df_all['year'].astype(int)
test = df_all

print('CIKM', train[train['label'] == 'CIKM']['label'].count())
print('SIGCHI', train[train['label'] == 'SIGCHI']['label'].count())
print('SIGKDD', train[train['label'] == 'SIGKDD']['label'].count())
print('SIGIR', train[train['label'] == 'SIGIR']['label'].count())
print('SIGCSE', train[train['label'] == 'SIGCSE']['label'].count())
print('WWW', train[train['label'] == 'WWW']['label'].count())
print('siggraph', train[train['label'] == 'SIGGRAPH']['label'].count())

rev_train, labels_train = train['abstract'], train['label']
rev_test, labels_test = test['abstract'], test['label']
counter = CountVectorizer()
counter.fit(rev_train)
counts_train = counter.transform(rev_train)  # transform the training data
counts_test = counter.transform(rev_test)  # transform the testing data

counts_train = hstack((counts_train, np.array(train['year'])[:, None]))
counts_test = hstack((counts_test, np.array(test['year'])[:, None]))
counts_train = hstack((counts_train, np.array(train['number_author'])[:, None]))
counts_test = hstack((counts_test, np.array(test['number_author'])[:, None]))
counts_train = hstack((counts_train, np.array(train['new_author_count'])[:, None]))
counts_test = hstack((counts_test, np.array(test['new_author_count'])[:, None]))

# train classifier
model1 = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model2 = LogisticRegression(solver='liblinear')
# model3 = ExtraTreesClassifier(n_estimators=1300, max_depth=None,min_samples_split=2, random_state=0)
model4 = MLPClassifier(hidden_layer_sizes=(12,), max_iter=20, random_state=14, warm_start=True)
model5 = MLPClassifier(hidden_layer_sizes=(9,), max_iter=17, random_state=14, warm_start=True)
model6 = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=13, warm_start=True)



combos = [('Multinomial NB', model1), ('Logistic Regression', model2), ('MLP1', model4), ('MLP2', model5)]
VT = VotingClassifier(combos)
VT.fit(counts_train, labels_train)
# use hard voting to predict (majority voting)
pred = VT.predict(counts_test)
combo_accuracy = accuracy_score(pred, labels_test)
print('\nEnsemble Models: ,Multinomial NB,Logistic Regression,MLP1,MLP2')
print(' Accuracy: {0:0.1%}'.format(combo_accuracy))