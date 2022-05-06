from xml.dom import NotFoundErr
import tweepy
import csv
import pandas as pd
from PIL import Image
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
import tweepy
import csv
import re
import spacy
import nltk
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import contractions
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

st.write("""
# Depression Detection
Detect if some twitter user has depression using machine learning and python""")
image = Image.open('image.jpg')
st.image(image, caption='ML', use_column_width=True)
tweets=pd.read_csv('output.csv')
vectorizer = TfidfVectorizer(stop_words='english')
x = []
y = []
for row in tweets['SentimentText']:   
    x.append(row)
for rows in tweets['Sentiment']:
    y.append(rows)


def get_all_tweets(screen_name):  
    consumer_key = "FHSCcqycpgpHoFZ1OqZKtNLKE"
    consumer_secret = "YNdtiBJXuyMuTP0QyfAoEGbFvizyoIjCPZeUgDAwLqB2kJnOhc"
    access_key = "1501899564038881287-Mrd6cNZqIXYQGOxE0iRDPrW6oyhSbp"
    access_secret = "XoLZwCuVkogoxLZnL1Llxvc8GjJDIoykuxHX12YjZq20o"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    alltweets = []
    noRT = []
    new_tweets = api.user_timeline(screen_name = screen_name, tweet_mode = 'extended', count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1
    while len(new_tweets) >0:
        print("getting tweets before {}".format(oldest))
        new_tweets = api.user_timeline(screen_name = screen_name,tweet_mode = 'extended', count=200,max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1
        print("...{} tweets downloaded so far".format(len(alltweets)))
    for tweet in alltweets:
        if ('RT' in tweet.full_text or '@' in tweet.full_text):
            continue
        else:
            noRT.append([tweet.id_str, tweet.created_at, tweet.full_text])
    with open('{}_tweets.csv'.format(screen_name), 'w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(noRT)
        user=pd.read_csv('{}_tweets.csv'.format(screen_name))
        st.subheader('Data Information:')
        st.dataframe(user)
    pass


def word_is_negated(word):
    for child in word.children:
        if child.dep_ == 'neg':
            return True
    if word.pos_ in {'VERB'}:
        for ancestor in word.ancestors:
            if ancestor.pos_ in {'VERB'}:
                for child2 in ancestor.children:
                    if child2.dep_ == 'neg':
                        return True
    return False


def find_negated_wordSentIdxs_in_sent(sent, idxs_of_interest=None):
    negated_word_idxs = set()
    for word_sent_idx, word in enumerate(sent):
        if idxs_of_interest:
            if word_sent_idx not in idxs_of_interest:
                continue
        if word_is_negated(word):
            negated_word_idxs.add(word_sent_idx)
    return negated_word_idxs
    

def metric(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels.iloc[i] == 0 and predictions[i] == 0)
        true_neg += int(labels.iloc[i] == 1 and predictions[i] == 1)
        false_pos += int(labels.iloc[i] == 1 and predictions[i] == 0)
        false_neg += int(labels.iloc[i] == 0 and predictions[i] == 1)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    st.write("Precision: ", precision)
    st.write("Recall: ", recall)
    st.write("F-score: ", Fscore)
    st.write("Accuracy: ", accuracy)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    fig=plt.figure(figsize = (10, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)


def learns():    
    tweet = pd.read_csv('sentiment_tweets31.csv')
    tweet.drop(['Unnamed: 0'], axis = 1, inplace = True)
    z = []
    for col in tweet['message']:
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        col = RE_EMOJI.sub(r'', col)
        col = ''.join([c for c in col if ord(c) < 128])
        col=contractions.fix(col)
        z.append(col)
    start_timedt = time.time()
    train_featurestree = vectorizer.fit_transform(x)
    actual1 = y  
    #test_features1 = vectorizer.transform(z)
    test_features1 = vectorizer.transform(x)
    dtree = tree.DecisionTreeClassifier()
    dtree = dtree.fit(train_featurestree, [int(r) for r in y])
    prediction1 = dtree.predict(test_features1)
    st.subheader('Model Metrics:')
    #metric(tweet['label'], prediction1)
    metric(tweets['Sentiment'], prediction1)
    #st.write(metrics.confusion_matrix(tweet['label'], prediction1, labels=[0, 1]))
    #st.write(metrics.confusion_matrix(tweets['Sentiment'], prediction1, labels=[0, 1]))
    dt_matrix = confusion_matrix(actual1, prediction1)
    
    plot_confusion_matrix(dt_matrix, classes=[0,1], title='Confusion matrix')
    
   


def pred(inputtweet):
    tweet = pd.read_csv('{}_tweets.csv'.format(inputtweet))
    tweet["label"] = ""
    tweet.drop(['created_at'], axis = 1, inplace = True)
    tweet.drop(['id'], axis = 1, inplace = True)
    train_featurestree = vectorizer.fit_transform(x)
    dtree = tree.DecisionTreeClassifier()  
    dtree = dtree.fit(train_featurestree, [int(r) for r in y]) 
    i=0
    nlp = spacy.load('en_core_web_lg')
    for row in tweet['text']:
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        row = RE_EMOJI.sub(r'', row)
        row = ''.join([c for c in row if ord(c) < 128])
        row=contractions.fix(row)
        j=find_negated_wordSentIdxs_in_sent(nlp(row))      
        inputdtree= vectorizer.transform([row])
        predictt = dtree.predict(inputdtree)

        
        if (j != set()):
            tweet.loc[i,'label'] = int(not(predictt))
        else:
            tweet.loc[i,'label'] = int(predictt)
        i=i+1
    nodep=(tweet.label == 1).sum()
    dep=(tweet.label == 0).sum()
    sum=nodep+dep
    percentage=dep/sum
    st.subheader('Depression Level:')
    if (percentage>=0 and percentage<=0.25):
        st.write("Considered Normal")
    elif (percentage>=0.25 and percentage<=0.40):
        st.write("Mild Depression")
    elif (percentage>=0.40 and percentage<=0.55):
        st.write("Borderline Depression")
        st.video("C:/Users/Ameen Harafan/Desktop/Detecting-Depression-in-Tweets-master/Detecting-Depression-in-Tweets-master/DealingDepression.mp4")
    elif (percentage>=0.55 and percentage<=0.70):
        st.write("Moderate Depression")
        st.video("C:/Users/Ameen Harafan/Desktop/Detecting-Depression-in-Tweets-master/Detecting-Depression-in-Tweets-master/DealingDepression.mp4")
    elif (percentage>=0.70 and percentage<=0.85):
        st.write("Severe Depression")
        st.video("C:/Users/Ameen Harafan/Desktop/Detecting-Depression-in-Tweets-master/Detecting-Depression-in-Tweets-master/DealingDepression.mp4")
    else:
        st.write("Extreme Depression")
        st.video("C:/Users/Ameen Harafan/Desktop/Detecting-Depression-in-Tweets-master/Detecting-Depression-in-Tweets-master/DealingDepression.mp4")
    print (dep)
    print (percentage)

    st.subheader('Training Data Information:')
    st.write("Available [here](https://drive.google.com/file/d/1X6oA1Jvs4c-3tYJZwJJVmM0pzKabQKly/view?usp=sharing)")
 
    st.subheader('WordCloud Analysis of Training Data:')
    st.write("Available [here](https://drive.google.com/drive/folders/1VMG0MgJ-nZrEYRTLBxbKVwtwNVUKXq0N?usp=sharing)")
try:
    with st.form(key='my_form'):
        inputtweet = st.text_input(label='Input your twitter handle without @:')
        col1, col2, col3 , col4, col5 = st.columns(5)

        with col1:
            pass
        with col2:
            pass
        with col3:
            pass
        with col4:
            pass
        with col5 :      
            submit_button = st.form_submit_button(label='Check')
        
        get_all_tweets(inputtweet)

    learns()
    pred(inputtweet)
except:
    time.sleep(10)
    st.info("Waiting for your correct input...")
    
