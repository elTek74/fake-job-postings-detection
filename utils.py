import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import sys
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from gensim.models import word2vec
from gensim.utils import tokenize
import nltk.data
from nltk.corpus import stopwords
from bs4 import BeautifulSoup 
import re 

##########################################################################################
'''Preprocessing functions '''
##########################################################################################

def get_txt_data(X, num_series):
    """Extracts the columns in X containing text data
        and replaces the missing values with 'nan' string
        
        Parameter
        ----------
        X: dataframe
        num_series: list of numerical feature column names 
        
        Returns
        -------
        Xtxt: cleaned dataframe, only text remains. Same number of rows.
        """
    Xtxt = X.copy()
    Xtxt.drop(num_series,axis = 1, inplace = True)
    Xtxt.replace(np.nan, 'nan', inplace=True) #NaNs don't allow join columns (propagates)
    return Xtxt

def get_num_data(X, text_series):
    """Extracts the columns in X containing numerical data
        
        Parameter
        ----------
        X: dataframe
        text_series: list of text feature column names 

        Returns
        -------
        num_data: cleaned dataframe, only numbers. Same number of rows.
    """
    num_data = dataframe.copy()
    num_data.drop(text_series,axis = 1, inplace = True)
    return num_data

def generate_corpus(X_train, X_test, categories):
    """Merge/combine the columns/features given in categories of each dataset(train and test) 
        into one.
        
        Parameter
        ----------
        X_train: Train set as a dataframe 
        X_test: Test set as a dataframe 
        categories: list of feature names to merge

        Returns
        -------
        Xmerg_train: merged train dataframe
        Xmerg_test: merged test dataframe
    """
    Xmerg_train = X_train[categories].agg('-'.join, axis=1)
    Xmerg_test = X_test[categories].agg('-'.join, axis=1)
    return Xmerg_train, Xmerg_test


def clean_dataframe(X):
    """ Removes all "unmeaning" text from dataframe like html tags, stop words, etc
        If used for postings, the sentences are not readable anymore
    Parameter
    ----------
    X: dataframe
    
    Returns
    ----------
    clean_X: clean dataframe. Dimension is preserved.
    """
    index_train = [i for i in range(X.shape[0])]
    clean_X = pd.DataFrame(columns=['posting'], index=index_train)
    counter = 0
    for description in X:
        if counter%1000 == 0:
            print("Posting %d of %d"%(counter,X.shape[0]))
        clean_X.iloc[counter]= clean_X(description, mode='gathered')
        counter+=1
    return clean_X


def get_most_freq_grams(data, categ, n_most=10):
    """ Extract most common/frequent unigrams in a corpus (data)
        Uses a bag of word representation with an occurence matrix.
        Prints the result.
        
        Parameter
        ----------
        data: text sequences to process. Each posting is a 'document'.
               the set of documents is called a corpus.
        categ: feature column  name to analyze
        n_most: number of most occurent words to find
        
        Comments:
        ----------
        Ignores the stop words (built-in lookup in sklearn)
        Ignores the missing values

    """
    vectorizer = CountVectorizer(stop_words='english',ngram_range=(1, 1))
    corpus = data[categ]
    corpus = corpus.dropna(inplace=False)
    vec = vectorizer.fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    print(str(n_most)+' most frequent words (unigrams) in corpus: \n')
    for i in range(n_most):
        print(words_freq[i])
        
    
def posting_wordlist(posting,mode='w2v'):
    """ Converts a raw text to a sequence of words.
        Removes all ponctuation, html tags, most common words
        
        Parameter
        ----------
        posting: posting text. Can be a full text or a sentence.
        mode: 'w2v' returns a list of "clean" words per posting (text input)
              'gathered' returns the merged list of "clean" words in the posting
        
        Returns
        -------
        words: clean words list
    """
    # 1. Removing html tags
    posting_text = BeautifulSoup(posting,features="html.parser").get_text()
    # 2. Removing non-letter.
    posting_text = re.sub("[^a-zA-Z]"," ",posting_text)
    # 3. Converting to lower case and splitting
    words = posting_text.lower().split()
    stops = set(stopwords.words("english"))  
    if mode=='w2v': 
        print('stops')
        words = [w for w in words if not w in stops]
    if mode=='gathered':
            clean_text = ' '
            for w in words :
                if not w in stops: clean_text =  clean_text+' '+w
    return(words)


def posting_sentences(posting, tokenizer):
    """ splits a posting into sentences.
        Cleans the sentences by calling 'posting_wordlist' function.
        
        Parameter
        ----------
        posting: posting text
        tokenizer: nltk tokenizer to split sentences
        remove_stopwords: Boolean to enable/disable the removal of stop words
        
        Returns
        -------
        sentences: clean list of sentences in posting
    """
    # 1. Using nltk tokenizer
    raw_sentences = tokenizer.tokenize(posting.strip())
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(posting_wordlist(raw_sentence))                                         
    # This returns the list of lists
    return sentences

def simple_split(X, y, test_size):
    """ splits the dataset X in train and test set.
        Corresponding labels splitted as well.
        
        Parameter
        ----------
        X: dataset as pandas dataframe
        y: class labels as list of pandas series
        test_size: proportion of original dataset assigned to test data
        
        Returns
        -------
        X_train: train set
        y_train: train labels
        X_test: test set
        y_test: test labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("fake class rate in train: %.2f / test: %.2f" % (y_train.sum()/y_train.shape[0],
                                                            y_test.sum()/y_test.shape[0]))
    print("Proportion of total fakes in train set: %.1f \n"  % (y_train.sum()/(y_train.sum()+y_test.sum())))
    return X_train, X_test, y_train, y_test

def process_nans(fake_data, real_data, dataframe, series_names):
    """ Computes the rate of missing values across feature columns
        Computes rate of missing values across postings in each class
        
        Parameter
        ----------
        fake_data: Dataframe of postings labeled as fake (1)
        real_data:  Dataframe postings labeled as real (0)
        dataframe: original dataset (all postings)
        series_names: list of features col names
        
        Returns
        -------
        real_counts: array-like. discrete distribution of missing values rate in real postings
        fake_counts:  array-like. discrete distribution of missing values rate in fake postings
        nan_prop_fake: array-like. rate of fakes in missing values for some feature columns
        feature_with_nans: column names that contain Nans for both classes. Same dimension as nan_prop_fake.
    """
    
    nan_prop_fake = []
    nan_cumul_fake = np.zeros(fake_data.shape[0])
    nan_cumul_real = np.zeros(real_data.shape[0])
    feature_with_nans = []
    for idx in range(len(series_names)):
        #extract class quantity in missing values for each category
        name = series_names[idx]
        nan_list = dataframe.fraudulent[np.where(dataframe[name].isnull()==True)[0]]
        unique, counts = np.unique(nan_list,return_counts=True)

        #extract missing value quantity for each class(real/fake) 
        nan_cumul_fake[np.where(fake_data[name].isnull()==True)[0]] += 1
        nan_cumul_real[np.where(real_data[name].isnull()==True)[0]] += 1

        #account for categories with missing values for both classes
        if counts.shape[0]==2:
            nan_prop_fake.append(counts[1]/counts[0]) # %of fakes in missing values
            feature_with_nans.append(name)

    real_uni, real_counts = np.unique(nan_cumul_real, return_counts=True)
    fake_uni, fake_counts = np.unique(nan_cumul_fake, return_counts=True)
    real_counts = real_counts/real_data.shape[0]
    fake_counts = fake_counts/fake_data.shape[0]
    return real_counts, fake_counts, nan_prop_fake, feature_with_nans
        


##########################################################################################
'''Bag of Words functions '''
##########################################################################################

# Function to vectorize the n-grams in a text and retrun frequency matrix using tf-idf 
# Performs the transformation for both train and test set.
# sklearn methods
def vectorize(Xraw_train, Xraw_test, params):
    if len(params)<3: max_df = 0.5
    else: max_df = params[2]
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=max_df,
                                 stop_words='english', ngram_range = params[0])
    Xvec_train = vectorizer.fit_transform(Xraw_train)
    Xvec_test  = vectorizer.transform(Xraw_test)
    feature_names = vectorizer.get_feature_names()
    if params[1] == 1:
        print('_' * 80)
        print("Vectorizing: ")
        print("Number of features: %d" % (Xvec_train.shape[1]))
    
    return Xvec_train, Xvec_test, feature_names

# Function to vectorize the n-grams in a text and retrun frequency matrix using tf-idf 
# Performs the transformation for both train and test set.
# sklearn methods
def vectorize_ch2(Xraw_train, Xraw_test, y_train, feature_names, n_best):
    
    print("Extracting %d best features by a chi-squared test" % n_best)
    ch2 = SelectKBest(chi2, k=n_best)
    Xch2_train = ch2.fit_transform(Xraw_train, y_train)
    Xch2_test = ch2.transform(Xraw_test)

    feature_names = [feature_names[i] for i
                      in ch2.get_support(indices=True)]
    print()
    print(feature_names)
    return Xch2_train, Xch2_test

def classify_data(clf, X_train, X_test, y_train, y_test, params, clf_name='SGD'):    
    """ Computes the predictions for a given classifier
        Vectorizes the dataset containing text sequences
        Fits the train data, then predicts the test data class labels.
        Prints the results and specs.
        
        Parameter
        ----------
        clf: classif
        X_train: Train data
        X_test: Test data
        params contains a list of parameters to feed to vectorizer or run parts of code
        params[0] : number of n-grams (tuple)
        params[1] : boolean to enable metric report printing
        params[2] : max_df. i.e. frequency threshold for the vectorizer
        
        
        Returns
        -------
        scores: scoring metrics in an array,i.e. Bal. accu, F1 score, accuracy
    """
    #convert text into numerical data: 1 vector per document/posting
    Xvec_train, Xvec_test, feature_names = vectorize(X_train, X_test, params) 
    
    #train the classifier
    if params[1] == 1:
        print('_' * 80)
        print("Training: ")
        print(clf)
    t0 = time()
    clf.fit(Xvec_train, y_train)
    train_time = time() - t0
    if params[1] == 1: print("train time: %0.3fs" % train_time)

    #predict the posting labels in the test set
    t0 = time()
    pred = clf.predict(Xvec_test)
    test_time = time() - t0
    if params[1] == 1: print("test time:  %0.3fs" % test_time)

    scores = compute_clf_scores([pred],[clf_name],y_test)

    feature_names = np.asarray(feature_names)
    target_names = ['real postings', 'fake postings']

    if params[1] == 1:
        print("top 10 determining keywords/features ")
        # extract the 10 highest weighted features
        top10 = np.argsort(clf.coef_[0])[-10:]
        print("%s" % ( " ;".join(feature_names[top10])))
        print()
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
    return scores


# Function to compute different classification scores for predictions
# Can deal simultaneously with results from various sources
# Needs the predictions and ground truth class labels
def compute_clf_scores(preds, clf_names, true_y):
    #preds: list of lists (one for each classif)
    scores = []
    for idx in range(len(preds)):
        accu = metrics.accuracy_score(true_y, preds[idx])
        bld_accu = metrics.balanced_accuracy_score(true_y, preds[idx])
        f1 = metrics.f1_score(true_y, preds[idx])
        #auc = metrics.roc_auc_score() #write function to extract probas
        scores.append([accu, bld_accu, f1])
        print(clf_names[idx],':')
        print('accu: %.3f   bld_accu: %.3f   f1: %.3f \n' % (accu,bld_accu,f1))
    scores = np.asarray(scores)
    return scores



##########################################################################################
'''Word2Vec functions '''
##########################################################################################

# Function to average a posting from its words/context embeddings  
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    if nwords:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

# Function to average all the postings and return them as a train/test dataset matrix
def getAvgFeatureVecs(postings, model, num_features):
    counter = 0
    postingFeatureVecs = np.zeros((len(postings),num_features),dtype="float32")
    for posting in postings:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Posting %d of %d"%(counter,len(postings)))
            
        postingFeatureVecs[counter] = featureVecMethod(posting, model, num_features)
        counter += 1
        
    return postingFeatureVecs


##########################################################################################
'''Plotting functions'''
##########################################################################################

#function to display the bar heights (scores) on top
def autolabel(scores, ax1, dx, color):
    for i, v in enumerate(scores):
        ax1.text(i - dx, v + 0.02, str(round(v,2)),color=color)

#function to plot the different metrics as a bar plot for each classifier
def plot_scores(ax,scores,clf_names, col_name):
    width = 0.25
    title = 'Results for w2v embedding of '+col_name+' feature col' 
    rect1 = ax.bar(np.arange(len(clf_names))-width, scores[:,0], width,label='Accuracy')
    rect2 = ax.bar(np.arange(len(clf_names)), scores[:,1], width, label='Balanced accuracy')
    rect3 = ax.bar(np.arange(len(clf_names))+width, scores[:,2], width, label='F1')
    ax.set_title(title,fontsize=14)
    ax.set_ylabel('Score',fontsize=12)
    ax.set_xticklabels(clf_names, fontsize=12)
    ax.set_xticks(np.arange(len(clf_names)))
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    autolabel(scores[:,0], ax, .35, 'blue')
    autolabel(scores[:,1], ax, .1, 'orange')
    autolabel(scores[:,2], ax, -.15, 'green')

