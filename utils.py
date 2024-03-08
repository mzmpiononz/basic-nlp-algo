from __future__ import unicode_literals
import csv
import pickle
import numpy as np
from time import time
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.feature_selection import SelectFromModel

# import differents models
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import warnings # Suppress the FutureWarning from SciPy
warnings.simplefilter(action='ignore', category=FutureWarning)

# =================================DATASET ===========================================================
def read_csv(filename, intent_dict):
    X, y = [], []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            X.append(row[0])
            y.append(intent_dict[row[1]])
    return X, y

# SEMANTIC HASHING=====================================================================================
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens

def semhash_corpus(corpus):
    new_corpus = []
    for sentence in corpus:
        tokens = semhash_tokenizer(sentence)
        new_corpus.append(" ".join(map(str,tokens)))
    return new_corpus

def inference_preprocess(phrase):
    # set the desired dimensionality of HD vectors
    N = 1000 
    # n-gram size
    n_size=3 
    #fix the alphabet. Note, we assume that capital letters are not in use 
    aphabet = 'abcdefghijklmnopqrstuvwxyz#'
    # for reproducibility
    np.random.seed(1)
    # generates bipolar {-1, +1}^N HD vectors; one random HD vector per symbol in the alphabet
    HD_aphabet = 2 * (np.random.randn(len(aphabet), N) < 0) - 1 
    
    phrase = semhash_corpus([phrase])
    # phrase_raw = data_for_training(phrase)
    for i in range(len(phrase)):
        phrase[i] = ngram_encode(phrase[i], HD_aphabet, aphabet, n_size)
    return phrase

# VECTORIZER===========================================================================================
def get_vectorizer(corpus, preprocessor=None, tokenizer=None):
    vectorizer = CountVectorizer(ngram_range=(2,4),analyzer='char')
    vectorizer.fit(corpus)
    return vectorizer, vectorizer.get_feature_names_out()


# BENCHMARK CLASSIFIERS ===============================================================================
def benchmark(clf, X_train, y_train, X_test, y_test, intent_names,
              print_report=True, feature_names=None, print_top10=False,
              print_cm=True):
    print('_' * 80)
    # print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    return clf, score

# DATA FOR TRAINING ==========================================================================================
def data_for_training(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    vectorizer, feature_names = get_vectorizer(X_train_raw, preprocessor=None, tokenizer=None)
    
    X_train_no_HD = vectorizer.transform(X_train_raw).toarray()
    X_test_no_HD = vectorizer.transform(X_test_raw).toarray()
            
    return X_train_no_HD, y_train_raw, X_test_no_HD, y_test_raw, feature_names


# NGRAM ENCODE ================================================================================================
# method for mapping n-gram statistics of a word to an N-dimensional HD vector
def ngram_encode(str_test, HD_aphabet, aphabet, n_size): 
    # will store n-gram statistics mapped to HD vector
    HD_ngram = np.zeros(HD_aphabet.shape[1])
    # include extra symbols to the string
    full_str = '#' + str_test + '#' 

    # loops through all n-grams
    for il, l in enumerate(full_str[:-(n_size-1)]):
        # picks HD vector for the first symbol in the current n-gram
        hdgram = HD_aphabet[aphabet.find(full_str[il]), :]
        #loops through the rest of symbols in the current n-gram
        for ng in range(1, n_size): 
            # two operations simultaneously; binding via elementvise multiplication; rotation via cyclic shift
            hdgram = hdgram * np.roll(HD_aphabet[aphabet.find(full_str[il+ng]), :], ng) 
        # increments HD vector of n-gram statistics with the HD vector for the currently observed n-gram   
        HD_ngram += hdgram
    # normalizes HD-vector so that its norm equals sqrt(N)       
    HD_ngram_norm = np.sqrt(HD_aphabet.shape[1]) * (HD_ngram/ np.linalg.norm(HD_ngram) )  
    # output normalized HD mapping
    return HD_ngram_norm 


# TRAINING ALL THE MODELS
def semhash_training(filename_train, filename_test, intent_names, model_path):
    
    # READ THE DATASET ==================================================================================================
    intent_dict = {k: v for k, v in zip(intent_names, [i for i in range(len(intent_names))])}
    X_train_raw, y_train_raw = read_csv(filename_train, intent_dict)
    X_test_raw, y_test_raw = read_csv(filename_test, intent_dict)

    # SEMHASHING THE CORPUS =============================================================================================
    X_train_raw = semhash_corpus(X_train_raw)
    X_test_raw = semhash_corpus(X_test_raw)
    X_train, y_train, X_test, y_test, feature_names = data_for_training(X_train_raw, X_test_raw, y_train_raw, y_test_raw)

    # N-GRAM PROCESS ====================================================================================================
    # set the desired dimensionality of HD vectors
    N = 1000 
    # n-gram size
    n_size=3 
    #fix the alphabet. Note, we assume that capital letters are not in use 
    aphabet = 'abcdefghijklmnopqrstuvwxyz#'
    # for reproducibility
    np.random.seed(1) 
    # generates bipolar {-1, +1}^N HD vectors; one random HD vector per symbol in the alphabet
    HD_aphabet = 2 * (np.random.randn(len(aphabet), N) < 0) - 1 
    
    # str='High like a basketball jump' # example string to represent using n-grams
    # HD_ngram is a projection of n-gram statistics for str to N-dimensional space. It can be used to learn the word embedding
    
    for i in range(len(X_train_raw)):
         X_train_raw[i] = ngram_encode(X_train_raw[i], HD_aphabet, aphabet, n_size) 
    for i in range(len(X_test_raw)):
        X_test_raw[i] = ngram_encode(X_test_raw[i], HD_aphabet, aphabet, n_size) 
    X_train, y_train, X_test, y_test = X_train_raw, y_train_raw, X_test_raw, y_test_raw
    
    # TRAINING ALL THE MODELS ================================================================================================
    for i_s, split in enumerate(range(1)):
        print("Evaluating Split {}".format(i_s))
        results = []
        parameters_mlp={'hidden_layer_sizes':[(100,50), (300, 100),(300,200,100)]}
        parameters_RF={ "n_estimators" : [50,60,70],
               "min_samples_leaf" : [1, 11]}
        k_range = list(range(3,7))
        parameters_knn = {'n_neighbors':k_range}
        knn, model_idx = KNeighborsClassifier(n_neighbors=5), 1
        for clf, name in [  
                (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                (GridSearchCV(knn,parameters_knn, cv=5),"gridsearchknn"),
                (GridSearchCV(MLPClassifier(activation='tanh'),parameters_mlp, cv=5),"gridsearchmlp"),
                (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
                (GridSearchCV(RandomForestClassifier(n_estimators=10),parameters_RF, cv=5),"gridsearchRF")
        ]:
               
            print('=' * 80)
            print("MODEL "+str(model_idx)+" >>> "+name)
            results.append(benchmark(clf, X_train, y_train, X_test, y_test, intent_names,
                                     feature_names=feature_names))
            model_idx = model_idx +1
    
        #parameters_Linearsvc = [{'C': [1, 10], 'gamma': [0.1,1.0]}]
        model_idx = 6
        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("MODEL "+str(model_idx)+" & MODEL "+str(model_idx + 1)+" >>> %s penalty" % penalty.upper())
            results.append(benchmark(LinearSVC(penalty=penalty, dual=False,tol=1e-3),
                                     X_train, y_train, X_test, y_test, intent_names,
                                     feature_names=feature_names))
    
            # Train SGD model
            results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                                   penalty=penalty),
                                     X_train, y_train, X_test, y_test, intent_names,
                                     feature_names=feature_names))
            model_idx = model_idx + 2 
    
        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("MODEL 10 >>> Elastic-Net penalty")
        results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                               penalty="elasticnet"),
                                 X_train, y_train, X_test, y_test, intent_names,
                                 feature_names=feature_names))
    
        # Train NearestCentroid without threshold
        print('=' * 80)
        print("MODEL 11 >>> NearestCentroid (aka Rocchio classifier)")
        results.append(benchmark(NearestCentroid(),
                                 X_train, y_train, X_test, y_test, intent_names,
                                 feature_names=feature_names))
    
        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("MODEL 12 & MODEL 13 >>> Naive Bayes")
        scaler = MinMaxScaler()
        results.append(benchmark(MultinomialNB(alpha=.01),
                                 scaler.fit_transform(X_train), y_train, scaler.fit_transform(X_test), y_test, intent_names,
                                 feature_names=feature_names))
        results.append(benchmark(BernoulliNB(alpha=.01),
                                 X_train, y_train, X_test, y_test, intent_names,
                                 feature_names=feature_names))
    
        print('=' * 80)
        print("MODEL 14 >>> LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        
        # uncommenting more parameters will give better exploring power but will
        # increase processing time in a combinatorial way
        results.append(benchmark(Pipeline([
                                      ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                                                      tol=1e-3))),
                                      ('classification', LinearSVC(penalty="l2"))]),
                                 X_train, y_train, X_test, y_test, intent_names,
                                 feature_names=feature_names))
        
       #KMeans clustering algorithm 
        print('=' * 80)
        print("MODEL 15 >>> KMeans")
        results.append(benchmark(KMeans(n_clusters=2, init='k-means++', max_iter=300,
                    verbose=0, random_state=0, tol=1e-4),
                                 X_train, y_train, X_test, y_test, intent_names,
                                 feature_names=feature_names))
        
        print('=' * 80)
        print("MODEL 16 >>> LogisticRegression")
        results.append(benchmark(LogisticRegression(C=1.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
                                 X_train, y_train, X_test, y_test, intent_names,
                                 feature_names=feature_names))
    
    # FIND & AND SAVE THE BEST MODEL ================================================================================================ 
    print('=' * 80)
    acc_list = [results[i][1] for i in range(len(results))]
    best_model_idx = [index for index, element in enumerate(acc_list) if element == max(acc_list)]
    if len(best_model_idx) == 1:
        best_model = results[best_model_idx[0]]
        pickle.dump(best_model[0], open(model_path, 'wb'))
        print("THE BEST MODEL IS:\nMODEL {} : {}\nAccuracy = {:.3f}".format(best_model_idx[0]+1, str(best_model[0]), best_model[1]))
    else:
        print("THE BEST MODEL ARE:")
        for i in range(len(best_model_idx)):
            best_model = results[best_model_idx[i]]
            pickle.dump(best_model[0], open(model_path[:-4]+'_model_{}.pkl'.format(best_model_idx[i]+1), 'wb'))
            print("\nMODEL {} : {}\nAccuracy = {:.3f}".format(best_model_idx[i]+1, str(best_model[0]), best_model[1]))