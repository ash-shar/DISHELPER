import sys
import os
import random
import re
import codecs
import numpy as np
import string
from nltk.corpus import stopwords
from textblob import *
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import *
from sklearn.svm import *
from sklearn.linear_model import *


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

import gzip
#from happyfuntokenizing import *

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)

PRONOUN_PATH = 'global_view/english_pronoun.txt'
WHWORD_PATH = 'global_view/english_whwords.txt'
SLANG_PATH = 'global_view/english_slang.txt'
INTENSIFIER_PATH = 'global_view/english_intensifier.txt'
SUBJECTIVE_PATH = 'global_view/subjclueslen1-HLTEMNLP05.tff'
EVENT_PATH = 'global_view/english_nonsituational_phrase.txt'
MODAL_VERB_PATH = 'global_view/english_modal_verb.txt'

'''TRAIN1 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hydb_True_Split.txt'
TRAIN2 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/utkd_True_Split.txt'
TRAIN3 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/sandy_hook_True_Split.txt'
TRAIN4 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hagupit_True_Split.txt' '''


'''TRAIN1 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hydb_RAW_CLASS.txt'
TRAIN2 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/utkd_RAW_CLASS.txt'
TRAIN3 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/sandy_hook_RAW_CLASS.txt'
TRAIN4 = '/home/krudra/summarization/codetest/TWEB_REVISION/Training_Data/hagupit_RAW_CLASS.txt' '''

TRAIN1 = 'Training_Data/hydb_fragment_train.txt'
TRAIN2 = 'Training_Data/utkd_fragment_train.txt'
TRAIN3 = 'Training_Data/sandy_hook_fragment_train.txt'
TRAIN4 = 'Training_Data/hagupit_fragment_train.txt'



cachedstopwords = stopwords.words("english")
Tagger_Path = 'ark-tweet-nlp-0.3.2/'
#Tagger_Path = '/home/krudra/twitter_code/aaai/characterize_user/wordcloud/ark-tweet-nlp-0.3.2/'
lmtzr = WordNetLemmatizer()
PRONOUN = {}
SUBJECTIVE = {}

def READ_FILES():

    fp = open(PRONOUN_PATH,'r')
    for l in fp:
        PRONOUN[l.strip(' \t\n\r').lower()] = 1
        #temp.add(l.strip(' \t\n\r').lower())
    fp.close()

    fp = open(SUBJECTIVE_PATH,'r')
    for l in fp:
        wl = l.split()
        x = wl[0].split('=')[1].strip(' \t\n\r')
        if x=='strongsubj':
            y = wl[2].split('=')[1].strip(' \t\n\r')
            SUBJECTIVE[y.lower()] = 1
            #temp.append(y.lower())

def pronoun(sen):
    for x in sen:
        if PRONOUN.__contains__(x)==True:
            return 1
    return 0

def subjectivity(sen):

    c = 0
    for x in sen:
        if SUBJECTIVE.__contains__(x)==True:
            c+=1
    tot = len(sen) + 4.0 - 4.0
    num = c + 4.0 - 4.0
    s = round(num/tot,4)
    #print(sen,s,num,tot)
    #sys.exit(0)
    #return s
    return c


def predict(curr_classifier1, curr_classifier2, curr_classifier3, curr_classifier4, classifier_name):

    #tag = ['~',',']
    TAGREJECT = ['U','#','@','~','E',',']
    READ_FILES()

    ''' Hyderabad Blast '''
    HB = []
    hydb_word_dic = {}
    hydb_pos_dic = {}
    fp = open(TRAIN1,'r')
    fo = open('temp.txt','w')
    hydb_class_label = []
    for l in fp:
        wl = l.split('\t')
        fo.write(wl[0].strip(' \t\n\r'))
        fo.write('\n')
        hydb_class_label.append(int(wl[1].strip('\t\n\r')))
    fp.close()
    fo.close()

    command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
    os.system(command)

    fp = open('tag.txt','r')
    temp = []
    temp_pro = []
    temp_pos = []
    tweet_count = 0
    for l in fp:
        wl = l.split()
        if len(wl)>1:
            t = (wl[0].strip(' \t\n\r').lower(),wl[1].strip(' \t\n\r'))
            temp_pos.append(t[1])
            temp_pro.append(t[0])
            if t[1] not in TAGREJECT and t[0] not in cachedstopwords:
                temp.append(t[0])
            if hydb_pos_dic.__contains__(t[1])==False:
                hydb_pos_dic[t[1]] = 1
            else:
                count = hydb_pos_dic[t[1]]
                count+=1
                hydb_pos_dic[t[1]] = count
        else:
            bigram = []
            if len(temp)>1:
                for i in range(0,len(temp)-1,1):
                    s = temp[i] + ' ' +  temp[i+1]
                    bigram.append(s)
            W = temp + bigram
            for x in W:
                if hydb_word_dic.__contains__(x)==False:
                    hydb_word_dic[x] = 1
                else:
                    v = hydb_word_dic[x]
                    v+=1
                    hydb_word_dic[x] = v
            S = subjectivity(temp)
            P = pronoun(temp_pro)
            t = (hydb_class_label[tweet_count],temp,temp_pos,S,P,W)
            HB.append(t)
            temp = []
            temp_pro = []
            temp_pos = []
            tweet_count+=1
    fp.close()

    print('Complete Hyderabad')

    ''' Uttarakhand Flood '''

    UF = []
    utkd_word_dic = {}
    utkd_pos_dic = {}
    fp = open(TRAIN2,'r')
    fo = open('temp.txt','w')
    utkd_class_label = []
    for l in fp:
        wl = l.split('\t')
        fo.write(wl[0].strip(' \t\n\r'))
        fo.write('\n')
        utkd_class_label.append(int(wl[1].strip('\t\n\r')))
    fp.close()
    fo.close()

    command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
    os.system(command)

    fp = open('tag.txt','r')
    temp = []
    temp_pro = []
    temp_pos = []
    tweet_count = 0
    for l in fp:
        wl = l.split()
        if len(wl)>1:
            t = (wl[0].strip(' \t\n\r').lower(),wl[1].strip(' \t\n\r'))
            temp_pos.append(t[1])
            temp_pro.append(t[0])
            if t[1] not in TAGREJECT and t[0] not in cachedstopwords:
                temp.append(t[0])
            if utkd_pos_dic.__contains__(t[1])==False:
                utkd_pos_dic[t[1]] = 1
            else:
                count = utkd_pos_dic[t[1]]
                count+=1
                utkd_pos_dic[t[1]] = count
        else:
            bigram = []
            if len(temp)>1:
                for i in range(0,len(temp)-1,1):
                    s = temp[i] + ' ' +  temp[i+1]
                    bigram.append(s)
            W = temp + bigram
            for x in W:
                if utkd_word_dic.__contains__(x)==False:
                    utkd_word_dic[x] = 1
                else:
                    v = utkd_word_dic[x]
                    v+=1
                    utkd_word_dic[x] = v

            S = subjectivity(temp)
            P = pronoun(temp_pro)
            t = (utkd_class_label[tweet_count],temp,temp_pos,S,P,W)
            UF.append(t)
            temp = []
            temp_pro = []
            temp_pos = []
            tweet_count+=1
    fp.close()

    print('Complete Uttarakhand')


    ''' Sandy Hook Shootout '''

    SH = []
    sandy_word_dic = {}
    sandy_pos_dic = {}
    fp = open(TRAIN3,'r')
    fo = open('temp.txt','w')
    sandy_class_label = []
    for l in fp:
        wl = l.split('\t')
        fo.write(wl[0].strip(' \t\n\r'))
        fo.write('\n')
        sandy_class_label.append(int(wl[1].strip('\t\n\r')))
    fp.close()
    fo.close()

    command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
    os.system(command)

    fp = open('tag.txt','r')
    temp = []
    temp_pro = []
    temp_pos = []
    tweet_count = 0
    for l in fp:
        wl = l.split()
        if len(wl)>1:
            t = (wl[0].strip(' \t\n\r').lower(),wl[1].strip(' \t\n\r'))
            temp_pos.append(t[1])
            temp_pro.append(t[0])
            if t[1] not in TAGREJECT and t[0] not in cachedstopwords:
                temp.append(t[0])
            if sandy_pos_dic.__contains__(t[1])==False:
                sandy_pos_dic[t[1]] = 1
            else:
                count = sandy_pos_dic[t[1]]
                count+=1
                utkd_pos_dic[t[1]] = count
        else:
            bigram = []
            if len(temp)>1:
                for i in range(0,len(temp)-1,1):
                    s = temp[i] + ' ' +  temp[i+1]
                    bigram.append(s)
            W = temp + bigram
            for x in W:
                if sandy_word_dic.__contains__(x)==False:
                    sandy_word_dic[x] = 1
                else:
                    v = sandy_word_dic[x]
                    v+=1
                    sandy_word_dic[x] = v

            S = subjectivity(temp)
            P = pronoun(temp_pro)
            t = (sandy_class_label[tweet_count],temp,temp_pos,S,P,W)
            SH.append(t)
            temp = []
            temp_pro = []
            temp_pos = []
            tweet_count+=1
    fp.close()

    print('Complete Sandy Hook Shootout')


    ''' Typhoon Hagupit '''

    HG = []
    hagupit_word_dic = {}
    hagupit_pos_dic = {}
    fp = open(TRAIN4,'r')
    fo = open('temp.txt','w')
    hagupit_class_label = []
    for l in fp:
        wl = l.split('\t')
        fo.write(wl[0].strip(' \t\n\r'))
        fo.write('\n')
        hagupit_class_label.append(int(wl[1].strip('\t\n\r')))
    fp.close()
    fo.close()

    command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
    os.system(command)

    fp = open('tag.txt','r')
    temp = []
    temp_pro = []
    temp_pos = []
    tweet_count = 0
    for l in fp:
        wl = l.split()
        if len(wl)>1:
            t = (wl[0].strip(' \t\n\r').lower(),wl[1].strip(' \t\n\r'))
            temp_pos.append(t[1])
            temp_pro.append(t[0])
            if t[1] not in TAGREJECT and t[0] not in cachedstopwords:
                temp.append(t[0])
            if hagupit_pos_dic.__contains__(t[1])==False:
                hagupit_pos_dic[t[1]] = 1
            else:
                count = hagupit_pos_dic[t[1]]
                count+=1
                hagupit_pos_dic[t[1]] = count
        else:
            bigram = []
            if len(temp)>1:
                for i in range(0,len(temp)-1,1):
                    s = temp[i] + ' ' +  temp[i+1]
                    bigram.append(s)
            W = temp + bigram
            for x in W:
                if hagupit_word_dic.__contains__(x)==False:
                    hagupit_word_dic[x] = 1
                else:
                    v = hagupit_word_dic[x]
                    v+=1
                    hagupit_word_dic[x] = v

            S = subjectivity(temp)
            P = pronoun(temp_pro)
            t = (hagupit_class_label[tweet_count],temp,temp_pos,S,P,W)
            HG.append(t)
            temp = []
            temp_pro = []
            temp_pos = []
            tweet_count+=1
    fp.close()

    print('Complete Hagupit')

    ''' First Cross Validation '''

    keys = hydb_word_dic.keys()
    pos_keys = hydb_pos_dic.keys()
    F = []
    L = []
    for x in HB:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hydb_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])

        #for i in range(0,len(x[5]),1):
        #       C.append(x[5][i])
        F.append(C)
        L.append(int(x[0]))

    #print(len(hydb_word_dic.keys()))
    #print(len(hydb_pos_dic.keys()))
    '''fo = open('hydb_single.txt','w')
	for i in range(0,len(L),1):
		s = str(L[i]) + ' '
		for j in range(0,len(F[i]),1):
			s = s + str(j) + ':' + str(F[i][j]) + ' '
		fo.write(s.strip(' '))
		fo.write('\n')
	fo.close()'''
    IN = []
    CR = []
    PR = []
    RC = []
    FS = []

    #clf_hydb = svm.SVC()
    #clf_hydb = svm.SVC(kernel='rbf',gamma=0.5)
    #clf_hydb = svm.LinearSVC()
    #clf_hydb = LogisticRegression()
    clf_hydb = curr_classifier1
    #clf_hydb = RandomForestClassifier()
    #clf_hydb = MultinomialNB()
    clf_hydb.fit(F,L)
    scores = cross_validation.cross_val_score(clf_hydb,F,L,cv=10)
    print(classifier_name ,' hydb CrossValidation: ',scores.mean(),scores.std())
    '''recall = cross_validation.cross_val_score(clf_hydb, F, L, cv=10, scoring='recall')
    print('Recall', np.mean(recall), recall)
    precision = cross_validation.cross_val_score(clf_hydb, F, L, cv=10, scoring='precision')
    print('Precision', np.mean(precision), precision)
    f1 = cross_validation.cross_val_score(clf_hydb, F, L, cv=10, scoring='f1')
    print('F1', np.mean(f1), f1)'''

    IN.append(scores.mean())

    keys = utkd_word_dic.keys()
    pos_keys = utkd_pos_dic.keys()
    F = []
    L = []
    for x in UF:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(utkd_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(utkd_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])

        #for i in range(0,len(x[5]),1):
        #        C.append(x[5][i])
        F.append(C)
        L.append(int(x[0]))

    '''fo = open('utkd_single.txt','w')
	for i in range(0,len(L),1):
		s = str(L[i]) + ' '
		for j in range(0,len(F[i]),1):
			s = s + str(j) + ':' + str(F[i][j]) + ' '
		fo.write(s.strip(' '))
		fo.write('\n')
	fo.close()'''

    #clf_utkd = svm.SVC()
    #clf_utkd = svm.SVC(kernel='rbf',gamma=0.5)
    #clf_utkd = svm.LinearSVC()
    #clf_utkd = LogisticRegression()
    clf_utkd = curr_classifier2
    #clf_utkd = RandomForestClassifier()
    #clf_utkd = MultinomialNB()
    clf_utkd.fit(F,L)
    scores = cross_validation.cross_val_score(clf_utkd,F,L,cv=10)
    print(classifier_name,' utkd CrossValidation: ',scores.mean(),scores.std())
    '''recall = cross_validation.cross_val_score(clf_utkd, F, L, cv=10, scoring='recall')
    print('Recall', np.mean(recall), recall)
    precision = cross_validation.cross_val_score(clf_utkd, F, L, cv=10, scoring='precision')
    print('Precision', np.mean(precision), precision)
    f1 = cross_validation.cross_val_score(clf_utkd, F, L, cv=10, scoring='f1')
    print('F1', np.mean(f1), f1)'''
    IN.append(scores.mean())

    keys = sandy_word_dic.keys()
    pos_keys = sandy_pos_dic.keys()
    F = []
    L = []
    for x in SH:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(sandy_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(sandy_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])

        #for i in range(0,len(x[5]),1):
        #        C.append(x[5][i])
        F.append(C)
        L.append(int(x[0]))

    '''fo = open('sandy_hook_single.txt','w')
	for i in range(0,len(L),1):
		s = str(L[i]) + ' '
		for j in range(0,len(F[i]),1):
			s = s + str(j) + ':' + str(F[i][j]) + ' '
		fo.write(s.strip(' '))
		fo.write('\n')
	fo.close()'''

    #clf_sandy = svm.SVC()
    #clf_sandy = svm.SVC(kernel='rbf',gamma=0.5)
    #clf_sandy = svm.LinearSVC()
    #clf_sandy = LogisticRegression()
    clf_sandy = curr_classifier3
    #clf_sandy = RandomForestClassifier()
    #clf_sandy = MultinomialNB()
    clf_sandy.fit(F,L)
    scores = cross_validation.cross_val_score(clf_sandy,F,L,cv=10)
    print(classifier_name,' sandy hook CrossValidation: ',scores.mean(),scores.std())
    '''recall = cross_validation.cross_val_score(clf_sandy, F, L, cv=10, scoring='recall')
    print('Recall', np.mean(recall), recall)
    precision = cross_validation.cross_val_score(clf_sandy, F, L, cv=10, scoring='precision')
    print('Precision', np.mean(precision), precision)
    f1 = cross_validation.cross_val_score(clf_sandy, F, L, cv=10, scoring='f1')
    print('F1', np.mean(f1), f1)'''
    IN.append(scores.mean())

    keys = hagupit_word_dic.keys()
    pos_keys = hagupit_pos_dic.keys()
    F = []
    L = []
    for x in HG:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hagupit_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hagupit_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    #clf_hagupit = svm.SVC()
    #clf_hagupit = svm.SVC(kernel='rbf',gamma=0.5)
    #clf_hagupit = svm.LinearSVC()
    #clf_hagupit = LogisticRegression()
    clf_hagupit = curr_classifier4
    #clf_hagupit = RandomForestClassifier()
    #clf_hagupit = MultinomialNB()
    clf_hagupit.fit(F,L)
    scores = cross_validation.cross_val_score(clf_hagupit,F,L,cv=10)
    print(classifier_name,' hagupit CrossValidation: ',scores.mean(),scores.std())
    '''recall = cross_validation.cross_val_score(clf_hagupit, F, L, cv=10, scoring='recall')
    print('Recall', np.mean(recall), recall)
    precision = cross_validation.cross_val_score(clf_hagupit, F, L, cv=10, scoring='precision')
    print('Precision', np.mean(precision), precision)
    f1 = cross_validation.cross_val_score(clf_hagupit, F, L, cv=10, scoring='f1')
    print('F1', np.mean(f1), f1)'''
    IN.append(scores.mean())

    ''' cross domain evaluation '''

    ################################ HYDB => OTHERS ################################

    keys = hydb_word_dic.keys()
    pos_keys = hydb_pos_dic.keys()

    F = []
    L = []
    for x in UF:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(utkd_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    predicted_label = clf_hydb.predict(F)
    count = 0
    for i in range(0,len(L),1):
        if L[i]==predicted_label[i]:
            count+=1
    #print(count,' ',len(L))
    print('Hydb-Utkd score: ',clf_hydb.score(F,L))
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    #sys.exit(0)
    CR.append(clf_hydb.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in SH:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(sandy_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Hydb-SandyHook score: ',clf_hydb.score(F,L))
    predicted_label = clf_hydb.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_hydb.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in HG:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hagupit_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Hydb-Hagupit score: ',clf_hydb.score(F,L))
    predicted_label = clf_hydb.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    #sys.exit(0)
    CR.append(clf_hydb.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    ################################ UTKD => OTHERS ################################

    keys = utkd_word_dic.keys()
    pos_keys = utkd_pos_dic.keys()

    F = []
    L = []
    for x in HB:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hydb_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Utkd-Hydb score: ',clf_utkd.score(F,L))
    predicted_label = clf_utkd.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_utkd.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in SH:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(sandy_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Utkd-SandyHook score: ',clf_utkd.score(F,L))
    predicted_label = clf_utkd.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_utkd.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in HG:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hagupit_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                #C.append(temp.count(p))
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Utkd-Hagupit score: ',clf_utkd.score(F,L))
    predicted_label = clf_utkd.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_utkd.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    ################################ SANDY_HOOK => OTHERS ################################

    keys = sandy_word_dic.keys()
    pos_keys = sandy_pos_dic.keys()

    F = []
    L = []
    for x in HB:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hydb_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('SandyHook-Hydb score: ',clf_sandy.score(F,L))
    predicted_label = clf_sandy.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_sandy.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in UF:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(utkd_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('SandyHook-Utkd score: ',clf_sandy.score(F,L))
    predicted_label = clf_sandy.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_sandy.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in HG:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hagupit_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('SandyHook-Hagupit score: ',clf_sandy.score(F,L))
    predicted_label = clf_sandy.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_sandy.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))


    ################################ Hagupit => OTHERS ################################

    keys = hagupit_word_dic.keys()
    pos_keys = hagupit_pos_dic.keys()

    F = []
    L = []
    for x in HB:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(hydb_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Hagupit-Hydb score: ',clf_hagupit.score(F,L))
    predicted_label = clf_hagupit.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_hagupit.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in UF:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(utkd_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Hagupit-Utkd score: ',clf_hagupit.score(F,L))
    predicted_label = clf_hagupit.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_hagupit.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    F = []
    L = []
    for x in SH:
        C = []
        temp = x[5]
        for p in keys:
            if p in temp:
                C.append(sandy_word_dic[p])
                #C.append(temp.count(p))
                #C.append(1)
            else:
                C.append(0)

        temp = x[2]
        for p in pos_keys:
            if p in temp:
                #C.append(hydb_pos_dic[p])
                C.append(1)
            else:
                C.append(0)
        C.append(x[3])
        C.append(x[4])
        F.append(C)
        L.append(int(x[0]))

    print('Hagupit-SandyHook score: ',clf_hagupit.score(F,L))
    predicted_label = clf_hagupit.predict(F)
    print('PRECISION: ',metrics.precision_score(L,predicted_label))
    print('RECALL: ',metrics.recall_score(L,predicted_label))
    print('F1-SCORE: ',metrics.f1_score(L,predicted_label))
    CR.append(clf_hagupit.score(F,L))
    PR.append(metrics.precision_score(L,predicted_label))
    RC.append(metrics.recall_score(L,predicted_label))
    FS.append(metrics.f1_score(L,predicted_label))

    print(len(IN),len(CR),len(PR),len(RC),len(FS))
    print(classifier_name,' Average Indomain Accuracy: ',np.mean(IN))
    print(classifier_name,' Average Cross-domain Accuracy: ',np.mean(CR))
    print(classifier_name,' Average and std Precision: ',np.mean(PR),np.std(PR))
    print(classifier_name,' Average and std Recall: ',np.mean(RC),np.std(RC))
    print(classifier_name,' Average and std F-score: ',np.mean(FS),np.std(FS))

def main():
    '''try:
		_, fn1, fn2, fn3, fn4 = sys.argv
	except Exception as e:
		print(e)
		sys.exit(0)
	predict(fn1,fn2,fn3,fn4)'''

    names = [
            "KNN", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes"]


    classifiers1 = [
        KNeighborsClassifier(weights='distance', n_neighbors=121),
        SVC(kernel="linear", C=1, probability=True),
        SVC(C=1, probability=True),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        BernoulliNB()
        ]

    classifiers2 = [
        KNeighborsClassifier(weights='distance', n_neighbors=121),
        SVC(kernel="linear", C=1, probability=True),
        SVC(C=1, probability=True),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        BernoulliNB()
        ]

    classifiers3 = [
        KNeighborsClassifier(weights='distance', n_neighbors=121),
        SVC(kernel="linear", C=1, probability=True),
        SVC(C=1, probability=True),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        BernoulliNB()
        ]

    classifiers4 = [
        KNeighborsClassifier(weights='distance', n_neighbors=121),
        SVC(kernel="linear", C=1, probability=True),
        SVC(C=1, probability=True),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        BernoulliNB()
        ]

    for i in range(len(classifiers1)):
        predict(classifiers1[i], classifiers2[i], classifiers3[i], classifiers4[i], names[i])

if __name__=='__main__':
    main()
