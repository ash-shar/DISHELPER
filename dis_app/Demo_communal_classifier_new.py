import sys
import os
print(os.path.dirname(__file__))
import random
import re
import codecs
import string
from happyfuntokenizing import *
from nltk.corpus import stopwords
# from textblob import *
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.cross_validation import KFold

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import numpy as np
import gzip
from sklearn.externals import joblib
import pickle



#basepath = "/home/du3/13CS30043/BTP/Classifier/"
basepath = '/home/du3/13CS30043/BTP/System/Gandhi_Award/communal/'
model_path = basepath+"ant_communal_trained.pkl"
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)

OFFSET = 2
CONF_THR = 0

##################### DICTIONARY FILE PATH ####################

SUBJECTIVE_PATH = basepath+'DICTIONARY/subjclueslen1-HLTEMNLP05.tff'
COMMUNAL_PATH = basepath+'DICTIONARY/communal_dictionary.txt'
RELIGION_PATH = basepath+'DICTIONARY/communal_race.txt'
SLANG_PATH = basepath+'DICTIONARY/english_slang.txt'
SWEAR_PATH = basepath+'DICTIONARY/english_swear.txt'
WH_PATH = basepath+'DICTIONARY/english_whwords.txt'
INTENSIFIER_PATH = basepath+'DICTIONARY/english_intensifier.txt'
COMMUNAL_HASHTAG_PATH = basepath+'DICTIONARY/communal_hashtag_dictionary.txt'
ANTICOMMUNAL_COLLOCATIONS_PATH = basepath+'DICTIONARY/anticommunal_collocations.txt'
ANTICOMMUNAL_HASHTAGS_PATH = basepath+'DICTIONARY/anticommunal_hashtags.txt'

###################### END OF DICTIONARY FILE PATH #############


cachedstopwords = stopwords.words("english")	# English Stop Words
Tagger_Path = '/home/du3/13CS30043/BTP/System/Gandhi_Award/ark-tweet-nlp-0.3.2/'
#Tagger_Path = '/home/krudra/twitter_code/aaai/characterize_user/wordcloud/ark-tweet-nlp-0.3.2/'
lmtzr = WordNetLemmatizer()		# Lemmatizer

SUBJECTIVE = {}
COMMUNAL = {}
RELIGION = {}
SLANG = {}
HASHTAG = {}
WH = {}
INTENSIFIER = {}
A_COLL = []
A_HASH = []

##################################
# Reads Dictionary files and stores them in respective dictionary
##################################

def Read_Files():

	fp = open(SUBJECTIVE_PATH,'r')
	for l in fp:
		wl = l.split()
		Type = wl[0].split('=')[1].strip(' \t\n\r')
		pos_tag = wl[3].split('=')[1].strip(' \t\n\r')
		Tag = wl[5].split('=')[1].strip(' \t\n\r')
		word = wl[2].split('=')[1].strip(' \t\n\r')


		if Type=='strongsubj':
			if SUBJECTIVE.__contains__(word)==False:
				if Tag=='negative':
					SUBJECTIVE[word] = -1
				elif Tag=='positive':
					SUBJECTIVE[word] = 1
				else:
					SUBJECTIVE[word] = 0
	fp.close()

	fp = open(COMMUNAL_HASHTAG_PATH,'r')
	for l in fp:
		w = l.strip(' #\t\n\r').lower()
		if HASHTAG.__contains__(w)==False:
			HASHTAG[w] = 1
	fp.close()
	
	fp = open(WH_PATH,'r')
	for l in fp:
		w = l.strip(' #\t\n\r').lower()
		if WH.__contains__(w)==False:
			WH[w] = 1
	fp.close()
	
	fp = open(INTENSIFIER_PATH,'r')
	for l in fp:
		w = l.strip(' \t\n\r').lower()
		if INTENSIFIER.__contains__(w)==False:
			INTENSIFIER[w] = 1
	fp.close()

	fp = open(RELIGION_PATH,'r')
	for l in fp:
		w = l.strip(' \t\n\r').lower()
		if RELIGION.__contains__(w)==False:
			RELIGION[w] = 1
	fp.close()
	
	fp = open(COMMUNAL_PATH,'r')
	for l in fp:
		wl = l.split('\t')
		w = wl[0].strip(' \t\n\r').lower()
		if COMMUNAL.__contains__(w)==False:
			x1 = wl[1].strip(' \t\n\r')
			x2 = wl[2].strip(' \t\n\r')
			x3 = wl[3].strip(' \t\n\r')
			if len(x2)==0:
				COMMUNAL[w] = (int(wl[1]),0)
			else:
				COMMUNAL[w] = (int(wl[1]),int(wl[2]))
			'''	
			try:
				x1 = int(wl[1])
			except Exception as e
			COMMUNAL[w] = (int(wl[1]),int(wl[2]),int(wl[3]))
			'''
	fp.close()
	

	fp = open(SLANG_PATH,'r')
	for l in fp:
		w = l.strip(' \t\n\r').lower()
		if SLANG.__contains__(w)==False:
			SLANG[w] = 1
	fp.close()
	
	fp = open(SWEAR_PATH,'r')
	for l in fp:
		w = l.strip(' \t\n\r').lower()
		if SLANG.__contains__(w)==False:
			SLANG[w] = 1
	fp.close()

	fp = open(ANTICOMMUNAL_COLLOCATIONS_PATH,'r')
	for l in fp:
		w = l.strip(' \t\n\r').lower()
		A_COLL.append(w)
		# print(w)
	fp.close()

	fp = open(ANTICOMMUNAL_HASHTAGS_PATH,'r')
	for l in fp:
		w = l.strip(' \t\n\r').lower()
		A_HASH.append(w)
	fp.close()


##################################
# check if a unigram is strongsub and negative
##################################
def getnegativesubjective(unigram):
	for x in unigram:
		if SUBJECTIVE.__contains__(x)==True:
			if SUBJECTIVE[x]==-1:
				return 1
	return 0


def getcollocationterm(unigram,bigram,trigram):
	flag = 0
	for u in unigram:
		if A_COLL.__contains__(u)==True:
			flag = 1
			break
	if flag==1:
		return 1

	for u in bigram:
		if A_COLL.__contains__(u)==True:
			flag = 1
			break
	if flag==1:
		return 1

	for u in trigram:
		if A_COLL.__contains__(u)==True:
			flag = 1
			break
	if flag==1:
		return 1

	return 0

def getreligiouscount(unigram):
	flag = 0
	temp = set([])
	for u in unigram:
		if RELIGION.__contains__(u)==True:
			temp.add(u)
			#flag+=1
	if len(temp)>2:
		return 1
	#if flag>2:
	#        return 1
	return 0


##################################
# check if unigram, bigram and trigram are communal slang or not
##################################
def getcommunalslang(unigram,bigram,trigram):
	flag = 0
	for u in unigram:
		if COMMUNAL.__contains__(u)==True:
			v = COMMUNAL[u]
			if v[1]==1:
				flag=1
				break
	if flag==1:
		return 1
	
	for u in bigram:
		if COMMUNAL.__contains__(u)==True:
			v = COMMUNAL[u]
			if v[1]==1:
				flag=1
				break
	if flag==1:
		return 1
	
	for u in trigram:
		if COMMUNAL.__contains__(u)==True:
			v = COMMUNAL[u]
			if v[1]==1:
				flag=1
				break
	if flag==1:
		return 1

	return 0

def get_religious_sarcasm(unigram):
        flag_REL = 0
        for u in unigram:
                if RELIGION.__contains__(u)==True:
                        flag_REL = 1
                        break

        flag_INT = 0
        for u in unigram:
                if INTENSIFIER.__contains__(u)==True:
                        flag_INT = 1
                        break

        flag_WH = 0
        for u in unigram:
                if WH.__contains__(u)==True:
                        flag_WH = 1
                        break

        if flag_REL==1 and flag_WH==1:
                return 1
        if flag_REL==1 and flag_INT==1:
                return 1
        return 0


##################################
# check if there is a communal or religious term and a slang or subjective negative term in a window of +-3 
##################################

def get_religious_slang(unigram,bigram,trigram):
	
	################################# First Check Unigrams ################################################################
	flag = 0
	for i in range(0,len(unigram),1):
		w = unigram[i]
		if COMMUNAL.__contains__(w)==True or RELIGION.__contains__(w)==True:
			L = i - OFFSET
			R = i + OFFSET
			if L<0:
				L = 0
			if R >= len(unigram):
				R = len(unigram) - 1
			for j in range(L,i,1):
				if SUBJECTIVE.__contains__(unigram[j])==True:
					if SUBJECTIVE[unigram[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(unigram[j])==True:
					flag=1
					break
			if flag==1:
				return 1
			for j in range(i+1,R+1,1):
				if SUBJECTIVE.__contains__(unigram[j])==True:
					if SUBJECTIVE[unigram[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(unigram[j])==True:
					flag=1
					break
			if flag==1:
				return 1

	################################# Second Check Bigrams ??? - we should have checked for unigrams ################################################################
	
	flag = 0
	for i in range(0,len(bigram),1):
		w = bigram[i]
		if COMMUNAL.__contains__(w)==True or RELIGION.__contains__(w)==True:
			L = i - OFFSET
			R = i + OFFSET
			if L<0:
				L = 0
			if R >= len(bigram):
				R = len(bigram) - 1

			str_before = bigram[L]

			for j in range(L+1,i-1,1):
				str_before = str_before+' '+(bigram[j].split(' '))[1]

			unigram_before = str_before.split(' ')

			trigram_before = []

			if len(unigram_before)>=3:
				for j in range(0,len(unigram_before)-2,1):
					s = unigram_before[j] + ' ' + unigram_before[j+1] + ' ' + unigram_before[j+2]
					trigram_before.append(s)

			for j in range(L,i,1):
				if SUBJECTIVE.__contains__(bigram[j])==True:
					if SUBJECTIVE[bigram[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(bigram[j])==True:
					flag=1
					break
			if flag==1:
				return 1

			for j in range(0,len(unigram_before),1):
				if SUBJECTIVE.__contains__(unigram_before[j])==True:
					if SUBJECTIVE[unigram_before[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(unigram_before[j])==True:
					flag=1
					break
			if flag==1:
				return 1



			for j in range(0,len(trigram_before),1):
				if SUBJECTIVE.__contains__(trigram_before[j])==True:
					if SUBJECTIVE[trigram_before[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(trigram_before[j])==True:
					flag=1
					break
			if flag==1:
				return 1


			str_after = ''


			for j in range(i+1,R+1,1):
				str_after = str_after+' '+(bigram[j].split(' '))[1]

			unigram_after = str_after.split(' ')

			trigram_after = []

			if len(unigram_after)>=3:
				for j in range(0,len(unigram_after)-2,1):
					s = unigram_after[j] + ' ' + unigram_after[j+1] + ' ' + unigram_after[j+2]
					trigram_after.append(s)


			for j in range(i+1,R+1,1):
				if SUBJECTIVE.__contains__(bigram[j])==True:
					if SUBJECTIVE[bigram[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(bigram[j])==True:
					flag=1
					break
			if flag==1:
				return 1


			for j in range(0,len(unigram_after),1):
				if SUBJECTIVE.__contains__(unigram_after[j])==True:
					if SUBJECTIVE[unigram_after[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(unigram_after[j])==True:
					flag=1
					break
			if flag==1:
				return 1


			# print(str_before, unigram_before, str_after, unigram_after)

			for j in range(0,len(trigram_after),1):
				if SUBJECTIVE.__contains__(trigram_after[j])==True:
					if SUBJECTIVE[trigram_after[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(trigram_after[j])==True:
					flag=1
					break
			if flag==1:
				return 1
	
	################################# Third Check Trigrams ??? - we should have checked for unigrams ################################################################
	
	flag = 0
	for i in range(0,len(trigram),1):
		w = trigram[i]
		# print(w)
		if COMMUNAL.__contains__(w)==True or RELIGION.__contains__(w)==True:
			L = i - OFFSET
			R = i + OFFSET

			unigram_before = []

			bigram_before = []

			if i>=3:
				str_before = trigram[i-3]

				unigram_before = str_before.split(' ')

				if len(unigram_before)>=3:
					for j in range(0,len(unigram_before)-1,1):
						s = unigram_before[j] + ' ' + unigram_before[j+1]
						bigram_before.append(s)




			if L<0:
				L = 0
			if R >= len(trigram):
				R = len(trigram) - 1



			for j in range(L,i,1):
				if SUBJECTIVE.__contains__(trigram[j])==True:
					if SUBJECTIVE[trigram[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(trigram[j])==True:
					flag=1
					break
			if flag==1:
				return 1



			for j in range(0,len(unigram_before),1):
				if SUBJECTIVE.__contains__(unigram_before[j])==True:
					if SUBJECTIVE[unigram_before[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(unigram_before[j])==True:
					flag=1
					break
			if flag==1:
				return 1


			for j in range(0,len(bigram_before),1):
				if SUBJECTIVE.__contains__(bigram_before[j])==True:
					if SUBJECTIVE[bigram_before[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(bigram_before[j])==True:
					flag=1
					break
			if flag==1:
				return 1


			unigram_after = []
			bigram_after = []

			if i<=len(trigram)-4:
				str_after = trigram[i+3]

				unigram_after = str_after.split(' ')

				if len(unigram_after)>=3:
					for j in range(0,len(unigram_after)-1,1):
						s = unigram_after[j] + ' ' + unigram_after[j+1]
						bigram_after.append(s)


			for j in range(i+1,R+1,1):
				if SUBJECTIVE.__contains__(trigram[j])==True:
					if SUBJECTIVE[trigram[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(trigram[j])==True:
					flag=1
					break
			if flag==1:
				return 1


			for j in range(0,len(unigram_after),1):
				if SUBJECTIVE.__contains__(unigram_after[j])==True:
					if SUBJECTIVE[unigram_after[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(unigram_after[j])==True:
					flag=1
					break
			if flag==1:
				return 1

			# print(str_before, unigram_before, str_after, unigram_after)

			for j in range(0,len(bigram_after),1):
				if SUBJECTIVE.__contains__(bigram_after[j])==True:
					if SUBJECTIVE[bigram_after[j]]==-1:
						flag=1
						break
				elif SLANG.__contains__(bigram_after[j])==True:
					flag=1
					break
			if flag==1:
				return 1

	return 0


##################################
# return 1 if there is a communal term in unigrm, bigram or trigram
##################################

def getreligiousterm(unigram,bigram,trigram):
	flag = 0
	for u in unigram:
		if COMMUNAL.__contains__(u)==True:
			v = COMMUNAL[u]
			if v[0]==1:
				flag=1
				break
	if flag==1:
		return 1
	
	for u in bigram:
		if COMMUNAL.__contains__(u)==True:
			v = COMMUNAL[u]
			if v[0]==1:
				flag=1
				break
	if flag==1:
		return 1
	
	for u in trigram:
		if COMMUNAL.__contains__(u)==True:
			v = COMMUNAL[u]
			if v[0]==1:
				flag=1
				break
	if flag==1:
		return 1

	for u in unigram:
		if RELIGION.__contains__(u)==True:
			return 1
	return 0

##################################
# return 1 if there is a slang in tweet
##################################

def getslang(unigram,bigram,trigram):
	for u in unigram:
		if SLANG.__contains__(u)==True:
			return 1

	for u in bigram:
		if SLANG.__contains__(u)==True:
			return 1
	
	for u in trigram:
		if SLANG.__contains__(u)==True:
			return 1
	return 0

##################################
# return 1 if there is a communal hashtag in tweet
##################################

def getcommunalhashtag(unigram):
	for u in unigram:
		if HASHTAG.__contains__(u)==True:
			return 1
	return 0


######################## LOAD DICTIONARIES ##############################
#Read_Files()
#########################################################################

def main(text):

	Read_Files()
	tok = Tokenizer(preserve_case=False)
	tagreject = ['U','@','#','~','E','~',',']
	fo = open('temp.txt','w')
	fo.write(text)
	fo.close()
	
	command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
	os.system(command)
	
	fp = open('tag.txt','r')
	s = ''
	h = 0
	ah = 0
	feature = []
	label = []
	for l in fp:
		wl = l.split('\t')
		if len(wl)>1:
			word = wl[0].strip(' #\t\n\r').lower()
			tag = wl[1].strip(' \t\n\r')
			# print("In if", word, tag)
			if tag not in tagreject:
				if tag=='N':
					try:
						w = lmtzr.lemmatize(word)
						word = w
					except Exception as e:
						pass
				elif tag=='V':
					try:
						w = Word(word)
						x = w.lemmatize("v")
					except Exception as e:
						x = word
					word = x.lower()
				else:
					pass
				try:
					s = s + word + ' '
				except Exception as e:
					pass
					# print(word)
				if h==0:
					if HASHTAG.__contains__(word)==True and wl[0].startswith('#')==True:
						h = 1
				if ah==0:
					if A_HASH.__contains__(word)==True and wl[0].startswith('#')==True:
						ah = 1
			else:
				if h==0:
					if HASHTAG.__contains__(word)==True and wl[0].startswith('#')==True:
						h = 1
				if ah==0:
					if A_HASH.__contains__(word)==True and wl[0].startswith('#')==True:
						ah = 1
		else:
			unigram = tok.tokenize(s.strip(' '))
			bigram = []
			if len(unigram)>=2:
				for i in range(0,len(unigram)-1,1):
					s = unigram[i] + ' ' + unigram[i+1]
					bigram.append(s)
			trigram = []
			if len(unigram)>=3:
				for i in range(0,len(unigram)-2,1):
					s = unigram[i] + ' ' + unigram[i+1] + ' ' + unigram[i+2]
					trigram.append(s)
			NEG_SUBJ = getnegativesubjective(unigram)
			REL_TERM = getreligiousterm(unigram,bigram,trigram)
			COM_SLNG = getcommunalslang(unigram,bigram,trigram)
			SLNG = getslang(unigram,bigram,trigram)
			REL_SLNG = get_religious_slang(unigram,bigram,trigram)
			REL_SARC = get_religious_sarcasm(unigram)
			REL_COUNT = getreligiouscount(unigram)
			COL_TERM = getcollocationterm(unigram,bigram,trigram)
			#t = (REL_TERM,SLNG,COM_SLNG,h)
			#feature.append(t)
			#if (REL_TERM==1 and SLNG==1) or COM_SLNG==1 or h==1:
			# print("In else", unigram)
			if REL_SLNG==1 or COM_SLNG==1 or h==1 or REL_SARC==1:
				label.append(1)
			elif REL_COUNT==1 or COL_TERM==1 or ah==1:
				label.append(3)
			else:
				label.append(2)
			s = ''
			h = 0
		ah = 0
	fp.close()


	if label[0]==1:
		print('1')
	elif label[0] == 3:
		print('2')
	else:
		print('0')

	# if label==1:
	# 	print('Communal Tweet')
	# elif label==3:
	# 	print('Anticommunal Tweet')
	# else:
	# 	print('General Tweet')
	
	#train_clf = joblib.load(modelname)
	#predicted_label = train_clf.predict(feature)
	#predicted_proba = train_clf.predict_proba(feature)
	#predicted_label = label

	'''fp = open(fn,'r')
        fo = open(ofname,'w')

        index = 0
        for l in fp:
                wl = l.split('\t')
                s = wl[0].strip(' \t\n\r') + '\t' + wl[1].strip(' \t\n\r') + '\t' + wl[2].strip(' \t\n\r') + '\t' + wl[3].strip(' \t\n\r') + '\t' + wl[4].strip(' \t\n\r') + '\t' + wl[5].strip(' \t\n\r') + '\t' + wl[6].strip(' \t\n\r') + '\t' + str(label[index])
                fo.write(s+'\n')
                index+=1

        fp.close()
        fo.close()

        print('Complete Future Event')'''

if __name__ == "__main__":
	try:
		_, text = sys.argv
	except Exception as e:
		print(str(e))
		sys.exit(0)
	# print("here")
	text = text.replace("#####"," ")
	# print(text)
	main(text)