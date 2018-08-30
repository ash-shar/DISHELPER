import socketserver
import time

import sys
import re
import codecs
import string
import os
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import *
from sklearn import metrics
from sklearn import cross_validation
import gzip
import numpy as np
import pickle
from sklearn.externals import joblib
#from happyfuntokenizing import *


mycompile = lambda pat:  re.compile(pat,  re.UNICODE)

basepath = ""

PRONOUN_PATH = basepath+'global_view/english_pronoun.txt'
WHWORD_PATH = basepath+'global_view/english_whwords.txt'
SLANG_PATH = basepath+'global_view/english_slang.txt'
INTENSIFIER_PATH = basepath+'global_view/english_intensifier.txt'
SUBJECTIVE_PATH = basepath+'global_view/subjclueslen1-HLTEMNLP05.tff'
EVENT_PATH = basepath+'global_view/english_nonsituational_phrase.txt'
MODAL_VERB_PATH = basepath+'global_view/english_modal_verb.txt'
#OPINION_HASHTAG_PATH = '/home/krudra/twitter_code/shared/language_overlap/code_mix_pitch/devanagari/devanagari_hashtag_opinion.txt'
#MENTION_PATH = '/home/krudra/twitter_code/shared/language_overlap/code_mix_pitch/devanagari/news_mention.txt'

emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

TAGGER_PATH = basepath+'ark-tweet-nlp-0.3.2'

############################ This Functions are used #############################################

def emoticons(s):
	return len(re.findall(u'[\U0001f600-\U0001f60f\U0001f617-\U0001f61d\U0001f632\U0001f633\U0001f638-\U0001f63e\U0001f642\U0001f646-\U0001f64f\U0001f612\U0001f613\U0001f615\U0001f616\U0001f61e-\U0001f629\U0001f62c\U0001f62d\U0001f630\U0001f631\U0001f636\U0001f637\U0001f63c\U0001f63f-\U0001f641\U0001f64d]', s))

def smileys(s):
        return len(re.findall(r':\-\)|:[\)\]\}]|:[dDpP]|:3|:c\)|:>|=\]|8\)|=\)|:\^\)|:\-D|[xX8]\-?D|=\-?D|=\-?3|B\^D|:\'\-?\)|>:\[|:\-?\(|:\-?c|:\-?<|:\-?\[|:\{|;\(|:\-\|\||:@|>:\(|:\'\-?\(|D:<?|D[8;=X]|v.v|D\-\':|>:[\/]|:\-[./]|:[\/LS]|=[\/L]|>.<|:\$|>:\-?\)|>;\)|\}:\-?\)|3:\-?\)|\(>_<\)>?|^_?^;|\(#\^.\^#\)|[Oo]_[Oo]|:\-?o',s))

def getNumberOfElongatedWords(s):
    	return len(re.findall('([a-zA-Z])\\1{2,}', s))
    
def pronoun(sen):
	fp = open(PRONOUN_PATH,'r')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_prn = set([])
	for x in sen:
		cur_prn.add(x.strip(' \t\n\r').lower())
	
	size = len(temp.intersection(cur_prn))
	if size>0:
		return 1
	return 0
	
	
def exclamation(s):
	c = len(re.findall(r"[!]", s))
	if c>=1:
		return 1
	return 0
	

def question(s):
	return len(re.findall(r"[?]", s))

def intensifier(sen):
	fp = open(INTENSIFIER_PATH,'r')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_int = set([])
	for x in sen:
		cur_int.add(x.strip(' \t\n\r').lower())
	
	size = len(temp.intersection(cur_int))
	if size>0:
		return 1
	return 0

def whword(sen):
	fp = open(WHWORD_PATH,'r')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_wh = set([])
	for x in sen:
		cur_wh.add(x.strip(' \t\n\r').lower())
	
	size = len(temp.intersection(cur_wh))
	if size>0:
		return 1
	return 0

def slang(sen):
	fp = open(SLANG_PATH,'r')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_slg = set([])
	for x in sen:
		cur_slg.add(x.strip(' \t\n\r').lower())
	
	size = len(temp.intersection(cur_slg))
	if size>0:
		return 1
	return 0

def event_phrase(sen):
	fp = open(EVENT_PATH,'r')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_slg = set([])
	for x in sen:
		cur_slg.add(x.strip(' \t\n\r').lower())
	
	size = len(temp.intersection(cur_slg))
	if size>0:
		return 1
	return 0

def getHashtagopinion(sen):
	fp = codecs.open(OPINION_HASHTAG_PATH,'r','utf-8')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_hash = set([])
	for x in sen:
		if x.startswith('#')==True:
			cur_hash.add(x.strip(' \t\n\r').lower())
	size = len(temp.intersection(cur_hash))
	if size>0:
		return 1
	return 0

def numeral(temp):
	c = 0
	for x in temp:
		if x.isdigit()==True:
			c+=1
	return c

def modal(sen):
	fp = open(MODAL_VERB_PATH,'r')
	temp = set([])
	for l in fp:
		temp.add(l.strip(' \t\n\r').lower())
	fp.close()

	cur_mod = set([])
	for x in sen:
		cur_mod.add(x.strip(' \t\n\r').lower())
	
	size = len(temp.intersection(cur_mod))
	if size>0:
		return 1
	return 0

def subjectivity(sen):
	
	fp = open(SUBJECTIVE_PATH,'r')

	temp = []

	for l in fp:
	        wl = l.split()
	        x = wl[0].split('=')[1].strip(' \t\n\r')
	        if x=='strongsubj':
	                y = wl[2].split('=')[1].strip(' \t\n\r')
	                temp.append(y)
	c = 0
	for x in sen:
	        if x in temp:
	                c+=1
	tot = len(sen) + 4.0 - 4.0
	num = c + 4.0 - 4.0
	s = round(num/tot,4)
	return s

	
def Classification(tweet_li):

	#self.text = text

	############### TEST ON FUTURE EVENTS #######################################
	cnt = 0
	train_clf = joblib.load(basepath+'CLMODEL.pkl')

	# fp = codecs.open(ifname,'r','utf-8')
	fo = codecs.open('temp.txt','w','utf-8')
	for l in tweet_li:
		fo.write(l.strip(' \t\n\r') + '\n')
		cnt+=1
		# if cnt>=num:
		# 	break
	# fp.close()
	fo.close()

	print("Cnt:", cnt)
	# if cnt

	command = TAGGER_PATH + '/./runTagger.sh --output-format conll temp.txt > tagfile.txt'

	os.system(command)


	Numeral = []
	fp = codecs.open('tagfile.txt','r','utf-8')
	c = 0
	for  l in fp:
	    	wl = l.split()
	    	if len(wl)>1:
	            	if wl[1].strip(' \t\n\r')=='$':
	                    	c+=1
	    	else:
	            	if c>=1:
	                    	Numeral.append(c)
	            	else:
	                    	Numeral.append(0)
	            	c = 0
	fp.close()


	fp = open('temp.txt','r')
	fs = codecs.open('temp.txt','r','utf-8')

	feature = []
	count = 0
	fs = codecs.open('temp.txt','r','utf-8')

	feature = []
	count = 0
	for l in fp:
		row = l.strip(' \t\n\r')
		org_tweet = fs.readline().strip(' \t\n\r')
		#print(getNumberOfElongatedWords('soooooooooooooooo'))
		#sys.exit(0)

		temp = row.split()
		N = Numeral[count]
		E = exclamation(row)
		Q = question(row)
		M = modal(temp)
		I = intensifier(temp)
		W = whword(temp)
		EP = event_phrase(temp)
		S = subjectivity(temp)
		SG = slang(temp)
		P = pronoun(temp)
		EL = getNumberOfElongatedWords(row)
		EM = emoticons(org_tweet)
		SM = smileys(row)
		#t = [N,E,Q,M,I,W,S,P,EP,SG,EM,SM]
		#t = [N,E,Q,M,I,W,S,P,EP,SG,EL,SM,EM]
		t = [N,E,Q,M,I,W,S,P,EP,SG,EL]
		feature.append(t)
		count+=1
		#if count<=5:
		#       print(row[1].strip(' \t\n\r'))
		#       print(t)


	fp.close()

	fs.close()

	predicted_label = train_clf.predict(feature)
	predicted_proba_label = train_clf.predict_proba(feature)


	out_str = ""	
	# fp = codecs.open(ifname,'r','utf-8')
	# fo = codecs.open(ofname,'w','utf-8')
	count = 0
	sit_cnt = 0
	cnt = 0
	for l in tweet_li:
		# wl = l.split('\t')
		# print(predicted_proba_label)
		temp = predicted_proba_label[count]
		if predicted_label[count] == 1:
			sit_cnt+=1

		s = l.strip(' \t\n\r')+ '\t' + str(predicted_label[count]) + '\t' + str(max(temp))
		out_str += s+'\n'
		count+=1

		cnt+=1
		# if cnt>=num:
		# 	break
	# fp.close()
	# fo.close()

	return out_str.strip()

	# print(sit_cnt)


class MyTCPHandler(socketserver.BaseRequestHandler):
	"""
	The request handler class for our server.

	It is instantiated once per connection to the server, and must
	override the handle() method to implement communication to the
	client.
	"""

	def handle(self):
		#print(dict_docs['998'])

		# self.request is the TCP socket connected to the client
		self.data = self.request.recv(10240).strip()
		
		#print "{} wrote:".format(self.client_address[0])
		# print("num -> "+str(number))
		# print("data -> "+str(self.data))
		query1 = self.data
		query = query1.decode('utf-8')
		# query = eval(query)
		print("Decoded: "+query)

		query_li = query.strip().split('\n')
		result = Classification(query_li)
		# just send back the same data, but upper-cased
		print(str(type(result)))

		to_send = str(result)
		#print(to_send)
		datalen = len(to_send)
		print(datalen)
		self.request.send(str(datalen).encode('utf-8'))
		time.sleep(1)
		sent = 0
		while(True):
			send = self.request.send(to_send.encode('utf-8'))
			#time.sleep(1)
			sent = sent + send
			datalen = datalen - send
			print("now sending : "+to_send[0:send])
			to_send = to_send[send:]
			if(datalen <= 0):
				break

		print("\nSend successful",sent)

		#self.request.sendall(to_send)

def serverRunner(HOST, PORT):
	# HOST, PORT = "10.5.18.104", 5010
	global X
	print("Start")
	# HOST = input("Enter HOST: ")
	 #Create the server, binding to localhost on port 9999
	while(True):
		try:
			server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
			break
		except:
			PORT_NO = input("Enter PORT: ")
			PORT = int(PORT_NO)

	print("HOST: "+HOST+"\tPORT: "+str(PORT))

	server.serve_forever()
	print("Server Started")

if __name__ == "__main__":
	serverRunner("10.5.18.103", 9954)