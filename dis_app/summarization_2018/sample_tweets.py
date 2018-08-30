from random import shuffle
import codecs
from datetime import datetime
from datetime import timezone    
from collections import OrderedDict
import time

data_name = {"hydb": 'Hyderabad Blasts', "utkd": 'Uttarakhand floods', "sandy_hook": 'Sandy Hook shootout', "hagupit": 'Typhoon Hagupit', "nepal": 'Nepal Earthquake', 'harda':'Harda Twin Train Derailment'}

for disaster in data_name:
	in_file = codecs.open(disaster+'_TWEB_CONCEPT.txt', 'r', 'utf-8')

	rows = in_file.read().strip().split('\n')

	in_file.close()

	shuffle(rows)

	out_li = rows[:1000]

	dict_time_tweet = {}

	for elem in out_li:
		try:
			date_created = datetime.strptime(elem.strip().split('\t')[1], '%a %b %d %H:%M:%S %z %Y')
		except:
			date_created = datetime.strptime(elem.strip().split('\t')[1], '%Y-%m-%d %H:%M:%S')


		secs = time.mktime(date_created.timetuple())

		dict_time_tweet[int(secs)] = elem.strip()

	dict_time_tweet = OrderedDict(sorted(dict_time_tweet.items()))

	out_file = codecs.open(disaster+'_TWEB_CONCEPT_sample.txt', 'w', 'utf-8')

	i = 0
	for elem in dict_time_tweet:
		row = dict_time_tweet[elem]

		s = row.strip().split('\t')[1:]

		out_str = str(i)+'\t'

		for elem in s:
			out_str += elem+'\t'


		print(out_str.strip(), file = out_file)

		i+=1
	out_file.close()

	print(disaster)