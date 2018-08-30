import os

from django.shortcuts import render
from django.http import HttpResponse

from django.core.files.storage import FileSystemStorage
from dishelper.settings import BASE_DIR

# import tarfile
# from io import BytesIO


def system_details(request):
	return render(request, "dis_app/system_details.html", {})

def performance(request):
	return render(request, "dis_app/performance.html", {})

def communal(request):
	from dis_app import Single_Tweet_Communal
	from dis_app import Communal_Multiple
	# return render(request, "dis_app/communal-identifier.html", {})

	if request.method == 'POST' and 'myfile_comm' in request.FILES and request.FILES['myfile_comm']:
		myfile = request.FILES['myfile_comm']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)

		Communal_Multiple.multiple_classify(uploaded_file_url, 'out_comm.txt')


		os.system("zip out_comm.zip out_comm.txt")

		if os.path.exists('out_comm.zip'):
			with open('out_comm.zip', 'rb') as fh:
				response = HttpResponse(fh.read(), content_type="application/zip")
				response['Content-Disposition'] = 'inline; filename=' + os.path.basename('out_comm.zip')

				return response
				# return render(request, 'dis_app/situational-classifier.html', {'response': response})
		raise Http404		



	elif request.method == 'POST': # If the form is submitted
		tweet = request.POST.get('tweet_comm', None)

		c = Single_Tweet_Communal.communal_classifier(tweet);

		return render(request, "dis_app/communal-identifier.html", {"class": [tweet,c]})
	else:
		return render(request, "dis_app/communal-identifier.html", {"class": ["", "-1"]})



def sc(request):
	from dis_app import Single_Tweet_Classifier 
	from dis_app import Tweet_Classifier

	if request.method == 'POST' and 'myfile' in request.FILES and request.FILES['myfile']:
		myfile = request.FILES['myfile']
		fs = FileSystemStorage()
		filename = fs.save(myfile.name, myfile)
		uploaded_file_url = fs.url(filename)

		Tweet_Classifier.Classification(uploaded_file_url, 'out.txt')


		os.system("zip out.zip out.txt")

		if os.path.exists('out.zip'):
			with open('out.zip', 'rb') as fh:
				response = HttpResponse(fh.read(), content_type="application/zip")
				response['Content-Disposition'] = 'inline; filename=' + os.path.basename('out.zip')

				return response
				# return render(request, 'dis_app/situational-classifier.html', {'response': response})
		raise Http404		



	elif request.method == 'POST': # If the form is submitted
		tweet = request.POST.get('tweet', None)

		c, confidence = Single_Tweet_Classifier.Classification(tweet)

		return render(request, "dis_app/situational-classifier.html", {"class": [tweet,c,round(confidence, 4)]})
	else:
		return render(request, "dis_app/situational-classifier.html", {"class": ["", "-1","0"]})

def ds(request):
	from dis_app.summarization import cowts

	if request.method == 'POST':
		dataset = request.POST.get('disaster', None)
		words = int(request.POST.get('words', None))

		ifname, wordcount = cowts.compute_similarity(dataset, words, words)

		return render(request, "dis_app/disaster-summarizer.html", {"dataset":ifname, "words":wordcount})

	else:
		return render(request, "dis_app/disaster-summarizer.html", {"dataset":"None", "words":"-1"})

def team(request):
	return render(request, "dis_app/our-team.html", {})

def index(request):
	return render(request, "dis_app/index.html", {})