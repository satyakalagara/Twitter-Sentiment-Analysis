"""
	Twitter sentiment analysis on tech related tweets using hashtag. Code does 
	the following..
		Stop words removal, tweets cleaning 
		Regex replace the string which starts with https/www
		Replace the string which starts with @ to AT_USER
		Tokenize tweets
"""

import twitter
import csv
import re
import nltk
import numpy as np

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC


class preprocesstweets:
	"""
		Consists of methods to perform 
		pre-processing tasks
	"""

	def __init__(self):
		self._stopwords = set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])
		#print(self._stopwords)

	def process_tweets(self,list_of_tweets):
		processed_tweets = []
		for tweet in list_of_tweets:
			processed_tweets.append((self._processTweets(tweet["tweet_text"]),tweet["label"]))
		return processed_tweets

	def _processTweets(self,tweet):
		tweet = tweet.lower()
		tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
		tweet = re.sub('@[^\s]+','AT_USER',tweet)
		tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
		tweet = word_tokenize(tweet)

		return [word for word in tweet if word not in self._stopwords]

'''
Create an account in https://apps.twitter.com/ 
Get the credentials to connect to twitter API inorder to fetch the required data
'''
api = twitter.Api(consumer_key='MC2dFxlEjONd46e7dqgwUN3SK',
				consumer_secret='nnke6ZaWGnzDWIRDWVVLbjud2bnmb6bdrX2vzNlxpOAfcDwXBO',
				access_token_key='834477706679054339-v5STA3B31IbSEEZc1oPRTFEMVhR4qse',
				access_token_secret='2ufO01QWVwlTqVbI7k6ZUHhXtQqK76vJq84hrUexKGbR3')


def Test_data(search_term):
	'''
		Download last 100 tweets by the given search term for test data
	'''

	try:
		tweets_fetched = api.GetSearch(search_term,count=100)
		#print("We have fetched " + str(len(tweets_fetched)) + "tweets from the twitter with the search string" + search_term + "!!!")

		return [{"tweet_text":status.text,"label":None} for status in tweets_fetched]
	except:
		print("Sorry there was an error")
		return None


def Training_data_limited(corpus_file,tweet_data_file):
	"""
		Downloaded the training data from neil sander's tweet corpus.
		Get tweet text from the respective tweet ID's
	"""

	corpus = []
	with open(corpus_file,"r",encoding="utf8") as csvfile:
		line_reader = csv.reader(csvfile,delimiter = ',')
		for row in line_reader:
			corpus.append({"tweet_text":row[3],"tweet_id":row[2],"label":row[1],"topic":row[0]})

	trainingdata = []
	for label in ["positive","negative"]:
		i = 1
		for line in corpus:
			if line["label"] == label and i<=150:
					tweet_text = api.GetStatus(line["tweet_id"])
					print("Tweet fetched" + tweet_text.text)
					line["text"] = tweet_text.text
				trainingdata.append(line)
				i = i+1
	
	with open(tweet_data_file,"w",encoding="utf-8") as csvfile:
		linewriter = csv.writer(csvfile,delimiter=",")
		for line in trainingdata:
			linewriter.writerow([line["tweet_id"],line["tweet_text"],line["label"],line["topic"]])
	return trainingdata


def buildVocabulary(pptrainingdata):
	"""
		Build vocabulary with the set of all words in all the tweets
	"""

	all_words = []
	for (words,sentiment) in pptrainingdata:
		all_words.extend(words)
	
	word_list = FreqDist(all_words)
	word_features = word_list.keys()
	return word_features


def extractFeatures(tweet):
	"""
		Extracting features
		It checks presence/absence of tweet words in the vocabulary and returns true/false
	"""

	tweet_words = set(tweet)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = word in tweet_words
	return features



if __name__ == "__main__":

	search_term = input("Hi there ! What are we searching for today?")
	testdata_arr_dic = Test_data(search_term)

	corpus_file = "C:/Users/Satya/Downloads/sanders-twitter-0.2/sanders-twitter-0.2/corpus.csv"
	tweet_data_file = "C:/Users/Satya/Downloads/sanders-twitter-0.2/sanders-twitter-0.2/data_tweet.csv"

	limited_data = Training_data_limited(corpus_file,tweet_data_file)

	tweetprocessor = preprocesstweets()
	
	pptrainingdata = tweetprocessor.process_tweets(limited_data)
	pptestdata = tweetprocessor.process_tweets(testdata_arr_dic)
	
	word_features = buildVocabulary(pptrainingdata)
	
	trainingfeature = nltk.classify.apply_features(extractFeatures,pptrainingdata)
	
	NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingfeature)
	svmtrainingdata = [' '.join(tweet[0]) for tweet in pptrainingdata]
	
	vectorizer = CountVectorizer()
	x = vectorizer.fit_transform(svmtrainingdata).toarray()
	
	vocabulary = vectorizer.get_feature_names()
	swn_weights = []
	for word in vocabulary:
		try:

			synset = list(swn.senti_synsets(word))
			common_meaning = synset[0]
			if common_meaning.pos_score() >= common_meaning.neg_score():
				weight = common_meaning.pos_score()
			elif common_meaning.pos_score() == common_meaning.neg_score():
				weight = 1
			else:
				weight = -common_meaning.neg_score()
		except:
			weight = 1
		swn_weights.append(weight)
	swn_X = []
	for row in x:
		swn_X.append(np.multiply(row,np.array(swn_weights)))

	swn_X = np.vstack(swn_X)


	labels_to_array = {"positive":1,"negative":2}
	labels = [labels_to_array[tweet[1]] for tweet in pptrainingdata]
	y = np.array(labels)

	SVMclassifier = SVC()
	SVMclassifier.fit(swn_X,y)

	NBResultlabels = [NBayesClassifier.classify(extractFeatures(tweet[0])) for tweet in pptestdata]

	SVMResultlabels = []
	for tweet in pptestdata:
		tweet_sentence = ' '.join(tweet[0])
		svmFeatures = np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
		SVMResultlabels.append(SVMclassifier.predict(svmFeatures))

	if NBResultlabels.count('positive')>NBResultlabels.count('negative'):
		print("NB Result positive sentiment : " + str(100*NBResultlabels.count('positive')/len(NBResultlabels)) + "%")
	else:
		print("NB Result negative sentiment : " + str(100*NBResultlabels.count('negative')/len(NBResultlabels)) + "%")

	if SVMResultlabels.count(1)>SVMResultlabels.count(2):
		print("SVM Result Positive Sentiment : " + str(100*SVMResultlabels.count(1)/len(SVMResultlabels)) + "%")
	else:
		print("SVM Result negative Sentiment : " + str(100*SVMResultlabels.count(2)/len(SVMResultlabels)) + "%")
