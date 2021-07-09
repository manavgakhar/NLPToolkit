import flair
import re

class nlpToolkit:

	def __init__(self):
		pass


	def clean_text(self,text):
		# lower case
		text = text.lower()
		# remove extra whitespaces
		text = re.sub("s+"," ", text)
		# remove punctuations
		text = re.sub("[^-9A-Za-z ]", "" , text)

		return text

	def sentiment(self,sentence):

		flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
		s = flair.data.Sentence(sentence)
		flair_sentiment.predict(s)
		total_sentiment = s.labels

		return total_sentiment
	
