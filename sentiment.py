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
	
	@st.cache()
	def sentiment(self,sentence):

		try:
			flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
			s = flair.data.Sentence(sentence)
			flair_sentiment.predict(s)
			total_sentiment = s.labels

			return total_sentiment

		except:
			print("flair sentiment error")
			return

	@st.cache()
	def pos(self,sentence):
		try:
			# make a sentence
			sentence = flair.data.Sentence(sentence)

			# load the NER tagger
			tagger = flair.models.SequenceTagger.load("flair/pos-english-fast")

			tagger.predict(sentence)

			entities = []
			# iterate over entities
			for entity in sentence.get_spans('pos'):
			    entities.append(entity)

			return (sentence,entities)

		except:
			print("flair POS error")
			return


	@st.cache()
	def ner(self,sentence):
		try:
			# make a sentence
			sentence = flair.data.Sentence(sentence)

			# load the NER tagger
			tagger = flair.models.SequenceTagger.load('flair/ner-english')

			tagger.predict(sentence)

			entities = []
			# iterate over entities
			for entity in sentence.get_spans('ner'):
			    entities.append(entity)

			return (sentence,entities)

		except:
			print("flair NER error")
			return
