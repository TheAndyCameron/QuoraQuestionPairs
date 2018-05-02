from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize
import string

class wordPreper:
	ps = PorterStemmer()

	def removeStopWords(self, s):
		"""Takes an array of string words and removes stop words"""
		filtered_words = [word for word in s if word not in stopwords.words('english')]
		if "" in filtered_words: filtered_words.remove("")
		return filtered_words
		
	def removePunctuation(self, s):
		"""Removes all punctuation from a list of strings"""
		t = "".maketrans('','',string.punctuation)
		for i in range(len(s)):
			s[i] = s[i].translate(t)
		return s

	def stemWords(self, s):
		"""Takes a list of words and returns the list of stemmed words"""
		stemmedwords = [ps.stem(word) for word in s ]
		return stemmedwords


































