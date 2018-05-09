import gensim
from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import stopwords
import csv
import sys
sys.path.insert(0, '../helpers')
import wordPrep


class wordvecclassifier:
	#flags:
	outputcsv = True
	
	model = 0;
	wp = wordPrep.wordPreper()
	
	def __init__(self, useGooglewv=False, outputcsv=False):
		self.outputcsv = outputcsv
		#set word2vec model (google one set to false due to mahoosive file
		self.model = gensim.models.KeyedVectors.load_word2vec_format('../wordVecData/GoogleNews-vectors-negative300.bin', binary=True) if useGooglewv else Word2Vec.load('training model')
	
	def prepareWords(self, s):
		s2 = self.wp.removePunctuation(s)
		s2 = self.wp.removeStopWords(s2)
		return s2

	def vectorise(self, s):
		#get vectors for each word
		arr = self.prepareWords(s.split(' '))
		vecs = []
		
		for w in arr:
			try:
				vecs.append(self.model.wv[w])
				#print(w)
			except:
				continue
		
		#average the vectors
		vec = vecs[0]
		for v in vecs:
			vec = vec+v
		
		vec = vec - vec[0]
		return vec/len(vecs)
		#return vec


	def getEstimates(self, s1, s2):
		"""Takes 2 raw sentences and gives the vector distance between them"""
		v1 = self.vectorise(s1)
		v2 = self.vectorise(s2)
	
		distance = np.linalg.norm(v1-v2)
	
		return distance

	def getVal(self,arr,i):
		return arr[i] if i < len(arr) else ''
	
	def _outputcsv(self):
		"""Best not use"""
		with open('output.csv', 'w') as out:
			writer = csv.writer(out, dialect='excel')
			for i in range(max(len(sameDists),len(diffDists))):
				row = [str(getVal(sameDists, i)), str(getVal(diffDists, i))]
				writer.writerow(row)
		

wvc = wordvecclassifier()
d = wvc.getEstimates("What is the capital of bangaldesh?", "What does a capital sigma look like?")
print(d)