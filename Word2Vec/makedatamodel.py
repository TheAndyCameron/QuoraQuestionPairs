from gensim.models import Word2Vec
import numpy as np
import csv
import sys
sys.path.insert(0, '../helpers')
import wordPrep

class modelmaker:
	fname = "training model"
	wp = wordPrep.wordPreper()
	
	
	def _prepareSentance(self, s):
		s1 = s.split(" ")
		s1 = self.wp.removePunctuation(s1)
		s1 = self.wp.removeStopWords(s1)
		return s1
	
	def makeModel(self, datasetpath):
		with open('../data/trainSmall.csv', 'r', encoding='utf-8') as f:
			sentences = []
			reader = csv.reader(f)
			
			count = 0
			for row in reader:
				s1 = self._prepareSentance(row[3])
				s2 = self._prepareSentance(row[4])

				sentences.append(s1)
				sentences.append(s2)
				
				count = count +1
				if count%100 == 0:
					print(count)

			model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
			
			model.save(self.fname)
			

			
mm = modelmaker()
mm.makeModel('../data/trainSmall.csv')