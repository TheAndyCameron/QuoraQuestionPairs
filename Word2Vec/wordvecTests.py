import gensim
from gensim.models import Word2Vec
import numpy as np
from nltk.corpus import stopwords
import csv
import sys
sys.path.insert(0, '../helpers')
import wordPrep

#flags:
outputcsv = True



#model = gensim.models.KeyedVectors.load_word2vec_format('../wordVecData/GoogleNews-vectors-negative300.bin', binary=True)
model = Word2Vec.load('training model')
wp = wordPrep.wordPreper()

def prepareWords(s):
	s2 = wp.removePunctuation(s)
	s2 = wp.removeStopWords(s2)
	return s2

def vectorise(s):
	#get vectors for each word
	arr = prepareWords(s.split(' '))
	vecs = []
	
	for w in arr:
		vecs.append(model.wv[w])
		#print(w)
		
	#average the vectors
	vec = vecs[0]
	for v in vecs:
		vec = vec+v
		
	vec = vec - vec[0]
	return vec/len(vecs)
	#return vec


sameDists = []
diffDists = []

with open('../data/trainSmall.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    count = 0
    #Collect zipf-y data
    for row in reader:
        count += 1
		
        try:
            v1 = vectorise(row[3])
            v2 = vectorise(row[4])
        except KeyError as e:
            #print("ERROR!")
            #print(str(e))
            continue
        except IndexError:
            continue
		
        if row[5] == "1":
            sameDists.append(np.linalg.norm(v1-v2))
        else:
            diffDists.append(np.linalg.norm(v1-v2))#
		
		
        if count%100 == 0:
            print("\t\t\t\t\t\t\t" + str(count) + " rows")
			
    f.close()

#Means	
print("\nSame Lists Distance")
sm = sum(sameDists)/len(sameDists)
print(str(sm))
print("\nDiff lists Distance")
dm = sum(diffDists)/len(diffDists)
print(str(dm))
	
#Standard Deviation
print("\nSame Lists Standard Deviation")
sameDevs = []
for n in sameDists:
	sameDevs.append(abs(n-sm))
ssd = sum(sameDevs)/len(sameDevs)
print(str(ssd))

print("\nDiff Lists Standard Deviation")
diffDevs = []
for n in diffDists:
	diffDevs.append(abs(n-dm))
dsd = sum(diffDevs)/len(diffDevs)
print(str(dsd))



def getVal(arr,i):
	return arr[i] if i < len(arr) else ''
	
if outputcsv:
	with open('output.csv', 'w') as out:
		writer = csv.writer(out, dialect='excel')
		for i in range(max(len(sameDists),len(diffDists))):
			row = [str(getVal(sameDists, i)), str(getVal(diffDists, i))]
			writer.writerow(row)
		

