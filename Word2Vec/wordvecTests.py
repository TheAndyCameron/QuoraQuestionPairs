import gensim
import numpy as np
from nltk.corpus import stopwords
import csv

s1 = "what should I do be great geologist"
s2 = "how can I be good geologist"
s3 = "what was your first sexual experience"

model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)

def removeStopWords(s):
	filtered_words = [word for word in s if word not in stopwords.words('english')]
	if "" in filtered_words: filtered_words.remove("")
	return filtered_words

def cleanword(w):
	w = w.replace("?", "")
	w = w.replace(".","")
	w = w.replace(",","")
	w = w.replace(";","")
	w = w.replace(":","")
	w = w.replace("(","")
	w = w.replace(")","")
	w = w.replace("\"","")
	w = w.replace("'","")
	return w

def vectorise(s):
	#get vectors for each word
	arr = removeStopWords(s.split(' '))
	vecs = []
	
	for w in arr:
		#print(w)
		w = cleanword(w)
		try:
		    vecs.append(model.wv[w])
		except KeyError:
		    #ignore the error
		    continue

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
            print(str(e))
            continue
        except IndexError:
            continue
		
        if row[5] == "1":
            sameDists.append(np.linalg.norm(v1-v2))
        else:
            diffDists.append(np.linalg.norm(v1-v2))
		
		
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
