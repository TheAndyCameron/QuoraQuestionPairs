import csv
from zipfData import zipfData
import numpy as np
import math

with open('../data/trainSmall.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)

    zipf = zipfData()

    count = 0

    #Collect zipf-y data
    for row in reader:
        count += 1
        zipf.addSentence(row[3])
        zipf.addSentence(row[4])

        if count%1000 == 0:
            print(str(count) + " rows")

    print("finished collecting!")
    f.close()

with open('../data/trainSmall.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    #get scores and compare to actual result

    sameScoreList = []
    diffScoreList = []
    
    for row in reader:
        score = zipf.getSentenceSimilarityScore(row[3],row[4])
        if row[5] == "1":
            sameScoreList.append(score)
        else:
            diffScoreList.append(score)
        
    #histograms yo!
    bins = [0,0.00001,0.000031,0.0001,0.00031,0.001,0.0031,0.01,0.025,0.05,0.075,0.1,0.2,0.3,0.4]
    bins2 = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,1]
    samehist = np.histogram(sameScoreList, bins)
    diffhist = np.histogram(diffScoreList, bins)

    print("same: " + str(samehist[0]))
    print("diff: " + str(diffhist[0]))

    print("\nSize of Arrays: " + str(len(sameScoreList)) + "\t" + str(len(diffScoreList)))

    ratio = len(diffScoreList)/len(sameScoreList)
    samehist2 = []
    for i in range(len(samehist[0])):
        samehist2.append(math.floor(samehist[0][i]*ratio))
    
    print("same wrt ratio: " + str(samehist2))
    print("diff:           " + str(diffhist[0]))

    print("Finished!")
    f.close()
