import csv
import sys
sys.path.insert(0, '../helpers')
sys.path.insert(0, '../Word2Vec')
sys.path.insert(0, '../aligning')
import wordPrep
import wordvecmodule as wvm
import align

wvc = wvm.wordvecclassifier()


with open('../data/trainSmall.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(f)
	
    wvvals1 = [[],[]]
    wvvals2 = [[],[]]
    alvals = [[],[]]
	
    count = 0
    for row in reader:
        wvValue1, wvValue2 = wvc.getEstimates(row[3],row[4])
        alValue = align.score(row[3],row[4])
		
        print(wvvals1[int(row[5])])
	
        wvvals1[int(row[5])].append(wvValue1)
        wvvals2[int(row[5])].append(wvValue2)
        alvals[int(row[5])].append(alValue)
		
        count = count +1
        if count%100 == 0:
            print(count)
            break
            
	
    with open('output2.csv', 'w', newline='') as out:
        writer = csv.writer(out, dialect='excel')
        for i in range(len(wvvals1[0])):
            row = [str(wvvals1[0][i]), str(wvvals2[0][i]), str(alvals[0][i])]
            writer.writerow(row)
	
        writer.writerow([])
        for i in range(len(wvvals1[1])):
            row = [str(wvvals1[1][i]), str(wvvals2[1][i]), str(alvals[1][i])]
            writer.writerow(row)
