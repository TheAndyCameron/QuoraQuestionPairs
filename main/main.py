import csv
import sys
sys.path.insert(0, '../helpers')
sys.path.insert(0, '../Word2Vec')
sys.path.insert(0, '../aligning')
import wordPrep
import wordvecmodule as wvm
import align

wvc = wvm.wordvecclassifier(useGooglewv=True)


with open('../data/train.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(f)
	
    wvvals1 = [[],[]]
    wvvals2 = [[],[]]
    alvals = [[],[]]
	
    count = 0
    
    with open('output3.csv', 'w', newline='') as out:
        writer = csv.writer(out, dialect='excel')
        for row in reader:
            wvValue1, wvValue2 = wvc.getEstimates(row[3],row[4])
            alValue = align.score(row[3],row[4])
        
            wvvals1[int(row[5])].append(wvValue1)
            wvvals2[int(row[5])].append(wvValue2)
            alvals[int(row[5])].append(alValue)
		
            count = count +1
            if count%100 == 0:
                print(count)
                #break
            
            row = [str(wvValue1), str(wvValue2), str(alValue), row[5]]
            writer.writerow(row)