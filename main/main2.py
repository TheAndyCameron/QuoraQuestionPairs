import csv
import sys
sys.path.insert(0, '../helpers')
sys.path.insert(0, '../Word2Vec')
sys.path.insert(0, '../aligning')
sys.path.insert(0, '../neuralNetFramework')
import wordPrep
import wordvecmodule as wvm
import align
import nnet
import statshelp

nn = nnet.MLP([3,3,1])
print("training NN...")

#train NN
for epoch in range(0,5):
    with open('../main/NNtraining.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        count = 0
        print(epoch)
        for row in reader:
            out = nn.feedForward([float(row[0]),float(row[1]),float(row[2])])
            nn.backProp(out, [float(row[3])])
    
            count = count + 1
            if count%1000 == 0:
                print(str(epoch) + " -- " + str(count))
    
        f.close()

print("finished training NN")

print("loading word vec...")
wvc = wvm.wordvecclassifier(useGooglewv=True)
print("done")

print("Now for tests!!")
    
with open('../data/trainSmall.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(f)
    
    with open('combinedout.csv', 'w', newline='') as out:
        writer = csv.writer(out, dialect='excel')
    
        count = 0
        for row in reader:
            v1,v2 = wvc.getEstimates(row[3],row[4])
            v3 = align.score(row[3],row[4])
        
            label = int(row[5])
        
            ffval = nn.feedForward([v1,v2,v3])
            logloss = statshelp.logloss(label, ffval[0])
            correct = ((ffval[0] > 0.3) == (label==1))
            
            row = [ffval[0], logloss, correct, label]
            writer.writerow(row)
    
            count = count + 1
            if count%100 == 0:
                print(count)
    
    
    
    
    









