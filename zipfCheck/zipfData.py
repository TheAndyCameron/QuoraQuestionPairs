class zipfData:
    #words in wordlist are the ones seen so far, with corresponding
    #index in frequencylist
    wordlist = [];
    frequencylist = [];

    def addWord(self, word):
        try:
            i = self.wordlist.index(word)
            self.frequencylist[i] += 1
        except:
            self.wordlist.append(word)
            self.frequencylist.append(1)

    def addSentence(self, sentence):
        sentence = sentence.replace('?','')
        words = sentence.split(' ')
        for word in words:
            self.addWord(word)

    def getWordFrequency(self, word):
        try:
            i = self.wordlist.index(word)
            return self.frequencylist[i]
        except:
            return 0

    def getSentenceSimilarityScore(self, s1, s2):
        s1 = s1.replace('?','')
        s2 = s2.replace('?','')

        words1 = s1.split(" ")
        words2 = s2.split(" ")

        score = 0

        #matching words is good, infrequent words better
        for word in words1:
            if word in words2:
                freq = self.getWordFrequency(word)
                score += 1/freq

        #normalise score wrt length
        score = score / ((len(words1) + len(words2))/2)

        return score
