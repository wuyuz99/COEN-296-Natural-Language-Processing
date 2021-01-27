import nltk
import re
import os
#for text summarization
tokenizer = nltk.RegexpTokenizer(r"\w+")
stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.PorterStemmer()
class Node:
    def __init__(self, s):
        s = re.sub("\[[0-9]+\]", '', s)#remove any reference numbering
        self.origional = s
        self.content = tokenizer.tokenize(s)
        self.content = [w for w in self.content if not w in stopwords]
        for i in range(len(self.content)):
            self.content[i] = stemmer.stem(self.content[i])
        self.adjweight = []
        self.weightsum = 0.0
        self.score = 0.5

def distance(a, b):
    count = 0;
    for c in a.content:
        for d in b.content:
            if c == d:
                count = count + 1
    return float(count) / (float(len(a.content)) * float(len(b.content)))

class Graph:
    nodes = []

    def addNode(self, s):
        pos = len(self.nodes)
        self.nodes.append(Node(s))
        if(len(self.nodes[pos].content) == 0):
            self.nodes.pop()
            return
        for i in range(pos):
            weight = distance(self.nodes[pos], self.nodes[i])
            self.nodes[i].adjweight.append(weight)
            self.nodes[i].weightsum = self.nodes[i].weightsum + weight
            self.nodes[pos].adjweight.append(weight)
            self.nodes[pos].weightsum = self.nodes[pos].weightsum + weight
        self.nodes[pos].adjweight.append(0.0)

    def calcScore(self):
        #deal with totally irelevant ones
        n = 0
        i = 0
        while i + n < len(self.nodes):
            if self.nodes[i].weightsum == 0:
                self.nodes.pop(i)
                i = i - 1
                n = n + 1
            i = i + 1
        #calculate scores
        th = 0.0001
        beta = 0.875
        prev = 1.0
        current = 100000.0
        while abs(current - prev) > th:
            for i in range(len(self.nodes)):
                tmp = 0.0
                for j in range(len(self.nodes)):
                    tmp = tmp + self.nodes[j].adjweight[i] / self.nodes[j].weightsum
                self.nodes[i].score = 1 - beta + beta * tmp
            prev = current
            current = self.nodes[0].score

    def printMax(self, wc):
        c = 0
        arr = []
        for i in range(len(self.nodes)):
            arr.append((self.nodes[i].score, i))
        arr.sort(reverse = True)
        i = 0
        while c < wc:
            w = graph.nodes[arr[i][1]].origional
            c = c + len(w)
            i = i + 1


#main function
graph = Graph()
word = input()
searchword = word.replace(' ', '+')

address = "http://www.google.com/search?q='" + searchword +"'"
webpage = os.popen("lynx -accept_all_cookies -dump -width=10000 " + address).read()
ind = webpage.rfind("References")
content = webpage[:ind]
reference = webpage[ind:]

link = "[5]"
ref = "5."
ind = content.find("Showing results for")
if ind != -1: #if the input happends to be wrong
    content = content[ind+1:]
    link = "[7]"
    ref = "7."
    ind = content.find('\n')
    content = content[ind + 1:]
else:
    for i in range(4):
        ind = content.find('\n')
        content = content[ind + 1:]

ind = content.find(link)
answer = content[:ind]

if not(len(answer.strip()) == 0):#if the first is not a link to other site
    print(answer)
else:
    #find the first link
    ind = reference.find(ref)
    reference = reference[ind + 3:]
    ind = reference.find('\n')
    reference = reference[:ind]
    webpage = os.popen("lynx -accept_all_cookies -dump -width=10000 " + reference).read()
    #handle redurection of link
    ind = webpage.rfind("References")
    reference = webpage[ind:]
    ind = reference.find("1.")
    reference = reference[ind + 3:]
    ind = reference.find('\n')
    reference = reference[:ind]
    print("information was found on " + reference)
    print("the summary is:")
    webpage = os.popen("lynx -accept_all_cookies -dump -width=10000 " + reference).read()
    #handle the paper
    ind = webpage.rfind("References")
    webpage = webpage[:ind]
    ind = webpage.find("References")
    if ind != -1:
        webpage = webpage[:ind]
    ind = webpage.find("CONCLUSIONS")

    if ind != -1:
        webpage = webpage[ind:]
        ind = content.find('\n')
        content = content[ind + 1:]
        ind = content.find('\n')
        content = content[:ind  ]
        print(content)
    else:
        sent_text = nltk.sent_tokenize(webpage)
        wordCount = 0
        for s in sent_text:
            wordCount = wordCount + len(s)
            graph.addNode(s)
        graph.calcScore()
        wordCount = wordCount / 4
        if wordCount > 500:
            wordCount = 500
        graph.printMax(wordCount)
