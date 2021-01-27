#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

vector<string> stopwords = 
            {"most", "wouldn't", "he", "whom", "does", "didn't", "again", "into", "needn't", "am", "both", "that'll", "aren't", "shouldn't", "now", "is",
            "ma", "wasn't", "hasn't", "under", "here", "ain", "between", "don", "while", "couldn", "with", "there", "be", "until", "ours", "will", "such",
            "him", "mustn't", "wasn't", "can", "herself", "weren", "should've", "hadn't", "of", "had", "shan", "ve", "which", "re", "my", "when",
            "because", "himself", "through", "then", "couldn't", "few", "to", "just", "o", "you", "so", "don't", "yourself", "the", "but", "me", "you'd",
            "from", "she's", "it's", "do", "yourselves", "out", "below", "ll", "on", "your", "she", "how", "and", "won't", "hadn", "hers", "didn't", "themselves",
            "doesn", "been", "were", "at", "d", "over", "a", "more", "who", "should", "his", "isn't", "up", "before", "some", "any", "off", "have", "own", "our",
            "we", "not", "doesn't", "an", "aren", "mightn", "down", "other", "these", "it", "each", "i", "during", "if", "myself", "mustn", "they", "same", "them",
            "very", "wouldn't", "theirs", "weren't", "nor", "haven", "needn", "having", "shan't", "doing", "all", "than", "as", "by", "no", "only", "their", "that",
            "shouldn", "against", "won", "its", "her", "m", "did", "y", "mightn't", "you'll", "above", "are", "being", "itself", "once", "s", "you've", "was", "what",
            "yours", "haven't", "further", "after", "where", "this", "ourselves", "about", "you're", "for", "those", "hasn't", "has", "or", "why", "too", "in", "isn't"};
//this list of stopwords is taken from NLTK library

struct Node {
    vector<string> content;
    string origional;
    char ending;
    vector<float> adjweight;
    float weightsum = 0.0;
    float score = 0.5;
    Node(string str, char e) {
        string tmp= str;
        for (int i = 0; i < tmp.length(); ++i) {
            if (!isalnum(tmp[i]) && !isspace(tmp[i]) && tmp[i] != '\'') {
                tmp.erase(i, 1);
                i--;
            }
        }
        string s;
        stringstream ss(tmp);
        for (; ss >> s;)
            content.push_back(s);
        for (int i = 0; i < content.size(); ++i) {
            for (int j = 0; j < content[i].size(); ++j) {
                content[i][j] = tolower(content[i][j]);
            }
            if (find(stopwords.begin(), stopwords.end(), content[i]) != stopwords.end()) {
                content.erase(content.begin() + i);
                i--;
            }
        }

        origional = str;
        ending = e;
    }
};

bool isEndofSentence(string str) {
    char last = str[str.length() - 1];
    if (last == '.' || last == '!' || last == '?') {
        return true;
    }
    return false;
}
int wordCount(string str) {
    int count = 0;
    string s;
    stringstream ss(str);
    for (; ss >> s;) {
        count++;
    }
    return count;
}
float distance(Node a, Node b) {
    int count = 0;
    for (int i = 0; i < a.content.size(); ++i) {
        for (int j = 0; j < b.content.size(); ++j) {
            if (a.content[i] == b.content[j]) {
                count++;
            }
        }
    }
    return (float)count / ((float)a.content.size() * (float)b.content.size());
}

class Graph {
private:
    vector<Node> nodes;
    
public:
    void addNode(string str, char e) {
        int pos = nodes.size();
        nodes.push_back(Node(str, e));
        for (int i = 0; i < nodes.size() - 1; ++i) {
            float weight = distance(nodes[i], nodes[pos]);
            nodes[i].adjweight.push_back(weight);
            nodes[i].weightsum += weight;
            nodes[pos].adjweight.push_back(weight);
            nodes[pos].weightsum += weight;
        }
        nodes[pos].adjweight.push_back(0);
    }
    Node& getNode(int i) {
        return nodes[i];
    }
    int getNum() {
        return nodes.size();
    }
    void calcScore() {
        float th = 0.0001;
        float beta = 0.875;
        float prev = 1.0;
        float current = 100000.0;
        while (abs(current - prev) > th) {
            for (int i = 0; i < nodes.size(); ++i) {
                float tmp = 0;
                for (int j = 0; j < nodes.size(); ++j) {
                    tmp += nodes[j].score * nodes[j].adjweight[i] / nodes[j].weightsum;
                }
                nodes[i].score = 1 - beta + beta * tmp;
            }
            prev = current;
            current = nodes[0].score;
        }
    }
 
    void printTop(int words) {
        int count = 0;
        vector<pair<float, int>> list;
        for (int i = 0; i < nodes.size(); ++i) {
            list.push_back(pair<float, int>(nodes[i].score, i));
        }
        sort(list.begin(), list.end());
        reverse(list.begin(), list.end());
        string result = "";
        for (int i = 0; i < nodes.size(); ++i) {
            string next = nodes[list[i].second].origional + nodes[list[i].second].ending + " ";
            if (count + wordCount(next) > words) {
                break;
            }
            count += wordCount(next);
            result += (next);
        }
        cout << result;

    }
};

int main()
{
    int wordcount = 0;
    string tmp;
    string sentence = "";
    Graph graph;

    while (cin >> tmp) {
        wordcount++;
        if (sentence.length() == 0) {
            sentence += tmp;
        } else {
            sentence += ' ';
            sentence += tmp;
        }
        
        if (isEndofSentence(tmp)) {
            char e = sentence[sentence.length() - 1];
            sentence.pop_back();
            graph.addNode(sentence, e);
            sentence = "";
        }
    }
    graph.calcScore();
    graph.printTop(wordcount > 2500 ? 500 : (wordcount / 5));
}
