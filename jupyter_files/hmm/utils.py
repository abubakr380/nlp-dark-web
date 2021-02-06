from collections import defaultdict
import sys
import math


# a dictionary record the entry
def word_count(file):
    d = defaultdict(int)
    for l in file:
        line = l.strip()
        if line:  #Nonempty line
            fields = line.split(" ")
            if fields[1] == "WORDTAG":
                word = fields[-1]
                d[word] = fields[0]
            else:
                break
    return d

# import a file, output a dictionary, which has (word, label) as its key and e as its value
def emission_parameter(file):
    # create a dict with label as its key and count as its value
    label_count = defaultdict(int)
    # first traversal, record Count(y) into label_count
    for l in file:
        line = l.strip() # remove unnecessary space
        if line: # Nonempty line
            fields = line.split(" ")
            if fields[1] != "1-GRAM":
                continue
            else:
                # the line has format count 1-GRAM label
                label_count[fields[2]] = int(fields[0])
    d = defaultdict(float)

    # go back to the top of file
    file.seek(0)
    # second traversal, create a dict whose value is emission parameter
    for l in file:
        line = l.strip()
        if line: # Nonempty line
            fields = line.split(" ")
            # finish reading the WORDTAG part, don't care the remaining
            if fields[1] != "WORDTAG":
                break
            else:# have the format: count WORDTAG label word
                word = fields[3]
                label = fields[2]
                # find count(y) and calculate emission parameter
                e = (int(fields[0])) * 1.0 / label_count[label]
                d[(word, label)] = e

    return d

# take a countFile such as freqs.txt as input, return a dict of q(y_n | y_n-2, y_n-1)
def trigram_prob(countFile):
    bigram = defaultdict(int)
    trigram = defaultdict(int)
    # first traversal, reform entries into dictionary
    for l in countFile:
        line = l.strip()
        if line: # nonempty
            fields = line.split(" ")
            # add into bigram dict
            if fields[1] == "2-GRAM":
                bigram[(fields[2], fields[3])] = int(fields[0])
            # add into trigram dict
            elif fields[1] == "3-GRAM":
                trigram[(fields[2], fields[3], fields[4])] = int(fields[0])
    for key in trigram:
        trigram[key] = trigram[key] * 1.0 / bigram[(key[0], key[1])]
    return trigram


def K(index, length):
    # K_-1 = K_-2 = {*}
    if index == -1 or index == -2:
        return ["*"]
    # K_k = K for k = 1...n
    if index <= length:
        return ["O", "I-ORG", "B-LOC", "I-LOC", "I-PER", "I-MISC", "B-MISC", "B-ORG"]

def viterbi(sentence, tri_d, e_d, word_d, caps):
    n = len(sentence)
    for index in range(n):
        # the word does not appeared before, treat it as _RARE_
        if word_d[sentence[index]] == 0:
            sentence[index] = '_RARE_'
    pi_dict = defaultdict(float)
    bp_dict = defaultdict()
    # Set pi(-1, *, *) = 0
    pi_dict[(-1, "*", "*")] = 0.0
    # For k = 0...n-1
    for k in range(n):
        for u in K(k - 1, n):
            for v in K(k, n):
                largest = float("-inf")
                for w in K(k-2, n):
                    # there should be no log0 case, since all words do not existed are replaced with _RARE_, if the word
                    # does exist, then we should use that emission parameter
                    if e_d[(sentence[k], v)]== 0:
                        sumPi = float("-inf")
                    # in case log0
                    # if (w, u, v) not in tri_d:
                    elif tri_d[(w, u, v)] == 0:
                        sumPi = float("-inf")
                    else:
                        sumPi = pi_dict[(k-1, w, u)]+math.log(tri_d[(w, u, v)], 2) + math.log(e_d[(sentence[k], v)],2)
                        # print sumPi, k, w, u, v, pi_dict[(k-1, w, u)], math.log(tri_d[(w, u, v)], 2) ,math.log(e_d[(sentence[k], v)],2)
                    if sumPi >= largest:
                        largest = sumPi
                        pi_dict[(k, u, v)] = largest
                        bp_dict[(k, u, v)] = w
    # calculate yn-1 yn
    maxProb = float("-inf")
    maxLLast = ""
    maxL2ndLast = ""
    for u in K(n - 2, n):
        for v in K(n-1, n):
            if (u, v, "STOP") not in tri_d:
                sumPi = float("-inf")
            else:
                sumPi = pi_dict[(n-1, u, v)] + tri_d[(u, v, "STOP")]
            if sumPi >= maxProb:
                maxProb = sumPi
                maxLLast = v
                maxL2ndLast = u
    y = defaultdict()
    y[n-1] = maxLLast
    y[n-2] = maxL2ndLast
    for k in reversed(range(n - 2)):
        y[k] = bp_dict[(k + 2, y[k + 1], y[k + 2])]
        if word_d[sentence[k]] == 0:
            # search in wnet
            if sentence[k] in caps:
                y[k] = 'I-MISC'
    res = []
    # return labels
    for k in range(n):
        res.append(y[k])
    prob = []
    y[-2] = "*"
    y[-1] = "*"
    # return accumulated possibility
    for k in range(1,n+1):
        prob.append(pi_dict[(k-1, y[k-2], y[k-1])])
    return res, prob



# parse the file into sentences according to empty line
def into_sentence(msg, tri_d, e_d, word_d, caps):
    ii = 0
    results = []
    sentence = []
    for w in msg:
        word = w.strip()
        if word:  #nonempty
            sentence.append(word.split(" ")[0])
        else:  #empty line, end of one sentence
            original_sentence = sentence[:]
            res, prob = viterbi(sentence, tri_d, e_d, word_d, caps)
            for index in range(len(sentence)):
                #output.write("{} {} {}\n".format(original_sentence[index], res[index], prob[index]))
                results.append((original_sentence[index], res[index]))
                ii += 1
                #if (ii % 100000 == 0):
                #    print(ii, '/', 6017776)
            sentence = []
            #output.write("\n")

    return results