import glob
import re
import numpy as np

with open('data/ansar.pickle', 'rb') as handle:
    ansar = pickle.load(handle)
       
threads = getThreads(ansar)
message_files = glob.glob("messages/*.txt")

# get threads as np array, each element is a np array with posts inside
def getThreads(posts): # posts is a dataframe
    posts = posts.to_numpy()
    threads = []
    threadId = -1
    for i in range(posts.shape[0]):
        if posts[i][1] == threadId:
            threads[len(threads) - 1].append(posts[i])
        else:
            threadId = posts[i][1]
            threads.append([posts[i]])
    threads = np.asarray([np.array(thread) for thread in threads]) # convert 3d matrix to numpy array
    return threads

# read message files and remove O tagged words, save it into tags
# don't run if tags already exist
for file in message_files:
    save = ""
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.split()[1] is not 'O':
                save += line
    with open('tags/{}'.format(file.split('\\')[1]), 'w+') as f:
        f.write(save)
        
        
# calculate global frequencies(number of mentions of each entity)        
tag_files = glob.glob("tags/*.txt")
tag_sum = {}
for file in tag_files:
    with open(file, 'r') as f:
        for line in f.readlines():
            if line:
                if line in tag_sum:
                    tag_sum[line] += 1
                else:
                    tag_sum[line] = 1 # start from one, because we're trying to smooth the dist
                    
# create the empty structure that will hold frequencies between users                    
sims = {}
for index, row in ansar.iterrows():
    uid = row['MemberID']
    if uid not in sims:
        sims[uid] = {}
        
for uid in sims:
    for u in sims:
        if uid != u:
            sims[uid][u] = {}
            
# thread_groups is a dict which contains list of memberId's which contributed to each thread            
thread_groups = {}
for t in threads:
    group = []
    tid = t[0][1]
    for m in t:
        uid = m[3]
        if uid not in group:
            group.append(uid)
    thread_groups[tid] = group
    

    
# fill the sims dict, basically count number of mentions
for index, row in ansar.iterrows():
    uid = row['MemberID']
    tid = row['ThreadID']
    mid = row['MessageID']
    with open("tags/{}.txt".format(mid), 'r') as f:
        lines = f.readlines()
        for l in lines:
            for u in thread_groups[tid]:
                if uid != u:
                    if l in sims[uid][u]:
                        sims[uid][u][l] += 1
                    else:
                        sims[uid][u][l] = 1
                        
                    if l in sims[u][uid]:
                        sims[u][uid][l] += 1
                    else:
                        sims[u][uid][l] = 1

# scores will hold the actual weights
# create its structure first
scores = {}
for index, row in ansar.iterrows():
    uid = row['MemberID']
    if uid not in scores:
        scores[uid] = {}

# initialize NE weights with 0 
for uid in scores:
    for u in scores:
        if uid != u:
            scores[uid][u] = 0

# here, we do the weight calculation between users
for user1 in sims:
    for user2 in sims[user1]:
        for ne in sims[user1][user2]:
            freq = sims[user1][user2][ne]
            if freq > 2: # drop rare named entities, some urls are not stripped, this will eliminate them
                score = float(freq) / tag_sum[ne]
                scores[user1][user2] += score

# save the weights
with open('sim_scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)