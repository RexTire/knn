from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter
from numpy.core.numeric import full
import pandas as pd
import random

# test dataset
# dataset =  {'k':[[1,2], [2,3], [3,1]], 'r':[[6,5], [7,7], [8,6]]}
# new_feature = [1,1]

def knn(data, predict, k):
    if len(data) >= k:
        warnings.warn('K value is less than total voting groups')
    
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = np.sqrt( np.sum( (np.array(features) - np.array(predict))**2 ) )
            euclidean_distance = np.linalg.norm( np.array(features) - np.array(predict) )
            distances.append([euclidean_distance, group])
    
    votes = [i[1] for i in sorted(distances)[:k]]
    # visualize the algo
    # print('votes = ', votes)
    # print('distance = ' , distances)
    # print('sorted distance = ' , sorted(distances))
    # print('sorted distance = ' , sorted(distances)[:k])
    # print('counter(votes) = ' , Counter(votes))
    # print('Counter(votes).most_common() = ' , Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    # print(vote_result, confidence)
    return vote_result, confidence

# test dataset result
# result = knn(dataset, new_feature, 3)
# print(result)

# printing the result for test dataset
# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0], ii[1], s=100, color=i)

# plt.scatter(new_feature[0], new_feature[1], color=result)
# plt.show()

# real dataset

df = pd.read_csv(r'C:\Users\RexTire\Desktop\ML\KNN\breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
test_set = {2:[], 4:[]}
train_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# knn(data, predict, k)

for group in test_set:
    for data in test_set[group]:
        vote, confidence = knn(train_set, data, 5)
        if group == vote:
            correct += 1
        else:
            print(confidence)
        total += 1

accuracy = correct/total
print(accuracy)