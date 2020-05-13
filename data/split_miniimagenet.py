import pickle
import numpy as np
import os

np.random.seed(1234)

# we want 500 for training, 100 for test for wach class
n = 500

def get_total(data):
    data_x, data_y = [], []
    for k, v in data.items():
        for i in range(len(v)):
            data_x.append(v[i])
            data_y.append(k)
    d = {}
    d['images'] = data_x
    d['labels'] = data_y
    return d


# loading the pickled data
with open(os.path.join('../data/miniimagenet/data.pkl'), 'rb') as f:
    data_dict = pickle.load(f)
data = data_dict['images']
labels = data_dict['labels']

# split data into classes, 600 images per class
class_dict = {}
for i in range(len(set(labels))):
    class_dict[i] = []

for i in range(len(data)):
    class_dict[labels[i]].append(data[i])

# Split data for each class to 500 and 100
x_train, x_test = {}, {}
for i in range(len(set(labels))):
    np.random.shuffle(class_dict[i])
    x_test[i] = class_dict[i][n:]
    x_train[i] = class_dict[i][:n]

# mix the data
d_train = get_total(x_train)
d_test = get_total(x_test)

with open(os.path.join('../data/miniimagenet/train.pkl'), 'wb') as f:
    pickle.dump(d_train, f)
with open(os.path.join('../data/miniimagenet/test.pkl'), 'wb') as f:
    pickle.dump(d_test, f)     