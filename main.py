import numpy as np
import csv

def load_file(filename, has_label=True):
    data = np.genfromtxt(filename, dtype=np.str, delimiter=",")
    data = data.astype(float)

    if has_label:
        label = data[:, :1] # extract labels
        data = np.delete(data, 0, axis=1) # delete the labels
        for i in label:
            if i[0] == 3: i[0] = 1
            else: i[0] = -1;

    data = np.insert(data, 0, values=1.0, axis=1) # insert bias term

    if has_label:
        return data, label
    else:
        return data


def online_perceptron(data, label, iter):
    w = np.zeros((len(data[0]), 1)) # 785*1 matrix

    for iter in range(iter):
        for i in range(len(data)):
            x = np.matrix(data[i]) # 1*785 matrix
            g = np.matmul(x, w)  # 1*1 matrix

            u = 1 if g >= 0 else -1

            if label[i][0] * u <= 0:
                a = np.dot(label[i], x) # 1*785 matrix
                w = w + a.T

    return w

def online_predict(data, w):
    predictions = []
    for i in range(len(data)):
        x = np.matrix(data[i])
        predict = 1 if np.matmul(x, w) >=0 else -1
        predictions.append(predict)
    return predictions

def check_predictions(predict, label):
    error = 0
    for i in range(len(label)):
        if predict[i] != label[i][0]: error += 1
    return 1 - error/len(label)

def average_perception(data, label, iter):
    w = np.zeros((len(data[0]), 1))  # 785*1 matrix
    w_ = np.zeros((len(data[0]), 1))  # 785*1 matrix
    c = 1
    s = 0

    for iter in range(iter):
        for i in range(len(data)):
            x = np.matrix(data[i])  # 1*785 matrix
            g = np.matmul(x, w)  # 1*1 matrix

            u = 1 if g >= 0 else -1

            if label[i][0] * u <= 0:
                if s + c > 0:
                    w_ = ((s * w_ + c * w) / (s + c))
                s += c
                a = np.dot(label[i], x)  # 1*785 matrix
                w = w + a.T
                c = 0
            else: c += 1
    if c > 0:
        w_ = ((s * w_ + c * w) / (s + c))

    return w_, c

def average_predict( x, w, c):
    return np.sign(np.dot(x, c*w))

train, label_train = load_file("pa2_train.csv", has_label=True)
# train_w = online_perceptron(train, label_train, 15)
valid, label_valid = load_file("pa2_valid.csv", has_label=True)
# p_valid = online_predict(valid, train_w)
# print(check_predictions(p_valid, label_valid))

for i in range(16):
    train_w = online_perceptron(train, label_train, i)
    p_valid = online_predict(valid, train_w)
    print("i:", i, "Accuracy",round(check_predictions(p_valid, label_valid) * 100, 3), " %")

print("------------------")

for i in range(16):
    train_w, c = average_perception(train, label_train, i)
    p_valid = average_predict(valid, train_w, c)
    print("i:", i, "Accuracy", round(check_predictions(p_valid, label_valid) * 100, 3), " %")