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
    t = 0

    for iter in range(iter):
        for i in range(len(data)):
            x = np.matrix(data[i]) # 1*785 matrix
            g = np.matmul(x, w)  # 1*1 matrix
            # print("X: ", x.shape," W: ", w.shape)

            u = 1 if g >= 0 else -1

            if label[i][0] * u <= 0:
                # temp = np.transpose(data[i])
                # print(len(data[i]), data[i][0])
                # print("label: ", label[i].shape, " X: ", x.shape)
                a = np.dot(label[i], x) # 1*785 matrix
                # print("A: ", a.shape)
                w = w + a.T
            # print("-------------------")
    return w

def online_predict(data, w):
    predictions = []
    for i in range(len(data)):
        x = np.matrix(data[i])
        # print(x.shape, w.shape)
        predict = 1 if np.matmul(x, w) >=0 else -1
        predictions.append(predict)
    return predictions

def check_predictions(predict, label):
    error = 0
    # print(len(label))
    # print(len(predict))
    for i in range(len(label)):
        if predict[i] != label[i][0]: error += 1
    return error/len(label)


train, label_train = load_file("pa2_train.csv", has_label=True)
train_w = online_perceptron(train, label_train, 15)
valid, label_valid = load_file("pa2_valid.csv", has_label=True)
p_valid = online_predict(valid, train_w)
print(check_predictions(p_valid, label_valid))