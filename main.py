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


def online_perceptron(data, label, iters):
    w = np.zeros((len(data[0]), 1)) # 785*1 matrix

    for iter in range(iters):
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
        # print(len(predict), len(predict[i]), len(label[i]))
        if predict[i] != label[i][0]: error += 1
    return 1 - error/len(label)

def average_perception(data, label, iters):
    w = np.zeros((len(data[0]), 1))  # 785*1 matrix
    w_ = np.zeros((len(data[0]), 1))  # 785*1 matrix
    c = 1
    s = 0

    for iter in range(iters):
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

    return w_

def average_predict( x, w):
    return np.sign(np.dot(x, w))


def kernel_perceptron(data, label, iters, p):
    a = np.zeros((len(data), 1))  # 4888*1 matrix
    K = np.power((np.matmul(data, data.T) + 1 ), p)
    for iter in range(iters):
        for i, (k, y) in enumerate(zip(K, label)):
            k = np.matrix(k)
            # print(k.shape, label.shape, a.shape)
            u = np.sign(np.dot(k, np.multiply(label, a)))
            if y[0] * u <= 0:
                a[i] += 1
    return a

def kernel_predict(train_x, test_x, train_y, a, p):
    K = np.power((np.matmul(test_x, train_x.T) + 1), p)
    return np.sign(np.matmul(K, np.multiply(train_y, a)))

train, label_train = load_file("pa2_train.csv", has_label=True)
# train_w = online_perceptron(train, label_train, 15)
valid, label_valid = load_file("pa2_valid.csv", has_label=True)
# p_valid = online_predict(valid, train_w)
# print(check_predictions(p_valid, label_valid))

def check_kernel_predictions(predict, label):
    error = 0
    for i in range(len(label)):
        # print(len(predict), len(predict[i]), len(label[i]))
        if predict[i][0] != label[i][0]: error += 1
    return 1 - error/len(label)

# print("------------------ONLINE")
#
# for i in range(16):
#     train_w = online_perceptron(train, label_train, i)
#     p_train = online_predict(train, train_w)
#     p_valid = online_predict(valid, train_w)
#     print("i:", i, "Train: ",round(check_predictions(p_train, label_train) * 100, 3), "  Valid: ", round(check_predictions(p_valid, label_valid) * 100, 3))
#
# print("------------------AVERAGE")

# for i in range(16):
    # train_w = average_perception(train, label_train, i)
    # p_train = average_predict(train, train_w)
    # p_valid = average_predict(valid, train_w)
    # print("i:", i, "Train: ", round(check_predictions(p_train, label_train) * 100, 3), "  Valid: ", round(check_predictions(p_valid, label_valid) * 100, 3))

print("------------------KERNEL")
ps = [1, 2, 3, 7, 15]
for p in ps :
    print("  ------------------P: ", p)
    for i in range(16):
        a_train = kernel_perceptron(train, label_train, i, p)
        p_train = kernel_predict(train, train, label_train, a_train, p)
        print("i:", i, "Train: ", round(check_predictions(p_train, label_train) * 100, 3))