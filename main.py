import numpy as np
import csv

"""
loads the csv file 

filename: string, the filename of csv, has to include ".csv"
has_label: boolean, true if the data contains label to verify prediction

"""

def load_file(filename, has_label=True):
    data = np.genfromtxt(filename, dtype=np.str, delimiter=",")
    data = data.astype(float) # sets the type of values to float

    if has_label:
        label = data[:, :1] # extract labels from data
        data = np.delete(data, 0, axis=1) # delete the labels
        for i in label:
            if i[0] == 3: i[0] = 1 # changes label 3 to 1, 5 to -1
            else: i[0] = -1;

    data = np.insert(data, 0, values=1.0, axis=1) # insert bias term

    if has_label:
        return data, label
    else:
        return data

"""
trains and outputs the trained weight using online perceptron

data: matrix, the data used for training
label: matrix, the label (actual y) of the data
iters: integer, the number of iterations to train the model
"""

def online_perceptron(data, label, iters):
    w = np.zeros((len(data[0]), 1)) # 785*1 matrix

    for iter in range(iters):
        for i in range(len(data)):
            x = np.matrix(data[i]) # 1*785 matrix
            g = np.matmul(x, w)  # 1*1 matrix

            u = 1 if g >= 0 else -1 # the prediction for each i

            if label[i][0] * u <= 0: # if prediction was wrong
                a = np.dot(label[i], x) # 1*785 matrix
                w = w + a.T # updates the w

    return w



"""
generates predictions using weights trained by online perceptron method

data: matrix, the data used for prediction
w: matrix, trained weights
"""

def online_predict(data, w):
    predictions = [] # an array to store predictions
    for i in range(len(data)):
        x = np.matrix(data[i]) # this makes sure the 1 dimensional array behaves like a matrix
        predict = 1 if np.matmul(x, w) >=0 else -1
        predictions.append(predict)
    return predictions




"""
trains and outputs the trained weight using average perceptron

data: matrix, the data used for training
label: matrix, the label (actual y) of the data
iters: integer, the number of iterations to train the model
"""
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

            if label[i][0] * u <= 0: # if the prediction is wrong
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

"""
generates predictions using weights trained by average perceptron method

data: matrix, the data used for prediction
w: matrix, trained weights
"""
def average_predict(data, w):
    return np.sign(np.dot(data, w))

"""
trains and outputs the trained weight using kernel perceptron

data: matrix, the data used for training
label: matrix, the label (actual y) of the data
iters: integer, the number of iterations to train the model
p: the dimensions of the kernel method
"""
def kernel_perceptron(data, label, iters, p):
    a = np.zeros((len(data), 1))  # 4888*1 matrix
    K = np.power((np.matmul(data, data.T) + 1 ), p)
    for iter in range(iters):
        for i, (k, y) in enumerate(zip(K, label)):
            u = np.sign(np.dot(k, np.multiply(label, a)))
            if y[0] * u <= 0:
                a[i] += 1
    return a

"""
generates predictions using weights trained by kernel perceptron method

train_x: matrix, the data set from training
test_x: matrix, the data set from test
train_y: matrix, true y values for the training set
a:  matrix, the alpha matrix
p: integer, number of dimensions in the kernel function
"""
def kernel_predict(train_x, test_x, train_y, a, p):
    K = np.power((np.matmul(test_x, train_x.T) + 1), p)
    return np.sign(np.matmul(K, np.multiply(train_y, a)))


"""
loads train and valid data
"""
train, label_train = load_file("pa2_train.csv", has_label=True)
valid, label_valid = load_file("pa2_valid.csv", has_label=True)


"""
checks the prediction and calculate accuracy

predict: matrix, the predictions without label
label: matrix, the true y values
"""
def check_predictions(predict, label):
    error = 0 # counts the number of wrong predictions
    for i in range(len(label)):
        if predict[i] != label[i][0]: error += 1
    return 1 - error/len(label) # returns the percentage of correct predictions


"""
prepare to write out csvs
"""
with open('accuracy.csv', mode='w') as out:
    write = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    print("------------------ONLINE")
    write.writerow(['i','Train','Valid'])
    for i in range(16):
        train_w = online_perceptron(train, label_train, i)
        p_train = online_predict(train, train_w)
        p_valid = online_predict(valid, train_w)
        acc_train = round(check_predictions(p_train, label_train) * 100, 3)
        acc_valid = round(check_predictions(p_valid, label_valid) * 100, 3)
        write.writerow([i, acc_train, acc_valid])
        print("i:", i, "Train: ",acc_train, "  Valid: ", acc_valid)

    print("------------------AVERAGE")
    write.writerow(['AVERAGE'])
    for i in range(16):
        train_w = average_perception(train, label_train, i)
        p_train = average_predict(train, train_w)
        p_valid = average_predict(valid, train_w)
        acc_train = round(check_predictions(p_train, label_train) * 100, 3)
        acc_valid = round(check_predictions(p_valid, label_valid) * 100, 3)
        write.writerow([i, acc_train, acc_valid])
        print("i:", i, "Train: ", acc_train, "  Valid: ", acc_valid)

    print("------------------KERNEL")
    ps = [1, 2, 3, 7, 15]
    write.writerow(['KERNEL'])
    for p in ps :
        print("  ------------------P: ", p)
        for i in range(16):
            a_train = kernel_perceptron(train, label_train, i, p)
            p_train = kernel_predict(train, train, label_train, a_train, p)
            p_valid = kernel_predict(train, valid, label_train, a_train, p)
            acc_train = round(check_predictions(p_train, label_train) * 100, 3)
            acc_valid = round(check_predictions(p_valid, label_valid) * 100, 3)
            write.writerow([p, i, acc_train, acc_valid])
            print("i:", i, "Train: ", acc_train, "  Valid: ", acc_valid)
            print("i:", i, "Train: ", round(check_predictions(p_train, label_train) * 100, 3), "  Valid: ", round(check_predictions(p_valid, label_valid) * 100, 3))


"""
predicitng test data
"""


test = load_file("pa2_test_no_label.csv", has_label=False)
w_train = online_perceptron(train, label_train, 10)
p_test = online_predict(test, w_train)
with open('oplabel.csv', mode='w') as out:
    write = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in p_test:
        write.writerow([i])

a_train = kernel_perceptron(train, label_train, 10, 3)
p_test_k = kernel_predict(train, test, label_train, a_train, 3)
with open('kplabel.csv', mode='w') as out:
    write = csv.writer(out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in p_test_k:
        write.writerow(i)


"""
for personal testing only
compare the difference between 2 training methods
"""
# def load (filename):
#     data = np.genfromtxt(filename, dtype=np.str, delimiter=",")
#     data = data.astype(float)
#     return data
#
# x1 = load('oplabel.csv')
# x2 = load('kplabel.csv')
#
# err = 0
# for i, _ in enumerate(x1):
#     # print(x1)
#     # print(x2[i])
#     if x1[i] != x2[i]:
#         err += 1
#
# print(err / len(x1))