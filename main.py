import numpy as np
import pandas as pd
input_ner = 3
out_ner = 2
neur = 10

def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)

def sparse_cross_entropy(z, y):
    return -np.log((z[0, y]))

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def relu_deriv(t):
    return (t >= 0).astype(float)


data = pd.read_csv(r'C:\Users\Eugenix\PycharmProjects\untitled1\AAPL.csv')[:-1]
close_price = data.loc[:, 'Open'].tolist()
close_price = np.array(close_price)
open_price = data.loc[:, 'High'].tolist()
open_price = np.array(open_price)
high_price = data.loc[:, 'Adj Close'].tolist()
high_price = np.array(high_price)
a = np.column_stack([open_price, high_price])
a = np.column_stack([a, high_price])

bool_p = []

bool_p = close_price
mom = close_price[0]
for i in range(len(close_price)):
    if (mom < close_price[i]):
        bool_p[i] = 1
    else:
        bool_p[i] = 0

dataset = [(np.array([a[i]]), bool_p[i]) for i in range(len(a))]

W1 = np.random.rand(input_ner, neur)
b1 = np.random.rand(1, neur)
W2 = np.random.rand(neur, out_ner)
b2 = np.random.rand(1, out_ner)

alpha = 0.002
epoh = 4
luz = []

for j in range(epoh):
    for i in range(len(dataset)):
        x,y = dataset[i]
        y=int(y)


        # Прямое распространение
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax(t2)
        E = np.sum(sparse_cross_entropy(z, y))

        # Обратное
        y_full = to_full(y, out_ner)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = dE_dt1

        # Обновление весов
        W1 = W1 - alpha * dE_dW1
        b1 = b1 - alpha * dE_db1
        W2 = W2 - alpha * dE_dW2
        b2 = b2 - alpha * dE_db2

        luz.append(E)

def neuron(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z

def errors():
    correct = 0
    for x, y in dataset:
        z = neuron(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    err = correct / len(dataset)
    return err

osh = errors()
print("Ошибооочки:", osh)
import matplotlib.pyplot as plt
plt.plot(luz)
plt.show()

m = np.array([35.74    , 35.84    , 33.43    ])
g = neuron(m)
gadalka = np.argmax(g)
class_names = ['UP', 'DOWN']
print('Neuron class:', class_names[gadalka])
