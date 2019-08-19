import numpy as np
from dlgo.nn.load_mnist import load_data
from dlgo.nn.layers import sigmoid_double

def average_digit(data, digit):
    filtered_data = [x[0] for x in data if np.argmax(x[1])==digit]
    filtered_array = np.asarray(filtered_data)
    return np.average(filtered_array, axis = 0)

train,test = load_data()
avg_eight = average_digit(train, 8)

from matplotlib import pyplot as plt

img = (np.reshape(avg_eight, (28,28)))
plt.imshow(img)
plt.show()

x_3 = train[2][0]
x_18 = train[17][0]

W = np.transpose(avg_eight)
print(np.dot(W,x_3))
print(np.dot(W,x_18))