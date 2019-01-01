import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#height
x_data = np.asarray([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183], np.float32)
#weight
y_data = np.asarray([49 , 50 , 51 , 54 , 58 , 59 , 60 , 62 , 63 , 64 , 66 , 67 , 68], np.float32)


# x_data = np.random.rand(13).astype(np.float32)
# y_data = x_data * 2 + -32

# print(x_data, y_data)
# print(type(x_data), type(y_data))

#init weight and bias
a = tf.Variable(1.0)
b = tf.Variable(2.0)
y = a * x_data + b

print(y)

#loss function
loss = tf.reduce_mean(tf.square(y - y_data))

#with this data recommanding using adam optimizer
optimizer = tf.train.AdamOptimizer(0.9)
# optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#training data and need 5000 epochs
train_data = []
for step in range(5000):
    evals = sess.run([train, a, b, loss])[1:]
    if step % 500 == 0:
        print(step, evals)
        train_data.append(evals)

for f in train_data:

    [a, b] = f[:-1]

    x0 = x_data
    y0 = b + a*x0

    # plt.plot(x0, y0)


plt.plot(x_data, y_data, 'ro')
plt.plot(x0, y0)

plt.xlabel("X")
plt.ylabel('Y')

plt.show()

#test with x = 160, expected result = 56
[a, b] = train_data[-1][:-1]
print(b + a*160)
