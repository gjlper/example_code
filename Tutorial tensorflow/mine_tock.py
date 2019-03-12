import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preproccessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# reading the datashet
def read():
    df = pd.read_csv("path")
    X= df[df.columns[0:60]].values
    y = df[df.columns[60]]

    # encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return (X,Y)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    ohe = np.zeros((n_labels,n_unique_labels))
    ohe[np.arange(n_labels),labels] = 1
    return ohe


# read the datashet
X,Y = read()
# shuffle datashet
X,Y = shuffle(X,Y,random_state=1)

#convert the datashet
train_x,test_x,train_y,test_y = train_test_ssplit(X,Y,test_size=0.20,random_state=415)

# inspect elements
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

# define variables
learning_rate = 0.3
epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)
n_dim = X.shape[1]
n_class = 2

model_path = " "

# define the number of hidden layers and number of neurons for each
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60


x = tf.placeholder(tf.float32,[None,n_dim])
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32,[None,n_class])


# define model
def multilayer_perceptron(x,weights,biases):

    layer_1 = tf.add(tf.matmul(x,weights['h1'],biases['b1']))
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1,weights['h2'],biases['b2']))
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2,weights['h3'],biases['b3']))
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3,weights['h4'],biases['b4']))
    layer_4 = tf.nn.relu(layer_4)

    # output layer
    out_layer = tf.matmul(layer_4,weights['out']+biases['out'])

    return out_layer

#define weights and biases
weights = {
'h1': tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
'h2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
'h3': tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
'h4': tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
'out': tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))

}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}

# initialize all variables
init tf.global_variables_initializer()

saver = tf.train.Saver()


# call your model
y = multilayer_perceptron(x,weights,biases)

# define the cost
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# calculate the cost and the accuracy for each epoch

mse_history = []
accuracy_history = []

for ep in range(epochs):
    sess.run(training_step,feed_dict={x: train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x: train_x,y_:train_y})
    cost_history = np.append(cost_history,cost)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    pred_y = sess.run(y,feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y-test_y))
    mse_= sess.run(mse)
    mse_history.append(mse_)

    accuracy = (sess.run(accuracy, feed_dict={x:train_x,y_:train_y}))
    accuracy_history.append(accuracy)

    print('epoch : ',ep,'-','cost: ',cost,'-MSE: ',mse_,'-Train_accuracy: ',accuracy)

save_path = saver.save(sess=,model_path)
print("model saved in file: %s" %save_path)

#plot
plt.plot(mse_history,'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

# print the final accuracy
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("Test accuracy: ",sess.run(accuracy,feed_dict={x:test_x,y_:test_y}))

pred_y = sess.run(y,feed_dict={x:test_x})
mse = tf.reduce_mean(tf.square(pred_y-test_y))
print(" MSE: %.4f"%sess.run(mse))




# Restore code
#same code until line 117 then
saver.restore(sess,model_path)

pred = tf.argmax(y,1)
correct_prediction = tf.equal(prediction,tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

for i in range(93,101):
    pred_run = sess.run(pred,feed_dict={x: X[i].reshape(1,60)})
    accuracy_run = sess.run(accuracy, feed_dict={x: X[i].reshape(1,60),y_:??})
    print("Original class: ",y1[i],"Predicted Values: ",pred_run," Accuracy: ",accuracy_run)
