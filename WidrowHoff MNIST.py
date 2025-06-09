# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
mnist.data.shape
Xraw = mnist.data.astype('float32')
Yraw = mnist.target.astype('int64')
Xraw = mnist.data.astype('float32')
Yraw = mnist.target.astype('int64')
for k in range(10):
  plt.subplot(2,5,k+1)
  for j in range(70000):
    if Yraw[j]==k:
      plt.imshow(Xraw[j,:].reshape(28,28))
      break
plt.show()

Xraw = mnist.data.astype('float32')
Yraw = mnist.target.astype('int64')
for k in range(10):
  plt.subplot(2,5,k+1)
  for j in range(70000):
    if Yraw[j]==k:
      plt.imshow(Xraw[j,:].reshape(28,28))
      break
plt.show()


def mistakes(X,Y,W):
  Ypred_onehot = np.matmul(W,X)
  Ypred = np.zeros((70000,))
  m = 0
  for i in range(70000):
    Ypred[i] = np.argmax(Ypred_onehot[:,i])
    m += (Ypred[i]!=Yraw[i])
  return m


for d in (10, 50, 100, 200, 500):
  M = np.random.uniform(0,1,(d,28*28))
  X = np.zeros((d,70000))
  Y = np.zeros((10,70000))
  for i in range(70000):
    X[:,i] = np.matmul(M,np.transpose(Xraw[i,:]))/255/d
    Y[Yraw[i],i] = 1
  # Moore-Penrose Pseudoinverse
  # W = np.matmul(np.matmul(Y,np.transpose(X)),np.linalg.inv(np.matmul(X,np.transpose(X))))
  W = np.matmul(Y,np.linalg.pinv(X))
  MSE = np.sum(np.power(Y - np.matmul(W,X),2))
  m = mistakes(X,Y,W)
  print("d = ", d , ", MSE = ", MSE, " # mistakes = ", m)

  # d =  10 , MSE =  52029.49517530712  # mistakes =  35913
  # d =  50 , MSE =  36843.312471295925  # mistakes =  15458
  # d =  100 , MSE =  32561.208371052482  # mistakes =  12302
  # d =  200 , MSE =  29559.368966579987  # mistakes =  10940
  # d =  500 , MSE =  27594.135464287912  # mistakes =  10413
  # Random guess => we'll be off 90% of the time, so we'd do 63000 mistakes.
  # All these numbers are much better than that.
  # It makes sense to stop at d=200, since the additional computation
  # beyond that is not worth it.

d = 100
M = np.random.uniform(0,1,(d,28*28))
X = np.zeros((d,70000))
Y = np.zeros((10,70000))
for i in range(70000):
  X[:,i] = np.matmul(M,np.transpose(Xraw[i,:]))/255/d
  Y[Yraw[i],i] = 1



# Widrow-Hoff algorithm
eta = 0.01
# choice = "zero"
choice = "randomized"
if choice == "zero":
  W = np.zeros((10,d))
elif choice == "randomized":
  W = np.random.normal(0,1,(10,d))/np.sqrt(d)
L = 50
mse_array = np.zeros((L,))
m_array = np.zeros((L,))
for j in range(L):
  for i in range(70000):
    x = X[:,i].reshape(d,1)
    y = Y[:,i].reshape(10,1)
    W = W + eta*(y-np.matmul(W,x))*np.transpose(x)
  mse_array[j] = np.sum((Y-np.matmul(W,X))**2)
  m_array[j] = mistakes(X,Y,W)

plt.plot(mse_array)
plt.title("MSE vs. epochs")
plt.show()

print("Widrow-Hoff with " + choice + " initializations and eta = ",eta,":")
print("Number of mistakes after", L, "epochs =", m_array[-1])

plt.plot(m_array)
plt.title("# of mistakes vs. epochs")
plt.show()

# Note that with 10 epochs and eta=0.001, we're not as good as Moore-Penrose,
# when we initialize at 0.
# With 100 epochs and eta = 0.05, we get much closer.
# Initializing W to small random numbers, instead of 0, can help.
# Pitfalls: If we try to pick eta too big (say 0.1), we don't converge.



