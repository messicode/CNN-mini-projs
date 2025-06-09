import numpy as np
import matplotlib.pyplot as plt

# Data Generation
N = 1000
x1 = np.random.uniform(-2,2,1000)
x2 = np.random.uniform(-2,2,1000)
z1 = np.heaviside(x1-x2+1,1)
z2 = np.heaviside(-x1-x2+1,1)
z3 = np.heaviside(-x2-1,1)
y = np.heaviside(z1+z2-z3-1.5,1)
X = np.vstack((x1,x2))
Y = y.reshape((1,1000))
# plt.scatter(x1,x2,c=y,cmap='viridis')
# plt.show()

# Question 2(a)
def phi(v):
  return 1./(1.+np.exp(-5*v))

def phi_prime(v):
  return 5*phi(v)*(1-phi(v))

# Number of neurons in the hidden layer (to experiment with more than 3.)
k=3
# Question 2(d)
sigma=0.1
# Initialization
W = np.random.randn(k,X.shape[0])*sigma
b = np.random.randn(k,1)*sigma
U = np.random.randn(1,k)*sigma
c = np.random.randn(1,1)*sigma
eta = 0.01
epochs = 100
risk = np.zeros(epochs)
for epoch in range(epochs):
  for i in range(N):
    x = X[:,i].reshape(x.shape)
    y = Y[0,i].reshape(f.shape)
    # Forward - Question 2(b)
    v_z = np.matmul(W,x)+b
    z = phi(v_z)
    v_f = np.matmul(U,z)+c
    f = phi(v_f)
    # Backward - Question 2(c)
    dloss_df = -2*(y-f)
    delta_f = dloss_df * phi_prime(v_f)
    dloss_dz = np.matmul(np.transpose(U),delta_f)
    delta_z = dloss_dz * phi_prime(v_z)
    grad_W = np.matmul(delta_z, np.transpose(x))
    grad_b = delta_z * 1
    grad_U = np.matmul(delta_f, np.transpose(z))
    grad_c = delta_f * 1
    # SGD 
    W = W - eta * grad_W
    b = b - eta * grad_b
    U = U - eta * grad_U
    c = c - eta * grad_c
  Y_predicted = phi(np.matmul(U,phi(np.matmul(W,X)+b))+c)
  risk[epoch] = np.sum((Y-Y_predicted)**2)/N
plt.plot(risk)
plt.show()

# Question 2(e)
fig = plt.figure(figsize = (7, 10))
ax = plt.axes(projection ="3d")
ax.scatter3D(x1, x2, Y_predicted, color = "green")
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
ax.zaxis._axinfo['juggled'] = (2, 2, 1)
plt.show()


# Question 2(f)
def phi(v):
  return 1./(1.+np.exp(-10*v))
def phi_prime(v):
  return 10*phi(v)*(1-phi(v))

sigma=0.1
# Initialization
W = np.random.randn(k,X.shape[0])*sigma
b = np.random.randn(k,1)*sigma
U = np.random.randn(1,k)*sigma
c = np.random.randn(1,1)*sigma
# W=W*0 # Zero initialization
# b=b*0
# U=U*0
# c=c*0
eta = 0.1
epochs = 100
batch_size = 5
risk = np.zeros(epochs)
grad_W = 0*W
grad_b = 0*b
grad_U = 0*U
grad_c = 0*c
for epoch in range(epochs):
  for i in range(N):
    x = X[:,i].reshape(x.shape)
    y = Y[0,i].reshape(f.shape)
    # Forward
    v_z = np.matmul(W,x)+b
    z = phi(v_z)
    v_f = np.matmul(U,z)+c
    f = phi(v_f)
    # Backward
    dloss_df = -2*(y-f)
    delta_f = dloss_df * phi_prime(v_f)
    dloss_dz = np.matmul(np.transpose(U),delta_f)
    delta_z = dloss_dz * phi_prime(v_z)
    grad_W += np.matmul(delta_z, np.transpose(x))
    grad_b += delta_z * 1
    grad_U += np.matmul(delta_f, np.transpose(z))
    grad_c += delta_f * 1
    # Minibatch SGD
    if (i+1)%batch_size==0:
      W = W - eta * grad_W/batch_size # Averaged
      b = b - eta * grad_b/batch_size # gradients
      U = U - eta * grad_U/batch_size
      c = c - eta * grad_c/batch_size
      grad_W = 0*W # Zeroing out the gradients
      grad_b = 0*b
      grad_U = 0*U
      grad_c = 0*c      
  Y_predicted = phi(np.matmul(U,phi(np.matmul(W,X)+b))+c)
  risk[epoch] = np.sum((Y-Y_predicted)**2)/N
  if epoch > 1:
    if risk[epoch]>risk[epoch-1]:
      eta = eta*0.9
plt.plot(risk)
plt.show()

# Visualization
fig = plt.figure(figsize = (7, 10))
ax = plt.axes(projection ="3d")
ax.scatter3D(x1, x2, Y_predicted, color = "green")
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
ax.zaxis._axinfo['juggled'] = (2, 2, 1)
plt.show()
