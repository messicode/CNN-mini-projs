import numpy as np
import matplotlib.pyplot as plt

def step(v):
    return np.where(v>=0,1,0)

# for i in range(100):
#     print(f"Point {i+1}: x1 = {X[i, 0]:.2f}, x2 = {X[i, 1]:.2f}")

def neural_network(X,W,b,U,c):
    # Layer 1
    new_b=b.reshape((3,1))
    z=step(np.dot(W,X.T)+new_b) # broadcasting
    # Layer 2
    y=step(np.dot(U,z)+c)
    return y


X=np.random.uniform(-2,2,size=(1000,2))
W=np.array([[1,-1],[-1,-1],[0,-1]])
b=np.array([1,1,-1])
U=np.array([1,1,-1])
c=np.array([-1.5])
y=neural_network(X,W,b,U,c).flatten()


# Plotting


x1=X[:,0]
x2=X[:,1]
plt.scatter(x1[np.where(y == 1)], x2[np.where(y == 1)], c='red', marker='x',label='y = 1')
plt.scatter(x1[np.where(y == 0)], x2[np.where(y == 0)], c='blue', marker='o',label='y = 0')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Scatter plot")
plt.legend()
plt.grid(True)
plt.show()
