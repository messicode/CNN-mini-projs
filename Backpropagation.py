import numpy as np
import matplotlib.pyplot as plt

def smd(v,a=5):
    return ( 1 / (1+np.exp(-a*v)) )

def smd_drv(v,a=5):
    s=smd(v,a)
    return a*s*(1-s)

def forward_pass(X,W,b,U,c,a):
    #1st Layer
    # print(X.shape,W.shape,b.shape)
    vz=np.dot(W,X)+b
    # print(vz.shape)
    z=smd(vz,a)
    #2nd Layer
    # print(z.shape, U.shape, c.shape)
    vf=np.dot(U,z)+c
    # print(vf.shape)
    f=smd(vf,a)
    return vz,z,vf,f

def backward_pass(X,y,vz,z,vf,f,W,U,a):
    # print(X.shape)
    grad_f=2*(f[0][0]-y)
    delf=grad_f*smd_drv(vf,a)
    delz=delf*smd_drv(vz,a)
    # print(grad_f.shape,delf.shape,delz.shape)
    grad_W=np.dot(delz,X.T)
    grad_b=np.sum(delz,axis=0)
    grad_U=np.dot(delf,z.T)
    grad_c=np.sum(delf,axis=0)
    return grad_f,grad_W,grad_b,grad_U,grad_c

np.random.seed(3)
x1=np.random.uniform(-2,2,1000)
x2=np.random.uniform(-2,2,1000)
X=np.column_stack((x1,x2))#1000*2
# print(X.shape)
sd=0.1
W=np.random.normal(0,sd,(3,2))
b=np.random.normal(0,sd,(3,1))
U=np.random.normal(0,sd,(1,3))
c=np.random.normal(0,sd,(1,1))

z1=(x1-x2+1>=0)*1.
z2=(-x2-x1+1>=0)*1.
z3=(-x2-1>=0)*1.
y=np.heaviside(z1+z2-z3-1.5,1)

a=5
eta=0.01
epochs=100
mses=[]

for epoch in range(epochs):
    mse=0
    for i in range(1000):
        vz, z, vf, f=forward_pass(X[i].reshape(2,1),W,b,U,c,a)
        # print(vz.shape,z.shape,vf.shape,f.shape)
        mse+=(f[0][0]-y[i])**2
        # print("halo",mse.shape)
        grad_f,grad_W,grad_b,grad_U,grad_c=backward_pass(X[i].reshape(2,1),y[i],vz,z,vf,f,W,U,a)
        # print(grad_W.shape,grad_b.shape,grad_U.shape,grad_c.shape)
        W-=eta*grad_W
        b-=eta*grad_b
        U-=eta*grad_U
        c-=eta*grad_c
    mse/=1000
    mses.append(mse)


plt.plot(range(epochs),mses)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title(f'MSE vs Epoch for eta={eta}')


y_pred = []
for x in X:
    _, _, _, f = forward_pass(x, W, b, U, c,a)
    y_pred.append(f[0][0])

y_pred = np.array(y_pred)

# 3D Scatter plot
fig = plt.figure(figsize=(7, 10))
plt.title(f'3d Decision boundary scatter Plot for eta={eta}')
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y_pred, color='green')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
ax.zaxis._axinfo['juggled'] = (2, 2, 1)

#Hacks

plt.show()