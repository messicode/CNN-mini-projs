import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

#Q1A-Load mnist

def mnist_plot_dig(X,Y):
    for dig in range(10):
        ids=np.where(Y==dig)
        # img=X[ids[0][0]].reshape(28,28)
        # plt.imshow(img, cmap='gray')
        # plt.show()
    plt.tight_layout()

mnist=fetch_openml('mnist_784', version=1)
Xraw=mnist.data.values
Yraw=mnist.target.astype(int).values
# print(Xraw.shape[0])
mnist_plot_dig(Xraw,Yraw)

# Part b and c MPI

for d in [10, 50, 100, 200, 500]:
    M=np.random.rand(d,784) / (255*d)
    X=M @ Xraw.T
    # print(X.shape)  # d*70000

    Y=np.zeros((10,70000))#10*70000
    Y[Yraw,np.arange(70000)]=1
    print(Y.shape)
    X_prose=np.linalg.pinv(X.T)
    # print(X_prose.shape)#70000*d
    # print(Y.shape)
    W=X_prose @ Y.T #d*10
    # print(W.shape)
    fx=W.T@X
    # print(fx.shape)

    mse=np.sum(np.power((Y-fx),2)) / 70000 # square check
    preds=np.argmax(fx,axis=0)
    mstks=np.sum(preds!=Yraw)
    print(f"d={d}; MSE={mse}; Mistakes={mstks}")

#Part d-LMS
#Feature extractor 100d
M=np.random.rand(100,784) / (255*100)
X_n=M @ Xraw.T
Y_n=np.zeros((10,70000))#10*70000
Y_n[Yraw,np.arange(70000)]=1


W_n=np.zeros((100,10))
mses=[]
eta=0.001
for epoch in range(10):
    for i in range(70000):
        x=X_n[:,i].reshape(-1,1)
        y=Y_n[:,i].reshape(-1,1)
        fx_n=W_n.T@x #Prediction
        err=(y-fx_n)
        W_n+=eta*(x@err.T)
    fx_final=W_n.T@X_n
    mse_curr=np.sum(np.power((Y_n - fx_final), 2))/70000
    mses.append(mse_curr) #Epoch to mse map
preds_n=np.argmax(fx_final,axis=0)
mstks_n=np.sum(preds_n!=Yraw)
print("Number of mistakes with LMS and d=100: ",mstks_n)
plt.plot(range(10), mses, marker='o')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

# # Improvement eta=0.01 and epochs=50
# M=np.random.rand(100,784) / (255*100)
# X_n=M @ Xraw.T
# Y_n=np.zeros((10,70000))#10*70000
# Y_n[Yraw,np.arange(70000)]=1
#
#
# W_n=np.zeros((100,10))
# mses=[]
# eta=0.01
# ep=50
# for epoch in range(ep):
#     for i in range(70000):
#         x=X_n[:,i].reshape(-1,1)
#         y=Y_n[:,i].reshape(-1,1)
#         fx_n=W_n.T@x #Prediction
#         err=(y-fx_n)
#         W_n+=eta*(x@err.T)
#     fx_final=W_n.T@X_n
#     mse_curr=np.sum(np.power((Y_n - fx_final), 2))/70000
#     mses.append(mse_curr) #Epoch to mse map
# preds_n=np.argmax(fx_final,axis=0)
# mstks_n=np.sum(preds_n!=Yraw)
# print("Number of mistakes with LMS and d=100: ",mstks_n)
# plt.plot(range(ep), mses, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.show()