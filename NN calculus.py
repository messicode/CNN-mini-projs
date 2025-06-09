import numpy as np
import matplotlib.pyplot as plt


max_epochs=50
def step(v):
    return np.where(v>=0,1,0)

#Q2) a&c
def ptronAlgo(X,y,w,eta):
    errs=[]
    for epoch in range(max_epochs):# Stop after max epochs
        err=0
        for x in range(100): # 100 data points
            pred=step(np.dot(w,X[x]))
            if pred!=y[x]:
                w=w+(eta*X[x]*(y[x]-pred))
                err+=1
        errs.append(err)
        if err==0:
            break
    return w,errs


#Q1a
w_star0=np.random.uniform(-.25,.25)
w_star1=np.random.uniform(-1,1)
w_star2=np.random.uniform(-1,1)
w_star=np.array([w_star0,w_star1,w_star2]) # 3*1
print(f"Random Weight Vector:{w_star}")

x0=np.ones(100)
x2=np.random.uniform(-1,1,100)
x1=np.random.uniform(-1,1,100)

X=np.vstack([x0,x1,x2]).T
y=step(np.dot(w_star,X.T))
x=np.linspace(-1,1,100)

y_eqn=-(w_star0+(w_star1*x)) / w_star2

#Q2a
w=np.array([1,1,1])
w_ptron=[None]*3
errs={}
for i in range(3):
    w_ptron[i],errs[i]=ptronAlgo(X,y,w,0.1*pow(10,i))
print("Perceptron weights predicted for eta=1 and w.init=[1,1,1]: ",w_ptron[1])
y_ptron=-(w_ptron[1][0]+(w_ptron[1][1]*x))/w_ptron[1][2]



#Q2c-1000 points


xn0=np.ones(1000)
xn2=np.random.uniform(-1,1,1000)
xn1=np.random.uniform(-1,1,1000)
Xn=np.vstack([xn0,xn1,xn2]).T
yn=step(np.dot(w_star,Xn.T))
xn=np.linspace(-1,1,1000)
wn_ptron,err_n=ptronAlgo(Xn,yn,w,1)
yn_ptron=-(wn_ptron[0]+(wn_ptron[1]*xn))/wn_ptron[2]




#Plotting

plt.figure()
plt.scatter(x1[np.where(y == 1)], x2[np.where(y == 1)], c='red', marker='x',label='y = 1')
plt.scatter(x1[np.where(y == 0)], x2[np.where(y == 0)], c='blue', marker='o',label='y = 0')
plt.plot(x,y_eqn,label='Decision Boundary')

plt.quiver(0,0,w_star1,w_star2,color='black',label='w-vector')

plt.plot(x,y_ptron,label='Ptron Boundary')
plt.plot(xn,yn_ptron,label='1000 points Ptron Boundary')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Scatter plot")
plt.grid(True)
plt.legend()


#Q2bEpochs vs errors
plt.figure()
plt.title('Number of Errors vs. Epoch for Different Learning Rates')

for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(range(1, len(errs[i]) + 1), errs[i], label=f'eta = {0.1*pow(10,i)}')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Number of Errors')
    plt.legend()


#Q2d

plt.figure()
all_etas={eta:[] for eta in [0.1,1,10]}

for _ in range(100):
    w_init = np.random.uniform(-1, 1, 3)  # Randomly initialize w
    for i in range(3):
        _, er = ptronAlgo(X, y, w_init, 0.1 * pow(10, i))
        er += [0] * (max_epochs - len(er))
        all_etas[0.1 * pow(10, i)].append(er)

for i in range(3):
    avg_er=np.mean(all_etas[0.1 * pow(10, i)],axis=0)
    lp = np.percentile(all_etas[0.1 * pow(10, i)], 10, axis=0)
    up = np.percentile(all_etas[0.1 * pow(10, i)], 90, axis=0)

    plt.subplot(3,1,i+1)
    plt.plot(range(1, max_epochs+1), avg_er, label=f'eta = {0.1*pow(10,i)}')
    plt.fill_between(range(1, max_epochs+1), lp, up, color='lightblue', label='10th-90th Percentile')

    plt.xlabel('Epoch')
    plt.ylabel('Number of Errors')
    plt.title(f'Avg Errors vs. Epoch for n={0.1 * pow(10, i)}')
    plt.legend()
    plt.grid(True)


plt.tight_layout()
plt.show()


