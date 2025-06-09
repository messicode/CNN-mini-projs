import numpy as np
import matplotlib.pyplot as plt


def eqn(w1,w2):
    r= 13*(w1**2) - 10*w1*w2 + 4*w1 + 2*(w2**2) - 2*w2 +1
    return r




w=np.array([0,0]) #Initial weights
r_opt=eqn(1,3) #Theoretically found w1=1,w2=3
# print(r_opt)
# print(w)

for eta in [0.02,0.05,0.1]:
    distances = []
    for i in range(0,500):
        grad=np.array( [ 26*w[0] - 10*w[1] + 4, 4*w[1] -10*w[0]-2])
        w=w-eta*grad
        r=eqn(w[0],w[1])
        l1=abs(r-r_opt)
        distances.append(l1)
    plt.plot(range(500),distances)
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.title(f'Distance vs Iterations for eta={eta}')
    plt.show()
