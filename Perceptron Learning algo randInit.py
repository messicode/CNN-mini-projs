# -*- coding: utf-8 -*-
"""ECE/CS 559 Fall 2024 HW3
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11111559)

wstar = np.zeros(3)
wstar[0] = np.random.uniform(-1/4,1/4)
wstar[1] = np.random.uniform(-1,1)
wstar[2] = np.random.uniform(-1,1)
print("Question 1 (a): wstar = ",wstar)

n=100
x = np.ones((3,n))
x[1:3,:] = np.random.uniform(-1,1,(2,n))
y= np.heaviside(np.matmul(wstar,x),1)
fig = plt.figure(figsize=(5,5))
plt.scatter(x[1,np.where(y==1)], x[2,np.where(y==1)], c='red')
plt.scatter(x[1,np.where(y==0)], x[2,np.where(y==0)], c='blue')
t = np.linspace(-1,1,100)
plt.plot(t,-(wstar[0]+wstar[1]*t)/wstar[2], c='green')
j=25 # where on the line to place the arrow.
plt.arrow(t[j],-(wstar[0]+wstar[1]*t[j])/wstar[2],wstar[1],wstar[2],head_width=0.05, head_length=0.1)
wstar_length = np.sqrt(wstar[1]**2+wstar[2]**2)
magnitude = - wstar[0] / wstar_length
plt.arrow(0,0,magnitude*wstar[1]/wstar_length,magnitude*wstar[2]/wstar_length)
plt.axis('equal')
plt.xlim((-1,1)); plt.ylim((-1,1))
plt.xlabel('x1'); plt.ylabel('x2')
plt.grid()
plt.title("Question 1 (b) and (c):")
plt.show()

# Proof for (c):
# ~~~~~~~~~~~~~~
# The length of (wstar[1],wstar[2]) is: wstar_length = np.sqrt(wstar[1]**2+wstar[2]**2).
# The vector (u1,u2) = (wstar[1]/wstar_length,wstar[2]/wstar_length) is a unit vector in the same direction.
# If we start at the origin and we move s along this direction, wstare end up
#   at point (u1*s, u2*s). When do wstare hit the line?
# We touch the line when the line's equations are satisfied:
#   wstar[0] + wstar[1]*u1*s + wstar[2]*u2*s = 0
#   => wstar[0] + s*wstar[1]*wstar[1]/wstar_length + s*wstar[2]*wstar[2]/wstar_length = 0
#   => wstar[0] + s*(wstar[1]**2 + wstar[2]**2)/wstar_length = 0
#   => wstar[0] + s*wstar_length = 0
#   => s = -wstar[0]/wstar_length


def perceptron(x,y,eta,w_init):
  w = w_init
  converged = False
  epoch_errors = []
  while not converged:
    errors = 0
    for i in range(n):
      f = np.heaviside(np.dot(w,x[:,i]),1)
      errors += 1.*(y[i]!=f)
      w += eta*(y[i]-f)*x[:,i]
    converged = (errors == 0)
    epoch_errors.append(errors)
  return w, epoch_errors
w, epoch_errors = perceptron(x,y,1, np.ones(3))
print("Question 2(a): w =", w)
print("Not the same as wstar. One issue is that they have different scaled. We can normalize by the length of (w1,w2), to compare better:")
print("Normalized wstar = ", wstar/wstar_length)
w_length = np.sqrt(w[1]**2+w[2]**2)
print("Normalized w = ", w/w_length)
w_eta1 = w
print("Still not the same, but close. The reason is that the data has gaps, so many w's would separate y=0 and y=1 points.")
fig = plt.figure(figsize=(5,5))

plt.plot(epoch_errors)
plt.title("Question 2 (b): eta=1")
plt.show()

w, epoch_errors = perceptron(x,y,0.1, np.ones(3))
plt.plot(epoch_errors)
plt.title("Question 2 (b): eta=0.1")
plt.show()

w, epoch_errors = perceptron(x,y,10, np.ones(3))
plt.plot(epoch_errors)
plt.title("Question 2 (b): eta=10")
plt.show()

# Question 2(b) Not evident from a single run, but usually:
    # eta small => slower convergence, but smoother reduction in errors
    # eta large => can be fast, but can also be very slow, too volatile
    # eta just right, converge at a reasonable speed, smoothly



n=1000
x = np.ones((3,n))
x[1:3,:] = np.random.uniform(-1,1,(2,n))
y= np.heaviside(np.matmul(wstar,x),1)
wnew, epoch_errors = perceptron(x,y,1, np.ones(3))
fig = plt.figure(figsize=(5,5))
plt.scatter(x[1,np.where(y==1)], x[2,np.where(y==1)], c='red')
plt.scatter(x[1,np.where(y==0)], x[2,np.where(y==0)], c='blue')
t = np.linspace(-1,1,100)
plt.plot(t,-(wstar[0]+wstar[1]*t)/wstar[2], c='green')
plt.plot(t,-(w_eta1[0]+w_eta1[1]*t)/w_eta1[2], c='purple')
plt.plot(t,-(wnew[0]+wnew[1]*t)/wnew[2], c='orange')
plt.axis('equal')
plt.xlim((-1,1)); plt.ylim((-1,1))
plt.xlabel('x1'); plt.ylabel('x2')
plt.grid()
plt.title("Question 2 (c): Purple (n=100), Orange (n=1000)")
plt.show()

print("Question 2(c): The new (orange) w is closer to the (green) wstar than the old (pruple) w.")
wnew_length = np.sqrt(wnew[1]**2+wnew[2]**2)
print("Normalized wnew = ", wnew/wnew_length)
print("Normalized wstar = ", wstar/wstar_length)
print("Normalized w = ", w_eta1/w_length)

max_epochs = 100
repeats = 100
errors_01 = np.zeros(max_epochs)
errors_1 = np.zeros(max_epochs)
errors_10 = np.zeros(max_epochs)

for _ in range(repeats):
  w_init = np.random.uniform(0,1,3)
  _, epoch_errors = perceptron(x,y,0.1, w_init)
  epoch_errors += [0]*(max_epochs - len(epoch_errors))
  errors_01 += epoch_errors
  w_init = np.random.uniform(0,1,3)
  _, epoch_errors = perceptron(x,y,1, w_init)
  epoch_errors += [0]*(max_epochs - len(epoch_errors))
  errors_1 += epoch_errors
  w_init = np.random.uniform(0,1,3)
  _, epoch_errors = perceptron(x,y,10, w_init)
  epoch_errors += [0]*(max_epochs - len(epoch_errors))
  errors_10 += epoch_errors

plt.plot(errors_01/repeats)
plt.title("Question 3 (b): eta=1")
plt.show()
plt.plot(errors_01/repeats)
plt.title("Question 3 (b): eta=0.1")
plt.show()
plt.plot(errors_10/repeats)
plt.title("Question 3 (b): eta=10")
plt.show()

# Question 2(d): Randomizing over initializations smoothes out the behavior
# we noted with eta small to large (slow/smooth to fast/unstable). However,
# for any specific initialization, we do seem to have roughly the same
# behavior at any eta, with mild variations.

np.random.uniform(3)

