
import matplotlib.pyplot as plt 
import numpy as np 



def Gradient(x,y,theta,alpha,m,Iteration):
	if len(x) == len(y):
		for i in range(0,Iteration):
			hypothesis = np.dot(x,theta)
			loss = (hypothesis - y).astype(int)
			cost = np.sum(pow(loss,2))/m
			#print("Iteration : {0} || Cost : {1}".format(i,cost))
			GradientD = np.dot(x.transpose(),loss)/m
			theta = theta - alpha*GradientD
		return theta
	else:
		print("test-datasets are not equal")

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
y = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
m = np.shape(x)
theta = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
alpha = 0.0001

def hyp(x,t):
	y = t[0] + t[1]*x 
	return y


t = Gradient(x,y,theta,alpha,m,10000)
print(t[0],t[1])

for x in range(1,10):
	print(x,hyp(x,t))

#plt.scatter(x_dataset,y_dataset)
#plt.xlabel('X-axis')
#plt.ylabel('y-axis')
#plt.show()

