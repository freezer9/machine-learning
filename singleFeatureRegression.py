
import matplotlib.pyplot as plt 
import numpy as np 
import csv
import random

# This model predicts the cost which is some-what close to the real value
# since this uses only 1 feature of x, it predicts a value some-what closer to y
# the real value's are the result of multiple linear regression

def LoadCSVData(filename):
	#  loads the csv data into lists and returns a list
	SizeInft = []
	HousePrice = []
	with open(filename,'r') as data:
		csvData = csv.reader(data)
		for x in csvData:
			SizeInft.append(float(x[5]))
			HousePrice.append(float(x[2]))
		print("Data Loaded from {0}".format(filename))
	return (SizeInft,HousePrice)

def Gradient(x,y,theta,alpha,m,Iteration):
	if len(x) == len(y):
		for i in range(0,Iteration):
			hypothesis = np.dot(x,theta)
			loss = (hypothesis - y).astype(int)
			cost = np.sum(pow(loss,2))/m
			print("Iteration : {0} || Cost : {1}".format(i,cost))
			GradientD = np.dot(x.transpose(),loss)/m
			theta = theta - alpha*GradientD
		return theta
	else:
		print("test-datasets are not equal")

def hyp(x,t):
	y = t[0] + t[1]*x 
	return y

data = LoadCSVData('RealEstate.csv')
x = np.array(data[0])
y = np.array(data[1])
m = np.shape(x)
theta = np.array([0 for i in range(len(x))])
theta[0],theta[1] = 1,1
alpha = 0.0000000001
t = Gradient(x,y,theta,alpha,m,100)
regline = [t[0] + (t[1]*xi) for xi in x]
print("\n\n\nLearning rate = {0}".format(alpha))
print("Weights = {0},{1}".format(t[0],t[1]))
xValue = 256
predictedY = hyp(xValue,t)
print("\n>> Predicted value for {0}sqft = ${1}".format(xValue,predictedY*1000)) #y value in 1000's

#plt.plot(x,regline,color='m')
plt.scatter(x,y)
plt.xlabel('X-axis')
plt.ylabel('y-axis')
#plt.show()

