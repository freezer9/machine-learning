
#MLS,Location,Price,Bedrooms,Bathrooms,Size,Price/SQ.Ft,Status

# linear regression with multiple features 

import numpy as np 
import matplotlib.pyplot as plt 
import csv 

def LoadFrom(path):
	x_dataset = []
	y_dataset = []
	with open(path,'r') as file:
		data = csv.reader(file)
		print('data loaded from {0}\n\n'.format(path))
		for item in data:
			# x[0] is set to one just so that the dot product runs
			x_dataset.append([1,item[3],item[4],item[6]])
			y_dataset.append(item[2])
	return x_dataset,y_dataset

def SplitIntoTestandTrain(Lx,Ly):
	training_xdata = []
	training_ydata = []
	testing_xdata = []
	testing_ydata = []
	training_xdata = Lx[0:600]
	training_ydata = Ly[0:600]
	testing_xdata = Lx[600:780]
	testing_ydata = Ly[600:780]
	return (training_xdata,training_ydata,testing_xdata,testing_ydata)

def Hypothesis(x,t):
	hx = np.dot(t,x)
	return hx


def Gradient(x,y,t,m,a,numInter):
	hyp = Hypothesis(x,t.transpose())
	TempHype = [0 for zero in range(len(y))]
	TempHype[0],TempHype[1],TempHype[2],TempHype[3] = hyp[0],hyp[1],hyp[2],hyp[3]
	for i in range(numInter):
		loss = (TempHype - y)
		cost = np.sum(pow(loss,2))/m
		print("Iteration : {0} || Cost : {1}".format(i,cost))
		GD = np.dot(x.transpose(),loss)/m
		t = t- a*GD
	return t


raw_x,raw_y = LoadFrom('RealEstate.csv')
FinalData = SplitIntoTestandTrain(raw_x,raw_y)
xtrain = []
ytrain = []
xtrain = np.array(FinalData[0],dtype = float)
ytrain = np.array(FinalData[1],dtype = float)
theta = np.array( [[0,0,0,0] for i in range(600) ] )
#theta[0][0][0],theta[1][1][0],theta[2][2][0],theta[3][3][0] = 1,1,1,1
alpha_rate = 0.0000000011
numberOfIter = 1000

thetaValues = Gradient(xtrain,ytrain,theta,600,alpha_rate,numberOfIter)
#print(thetaValues[0][0][0],thetaValues[1][1][0],thetaValues[2][2][0],thetaValues[3][3][0])
tvlist = [thetaValues[0][0][0],thetaValues[1][1][0],thetaValues[2][2][0],thetaValues[3][3][0]]
print(tvlist)
x = [1,3,3,335.30]
y = tvlist[0] + x[1]*tvlist[1]+x[2]*tvlist[2]+x[3]*tvlist[3]
print(y)
