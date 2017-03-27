
# Linear regression to predict the body weight along with the brain weight
# DATA

# INDEX BRAIN-WEIGHT BODY-WEIGHT
# 1		   3.385      44.500

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import csv

def ReadFile(fn='x01.txt'):
	data_list = []
	with open(fn,'r') as data:
		file = csv.reader(data)
		for x in file:
			data_list.append([x[1],x[2]])
	return data_list

scatter_data_points = ReadFile()
		
xList = [scatter_data_points[x][0] for x in range(len(scatter_data_points))]
yList = [scatter_data_points[y][1] for y in range(len(scatter_data_points))]


xs = np.array(xList,dtype=np.float64)
ys = np.array(yList,dtype=np.float64)

def Best_fit_slope(xs,ys):
	m = ( (mean(xs)*mean(ys))-(mean(xs*ys)) ) / (pow(mean(xs),2) - mean(pow(xs,2)))
	return m

def Intercept(xs,ys,m):
	b = mean(ys) - m*mean(xs)
	return b

m = Best_fit_slope(xs,ys)
b = Intercept(xs,ys,m)
reg_line = [(m*x)+b for x in xs]

plt.scatter(xs,ys)

predict_list = [15,5,14,28,10,3,61]
predict_list_resp = [(m*x)+b for x in predict_list ]
for predX in predict_list:
	for resp in predict_list_resp:
		print("predicted body weight for {0} is :{1}".format(predX,resp))

for yPlt in predict_list_resp:
	for xPlt in predict_list:
		plt.scatter(xPlt,yPlt,color='r')
plt.plot(xs,reg_line)
plt.show()