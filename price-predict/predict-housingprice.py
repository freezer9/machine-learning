

# predict housing prices based on the size of the house in square ft
# x -> size in square feet
# y -> price of the house
# RealEstate.csv is arranged in the following order
#MLS,Location,Price,Bedrooms,Bathrooms,Size,Price/SQ.Ft,Status

from statistics import mean
import matplotlib.pyplot as plt
import numpy as np 
import csv,random


trainingListSize = []  # ---> x
trainingListPrice = [] # ---> y

testingListSize = []   #----> x
testingListPrice = []  #----> y


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
	return [SizeInft,HousePrice]

def SplitToTrainAndTest(FtrainingListSize,FtestingListSize,FtrainingListPrice,FtestingListPrice,REdata,splitRatio = 0.63,Ftype = 'eq'):
	# the prefix 'F' is added to the function varialbles just to avoid confusion, because without them they have the same
	# names as the lists
		if len(REdata[0]) == len(REdata[1]):
			# split House size data into training and testing data
			for x in range(len(REdata[0])):
				if random.random() < splitRatio:
					FtrainingListSize.append(float(REdata[0][x]))
				else:
					FtestingListSize.append(float(REdata[1][x]))
			print("Splited House size data into T&T")
			# split price data into training and testing data
			for x in range(len(REdata[1])):
				if random.random() < splitRatio:
					FtrainingListPrice.append(float(REdata[0][x]))
				else:
					FtestingListPrice.append(float(REdata[1][x]))
			print("Splited House price data into T&T")

def Equalize2Lists(xs,ys):
	# Equalizes the list if either of the list is smaller than the other
	#  Takes 2 normal lists and returns 2 lists
	gap = 0
	if len(xs) > len(ys):
		gap = len(xs) - len(ys)
		for y in range(gap):
			ys.append(0)
	elif len(ys) > len(xs):
		gap = len(ys) - len(xs)
		for x in range(gap):
			xs.append(0)
	return xs,ys



def BestFitLine(xs,ys):
	slope = ( (mean(xs)*mean(ys))-(mean(xs*ys)) ) / (pow(mean(xs),2) - mean(pow(xs,2)))
	return slope

def InterceptOfTheLine(xs,ys,m):
	b = mean(ys) - m*mean(xs)
	return b


def main():
	CSVFileData= LoadCSVData('RealEstate.csv') # the housing data is returned as a list of 2 items
	SplitToTrainAndTest(trainingListSize,testingListSize,trainingListPrice,testingListPrice,CSVFileData,splitRatio = 0.76)
	# equalize the list
	trainingSetEqSize,trainingSetEqPrice = Equalize2Lists(trainingListSize,trainingListPrice)
	# the data points in np array
	xsTrain = np.array(trainingSetEqSize,dtype=np.float64) 
	ysTrain = np.array(trainingSetEqPrice,dtype=np.float64)
	xsTest = np.array(testingListSize,dtype=np.float64)
	ysTest = np.array(testingListPrice,dtype=np.float64)

	print("SizeHouse Training Data: {0}".format(len(trainingListSize)))
	print("SizeHouse Testing Data : {0}".format(len(testingListSize)))
	print("Price Training Data: {0}".format(len(trainingListPrice)))
	print("Price Testing Data : {0}".format(len(testingListPrice)))

	slopeM = BestFitLine(xsTrain,ysTrain)
	InterceptB = InterceptOfTheLine(xsTrain,ysTrain,slopeM)
	reg_line = [(slopeM*x)+InterceptB for x in xsTrain]
	print("Slope of training data :",slopeM)
	print("Intercept of training data :",InterceptB)
	predx = 256
	predict_y = (slopeM*predx) + InterceptB
	print("\n\n\n>> Predicted Price for {0}sqft is ${1}\n\n\n".format(predx,predict_y*1000))
	#plot the data
	PlotColors = [[0.35,0.21,0.54],[0.225,0.45,0.55]]
	fig = plt.figure()
	fig.patch.set_facecolor('MidnightBlue')
	plt.scatter(xsTrain,ysTrain,s = 50,c = PlotColors,alpha = 0.5)
	# plot the predicted value 
	plt.scatter(predx,predict_y,s = 50,color='r',alpha = 1)
	plt.plot(xsTrain,reg_line,color = 'm')
	plt.xlabel('Size in sqft',color='w')
	plt.ylabel('Price in 1000s',color = 'w')
	plt.show()


if __name__ == '__main__':
	main()

