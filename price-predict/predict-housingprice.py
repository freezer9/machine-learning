

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
			SizeInft.append(x[5])
			HousePrice.append(x[2])
		print("Data Loaded from {0}".format(filename))
	return [SizeInft,HousePrice]

def SplitToTrainAndTest(FtrainingListSize,FtestingListSize,FtrainingListPrice,FtestingListPrice,REdata,splitRatio = 0.63):
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

#def AdjustDataToEqual(x,y):
	# To plot the data on the scatter plot they should be equal
#	gap = max(x,y) - min(x,y)
#	for x in range(gap):
#		y.append(0)
#	print(x,y)


def main():
	CSVFileData= LoadCSVData('RealEstate.csv') # the housing data is returned as a list of 2 items
	SplitToTrainAndTest(trainingListSize,testingListSize,trainingListPrice,testingListPrice,CSVFileData,splitRatio = 0.76)

	# the data points in np array
	xsTrain = np.array(trainingListSize,dtype=np.float64) 
	ysTrain = np.array(trainingListPrice,dtype=np.float64)
	xsTest = np.array(testingListSize,dtype=np.float64)
	ysTest = np.array(testingListPrice,dtype=np.float64)

	print("SizeHouse Training Data: {0}".format(len(trainingListSize)))
	print("SizeHouse Testing Data : {0}".format(len(testingListSize)))
	print("Price Training Data: {0}".format(len(trainingListPrice)))
	print("Price Testing Data : {0}".format(len(testingListPrice)))
	print(len(xsTrain))
	print(len(ysTrain))
	AdjustDataToEqual(len(xsTrain),len(ysTrain))
	#plt.scatter(xsTrain,ysTrain)
	#plt.show()


if __name__ == '__main__':
	main()

