#########################################  
# kNN: k Nearest Neighbors  
  
# Input:      inX: vector to compare to existing dataset (1xN)  
#             dataSet: size m data set of known vectors (NxM)  
#             labels: data set labels (1xM vector)  
#             k: number of neighbors to use for comparison   
              
# Output:     the most popular class label  
#########################################  
  
from numpy import *  
import operator  
import os  
  
  
# classify using kNN  
def kNNClassify(newInput, dataSet, labels, k):  
	numSamples = dataSet.shape[0] # shape[0] stands for the num of row  
  
    ## step 1: calculate Euclidean distance  
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
	diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise  
	squaredDiff = diff ** 2 # squared for the subtract  
	squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row  
	distance = squaredDist ** 0.5  
  
    ## step 2: sort the distance  
    # argsort() returns the indices that would sort an array in a ascending order  
	sortedDistIndices = argsort(distance)  
  
	classCount = {} # define a dictionary (can be append element)  
	for i in range(k):  
        ## step 3: choose the min k distance  
		voteLabel = labels[sortedDistIndices[i]]  
  
        ## step 4: count the times labels occur  
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
  
    ## step 5: the max voted class will return  
	maxCount = 0  
	for key, value in classCount.items():  
		if value > maxCount:  
			maxCount = value  
			maxIndex = key  
  
	return maxIndex   
  
# convert image to vector 
#将图像转化为一个向量 
def  img2vector(filename):  
	rows = 32  
	cols = 32  
	imgVector = zeros((1, rows * cols))   
	fileIn = open(filename)  
	for row in range(rows):  
		lineStr = fileIn.readline()  
		for col in range(cols):  
			imgVector[0, row * 32 + col] = int(lineStr[col])  
  
	return imgVector  
  
# load dataSet  
def loadDataSet():  
    ## step 1: Getting training set  
	print ("---Getting training set..."  )
	dataSetDir = 'D:/xuepython/KNN/digits/'
	#trainingFileList是训练文件的列表
	trainingFileList = os.listdir(dataSetDir + 'trainingDigits') # load the training set  
	numSamples = len(trainingFileList)  
  
	train_x = zeros((numSamples, 1024))   #数组，用来保存所有文件，每个文件的数据保存为一行
	train_y = []  #保存文件中的数据的类别号
	for i in range(numSamples):  
		filename = trainingFileList[i]  
  
        # get train_x  
		train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename) #传入的是一个文件的路径  
  
        # get label from file name such as "1_18.txt"  
		label = int(filename.split('_')[0]) # return 1  
		train_y.append(label)  
  
    ## step 2: Getting testing set  
	print ("---Getting testing set..." ) 
	testingFileList = os.listdir(dataSetDir + 'testDigits') # load the testing set  
	numSamples = len(testingFileList)  
	test_x = zeros((numSamples, 1024))  
	test_y = []  
	for i in range(numSamples):  
		filename = testingFileList[i]  
  
        # get train_x  
		test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename)   
  
        # get label from file name such as "1_18.txt"  
		label = int(filename.split('_')[0]) # return 1  
		test_y.append(label)  
  
	return train_x, train_y, test_x, test_y  
  
# test hand writing class  
def testHandWritingClass():  
    ## step 1: load data  
	print ("step 1: load data..." ) 
	train_x, train_y, test_x, test_y = loadDataSet()  
  
    ## step 2: training...  
	print ("step 2: training...")  
	pass  
  
    ## step 3: testing  
	print ("step 3: testing..." ) 
	numTestSamples = test_x.shape[0]   #测试文件的数目
	matchCount = 0  #记录正确检测的个数
	for i in range(numTestSamples):  
		predict = kNNClassify(test_x[i], train_x, train_y, 3)  #返回的是判断的测试样本的类别
		if predict == test_y[i]:#如果测试所得到的类别号与原本他所对应的类别号相同，则为正确检测  
			matchCount += 1  
	accuracy = float(matchCount) / numTestSamples  #检测率 
  
    ## step 4: show the result  
	print ("step 4: show the result..."  )
	print ('The classify accuracy is: %.2f%%' % (accuracy * 100) )