"""Mini project to predict the sales of an e-commerce website based on attributes :
1)Avg. session length'
2)Time on mobile app
3)Membership length"""

# Loading the text data into a CSV file using pd library

import numpy
import pandas as pd


# Initialising the dataset as a csv file
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Dropping the unwanted columns

train_X = train.loc[:,['X1','X3','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']]
train_Y =  train.loc[:,'Y']

# Extracting the features and storing it in Input


# The matrix holding values for X0 whose all elements are 1

ones = numpy.ones( [train_X.shape[0] , 1 ] )

Input = numpy.concatenate( (ones, train_X), axis = 1)

# Extracting the expected output to Output variable


# Normalized equation used to obtain the  values of the parameters Theta

Theta = numpy.matmul(numpy.linalg.inv(numpy.matmul(train_X.transpose(),train_X)),numpy.matmul(train_X.transpose(),train_Y))

def hypothesis(X,theta):
    """The hypothesis function takes the values of the features as the input and returns a matrix of the predicted values using
    the obtained values of the parameters saved in matrix theta"""

    return numpy.matmul(X,theta)

test = test.loc[:,['X1','X3','X5','X6','X7','X8','X9','X11','X10','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']]


result = pd.DataFrame(hypothesis(test,Theta))
result.columns = ['Y']
result.index.name = 'ID'
result.index += 1
result.to_csv('output1.csv')

print (result.head())

def accuracy_checker(Input,Expected_Output):
    """The function is used to check the accuracy of the predicted output using the hypothesis function"""
    n = len(Input)
    Predicted_output = hypothesis(Input,Theta)
    check = numpy.zeros([n,1])
    for i in range(n):
        check[i] = (1 - abs(Predicted_output[i] - Expected_Output[i])/Expected_Output[i])*100
    return check.mean()

# print(accuracy_checker(Input[450:],Output[450:]))

