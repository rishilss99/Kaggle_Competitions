import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

"""Converting information from csv to pandas dataframe"""

df_train = pd.read_csv("Titanic_datasets/train.csv")
df_test = pd.read_csv("Titanic_datasets/test.csv")

# Initial analysis of the data

# print(df_train.describe())
# print(df_train.info())

def extract_title(name):
    """Used to extract title from the given name"""
    ind1 = name.find(',')
    ind2 = name.find('.')
    return name[ind1+2:ind2]

# Adds a new column to the dataframe 'Title'
df_train['Title'] = df_train['Name'].apply(extract_title)
df_test['Title'] = df_test['Name'].apply(extract_title)



# print(df_train)

# # Plotting the titles to visualize the data
# plt.figure(figsize=(12,5))
# sns.countplot(x='Title',data=df_train,palette='hls')
#
# plt.show()

Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Lady" :      "Royalty",
        "Mme":        "Mrs",
        "Ms":         "Mrs",
        "Mrs" :       "Mrs",
        "Mlle":       "Miss",
        "Miss" :      "Miss",
        "Mr" :        "Mr",
        "Master" :    "Master"
                   }

"""Titles have been replaced in both training set and test set"""

df_train['Title'] = df_train['Title'].map(Title_Dictionary)
df_test['Title'] = df_train['Title'].map(Title_Dictionary)
# df_train['Title'] = df_train['Title'].fillna("Mr")
df_test['Title'] = df_test['Title'].fillna("Mr")

# print(df_train)
# # Plotting the titles to visualize the data according to survivors
# plt.figure(figsize=(12,5))
# sns.countplot(x='Title',hue='Survived',data=df_train,palette='hls')
#
# plt.show()

# # Analyzing the age to find values of NaN
# age_group = df_test.groupby(['Sex','Pclass','Title'])
# print(age_group['Age'].median())

"""Age containing NaN values have been filled based on median values"""

df_train.loc[df_train['Age'].isnull(),'Age']=df_train.groupby(['Sex','Pclass','Title'])['Age'].transform('median')
df_test.loc[df_test['Age'].isnull(),'Age']=df_test.groupby(['Sex','Pclass','Title'])['Age'].transform('median')

"""Filling S in all NaN embarked"""

df_train['Embarked'] = df_train['Embarked'].fillna('S')

# print(df_train.info())
# print(df_test.info())

# plt.figure(figsize=(12,5))
# sns.distplot(df_train["Age"], bins=24)
# plt.title("Distribuition and density by Age")
# plt.xlabel("Age")
# plt.show()

"""Converting pandas dataframes to numpy arrays"""

X_train = df_train.iloc[:,[2,4,5,6,7,9,11,12]].values
Y_train = df_train.iloc[:,1].values
X_test = df_test.iloc[:,[1,3,4,5,6,8,10,11]].values

"""Filling the missing data once converted to numpy array"""
# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer = imputer.fit(X_test[:, [5]])
X_test[:, [5]] = imputer.transform(X_test[:, [5]])
pass


"""Encoding the categorical data"""
# Encoding the categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Encoding gender
labelencoder_X1 = LabelEncoder()
X_train[:,1] = labelencoder_X1.fit_transform(X_train[:,1])
X_test[:,1] = labelencoder_X1.transform(X_test[:,1])


# Encoding embarkment
labelencoder_X2 = LabelEncoder()
X_train[:,6] = labelencoder_X2.fit_transform(X_train[:,6])
X_test[:,6] = labelencoder_X2.transform(X_test[:,6])

# Encoding title
labelencoder_X3 = LabelEncoder()
X_train[:,7] = labelencoder_X3.fit_transform(X_train[:,7])
X_test[:,7] = labelencoder_X3.transform(X_test[:,7])

# Encoding the embarkment and removing first row
onehotencoderX1 = OneHotEncoder(categorical_features=[6])
X_train = onehotencoderX1.fit_transform(X_train).toarray()
X_train = X_train[:,1:]
X_test = onehotencoderX1.transform(X_test).toarray()
X_test = X_test[:,1:]


# Encoding the title and removing first row
onehotencoderX2 = OneHotEncoder(categorical_features=[8])
X_train = onehotencoderX2.fit_transform(X_train).toarray()
X_train = X_train[:,1:]
X_test = onehotencoderX2.transform(X_test).toarray()
X_test = X_test[:,1:]

"""Feature scaling using sklearn"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
pass

"""Building the ANN"""

# Importing the libraries
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

classifier = Sequential()

# Adding the first hidden layer
classifier.add(Dense(13,activation='relu',kernel_initializer='uniform',input_dim=13))

# Adding the second hidden layer
classifier.add(Dense(13,activation='relu',kernel_initializer='uniform'))

# Adding the output layer
classifier.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

# Compiling the model
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train, Y_train,  batch_size = 25,nb_epoch = 100, verbose=2)
scores = classifier.evaluate(X_train, Y_train, batch_size=25)
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))


