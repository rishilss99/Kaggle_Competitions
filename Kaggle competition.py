import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# print(train.head())
# print(test.head())
# print(type(train))

train_X = train.loc[:,['X1','X3','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']]
train_Y =  train.loc[:,'Y']

# print(train_X.head())
# print(train_Y.head())

rf = RandomForestClassifier(n_estimators=2000,min_samples_leaf=80)
rf.fit(train_X,train_Y)

# features = pd.Series(rf.feature_importances_,index=list(train.iloc[0:0,1:24])).sort_values(ascending=False)
# print(features)

test = test.loc[:,['X1','X3','X5','X6','X7','X8','X9','X11','X10','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']]
pred = rf.predict_proba(test)

result = pd.DataFrame(pred[:,1])
result.columns = ['Y']
result.index.name = 'ID'
result.index += 1
result.to_csv('output.csv')
print (result.head())