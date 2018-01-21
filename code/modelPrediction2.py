# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:09:53 2018

@author: Hubert
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

import datetime

def addCities(df):
    cityCode = ["C10001", "C10002", "C10003", "C10006","C10004", 
                "C10005",  "C10007",  "C10010",  "C10008", "C10009"]
    
    for city in cityCode:
        df[city] = (df['City_Code']== city)
    return df

def addSource(df):
    sourceCode = ["S122", "S133", "S143", "S134", "S159"]
    
    for source in sourceCode:
        df[source] = (df["Source"]== source)
    return df

def addBank(df):
    bankCode = ["B001", "B002", "B003", "B004", "B005", "B006", "B011"]
    
    for bank in bankCode:
        df[bank] = (df["Customer_Existing_Primary_Bank_Code"]== bank)
    return df

def addMoneyFeature(df):
    df["monthlyIncomeRatio"] = df["Monthly_Income"]/df["Loan_Amount"]
    
    return df


def trainXGModel(train):
    target = train['Approved']
    train['Approved'] = train['Approved'].astype('category')
    
    train = train.drop(['Approved', 'ID'], axis=1)   
    
    currentTime = datetime.datetime.now().isoformat()
    print("Train time begin:", currentTime)
    print("For model", train)
    xgModel = xgb.XGBClassifier(learning_rate = 0.3, n_estimators=20).fit(train, target)
    finishTime = datetime.datetime.now().isoformat()
    print("Train time finish:", finishTime)
    return xgModel

def predictXGModel(XGModel, test):
    testID = test['ID'] 
    test = test.drop(['ID'], axis = 1)
    
    predictions = XGModel.predict(test)

    
    results = pd.DataFrame()
    results['ID'] = testID
    results['Approved'] = predictions

    return results

def lastResults(test):
    results = pd.DataFrame()
    results['ID'] = test['ID']
    results['Approved'] = 0
    
    return results

def lastResults2(test):
    results = pd.DataFrame()
    results['ID'] = test['ID']
    results['Approved'] = 1
    
    return results
    
'''
trainSubset1 = pd.read_csv("../data/trainSubset1.csv")
trainSubset1a = pd.read_csv("../data/trainSubset1a.csv")
trainSubset2 = pd.read_csv("../data/trainSubset2.csv")
trainSubset3 = pd.read_csv("../data/trainSubset3.csv")
trainSubset4 = pd.read_csv("../data/trainSubset4.csv")


testSubset1 = pd.read_csv("../data/testSubset1.csv")
testSubset1a = pd.read_csv("../data/testSubset1a.csv")
testSubset2 = pd.read_csv("../data/testSubset2.csv")
testSubset3 = pd.read_csv("../data/testSubset3.csv")
testSubset4 = pd.read_csv("../data/testSubset4.csv")
'''
#dataFrameSet = [trainSubset1, trainSubset1a, trainSubset2, trainSubset3, trainSubset4,
#                testSubset1, testSubset1a, testSubset2, testSubset3, testSubset4]

trainDF = pd.read_csv("../data/train.csv")
test= pd.read_csv("../data/test.csv")

for col in ['Employer_Category2', 'Var1']:
     test[col] = test[col].astype('category')
     trainDF[col] = trainDF[col].astype('category')
cols_to_transform = ['Gender', 'City_Category', 'Employer_Category1', 'LeadCreationDOW',
                     'Employer_Category2', 'Source_Category', 'Var1', 'Primary_Bank_Type', 'Contacted']     
     
test = pd.get_dummies(test, columns = cols_to_transform)
trainDF = pd.get_dummies(trainDF, columns = cols_to_transform)

trainDF = addCities(trainDF)
trainDF = addSource(trainDF)
trainDF = addBank(trainDF)
test = addCities(test)
test = addSource(test)
test = addBank(test)

trainApprove = trainDF[trainDF["Approved"] == 1]
nrow = trainApprove.shape[0]
trainNill = trainDF[trainDF["Approved"] == 0]
trainNotApprove = trainNill.sample(nrow - 204)

train = pd.DataFrame()
train = train.append(trainApprove)
train = train.append(trainNotApprove)


train = train.drop(['City_Code', 'Source', 'Customer_Existing_Primary_Bank_Code'], axis = 1)
test = test.drop(['City_Code', 'Source', 'Customer_Existing_Primary_Bank_Code'], axis = 1)



trainCols = train.columns.tolist()
testCols = test.columns.tolist()

trainCols.remove("Approved")
test.columns = trainCols


trainSubset1  = train[train['Interest_Rate'].isnull() == 0]
trainSubset1a = train[(train['Existing_EMI'].isnull() == 0) & (train['Interest_Rate'].isnull() == 0)]
removeMissingEEMI = train[train['Existing_EMI'].isnull() == 0]
trainSubset2 = removeMissingEEMI[removeMissingEEMI['Loan_Amount'].isnull() == 0]
trainSubset3 = removeMissingEEMI
trainSubset4 = train[train['Interest_Rate'].isnull() & train['Existing_EMI'].isnull()]

testSubset1 = test[(test['Interest_Rate'].isnull() == 0) & (test['Existing_EMI'].isnull() == 1)] #Set so we don't need to use Existing EMI column to predict
testSubset1a = test[test['Interest_Rate'].isnull() == 0] #Set we can use EMI to predict
removeMissingEEMI = test[test['Existing_EMI'].isnull() == 0]
testSubset2 = removeMissingEEMI[removeMissingEEMI['Loan_Amount'].isnull() == 0 & removeMissingEEMI['Interest_Rate'].isnull()] 
testSubset3 = removeMissingEEMI[removeMissingEEMI['Loan_Amount'].isnull()]
testSubset4 = test[test['Interest_Rate'].isnull() & test['Existing_EMI'].isnull()] #All Values here are zero

for sets in [trainSubset1, trainSubset1a, trainSubset2, testSubset1, testSubset1a, testSubset2]:
    sets = addMoneyFeature(sets)



trainSubset1 = trainSubset1.drop('Existing_EMI', axis = 1)
xgModel1 = trainXGModel(trainSubset1)

xgModel1a = trainXGModel(trainSubset1a)

#trainSubset2 = trainSubset2.drop(['Interest_Rate', 'EMI'], axis = 1)
xgModel2 = trainXGModel(trainSubset2)

trainSubset3 = trainSubset3.drop(['Interest_Rate', 'EMI', 'Loan_Amount', 'Loan_Period'], axis = 1)
xgModel3 = trainXGModel(trainSubset3)

testSubset1 = testSubset1.drop('Existing_EMI', axis = 1)
results1 = predictXGModel(xgModel1, testSubset1)
results1a = predictXGModel(xgModel1a, testSubset1a)

#testSubset2 = testSubset2.drop(['Interest_Rate', 'EMI'], axis = 1)
results2 = predictXGModel(xgModel2, testSubset2)

testSubset3 = testSubset3.drop(['Interest_Rate', 'EMI', 'Loan_Amount', 'Loan_Period'], axis = 1)
results3 = predictXGModel(xgModel3, testSubset3)
results4 = lastResults(testSubset4)


results = pd.DataFrame()
results = results.append(results1)
results = results.append(results1a)
results = results.append(results2)
results = results.append(results3)
results = results.append(results4)


results.to_csv("../results/newFeatures.csv", index = False)
