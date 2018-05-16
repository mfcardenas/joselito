# Import libraries
import pandas as pd
import time
import math
import csv

exe = str(int(time.time()*100))
# Matrix similarity using corated similarity function
coratedSim = pd.read_csv('cfCoratedSimilarity.csv', sep=';', header=None)
coratedSim.columns = ['User-i', 'User-j', 'Similarity']
coratedSim.head(100).to_csv("coratedSim_" + exe + ".csv")

# Matrix similarity using item based similarity function
itemSim = pd.read_csv('itemsSimilarity.csv', sep=";", header=None)
itemSim.columns = ['Item-i', 'Item-j', 'Similarity']
itemSim.head(100).to_csv("itemSim_" + exe + ".csv")

# Evaluation data
evalData = pd.read_csv('data/eval.dat', sep=';', header=None)
evalData.columns = ['Item', 'User', 'Rating', 'Timestamp']
evalData.head(100).to_csv("evalData_" + exe + ".csv")

# Training Data
trainingData = pd.read_csv('data/training.dat', sep=';', header=None)
trainingData.columns = ['Item', 'User', 'Rating', 'Timestamp']
trainingData.head(100).to_csv("trainingData_" + exe + ".csv")

# Set the number of cases to retrieval
k = 5
def ratingPredicted(item, user):
    # Obtain the most similar users
    simUsers = coratedSim[coratedSim['User-i'] == user].iloc[:,1:3].rename(index=str, columns={"User-j": "User", "Similarity": "Similarity"})
    simUsers.append(coratedSim[coratedSim['User-j'] == user].iloc[:,[0, 2]].rename(index=str, columns={"User-i": "User", "Similarity": "Similarity"}))
    itemRatings = trainingData[trainingData.Item == item]
    itemRatings = itemRatings[itemRatings.User.isin(simUsers.User)]


for k in [10, 30, 50, 100]:

    ratingPred = list()
    for index, row in evalData.iterrows():
        ratingPred.append(ratingPredicted(row['Item'], row['User']))

    name = 'Predicted-K' + str(k)
    evalData[name] = ratingPred


## Calculate the error of each rating predicted
for k in [10, 30, 50, 100]:
    evalData['Error-K'+str(k)] = abs(evalData.Rating - evalData['Predicted-K' + str(k)])

evalData.head().to_csv("errorEachRatingPredicted_evalData_" + exe + ".csv")

listMAE = list()
## We calculate the Mean Absoulte Error
for k in [10, 30, 50, 100]:
    MAE = evalData['Error-K'+str(k)].mean()
    listMAE.append("Mean Absolute Error (MAE) in K=" + str(k) + " is " + str(MAE))

with open("MeanAbsoluteError_" + exe + ".csv", 'wb') as myFile:
    wr = csv.writer(myFile, quoting=csv.QUOTE_ALL)
    wr.writerow(listMAE)

## We calculate the Root Mean Absolute error
for k in [10, 30, 50, 100]:
    error2 = list()
    for index_, row in evalData.iterrows():
        error2.append(row['Error-K'+str(k)] ** 2) ## It is error^2
    evalData['Error2-K'+str(k)] = error2

evalData.head().to_csv('RootMeanAbsoluteError_" + exe + ".csv')

listRMA = list()
for k in [10, 30, 50, 100]:
    RMAE = math.sqrt(evalData['Error2-K'+str(k)].mean())
    listRMA.append("Root Mean Absolute Error (RMAE) in K=" + str(k) + " is " + str(RMAE))

with open("RootMeanAbsoluteError_" + exe + ".csv", 'wb') as myFile:
    wr = csv.writer(myFile, quoting=csv.QUOTE_ALL)
    wr.writerow(listRMA)