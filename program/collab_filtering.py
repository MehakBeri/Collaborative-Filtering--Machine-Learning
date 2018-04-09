import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
def readFile(filename):
    """reads file from filename and stores the lines in the file and returns them"""
    data=pd.read_csv(filename, sep=",", header=None, names=["MovieID","UserID","Rating"], encoding = "ISO-8859-1")
    return data

def grouped(data):
    """here data grouped by user id, along with their meanvotes calculation"""
    groups = data.groupby(by=['UserID'])
    meanVotes= pd.DataFrame(columns=['userId','meanVote'])
    for name, group in groups:
        m=calculateMeanVote(group)
        meanVotes=meanVotes.append({'userId':name,'meanVote':m},ignore_index=True)
    return meanVotes

def calculateMeanVote(data):
    den= len(data.index)
    num = data['Rating'].sum()
    return (num/den)

def predictionTest(testData, g_train, trainingData):
    size=testData.shape
    users = trainingData['UserID'].unique()
    # create an empty dataframe of user by user to store weights
    # w=pd.DataFrame(index=users,columns=users)
    y_actual=trainingData['Rating'].tolist()
    y_predicted=[]
    n=trainingData.shape[0]
    # rows=size[0]
    rows=10
    for i in range(rows):
        print("Considering test data",testData.iloc[[i]])
        a= testData.iloc[[i]]['UserID']
        j= testData.iloc[[i]]['MovieID']
        userA_g = g_train['userId'] == a.tolist()[0]
        v_a_bar= g_train[userA_g]['meanVote']
        sumW=0
        sumprod=0
        # n is number of users who could possibly vote=> so get a list of unique voter/user ids and iterate through that list

        for iter in users:
            temp=trainingData['UserID']==iter
            t2=trainingData['MovieID']==j.tolist()[0]
            v_i_j=trainingData[temp & t2]['Rating']
            if v_i_j.empty:
                # print("v_i_j empty, no vote from user ",iter," for movie ",j)
                continue
            t3= g_train['userId'] == iter
            v_i_bar = g_train[t3]['meanVote']
            # set of movies in testdata
            mov_test= testData['MovieID'].unique()
            # set of movies in training data
            mov_train= trainingData['MovieID'].unique()
            # intersection
            movs=pd.Series(list(set(mov_test).intersection(set(mov_train))))
            w = 0
            num=0
            den1=0
            den2=0
            for mId in movs:
                # vaj= rating of movie j by user a
                t1_=trainingData['UserID']==a.tolist()[0]
                # vabar= meanvote by user a-calculated already
                # vibar= calculated already
                # vij = rating of movie j by user i
                t2_ = trainingData['MovieID'] == mId
                vij = trainingData[temp & t2_]['Rating']
                if vij.empty:
                    # print("vij empty, no vote from user ", iter, " for movie ", j)
                    continue
                vaj= trainingData[t1_ & t2_]['Rating']
                if vaj.empty:
                    # print("vaj empty, no vote from user ", a[0], " for movie ", j)
                    continue
                one=(vaj.tolist()[0]-v_a_bar.tolist()[0])*(vij.tolist()[0]-v_i_bar.tolist()[0])
                num=num+one
                two=(vaj.tolist()[0]-v_a_bar.tolist()[0])*(vaj.tolist()[0]-v_a_bar.tolist()[0])
                den1=den1+two
                three=(vij.tolist()[0]-v_i_bar.tolist()[0])*(vij.tolist()[0]-v_i_bar.tolist()[0])
                den2=den2+three
            if (den1!=0) and (den2!=0):
               w = num / math.sqrt(den1 * den2)
               print("W", w)
               print("--------------------------------")
            sumprod = sumprod + w * (v_i_j.tolist()[0] - v_i_bar.tolist()[0])
            sumW = sumW + math.fabs(w)
        k=0
        if(sumW!=0):
            k=1/sumW
        p=v_a_bar+ (k)*(sumprod)
        print("predicted value:",p)
        y_predicted.append(p.tolist()[0])
    results(y_actual[0:rows],y_predicted)

def results(y_actual, y_predicted):
    print("Actual: ",y_actual)
    print("Predicted: ",y_predicted)
    rms = math.sqrt(mean_squared_error(y_actual, y_predicted))
    print("Root Mean Squared Error=",rms)
    mae  = mean_absolute_error(y_actual,y_predicted)
    print("Mean Absolute Error=",mae)

if __name__=="__main__":
    trainingDataFile="../netflix/TrainingRatings.txt"
    trainingData= readFile(trainingDataFile)
    print("got training data....")
    g_train= grouped(trainingData)
    print("done grouping...")
    testingDataFile ="../netflix/TestingRatings.txt"
    testingDataComplete=readFile(testingDataFile)
    testData=testingDataComplete.loc[:,["MovieID","UserID"]]
    print("got test data...")

    print("beginning prediction...")
    predictionTest(testingDataComplete, g_train, trainingData)