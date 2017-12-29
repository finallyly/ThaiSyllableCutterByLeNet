#-*-coding:utf8-*-
import sys;
import numpy as np;

def LoadTrainData(infilename):
    with open(infilename,"r") as f:
        labels = [];
        data=[];
        datapad=[];
        for line in f:
            line = line.strip();
            col =line.split("\t");
            labelVec=[0]*4;
            labelVec[int(col[-2])]=1;
            labels.append(labelVec);
            data.append([int(m) for m in col[:100]]);

    datapad=[m+[0]*(28*28-100) for m in data];
    return np.array(datapad),np.array(labels);

def LoadTestData(infilename):
    with open(infilename,"r") as f:
        X=[];
        Y=[];
        labels=[];
        TempX=[];
        TempY=[];
        TempLabels=[];

        for line in f:
            line = line.strip();
            col = line.split("\t");
            if len(col)==102:
                labelVec=[0]*4;
                labelVec[int(col[-2])]=1;
                TempLabels.append(col[-1]);
                TempY.append(labelVec);
                x1=[int(m) for m in col[:100]];
                x2=x1+[0]*(28*28-100);
                TempX.append(x2);
            else:
                X.append(TempX);
                Y.append(TempY);
                labels.append(TempLabels);
                TempX=[];
                TempY=[];
                TempLabels=[];
    return np.array(X),np.array(Y),np.array(labels);






def batch_data(source,target,batch_size):
    for batch_i in range(0,len(source)//batch_size):
        start_i = batch_i*batch_size;
        X = source[start_i:(batch_i+1)*batch_size];
        Y = target[start_i:(batch_i+1)*batch_size];
        #print len(X),len(Y);
        yield X,Y;

def GetLabelById(labelId):
    if labelId==0:
        return "S";
    elif labelId==1:
        return "B";
    elif labelId==2:
        return "M";
    else:
        return "E";






if __name__=="__main__":
    datapad,labels=LoadTrainData("train.vector.20171207.txt");
    #X,Y,labels=LoadTestData("test.vector.txt");
    #print labels[0];
    #print len(X);
    #print labels[0];
    for x,y in batch_data(datapad,labels,2767):
        print len(x);
        print len(y);

