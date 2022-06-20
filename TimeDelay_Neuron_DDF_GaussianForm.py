#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:34:54 2021

@author: randallclark
"""


import numpy as np
from sklearn.cluster import KMeans


class Gauss:    
    """
    First Make centers for your Training. It useful to do this step seperately as it can take a while to perform for large data sets,
    and if the user wants to perform multiple trainings to select good hyper parameters, it would be unnecessary to recalculate centers
    every time
    inputs:
        Xdata - 1 Dimensional Data that will be used for K-means clustering.
        P - number of centers you want
        D - The number of dimensions you want
        tau - the time delay you want
    """
    def KmeanCenter(self,Xdata,P,D,length,tau):
        XTau = np.zeros((D,length))
        #Here we generate the time delayed data
        for d in range(D):
            XTau[D-1-d] = Xdata[tau*d:length+tau*d]
        centers = KMeans(n_clusters=P, random_state=0).fit(XTau.T).cluster_centers_
        return centers
    
    """
    This is how Data Driven Forecasting is performed when a Gaussian function is used as the Radial basis Function for
    interpolation and an additional 1st order polynomial term is included for the stimuli. Ridge regression is used for training. 
    The Gaussian form is:
        e^[(-||X(n)-C(q)||^2)*R]
    inputs:
        Xdata - the 1 dimensional data to be used for training and centers
        length - amount of time to train for
        centers - the input for the centers (these can be made using the KmeanCenter function above)
        beta - the regularization term in Ridge Regression
        R - The coefficient in the exponent in the Gaussian RBF
        D - the Dimension (how many time delays there will be + the original data)
        stim - the stimulus that will be applied
        tau - the length of the time delay inbetween dimensions
    """
    def FuncApproxF(self,Xdata,length,centers,beta,R,D,stim,tau):
        #We will make our time delay data. This assumes Xdata and stim are 1 dimensional data sets
        #Xdata will need to be 1 point longer than length to make Y
        print("Length+1: "+str(length+1))
        print("Shape of Xdata: "+str(Xdata.shape))
        XTau = np.zeros((D,length+1))
        print("Shape of XTau: "+str(XTau.shape))
        print("Sample Xdata over length:" +str((tau*(D-1),length+1+tau*(D-1))))
        for d in range(D):
            XTau[D-1-d] = Xdata[tau*d:length+1+tau*d]
        
        
        #To Create the F(x) we will need to generate X and Y, then give both to the Func Approxer
        self.D = D
        self.tau = tau
        XdataT = XTau.T
        Ydata = self.GenY(XTau,length,D)
        NcLength = len(centers)     
        
        #Create the X matrix with those centers
        PhiMatVolt = self.CreatePhiMatVoltage(XdataT,centers,length,NcLength,R,stim,D)

        #Perform RidgeRegression
        YPhi = np.matmul(Ydata,PhiMatVolt.T)
        PhiPhi = np.linalg.inv(np.matmul(PhiMatVolt,PhiMatVolt.T)+beta*np.identity(NcLength+1))
        W = np.matmul(YPhi,PhiPhi)
        self.W = W
            
        #Now we want to put together a function to return
        def newfunc(x,stim):
            f = np.matmul(W[0:NcLength],np.exp(-(np.linalg.norm((x-centers),axis=1)**2)*R))
            f = f + W[NcLength]*(stim[0]+stim[1])*0.5
            return f
        
        self.FinalX = XTau.T[length-1]
        return newfunc
        
    """
    Predict ahead in time using the F(t)=dx/dt we just built
    input:
        F - This is the function created above, simply take the above functions output and put it into this input
        PreLength - choose how long you want to predict for
        stim - This will be the stimulus that will be applied for forecasting. stim must be 1 value longer than PreLength
        PData - This will be the set of data that will determine the first tau*(D-1)+1 values. This term is very specific because
                we need an initial condition for each of the time delay coordinates. PData is a 1 dimensinonal vector that must be at
                least tau*(D-1)+1 long.
    """
    def PredictIntoTheFuture(self,F,PreLength,stim,PData):         
        D = self.D
        tau = self.tau
        Prediction = np.zeros(PreLength+tau*(D-1)+1)
        #Set the first tau*(D-1)+1 values to be the initial condition
        Prediction[0:tau*(D-1)+1] = PData[0:tau*(D-1)+1]

        #Predict Forward PreLength steps. The leading time delay dimension is predicted upon, and the previous steps of the leading term
        #become the values for the time delayed dimensions.
        for t in range(0,PreLength):
            Input = np.flip(Prediction[t:t+tau*D:tau])
            Prediction[t+tau*(D-1)+1] = Prediction[t+tau*(D-1)]+F(Input,stim[t:t+2])
        return Prediction
    
    """
    These are secondary Functions used in the top function. You need not pay attention to these unless you wish to understand or alter
    the code.
    """
    def CreatePhiMatVoltage(self,X,C,Nlength,NcLength,R,stim,D):
        #This code is calculating the X matrix, the RBF value at all times and with all centers, and the last row is for the current coefficient
        Mat = np.zeros((NcLength+1,Nlength),dtype = 'float64')
        for i in range(NcLength):
            CC = np.zeros((Nlength,D))
            CC[:] =  C[i]
            Diff = X[0:Nlength]-CC
            Norm = np.linalg.norm(Diff,axis=1)
            Mat[i] = Norm
        Mat[0:NcLength][0:Nlength] = np.exp(-(Mat[0:NcLength][0:Nlength]**2)*R)
        Mat[NcLength] = 0.5*(stim[0:Nlength].astype('float64')+stim[1:1+Nlength].astype('float64'))
        return Mat


    def GenY(self,Xdata,length,D):
        #This code is self explanatory. Take the difference
        Y = Xdata[0][1:length+1]-Xdata[0][0:length]
        return Y
    
    
    
    
    
    
    
    
    
    
    