#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:10:25 2022

@author: bar
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
import warnings
from scipy.sparse import issparse
#%%
def ConvertTimeString(seconds):
    if seconds>60 and seconds<3600:
        seconds=np.round(seconds/60,1)
        return str(seconds)+" mins"
    elif seconds>3600:
        seconds=np.round(seconds/3600,1)
        return str(seconds)+" hours"
    else:
        seconds=np.round(seconds,1)
        return str(seconds)+" seconds"
#%%
def ComputeLogMaxLikelood(delta,sigma):
    """ delta is the difference between observation and best fitting value and sigma is the uncertinaity """
    #n=len(sigma)
    return -(np.pi/2)*np.log(2*np.pi)-len(delta)*np.log(sigma)-np.sum(delta**2)/(2*sigma)

#%%
def Compute_log_likelihood(x_u, standard_deviations):
    """
    Compute the log-likelihood for a Gaussian distribution, supporting both variable and constant standard deviations.
    
    Parameters:
    - x_u: NumPy array of differences (x_i - mu).
    - standard_deviations: NumPy array or scalar of standard deviations (sigma). If scalar, it's expanded to a vector.
    
    Returns:
    - The log-likelihood of the observations given mu and the standard_deviations.
    """
    # Convert standard_deviations to a NumPy array if it's not already
    standard_deviations = np.array(standard_deviations)
    
    # If standard_deviations is a scalar (len == 1), expand it to a vector of the same length as x_u
    if standard_deviations.size == 1:
        standard_deviations = np.full_like(x_u, standard_deviations)
    
    variance = standard_deviations ** 2
    n = len(x_u)
    
    # Compute the log-likelihood
    log_likelihood = - np.sum(np.log(standard_deviations * np.sqrt(2 * np.pi))) - np.sum((x_u ** 2) / (2 * variance))
    return log_likelihood


#%%
class newtonInversion:
    """ this class gets fowardmodeling package with Foward function that computes the result for paramters model m and computed the newtown
    inversion based on eq. 3.51 and logthrim denoted in the book Inverse Problem Theory by Tarantola. 
    It is bascailly computed the Jacobian based on papramter delta for problem and by that run until it converages"""
    
    def __init__(self,fowardModeling,m0,Cd,observation,Cm,delta=1e-6,maxIterations=20,mu=0.5,minStepImprovement=1e-8,Cdinv=None):
        
        """ fowardModeling class with function Foward that is capable of run model when getting vector of input paramters m
        mo - inital guests
        Cd - matrix len(m0)*len(mo) with uncertincity in m
        Observation - Matrix with observation. fowardModeling.Foward(m) will need to return a vector of the same length
        Cm - matrix with uncertinrty in observation
        delta - compute Jacobian with delta 
        maxIterations - stop once getting to this number of Iterations
        mu - fixed paratmer that is used in getting the final m
        minStepImprovement - once reached this impormpment stop run
        """
        
        self.m0=self.CheckSpace(np.array(m0))
        

        
        self.mu=mu
        #self.fowardModeling=fowardModeling
       
        if Cdinv is not None:
            self.Cdinv=Cdinv
        else:
            self.Cdinv=np.linalg.inv(Cd)
        
       
        self.delta=self.TurnDeltaToVector(delta)
        self.CheckForNans()
        if self.valueOfmNotToTest is not None:
            Cm=self.DeleteRowsAndCOlumusFromMatrix(Cm,self.valueOfmNotToTest)
        self.Cminv=self.CheckSpace(np.linalg.inv(Cm))
        self.Cm=Cm
        self.Cd=Cd
        
        self.observation=observation
        self.CheckForInputsErros()
        self.maxIterations=maxIterations
        self.minStepImprovement=minStepImprovement
        self.fowardModeling=fowardModeling
        
        print ("Loaded almost everything... keep your pants on!")
        self.RunInversion()
        
    def CheckSpace(self,array):
        return array
        
        
    def TurnDeltaToVector(self,delta):
        if isinstance(delta, float):
            delta=np.ones(len(self.m0))*delta
        
        elif isinstance(delta, list):
            delta=np.array(delta)          
          
        elif isinstance(delta, np.ndarray):
            pass
  
        else:
            raise TypeError ("delta is not reconginzed! It needs to be a float/np.array/list")
           
        return delta
        
        
    def ComputeJacobian(self,m,delta,soultion_i):
        J=np.zeros([len(self.observation),len(m)])
        
        for i in range(len(m)):

            m_plus=m.copy()
            m_minus=m.copy()
            
            m_plus[i]=m[i]+delta[i]/2
            m_minus[i]=m[i]-delta[i]/2
            #print(m_plus)
            forward=soultion_i.Foward(m_plus)
            backward=soultion_i.Foward(m_minus)
            #print(m_minus)
        
            J[:,i]=(forward-backward)/delta[i]
        
        return J
    
    
    def ComputeDataLoss(self,misfit,m):
        
        lossData=np.transpose(misfit) @ self.Cdinv @ misfit
        lossModel=np.transpose(m-self.m0) @ self.Cminv @ (m-self.m0)
        loss=np.abs(lossData+lossModel)
        
        return loss
    
    def ComputeNextStep(self,G,m,misfit):
        
        AUX1 =np.linalg.inv( np.transpose(G) @ self.Cdinv @ G + self.Cminv)
        AUX2 = np.transpose(G) @ self.Cdinv  @ misfit + self.Cminv @ ( m - self.m0)
          
        dm=AUX1@AUX2
         
        return dm 
    
    def ComputeReslutionMatrix(self,G=None):
        if G is None:
            G=self.SaveJacobieans[-1]
            
        AUX1=G@self.Cm@ np.transpose(G)+self.Cd 
        AUX1_inv=np.linalg.inv(AUX1)
        R=self.Cm @ np.transpose(G) @ AUX1_inv @ G
        
        return R
            
 
    def CheckForNans(self):
        
        self.valueOfmNotToTest=[]
        for i in range(len(self.m0)):
            if np.isnan(self.m0[i]):
                self.valueOfmNotToTest.append(i)
                
        if len(self.valueOfmNotToTest)==0:
            self.valueOfmNotToTest=None
        else:
            delFromVectors=np.array(self.valueOfmNotToTest.copy())
            delFromVectors[::-1].sort()
            
            for i in delFromVectors:
                self.m0=np.delete(self.m0, i,axis=0)
                self.delta=np.delete(self.delta, i,axis=0)

            
                
    def DeleteRowsAndCOlumusFromMatrix(self,matrix,vector):
        vectorOfIndex=np.array(vector.copy())
        vectorOfIndex[::-1].sort()
        
        for i in vectorOfIndex:
            matrix=np.delete(matrix, i,axis=0)
            matrix=np.delete(matrix, i,axis=1)
                
        return matrix
    
    def LoadSoultion(self,fowardModeling,m):
        soultionLoaded=soultion(fowardModeling,m,self.observation,self.valueOfmNotToTest)
        
        return soultionLoaded
        
    
    
    def RunInversion(self,m0=None):
        self.step=[]
        self.misfitForStep=[]
        self.SaveJacobieans=[]
        self.uncertaintyForStep=[]
        self.uncertaintyForStep.append(np.linalg.inv(self.Cminv))
        
        if m0 is None:
            m=self.m0.copy()
        else:
            m=m0.copy()
            self.m0=m0.copy()
        
            
        fowardModeling=self.fowardModeling
                
        i=1
        

        stepMisfitImprovment=10000
        
        self.soultion_0=self.LoadSoultion(fowardModeling, m)
        soultion_i=self.LoadSoultion(fowardModeling, m)
        
        totalImprovment=100
        absoulteNormOld=100
        absoulteNormInprovment=10
        print("Almost there...let the inversion begin!",flush=True)
        warnings.filterwarnings("once")
        while (i  < self.maxIterations) and ((np.abs(absoulteNormInprovment) > self.minStepImprovement) or (absoulteNormInprovment<0)) :
            
            self.step.append(soultion_i.m)
            self.misfitForStep.append(soultion_i.MisfitSum())              
            #misfit=self.ComputeMisfit(m)
            misfit=soultion_i.misfit
            #totalMisfitOld=soultion_i.MisfitSum()
            t1=time.time()
            G=self.ComputeJacobian(soultion_i.m,delta=self.delta,soultion_i=soultion_i)
            t2=time.time()
            print ("\t \t Computed Jacobiean in "+ConvertTimeString(t2-t1),flush=True)
            
            t1=time.time()
            dm=self.ComputeNextStep(G,m,misfit)
            t2=time.time()
            print ("\t \t Computed next point in "+ConvertTimeString(t2-t1),flush=True)
            
            dm=-1*dm*self.mu
            t1=time.time()
            soultion_i=soultion_i+dm
            t2=time.time()
            print ("\t \t Computed soultion for next  point in " +ConvertTimeString(t2-t1),flush=True)
            #m=m-self.mu*dm
            #loss=self.ComputeDataLoss(misfit,m)
            #loss=self.ComputeDataLoss(misfit,soultion_i.m)
            
            #stepImprovement=np.abs(loss-lossPrevious)/loss
            totalImprovment=(100*soultion_i.MisfitSum()/self.soultion_0.MisfitSum())
            #stepMisfitImprovment=totalImprovmentOld-totalImprovment
            if i>1:
                stepMisfitImprovment=(self.misfitForStep[-2]-self.misfitForStep[-1])/self.misfitForStep[-2]
                
            absoulteNormInprovment=absoulteNormOld-totalImprovment
            absoulteNormOld=totalImprovment.copy()
            #totalImprovmentOld=totalImprovment.copy()
            
            t1=time.time()
            self.uncertaintyForStep.append(self.ComputeUncertaintyForSoultion(G))
            t2=time.time()
            print ("\t \t Computed uncretinaity for soultions in " +ConvertTimeString(t2-t1),flush=True)
            
            
            print ("\t i= "+str(i) + " total misfit: " +str(np.round(totalImprovment,2)) +"%" +"     step improvment:" +str(np.round(absoulteNormInprovment,1))+"%",flush=True)
            self.SaveJacobiean(G)

            #lossPrevious=loss
            i=i+1
            
            
        
        self.step.append(soultion_i.m)
        self.misfitForStep.append(soultion_i.MisfitSum())    
        
        self.step=np.array(self.step)
        self.misfitForStep=np.array(self.misfitForStep)
        
        self.mBest=soultion_i.m
        self.bestSoultion=soultion_i
        
        self.mUncertainty=self.ComputeUncertaintyForSoultion(G)
        print ("Done!!! stoped at i  "+str(i) +" with best soultion!",flush=True)#" and  step improvement of "+str(np.round(stepImprovement,1)))
        #print(soultion_i.m)
        #print ("with uncertinty of ",flush=True)
        #print(np.diag(self.mUncertainty))
        print (" best fit soultion has misfit lower by " +str(np.round(totalImprovment,2)) + " step improvment:" +str(np.round(stepMisfitImprovment,3)),flush=True)
        
        self.uncertaintyForStep=np.array(self.uncertaintyForStep)
        
    def ReturnBestAndIntialSoultions(self):
        return self.soultion_0,self.bestSoultion
        
    def SaveJacobiean(self,G):
        self.SaveJacobieans.append(G)
        
    def PlotMisFitForStep(self,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
            
        ax.plot(100*self.misfitForStep/self.misfitForStep[0],marker='o',markersize=4)
        ax.set_xlabel("Iteration # ")
        ax.set_ylabel("Misfit improvement [%] ")



        
    def ComputeUncertaintyForSoultion(self,G):
        return np.linalg.inv(np.transpose(G) @ self.Cdinv @ G + self.Cminv)
    
    def CheckForInputsErros(self):
        if len(self.delta) != len(self.m0):
            print(self.delta)
            print(self.m0)
            raise TypeError("length of m0 and delta is not the same")

            
        if len(np.diagonal(self.Cminv))!= len(self.m0):
            raise TypeError("length of Cm and m0 is not the same")
            
        if issparse(self.Cdinv):
            if (self.Cdinv.shape[0] != len(self.observation)):
                raise TypeError("length of Cd and observations is not the same")
        else:
            
            if len((self.Cdinv))!= len(self.observation):
                raise TypeError("length of Cd and observations is not the same")
            
    
    def SaveInversionResult(self, filename, additional_data=None):
        """Save Inversion results, including optional additional data if not empty."""
        
        # Your existing attributes
        G = self.SaveJacobieans
        soultion = self.step
        std = self.uncertaintyForStep
        Cm = self.Cm
        Cminv = self.Cminv
        Cd = self.Cd
        Cdinv = self.Cdinv
        
        # Base data to save
        data_to_save = {
            'G': G, 
            'meanSoultion': soultion, 
            'stdSoultion': std,
            'Cm': Cm, 
            'Cminv': Cminv, 
            'Cd': Cd, 
            'Cdinv': Cdinv
        }
        
        # If additional_data is provided and not empty, merge it with the base data
        if additional_data and isinstance(additional_data, dict) and additional_data:
            data_to_save.update(additional_data)
        
        # Save the combined data
        np.savez_compressed(filename, **data_to_save)
#%%
class newtonInversionBetterJac(newtonInversion):
    def ComputeJacobian(self,m,delta,soultion_i):
        J=np.zeros([len(self.observation),len(m)])
        
        for i in range(len(m)):

            m_plus=m.copy()
            m_minus=m.copy()
            
            m_plus[i]=m[i]+delta[i]/2
            m_minus[i]=m[i]-delta[i]/2
            #print(m_plus)
            forward_misfit=soultion_i.Misfit(m_plus)
            backward_mistit=soultion_i.Misfit(m_minus)
            #print(m_minus)
        
            J[:,i]=(forward_misfit-backward_mistit)/delta[i]
        
        return J
    
#%%
class newtonInversionLogSpace(newtonInversion):

    
    def LoadSoultion(self,fowardModeling,m):
        soultionLoaded=soultionLogSpace(fowardModeling,m,self.observation,self.valueOfmNotToTest)
        
        return soultionLoaded
    
#%%
class newtonInversionWithMisfit(newtonInversion):
    def LoadSoultion(self,fowardModeling,m):
        soultionLoaded=soultionWithDifferentMisfit(fowardModeling,m,self.observation,self.valueOfmNotToTest)
        
        return soultionLoaded
            
#%%
class soultion:
    def __init__(self,foward,m,d_obs,addNantoM=None):
        self.foward=foward
        self.m=m
        self.d_obs=d_obs
        self.addNantoM=addNantoM
        self.RunSoultion(m)
        
    def RunSoultion(self,m):
        
        self.result=self.Foward(m)
        self.misfit=self.Misfit()
        #self.misfit=self.Foward(m)
        
    def __add__(self,dm):
        newM=self.m+dm

        #newSoultion=soultion(self.foward,newM,self.d_obs,addNantoM=self.addNantoM)
        self.m=newM
        self.RunSoultion(newM)
        
        return self
    
    def MisfitSum(self):
        #return np.sum(np.abs(self.misfit))
        #return np.sum(np.transpose(self.misfit) @ (self.misfit))
        return np.linalg.norm(self.misfit)
    
    def ComputeRMS(self,misfit=None):
            if misfit is None:
                misfit=self.misfit
            return np.sqrt(np.mean(misfit**2))
        
    def FromMisfitSumToRms(self,misfitSum):
        return misfitSum/ np.sqrt(self.misfit.size)
    
    def TakeCareOfm(self,m):
        m=np.array(m,dtype=float)
        self.addNantoM.sort()

        for i in self.addNantoM:    
            m=np.insert(m, i, np.nan)
            
        return m
            

    def Foward(self,m):
        if self.addNantoM is not None:
            m=self.TakeCareOfm(m)
            
        return self.foward.Foward(m)
        
    def Misfit(self,m=None):
        if m is None:
            return (self.result-self.d_obs)
        else:
            return (self.Foward(m)-self.d_obs)
    
    def PlotMisfit(self,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
        
        ax.set_title(str(self.m))
        ax.scatter(self.d_obs,self.misfit,s=1,c=self.misfit) 
        #ax.plot(self.n) 
        
        ax.set_xlabel("Observation")
        ax.set_ylabel("Misfit [predicted results minus observations]")
        
#%%
class soultionLogSpace(soultion):

        
    def Foward(self,m):
        if self.addNantoM is not None:
            m=self.TakeCareOfm(m)
        m=np.exp(m)    
        return self.foward.Foward(m)
    
    def CheckSpace(self,array):
        return array
    
        
        
#%%
class soultionWithDifferentMisfit(soultion):
    def Misfit(self,m=None):
        
        return self.foward.Misfit(m)
        
    
        
#%%
class gridSearch:
    
    """ This is a simple class get a list of parametes class that can compute fowardModeling and a vector of observation
    it then compute a matrix of all possible observations and find the min value by that. fowardModeling has the have the function Foward that 
    returns a vector of the same length as observation"""
    
    def __init__(self,*parameters,fowardModeling,observation,Cd):
    
        """ *paramters - list of paramters for example [1,2,4],[5,19]
    fowardModeling class with Foward that gets vector of parmters (has to be the same length of paramters) and return vector of the same length 
    observation
    observation - vector of observation
    Cd - vector of unceriticty in observation """
    
        self.fowardModeling=fowardModeling
        self.observation=observation
        self.Cd=Cd
        self.shapeOfMatrix=[]
        
        for par in parameters:
            self.shapeOfMatrix.append(len(par))
            
        arraysOfParamters=np.meshgrid(*parameters)
        self.paramtersArray=self.GetMeshParamtersReturnArray(arraysOfParamters)
        
        self.RunInversion(self.paramtersArray)
        
        
    def RunInversion(self,paramtersArray):
        self.misfit=np.zeros_like(self.paramtersArray[0,:])
        
        N=len(paramtersArray[0,:])
        
        for i in range(N):
            
            if i%10==0:
                print ("iteration " +str(i))
                
            m=paramtersArray[:,i]
            d=self.fowardModeling.Foward(m)
            self.misfit[i]=np.sum((d-self.observation)/self.Cd)
            
        self.FindBestFit(self.misfit)
            
    def FindBestFit(self,misfit):
        bestInd=np.where(np.abs(misfit)==np.min(np.abs(misfit)))
        self.mBest=self.paramtersArray[:,bestInd]
        
        self.misfit=misfit.reshape(self.shapeOfMatrix)
        
    
    def GetMeshParamtersReturnArray(self,arraysOfParamters):
        
        N=len(arraysOfParamters)
        numOfParamtersToRunIversion=len(arraysOfParamters[0].ravel())
        
        
        paramtersArray=np.zeros([N,numOfParamtersToRunIversion])
        
        for i in range(N):
            paramtersArray[i,:]=arraysOfParamters[i].ravel()
            
        return paramtersArray
     

#%% 
class misfitForGrid:
    def __init__(self,tablewithMisfit,STDvalue=0.68):
        print("Computing probbabilty for dataframe")
        tablewithMisfit=self.ComputeProbability(tablewithMisfit)
        print("Order dataframe by probabilty probbabilty for dataframe")
        self.tablewithMisfit=self.OrderValuesByChi(tablewithMisfit)
        print("Finding index for STD")
        self.indexToSTD=self.FindIndexThatCorrspondsToSumOfValue(self.tablewithMisfit['prob'],STDvalue)
        self.STDvalue=STDvalue
        
    def ComputeProbability(self,tablewithMisfit):
        tablewithMisfit['prob']=np.exp(-1*tablewithMisfit['chi']/2)
        tablewithMisfit['prob']=tablewithMisfit['prob']/np.sum(tablewithMisfit['prob'])
        
        return tablewithMisfit
        
    def OrderValuesByChi(self,dataFrame):
        
        dataFrame=dataFrame.sort_values('chi')
        dataFrame=dataFrame.reset_index()
        
        return dataFrame
        
        
    def PrintSTD(self):
        columns=self.tablewithMisfit.columns
        
        for column_i in columns:
            print(str(column_i)+": Best value "+str(self.tablewithMisfit.loc[0,column_i]))
            minValue=np.min(self.tablewithMisfit.loc[0:self.indexToSTD,column_i])
            maxValue=np.max(self.tablewithMisfit.loc[0:self.indexToSTD,column_i])
            print(str(column_i)+": min value "+str(minValue)+" max value " +str(maxValue) +" so STD is "+ str((maxValue-minValue)/2))
            print("\n")
        
        
        
    def FindIndexThatCorrspondsToSumOfValue(self,probVector,value=0.68):
        probability=0
        i=0
        while probability<value and i < len(probVector):
            probability=np.sum(probVector[0:i])
            i=i+1
            
        return i
    
    def PlotPairPlots(self,renameColumns=None,**args):
        tableToPlot=self.tablewithMisfit[0:self.indexToSTD]
        
        if renameColumns is not None:
            tableToPlot=tableToPlot.rename(columns=renameColumns)
        fig=sns.pairplot(tableToPlot,**args)
        
        return fig
    
    def AddSumToDataFrame(self):
        self.tablewithMisfit['sum']=0
        
        for i in range(len(self.tablewithMisfit)):
            self.tablewithMisfit.loc[i,'sum']=np.sum(self.tablewithMisfit.loc[0:i,'prob'])
            
            
    def GenerateNewMisfitObjectToSTDIndex(self):
        newObject=misfitForGrid(self.tablewithMisfit.loc[0:self.indexToSTD],STDvalue=self.STDvalue)
        
        return newObject
        
        
            
            
        
        
        
    
        
        
        
        
        