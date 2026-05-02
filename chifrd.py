#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:42:21 2022

@author: bar
"""
import numpy as np
import upliftFunctions
import numba as nb
import math
import time
import copy
import warnings
import inversions
from scipy.sparse import diags
#%%
def ComputeLogMaxLikelood(*args):
    return inversions.ComputeLogMaxLikelood(*args)

#%%

def ComputeAICandBIC(fowardModelFilename,kx,Cm_list,angle=None,splineDegree=3,**kwargs_for_inversion):
    AIC=np.zeros_like(kx)
    BIC=np.zeros_like(kx)
    inversions=[]
    
    
    for i,kx_i in enumerate(kx):
        print( " Inverting for knots:"+str(kx_i)+ " ! \n\n")
        f=LoadFowardModel(fowardModelFilename+".npz",kx=kx_i,splineDegree=splineDegree)
        Cm_i=Cm_list[i]
        if angle is not None:
            f=f.RotateFowardCorrd(angle)
            
        inversion=InvertLandScape(foward=f,Cm=Cm_i,**kwargs_for_inversion)
        inversions.append(inversion)
        delta=f.Foward(inversion.step[-1,:])-f.selectZ+f.minElevation
        
        maxLikelhood=ComputeLogMaxLikelood(delta,3)
        
        AIC[i]=maxLikelhood-len(inversion.step[0,:])
        BIC[i]=maxLikelhood-0.5*len(inversion.step[0,:])*np.log(len(inversion.Cd))
        
    return AIC,BIC,inversions
        
        
            

#%%
def InvertLandScape(foward,d_inital=10,m_initial=1,randomNoise=0,minStepImprovement=0.1,mu=0.1,numOf1DParamters=None,maxIterations=50,Cm=None,m0=None,spraseMatrix=False,Cdinv=None,Cd=None,std_m_n=0.1,std_as=0.5):
    print("Setting paramters for inversions",flush=True)
    m=0.5;n=1;slope=0.1
    #foward=fowardProp.LoafDietFowardBspline2D(fowardModelFilename,kx=kx,ky=ky)
    randomNoise=np.random.normal(0, randomNoise, len(foward.selectZ))
    errorPerPixel=d_inital
    
    if Cdinv is None:
        if spraseMatrix is False:
            Cdinv=np.diag((np.ones(len(foward.selectZ))*(1/errorPerPixel)**2))
            Cd=np.diag((np.ones(len(foward.selectZ))*(errorPerPixel)**2))
        else:
            diagonal_elements = (np.ones(len(foward.selectZ)) * (1 / errorPerPixel) ** 2)
            Cdinv = diags(diagonal_elements, 0)
            Cd=((np.ones(len(foward.selectZ))*(errorPerPixel)**2))
            Cd = diags(Cd, 0)
            
    numberOfK=foward.NumOfKParamters()
    
    if m0 is None:

        if isinstance(foward.UpliftFunction, upliftFunctions.bsplineUplift1D):
            m0=np.ones(foward.UpliftFunction.kx+foward.UpliftFunction.splineDegree+numberOfK)
        elif isinstance(foward.UpliftFunction, upliftFunctions.bsplineUplift2D):
            m0=np.ones((foward.UpliftFunction.ky+3)*(foward.UpliftFunction.kx+3)+numberOfK)
        elif foward.UpliftFunction is None:
            m0=np.array([])
            
        else:
            raise TypeError("problem with the type of uplift functions")
            
        if isinstance(foward,BsplineFoward1DOnlyUplift) or isinstance(foward,BsplineFoward2DOnlyUplift):
            m0=np.insert(m0,0,slope)
        elif isinstance(foward,BsplineFoward1DOnlyUpliftWithRatio) or isinstance(foward,BsplineFoward2DOnlyUpliftWithRatio):
            m0=np.insert(m0,0,m);m0=np.insert(m0,1,slope)
        else:
            m0=np.insert(m0,0,m);m0=np.insert(m0,1,n);m0=np.insert(m0,2,slope);
            
    
        
    
    
    std=np.ones_like(m0)*m_initial
    std=std.astype(np.float64)
    std[0]=std_m_n
    std[1]=std_m_n
    std[2]=std_as
    
    
    m0=m0.astype(np.float64)
    if Cm is None:
        Cm=np.diag(((std)**2)).astype(np.float64)
    Cdinv=Cdinv.astype(np.float64)
    
    delta=1e-6
    newtonInversion=inversions.newtonInversionBetterJac(fowardModeling=foward,m0=m0,Cd=Cd,
                                        observation=foward.selectZ+randomNoise-foward.minElevation,
                                            Cm=Cm,delta=delta,maxIterations=maxIterations,Cdinv=Cdinv,mu=mu,minStepImprovement=minStepImprovement)
    
    
    #newtonInversion=inversions.newtonInversion(fowardModeling=foward,m0=m0,Cd=Cd,
    #                                    observation=foward.selectZ+randomNoise-foward.minElevation,
    #                                        Cm=Cm,delta=delta,maxIterations=maxIterations,Cdinv=Cdinv,mu=mu,minStepImprovement=minStepImprovement)
    
    return newtonInversion

       
#%%

def LoadData(filename):
    """
    Load data from a file into a dictionary.

    :param filename: The path to the file containing the data.
    :return: Dictionary containing loaded data.
    """
    content = np.load(filename,allow_pickle=True)
    keysToLoad=['A','dX','Z','recs','stack','riverNodes','XXflat','YYflat','minElevation','pixelMask','shape','riverMask']
    data_dict = {key: content[key] for key in keysToLoad}
    return data_dict

def LoadFowardModel(filename,kx,spatialErodabilty=False,splineDegree=3,ky=None):
    fowardDict=LoadData(filename)
    
    if spatialErodabilty is False and ky is None:
        foward=BsplineFoward1D(**fowardDict,kx=kx,splineDegree=splineDegree)
    elif spatialErodabilty is True and ky is None:
        foward=BsplineFoward1DWithErodibailty(**fowardDict,kx=kx,splineDegree=splineDegree)
    elif spatialErodabilty is False and ky is not None:
        foward=BsplineFoward2D(**fowardDict,kx=kx,splineDegree=splineDegree,ky=ky)
    elif spatialErodabilty is True and ky is not None:
        foward=BsplineFoward2DWithErodibailty(**fowardDict,kx=kx,splineDegree=splineDegree,ky=ky)
    else:
        raise TypeError("Opps")
        
    return foward
        
def LoadDietFowardOnly_m_n_slope(filename,kx=None):
    fowardDict=LoadData(filename)
    return deitFowardOnly_m_n_slope(**fowardDict)

def LoadDietFowardOnlyUplift(filename,kx,splineDegree=3):
    print( " please notice you need to foward.m and foward.n")
    fowardDict=LoadData(filename)
    return BsplineFoward1DOnlyUplift(**fowardDict,kx=kx,splineDegree=splineDegree)

def LoadDietFowardOnlyRatio(filename,kx=None,splineDegree=3):
    print( " please notice you need to foward.mn_ratio")
    fowardDict=LoadData(filename)
    return BsplineFoward1DOnlyUpliftWithRatio(**fowardDict,kx=kx,splineDegree=splineDegree)

def LoadDietFowardOnly_m_n_slope_k(filename,kx=None):
    fowardDict=LoadData(filename)
    return deitFowardOnly_m_n_slope_k(**fowardDict)


    


#%% 
def RotateCorrdinates(theta,x,y):
        theta=np.deg2rad(theta)
        xRotated=x*np.cos(theta)+y*np.sin(theta)
        yRotated=-x*np.sin(theta)+y*np.cos(theta)
        
        return xRotated,yRotated


#%%
class dietFoward1D:
    def __init__(self,A, dX, Z, recs, stack, riverNodes,riverMask,XXflat,YYflat,minElevation,K=1,height=None,kx=None,ky=None,pixelMask=None,shape=None,splineDegree=3):
        
        ##### all of thses variables below remain constant for one inversion. They would change if I
        ##### ran this inversion for different rivers 
        ##### every file I load will have different variables 
        self.riverMask=riverMask
        self.CheckAllArraysAreOfTheSameLength([A,dX,Z,recs,stack,XXflat,YYflat])
        self.riverNodes=riverNodes # river nodes
        self.Z=Z # - Z: 1D array of elevationz
        #self.A=A
        self.K=K 
        self.CorrectForHeight(height)
        self.CorrectForAreaForRivers(A)
        self.dX=dX #  - dX: 1D array of distance to rec
        self.recs=recs # - recs: 1D array of receiver indices
        self.stack=stack # - stack: 1D array of indices in stack order
        self.XXflat=XXflat #  coords of the fault in x direction
        self.YYflat=YYflat #  coords of the fault in y direction
        self.minElevation=minElevation # min elevation 
        
        self.A0=np.min(A)
        self.selectZ=self.Z[self.riverMask]
        self.Xrivers=self.XXflat[self.riverMask]
        self.Yrivers=self.YYflat[self.riverMask]
        self.indForMinChi=np.argmin(self.selectZ)
        self.pixelMask=pixelMask
        
        self.shape=shape
        
        self.LoadUplift(kx,ky,splineDegree)
        
    def CorrectForAreaForRivers(self,A):
        self.Amin=np.max(A[self.riverMask])
        self.A=A/np.max(A[self.riverMask]) #  - A: 1D array of drainage area
        
        
    def CorrectForHeight(self,height):
        pass
        
    def CheckAllArraysAreOfTheSameLength(self,listOfArrays):
        lengthOfFirstArray=len(listOfArrays[0])
        
        for array_i in listOfArrays[1:]:
            if len(array_i) != lengthOfFirstArray:
                raise ValueError("Arrays Should all be of the same length")
        
        
    def LoadUplift(self,kx=None,ky=None): # load Uplift function with corr. of fault 
        #self.UpliftFunction=upliftFunctions.autoGassiuanFunctions1D(x=self.XXflat)
        self.UpliftFunction=upliftFunctions.autoGassiuanFunctions1DWithConstantUplift(x=self.XXflat)
  
    def Foward(self,mParamters):
        """ m mParamters[0]
        n mParamters[1]
        and then uniform uplift is usually mParamters[2] 
        """
        

        m=mParamters[0]
        n=mParamters[1]
        slope=mParamters[2]
        upliftParamters=mParamters[3:]


    
        return self.ComputeSyenteticElevation(upliftParamters,slope=slope,m=m,n=n)
    
    def ComputeRMSForSoultion(self,mParamters,observation=None):
        if observation is None:
            observation=self.selectZ-self.minElevation
            
        invertedZ=self.Foward(mParamters)
        
        return ComputeRMS(invertedZ-observation)
    
    def Check_slope(self,slope_value):
        if slope_value<1e-64:
            slope_value=1e-64
        
            warnings.warn(" Warning set slope to be very small setting it to min value of 1e-64",UserWarning)
    
        return slope_value
    def Check_m_n(self,m_n_value):
        if m_n_value<0.01: #3e-2:
            m_n_value=0.01 #3e-2
            warnings.warn(" Warning set m or n to be very small setting it to min value of 0.01",UserWarning)
            
        # if m_n_value>3:
        #     m_n_value=3
        #     warnings.warn(" Warning set m or n to be large setting it to max value of 3",UserWarning)
        
            
        return m_n_value
        
    
    def Misfit(self,mParamters):
        result=self.Foward(mParamters)
        return result-self.selectZ
    
    def MisfitSumAbs(self,mParamters):
        return np.sum(np.abs(self.Misfit(mParamters)))
    
    def Misfit2Sum(self,mParamters):
        misfit=self.Misfit(mParamters)
        return misfit @ np.transpose(misfit)
    
    def ComputeSyenteticElevation(self,upliftParamters,slope,m,n):
        """ this is the model where I will be using fitting the best linear line and then looking for the distance of all points from that line""" 

        #chi,z=self.ReturnNormalizedChiAndZ(upliftParamters=upliftParamters,m=m,n=n)
        #chi,z=self.ReturnChiForRiversPar(upliftParamters=upliftParamters,m=m,n=n)

        #print(m)
        #print(n)
        #print(upliftParamters)

        chi,_=self.GetUpliftParamtersReturnChi(upliftParamters=upliftParamters, m=m, n=n)
        chi = chi[:,np.newaxis]    
        chi=np.transpose(chi)
        chi=chi[0,:]
        slope=self.Check_slope(slope)
        result=slope*chi
        

        return result
    
    def GetUpliftParamtersReturnChi(self,upliftParamters,m,n):
        "get uplift Paramters,m,n and return chi and z"

        uplift=self.UpliftFunction.Uplift(upliftParamters)
        chi,z=self.ComputeChi(uplift=uplift,m=m,n=n)
        
        return chi,z
        
        
    def ComputeAStar(self,uStarFull,m):
        """ this function compute A star while taking into  account only uplift shape"""
        #A_star=self.A*((1/uStarFull)**(1/m))
        inv_m=1.0/m
        A_star = compute_A_starWithNumba(self.A, uStarFull, inv_m)
        return A_star
        
    
    def ComputeChi(self,uplift,m,n):
        """ this function gets uplift per pixel,m,n and return chi and z"""
        m=self.Check_m_n(m)
        n=self.Check_m_n(n)
        
        uStar=uplift
        uStarFull=np.ones_like(self.A)
        uStarFull[self.riverMask]=uStar
        A_star=self.ComputeAStar(uStarFull,m)
        
        #mask = np.zeros_like(self.A, dtype = np.bool)
        #mask[self.riverNodes] = True
        #t1=time.time()

        chi=calculate_chiP(A_star, self.dX, self.Z,self.recs, self.stack, 1, m,n,self.riverMask)
        #print (time.time()-t1)
        chi = chi[self.riverMask]
        return chi-chi[self.indForMinChi],self.selectZ-self.minElevation
    
    def CopyWithDifferentMask(self,mask):
        newFoward=copy.deepcopy(self)
        newFoward.riverMask=mask
        riverNodes=np.arange(len(self.XXflat))
        
        newFoward.riverNodes=riverNodes[mask]
        newFoward.selectZ=newFoward.Z[newFoward.riverMask]
        newFoward.Xrivers=newFoward.XXflat[newFoward.riverMask]
        newFoward.Yrivers=newFoward.YYflat[newFoward.riverMask]
        newFoward.LoadUplift(kx=self.UpliftFunction.kx,ky=self.UpliftFunction.ky)
        
        
        print(" copied the object and recomputed riverNodes, make sure it's right",flush=True)
        
        return newFoward
    
    def NumOfKParamters(self):
        return 0
    
    def RotateFowardCorrd(self,angle):
        newFoward=copy.deepcopy(self)
        xx,yy=RotateCorrdinates(angle,newFoward.XXflat,newFoward.YYflat)
        newFoward.XXflat=xx
        newFoward.YYflat=yy
        newFoward.Xrivers=newFoward.XXflat[newFoward.riverMask]
        newFoward.Yrivers=newFoward.YYflat[newFoward.riverMask]
        
        if isinstance(self.UpliftFunction,upliftFunctions.bsplineUplift1D):
            newFoward.LoadUplift(kx=self.UpliftFunction.kx,splineDegree=self.UpliftFunction.splineDegree)
        elif isinstance(self.UpliftFunction,upliftFunctions.bsplineUplift2D):
            newFoward.LoadUplift(kx=self.UpliftFunction.kx,ky=self.UpliftFunction.ky)
        else:
            raise TypeError("Not sure what to do now, please help me")

        
        print(" copied the object and recomputed xx,yy and river corrdiantes, make sure it's right",flush=True)    
        
        return newFoward
    
    def ComputeRiverGrad(self):
        
        
        riverElevationGrad=calculate_river_grad(self.Z,self.recs,self.stack,self.dX,self.riverMask)
        
        return riverElevationGrad[self.riverMask]
    
    
    def ComputeTravelTimePerPixel(self,k,m,n):
        grad=self.ComputeRiverGrad()
        return k*((self.Amin*self.A[self.riverMask])**m)*(grad**(n-1))
    
    def ComputeTravelTime(self,k,m,n):
        return calculate_travel_time(self.Z,self.recs,self.stack,self.dX,self.A*self.Amin,self.riverMask,k*np.ones_like(self.Z),m,n)[self.riverMask]
    
    def NumberOfTotalParamters(self):
        numOfParamtersUplift=self.UpliftFunction.NumOfParamters()
        numOfParamtersErodability=self.NumOfKParamters()
        
        return numOfParamtersUplift+numOfParamtersErodability+3
        
        
#%%    
class deitFowardOnly_m_n_slope(dietFoward1D):
    def Foward(self,mParamters):
        """ m mParamters[0]
        n mParamters[1]
        and then uniform uplift is usually mParamters[2] 
        """

        m=mParamters[0]
        n=mParamters[1]
        slope=mParamters[2]
        upliftParamters=None
        
        return self.ComputeSyenteticElevation(upliftParamters,slope=slope,m=m,n=n)
    
    def LoadUplift(self,*args):
        self.UpliftFunction=None
        
    def GetUpliftParamtersReturnChi(self,upliftParamters,m,n):
        "get uplift Paramters,m,n and return chi and z"

        
        chi,z=self.ComputeChi(uplift=None,m=m,n=n)
        
        return chi,z
    
    def ComputeAStar(self,uStarFull,m):
        """ this function compute A star while taking into  account only uplift shape"""
        uStarFull=1
        A_star=self.A*((1/uStarFull)**(1/m))
        return A_star


    
        
        
#%%
class dietFoward1DWithK(dietFoward1D):
    def ComputeAStar(self,uStarFull,m):
        """ this function compute A star but takes into account the erodibailty pattern and uplift shape"""
        #kStar=self.K[self.riverMask]
        #A_star=self.A*((self.K/uStarFull)**(1/m))
        inv_m=1.0/m
        A_star = compute_A_star_K_WithNumba(self.A, self.K, uStarFull, inv_m)
        
        #A_star_masked = compute_A_starWithNumba(self.A[self.riverMask], uStarFull[self.riverMask], inv_m)
        return A_star
    

    def LoadKStrcture(self,listOfMasks):
        """ This function test four things 
        1) It makes sure list of maks contains vector bool with same length as self.A or anything in the DEM for that matter
        2) It make sure there's no overlap with True values between all mask vectors 
        3) It make sure all river are covered by lithlogy
        4) It makes sure all lithogly section are at least covering by one river node
    
    
        It then load the litholghy strcture based on the masks and how it's order
        """
        self.validate_boolean_vectors(listOfMasks)
        riverMaskCheck=np.zeros_like(self.A,dtype=bool)
        self.check_no_overlap_between_combinations(listOfMasks) ## check no overlap between masks
        

        notCovered=[]
        for i,mask_i in enumerate(listOfMasks):
            riverMaskCheck[mask_i]=True
            if not np.any(mask_i & self.riverMask): 
                notCovered.append(i)
                
        if len(notCovered)>0:
            for notCovered_i in notCovered:
                print("array "+str(notCovered_i)+ "not covered")
            raise ValueError("item litholgy does not overlap with rivers") 
            
        if np.any(~riverMaskCheck[self.riverMask]):
            raise ValueError("some river nodes are not covered by litholghy section")  
            
            
        kStructureList=[]
        for i,mask_i in enumerate(listOfMasks):
            if i==0:
                toInvertFor=True
            else:
                toInvertFor=True
                
            kStructureList.append(erodibilityStructure(mask_i,value=1,invertFor=toInvertFor))
        
        maskList=[]
        
        for item in kStructureList:
           maskList.append(item.mask) 
           riverMaskCheck[item.mask]=True
           if not np.any(item.mask & self.riverMask): 
               raise ValueError("item litholgy does not overlap with rivers")  
            
        self.kStructureList=kStructureList
        
        self.K=np.ones_like(self.A)*np.nan
        
    def NumOfKParamters(self):
        c=0
        for item in self.kStructureList:
            if item.invertFor is True:
                c=c+1
                
        return c
        
        
        
    def Foward(self, mParamters):
        """ this function get paramters and set them in accordance with the ErodbiltyStrcture.
        if ErodbiltyStrcture has invertFor as True then it set based on one of the paramters. The way I move K is kinda stupid, istead of moving it around
        I jusst save it as self.K """
        m=mParamters[0]
        n=mParamters[1]
        slope=mParamters[2]
        upliftParamters=self.SetKVecotrReturnUpliftParams(mParamters) 

        return self.ComputeSyenteticElevation(upliftParamters,slope=slope,m=m,n=n)
    
    def CheckForK(self,k_value):
        if k_value<0.1:
            k_value=0.1
            warnings.warn(" Warning setk to be  small setting it to min value of 0.1",UserWarning)
         
        
        elif k_value>10:
            k_value=10
            warnings.warn(" Warning setk to be  large setting it to max value of 10",UserWarning)
        
        
            
        
            
        return k_value
    
    def SetKVecotrReturnUpliftParams(self,mParamters):
        i=0
        for kStructure_i in self.kStructureList:
            if kStructure_i.invertFor is True:
                self.K[kStructure_i.mask]=self.CheckForK(mParamters[3+i])
                i=i+1
            else:
                self.K[kStructure_i.mask]=kStructure_i.value

        upliftParamters=mParamters[3+i:]
        
        return upliftParamters
    
    
    def ComputeTravelTime(self,k,m,n):

        K=self.K
        K=K*k
        return calculate_travel_time(self.Z,self.recs,self.stack,self.dX,self.A*self.Amin,self.riverMask,K,m,n)[self.riverMask]
    
    
    def check_no_overlap_between_combinations(self,mask_list):
        num_masks = len(mask_list)
        
        for i in range(num_masks):
            for j in range(i + 1, num_masks):
                if np.any(mask_list[i] & mask_list[j]):
                    # If there's any overlap between mask i and mask j, return False
                    raise ValueError("masks overlap: "+str(i)+" "+str(j))
        
        # If no overlaps found in any combination, return True
        return True
    def validate_boolean_vectors(self,listOfVectors):
    
        for vec in listOfVectors:
            # Check if the vector is boolean
            if not np.issubdtype(vec.dtype, np.bool_):
                raise ValueError("All vectors must be of boolean type")
            
            # Check if the vector has length 1000
            if len(vec) != len(self.A):
                raise ValueError("All vectors must have a length like the DEM ")
                
#%%
class deitFowardOnly_m_n_slope_k(dietFoward1DWithK,deitFowardOnly_m_n_slope):
    def Foward(self, mParamters):
        """ this function get paramters and set them in accordance with the ErodbiltyStrcture.
        if ErodbiltyStrcture has invertFor as True then it set based on one of the paramters. The way I move K is kinda stupid, istead of moving it around
        I jusst save it as self.K """
        m=mParamters[0]
        n=mParamters[1]
        slope=mParamters[2]
        upliftParamters=self.SetKVecotrReturnUpliftParams(mParamters)
        upliftParamters=None

        return self.ComputeSyenteticElevation(upliftParamters,slope=slope,m=m,n=n)
    
    

    def ComputeAStar(self,uStarFull,m):
        """ this function compute A star but takes into account the erodibailty pattern and and upliftshape=1"""
        #kStar=self.K[self.riverMask]
        A_star=self.A*((self.K)**(1/m))
        return A_star
    
    
    def LoadUplift(self,*args):
        """ explicit call for ComputeAStar """
        return deitFowardOnly_m_n_slope.LoadUplift(self,*args)
        
        
        
#%%
class erodibilityStructure:
    def __init__(self,mask,value,invertFor=False):
        self.mask=mask
        self.invertFor=invertFor
        self.value=value
        
        

    
#%%
class dietFoward2D(dietFoward1D): # the same class just for 2D
    def LoadUplift(self,kx=None,ky=None,splineDegree=None):
        self.UpliftFunction=upliftFunctions.autoAsymmetricalGassiuanFunction2DWithConstantUplift(x=self.XXflat[self.riverMask],y=self.YYflat[self.riverMask])
        #self.UpliftFunction=upliftFunctions.autoAsymmetricalGassiuanFunction2DWithConstantUplift(x=self.Xrivers,y=self.Yrivers)
        
#%%
class BsplineFoward1D(dietFoward1D):
    def LoadUplift(self,kx=None,ky=None,splineDegree=3):
        self.UpliftFunction=upliftFunctions.bsplineUplift1D(x=self.Xrivers,splineDegree=splineDegree,kx=kx) # changed XXflat to Xrivers
#%%
class BsplineFoward1DOnlyUplift(BsplineFoward1D):
    def Foward(self,mParamters):
        """ m =self.m
        n =self.n
        slope= mParamters[0]
        upliftParamters = mParamters[1:]
        """
        

        m=self.m
        n=self.n
        slope=mParamters[0]
        upliftParamters=mParamters[1:]



        return self.ComputeSyenteticElevation(upliftParamters,slope=slope,m=m,n=n)
    
    
#%%
class BsplineFoward1DOnlyUpliftWithRatio(BsplineFoward1D):
    def Foward(self,mParamters):
        """ m =self.m
        n =self.n
        slope= mParamters[0]
        upliftParamters = mParamters[1:]
        """
        

        m=mParamters[0]
        n=m*self.mn_ratio
        slope=mParamters[1]
        upliftParamters=mParamters[2:]



        return self.ComputeSyenteticElevation(upliftParamters,slope=slope,m=m,n=n)
    
        
#%%
class BsplineFoward2D(dietFoward1D):
    def LoadUplift(self,kx,ky,splineDegree=3):
        print("Loading uplift functions, might take a bit longer than usual")
        self.UpliftFunction=upliftFunctions.bsplineUplift2D(x=self.Xrivers,y=self.Yrivers,kx=kx,ky=ky,splineDegree=splineDegree)
        
#%%
class BsplineFoward2DOnlyUplift(BsplineFoward1DOnlyUplift,BsplineFoward2D):
    def LoadUplift(*args):
        return BsplineFoward2D.LoadUplift(*args)
    
#%%
class BsplineFoward2DOnlyUpliftWithRatio(BsplineFoward1DOnlyUpliftWithRatio,BsplineFoward2D):
    def LoadUplift(*args):
        return BsplineFoward2D.LoadUplift(*args)   

#%%     
class BsplineFoward2DWithErodibailty(BsplineFoward2D,dietFoward1DWithK):
    """inheret all 2d functions that takes in account K sptail erodabilty"""
    def ComputeAStar(self,*args):
        """ explicit call for ComputeAStar """
        return dietFoward1DWithK.ComputeAStar(self,*args)
    
    def Foward(self,*args):
        return dietFoward1DWithK.Foward(self,*args)
    
class BsplineFoward1DWithErodibailty(BsplineFoward1D,dietFoward1DWithK):
    """inheret all 1d functions that takes in account K sptail erodabilty"""
    def ComputeAStar(self,*args):
        """ explicit call for ComputeAStar """
        return dietFoward1DWithK.ComputeAStar(self,*args)
    
    def Foward(self,*args):
        return dietFoward1DWithK.Foward(self,*args)
        
        

#%%
@nb.njit(parallel=True)
def calculate_chi(A, dX, Z, recs, stack, A0, m,n):
    '''
    Standalone chi calculation
    params:
        - A: 1D array of drainage area
        - dX: 1D array of distance to rec
        - Z: 1D array of elevationz
        - recs: 1D array of receiver indices
        - stack: 1D array of indices in stack order
        - A0: reference drainage area (=1 if you want dz/dchi = ksn)
        - theta: reference concavity index
    returns: 
        - 1D array of chi values
    '''

    ## Precomputing the output
    chi = np.zeros_like(A)
    theta=m/n
    ## Iterating from base levels to the top
    for node in stack:
        ## node is a node index
        
        ### Getting the receiver
        rnode = recs[node]
        
        ### if the receiver is the node itself, we are at a base level and chi == 0
        if rnode == node:
            continue # continue jumps tot the next iteration and code bellow in this loop is ignored
        
        # rectangular approximation
        # chi[node] = chi[rnode] + (math.pow(A0/A[node],theta)) * dX[node] # Rectangular approximation
        # Trapezoidal approximation approximation
        chi[node] = chi[rnode] + (math.pow(A0/A[node],theta) + math.pow(A0/A[rnode],theta))/2 * dX[node] # trapezoidal approximation
    # Done  
    return chi       


#%%
@nb.njit
def calculate_chiP(A, dX, Z, recs, stack, A0, m,n,mask):
    '''
    Standalone chi calculation
    params:
        - A: 1D array of drainage area
        - dX: 1D array of distance to rec
        - Z: 1D array of elevationz
        - recs: 1D array of receiver indices
        - stack: 1D array of indices in stack order
        - A0: reference drainage area (=1 if you want dz/dchi = ksn)
        - theta: reference concavity index
    returns: 
        - 1D array of chi values
    '''

    ## Precomputing the output
    chi = np.zeros_like(A)
    theta=m/n

    ## Iterating from base levels to the top
    for node in stack:
        ## node is a node index
        
        ### Getting the receiver
        rnode = recs[node]
        
        ### if the receiver is the node itself, we are at a base level and chi == 0
        if rnode == node:
            continue # continue jumps tot the next iteration and code bellow in this loop is ignored
            
   
        if(mask[node] == False):
            continue
        
        # rectangular approximation
        # chi[node] = chi[rnode] + (math.pow(A0/A[node],theta)) * dX[node] # Rectangular approximation
        # Trapezoidal approximation approximation
        chi[node] = chi[rnode] + (math.pow(A0/A[node],theta) + math.pow(A0/A[rnode],theta))/2 * dX[node] # trapezoidal approximation
    # Done  
    return chi 

#%%
@nb.njit
def calculate_river_grad(z,recs,stack,dx,mask):
    river_grad=np.zeros_like(z)
    for node in stack:
        ## node is a node index
        
        ### Getting the receiver
        rnode = recs[node]
        
        if rnode == node:
            continue # continue jumps tot the next iteration and code bellow in this loop is ignored
            
        if(mask[node] == False):
            continue
    
        river_grad[node]=(z[node]-z[rnode])/dx[node]
        
    return river_grad

#%%
@nb.njit
def calculate_travel_time(z,recs,stack,dx,A,mask,k,m,n):
    travelTime=np.zeros_like(z)

        
    for node in stack:
        ## node is a node index
        
        ### Getting the receiver
        rnode = recs[node]
        
        if rnode == node:
            continue # continue jumps tot the next iteration and code bellow in this loop is ignored
            
        if(mask[node] == False):
            continue
    
        #travelTime[node]=travelTime[rnode]+(dx[node]*(dx[node]**(n-1)))/(((z[node]-z[rnode])**(n-1))*(A[node]**m)*k[node])
        dz=z[node]-z[rnode]
        travelTime[node]=travelTime[rnode]+(dx[node]**n)/((dz**(n-1))*(A[node]**m)*k[node])
        
    return travelTime
        
#%%
@nb.njit(parallel=True)
def compute_A_starWithNumba(A, uStarFull, inv_m):
    A_star = np.empty_like(A)
    for i in nb.prange(len(A)):
        A_star[i] = A[i] * (1.0 / uStarFull[i])**inv_m
    return A_star
#%%
@nb.njit(parallel=True)      
def compute_A_star_K_WithNumba(A, K, uStarFull, inv_m):
    A_star = np.empty_like(A)  # Allocate an array for the result

    # Parallelized loop to compute A_star
    for i in nb.prange(len(A)):
        A_star[i] = A[i] * (K[i] / uStarFull[i]) ** inv_m
    
    return A_star

