#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:28:44 2022

@author: bar
"""
import numpy as np
#import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import BSpline 
import pandas as pd
import numba as nb
#%%
def gaussinFunction1D(a,b,c,x):
    """ a - amp
    b - middle
    c width """
    return (a)*np.exp(-1*((x-b)**2)/(2*c**2))    

def gaussinFunction2D(a,x0,y0,sigma_x,sigma_y,x,y):
    return (a)*np.exp(-1*(((x-x0)**2/(2*sigma_x**2))+(((y-y0)**2/(2*sigma_y**2)))))


def gaussinFunction2DAsymmetrical(amp,x0,y0,sigma_x,sigma_y,theta,x,y):
    theta=np.deg2rad(theta)
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = np.sin(2 * theta) / (4 * sigma_x**2) - np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    
    uplift= amp * np.exp(-(a * (x - x0)**2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0)**2))
    
    return uplift


#%% 
class upliftFunction:
    def __init__(self,x,y=None):
        self.x=x
        self.y=y


    def CorrectUplift(self,uplift):
        uplift=np.where(uplift<1e-4,1e-4,uplift)
        uplift=uplift/np.nanmax(uplift)
        

        #return CorrectUpliftWithNumba(uplift)
        return uplift
    
    def CorrectParamters(self,paramaters):
        #return np.abs(paramaters)
        paramaters=np.where(paramaters<1e-12,1e-12,paramaters)
        return paramaters
        
        
    def ReturnXarryWithUpliftFunction(self,xarray):
        
        x=xarray.x.values
        y=xarray.y.valuess
    
        
        data=xr.DataArray(data=self.uplift.reshape(xarray.shape),
                          dims=['y','x'],coords=[y,x],
                          attrs=dict(units='[ ]',description="Uplift Function"))
        
        if isinstance(xarray,xr.core.dataarray.DataArray):
            xarray=xarray.to_dataset(name='Z')
            
        xarray['uStar']=data

        return xarray
        
        
#%% 
class flexureUplift(upliftFunction):
    def Uplift(self,waveLength):
        
        uplift=((2.927-0.5897)*np.exp(-self.x/waveLength)*np.cos(self.x/waveLength)+0.5891)/2.9268
        
        uplift=self.CorrectUplift(uplift)
        
        return uplift
#%%
class simpleGuass(upliftFunction):
    def Uplift(self,a,b,c):
        """
        

        Parameters
        ----------
        a: float
            max of gaussin
        b : float
            middle of the guassin
        c : float
            width of the guass.
            please notice that this is the std^2

        Returns
        -------
        uplift function

        """


        uplift=gaussinFunction1D(a,b,c,self.x)
        
        uplift=self.CorrectUplift(uplift)
         
        return uplift
#%%
class twoSimpleGaussFunctions(upliftFunction):
    def Uplift(self,a1,b1,c1,a2,b2,c2):
        uplift1=gaussinFunction1D(a1,b1,c1,self.x)
        uplift2=gaussinFunction1D(a2,b2,c2,self.x)
        
        uplift=self.CorrectUplift(uplift1+uplift2)
        
        return uplift

#%%
class autoGassiuanFunctions1D(upliftFunction):
    def Uplift(self,gassuiansProperties):
        """
        a - amplitude [index 0]
        x0 - middle point x [index 1]
        width x - width  in x direction [index 2]
        sums the uplift for as many uplift function needed
        """
        
        x=self.x    
        if len(gassuiansProperties)%3 != 0 or len(gassuiansProperties) < 3 :
            raise TypeError ("Need 3,6,9,12,15... number of input arguments is not good")
        
        #print(gassuiansProperties)
        numberOfGaussians=int(len(gassuiansProperties)/3)
        
        uplift=np.zeros_like(x)
        
        for i in range(numberOfGaussians):
        
            a=gassuiansProperties[3*i]
            b=gassuiansProperties[3*i+1]
            c=gassuiansProperties[3*i+2]
            
            #print("a:"+str(a))
            #print("b:"+str(b))
            #print("c:"+str(c))

            
            
            uplift=gaussinFunction1D(a,b,c,x)+uplift
            
        uplift=self.CorrectUplift(uplift)
        
        
        return uplift
    
        
    
#%%
class autoGassiuanFunctions2D(upliftFunction):

    
    def Uplift(self,gassuiansProperties):
        """
        a - amplitude [index 0]
        x0 - middle point x [index 1]
        width x - width  in x direction [index 2]
        y0 - middle point y [index 3]
        width y - width in y direction [index 4]
        sums the uplift for as many uplift function needed
        """
        
        if len(gassuiansProperties)%5 != 0 or len(gassuiansProperties) < 5 :
            raise TypeError ("Need 5,10,15,20,25... number of input arguments is not good")
            
            
            
        numberOfGaussians=int(len(gassuiansProperties)/5)
        
        uplift=np.zeros_like(self.x)
        
        for i in range(numberOfGaussians):
        
            a=gassuiansProperties[5*i]
            x0=gassuiansProperties[5*i+1]
            sigma_x=gassuiansProperties[5*i+2]
            y0=gassuiansProperties[5*i+3]
            sigma_y=gassuiansProperties[5*i+4]
            
            
    
            uplift=gaussinFunction2D(a,x0,y0,sigma_x,sigma_y,self.x,self.y)+uplift
            
        uplift=self.CorrectUplift(uplift)
        
        
        
        return uplift
            

        
                                  
#%%
class autoGassiuanFunctions1DWithConstantUplift(upliftFunction):
    def Uplift(self,gassuiansProperties):
        """
        a - amplitude [index 0]
        x0 - middle point x [index 1]
        width x - width  in x direction [index 2]
        sums the uplift for as many uplift function needed
        """
        
        x=self.x  
        constantUplift=gassuiansProperties[0]
        gassuiansProperties=gassuiansProperties[1:]
        
        if len(gassuiansProperties)%3 != 0 or len(gassuiansProperties) < 3 :
            raise TypeError ("Need 3,6,9,12,15... number of input arguments is not good")
        
        #print(gassuiansProperties)
        numberOfGaussians=int(len(gassuiansProperties)/3)
        
        uplift=np.zeros_like(x)
        
        for i in range(numberOfGaussians):
        
            a=gassuiansProperties[3*i]
            b=gassuiansProperties[3*i+1]
            c=gassuiansProperties[3*i+2]
            
            #print("a:"+str(a))
            #print("b:"+str(b))
            #print("c:"+str(c))

            
            
            uplift=gaussinFunction1D(a,b,c,x)+uplift
        
        uplift=uplift+constantUplift
            
        uplift=self.CorrectUplift(uplift)
        
        
        return uplift                                
                                 
            
            
#%%
class autoAsymmetricalGassiuanFunction2DWithConstantUplift(upliftFunction):
    def Uplift(self,gassuiansProperties):
        """
        constantUplift - [index 0]
        amp - amplitude [index 1]
        x0 - middle point x [index 2]
        y0 - middle point y [index 3]
        width x - width  in x direction [index 4]
        width y - width in y direction [index 5]
        theta - rotation angle of the guassin [index 5]
        sums the uplift for as many uplift function needed
        """
        
        x=self.x  
        y=self.y
        constantUplift=gassuiansProperties[0]
        gassuiansProperties=gassuiansProperties[1:]
        
        if len(gassuiansProperties)%6 != 0 or len(gassuiansProperties) < 6 :
            raise TypeError ("Need 6,12,18... number of input arguments is not good")
        
        #print(gassuiansProperties)
        numberOfGaussians=int(len(gassuiansProperties)/6)
        
        uplift=np.zeros_like(x)
        
        for i in range(numberOfGaussians):
        
            amp=gassuiansProperties[6*i]
            x0=gassuiansProperties[6*i+1]
            y0=gassuiansProperties[6*i+2]
            sigma_x=gassuiansProperties[6*i+3]
            sigma_y=gassuiansProperties[6*i+4]
            theta=gassuiansProperties[6*i+5]
            
            #print("a:"+str(a))
            #print("b:"+str(b))
            #print("c:"+str(c))

            
            
            uplift=gaussinFunction2DAsymmetrical(amp,x0,y0,sigma_x,sigma_y,theta,x,y)+uplift
        
        uplift=uplift+constantUplift
            
        uplift=self.CorrectUplift(uplift)
        
        
        return uplift  
        
            
#%%
class bsplineUplift1D(upliftFunction): #bsplineUplift1D_noneUniform(upliftFunction):
    def __init__(self, x,splineDegree,kx):
        self.x=x#np.sort(x)
        self.splineDegree=splineDegree
        #knots=np.quantile(x,np.linspace(0,1,kx+splineDegree+1))
        knots=np.linspace(np.min(x), np.max(x),kx+1)
        modified_knots = np.concatenate(([knots[0]]*splineDegree, knots, [knots[-1]]*splineDegree))
        self.knots=modified_knots

        self.kx=kx
        
        
    def Uplift(self,upliftParamters,x=None):
        """ compute Uplift for self.x , please do not use x here and if you want to use it , use the function UpliftForDifferentX
        I don't use y, it's just for legacy issues"""
        if x is None:
            x=self.x
        #constantUplift=upliftParamters[0]
        
        
        #bSplineAmp=upliftParamters[1:]
        if len(upliftParamters) != (self.kx)+self.splineDegree:
            raise ValueError ("Number of paramters needs to coorspond to kx:"+str(self.kx+self.splineDegree)+" and I only have "+str(len(upliftParamters) ))
            
        spl=BSpline(self.knots, upliftParamters, self.splineDegree,extrapolate=False)
        
        uplift=spl(x)#+constantUplift
        
        uplift=self.CorrectUplift(uplift)
          
        return uplift  
    
    def UpliftForDifferentX(self,upliftParamters,x):
        upliftFull=np.ones_like(x)*np.nan
        mask=np.where(np.logical_or(x>np.max(self.x) , x< np.min(self.x)),False,True)
        correctedX=x[mask].copy()
        uplift=self.Uplift(upliftParamters=upliftParamters,x=correctedX)
        upliftFull[mask]=uplift
        
        return upliftFull
    
    def ComputeUpliftForDifferentXandY(self,*args):
        return self.UpliftForDifferentX(*args)
        
#%%
class bsplineUplift2D_noneUniform(upliftFunction):
    def __init__(self, x,y,kx,ky,splineDegree):
        self.x=x
        self.y=y
        xmin=self.ReturnValuePlusAtinyBit(x,maxValue=False);xmax=self.ReturnValuePlusAtinyBit(x,maxValue=True)
        ymin=self.ReturnValuePlusAtinyBit(y,maxValue=False);ymax=self.ReturnValuePlusAtinyBit(y,maxValue=True)
        self.nx_knots=kx
        self.ny_knots=ky
        kx=np.linspace(xmin, xmax,kx+1)
        ky=np.linspace(xmin, xmax,ky+1)
        self.kx = np.concatenate(([kx[0]]*(splineDegree), kx, [kx[-1]]*(splineDegree)))
        self.ky = np.concatenate(([ky[0]]*(splineDegree), ky, [ky[-1]]*(splineDegree)))
        self.splineDegree=splineDegree
        self.index_x=self.FindIndexInKnotsSpace(knots=self.kx,position=x)
        self.index_y=self.FindIndexInKnotsSpace(knots=self.ky,position=y)


    def ReturnValuePlusAtinyBit(self,vector,maxValue=True,ratio=1e-12):
        """ this function adds/subtracts 1*10-3 of the max/min value to make sure index includes everything """
        if maxValue is True:
            value=np.max(vector)
            value+=value*ratio
        else:
            value=np.min(vector)
            value-=value*ratio
            
        return value
        
        
    def FindIndexInKnotsSpace(self,knots, position):
        """
        Find indices i for each target value in 'position' such that knots[i] <= position < knots[i+1].
        Returns np.nan for target values that do not fit between any two elements in the vector.
        If the vector is not sorted, raises a ValueError.
    
        :param vector: NumPy array of numeric values
        :param target_values: NumPy array of numeric target values
        :return: NumPy array of indices (or np.nan) where each target value fits between vector[i] and vector[i+1]
        """
        if not np.all(np.diff(knots) >= 0):
            raise ValueError("The input vector is not sorted.")
    
        indices = np.searchsorted(knots, position, side='right') - 1
    
        # Adjust indices and set np.nan where the target value does not fit between any two elements
        valid_indices_mask = (indices >= 0) & (indices < knots.size - 1)
        result = np.full(position.shape, np.nan)
        result[valid_indices_mask] = indices[valid_indices_mask]
    
        return result.astype(int)
        
    
    def ComputeBsplineBasisAtPoint(self,knots,splineDegree,index,pointToCompute):
        """
     Computes the value of a B-spline basis function at a specific point.

     :param knots: Array of knot positions.
     :param splineDegree: Degree of the B-spline.
     :param index: Index of the basis function within the B-spline.
     :param pointToCompute: The point at which to evaluate the basis function.
     :return: Value of the basis function at the specified point.
     """
     
         # Initialize coefficients to zero, then set the coefficient at the given index to 1
        mask=np.zeros_like(knots,dtype=bool)
        mask[index-splineDegree:index+splineDegree+1]=True
        coefficients=np.zeros_like(knots);coefficients[index]=1
        spl = BSpline(knots, coefficients, splineDegree,extrapolate=False) # Create the B-spline object
        spl_value=spl(pointToCompute) # Evaluate the B-spline at the specified point
        
        return spl_value
    
    def BsplineSurfaceAtPoint(self,x,y,index_x,index_y,controlPoints,splineDegree,kx,ky):
        """
    Computes the value of a B-spline surface at a specific point (x, y) doing double summation ,
    follows doucmention in the NURBS book, page 100 section and eq 3.11 and ex 3.4.

    :param x: x-coordinate of the point.
    :param y: y-coordinate of the point.
    :param index_x: Index in the x-direction for the B-spline computation.
    :param index_y: Index in the y-direction for the B-spline computation.
    :param controlPoints: 2D array of control points for the B-spline surface.
    :param splineDegree: Degree of the B-spline.
    :param kx: Knot vector in the x-direction.
    :param ky: Knot vector in the y-direction.
    :return: The  value of the B-spline surface at the given point (x, y).
    """
        surfaceValue=0
         
        
        # Iterate over the relevant basis functions in the x-direction
        for i_x in range(index_x-splineDegree,index_x,1):
            
            spl_x=self.ComputeBsplineBasisAtPoint(kx,splineDegree,i_x,x)
            
            # Iterate over the relevant basis functions in the y-direction
            for j_y in range(index_y-splineDegree,index_y,1):

                spl_y=self.ComputeBsplineBasisAtPoint(ky,splineDegree,j_y,y)
                
                surfaceValue += spl_x*spl_y*controlPoints[i_x,j_y]
        
        return surfaceValue
    
    def ComputeUpliftForDifferentXandY(self,x,y,upliftParamters):
        """ this function can compute the position of x and y for different location than the river pos"""
        
        xIndToRemove=np.where((x<=np.min(self.kx)) | (x>= np.max(self.kx) ))[0]
        yIndToRemove=np.where((y<=np.min(self.ky)) | (y>= np.max(self.ky) ))[0]
        indToRemove=np.append(yIndToRemove, xIndToRemove)
        indToRemove=np.unique(indToRemove)
        mask=np.ones_like(x,dtype=bool)
        mask[indToRemove]=False
        upliftAll=np.zeros_like(x)*np.nan
        
        x=x[mask]
        y=y[mask]
        
        if len(x) != len(y):
            raise TypeError("x and y that are in raneg of nots are not equal in len")
            
        index_x=self.ReturnIndexAndChange(self.kx,x)
        index_y=self.ReturnIndexAndChange(self.ky,y)
        

        uplift=self.Uplift(upliftParamters=upliftParamters,x=x,y=y,index_x=index_x,index_y=index_y)
        
        upliftAll[mask]=uplift
        
        return upliftAll


    def Uplift(self,upliftParamters,x=None,y=None,index_x=None,index_y=None):

        if x is None:
            x=self.x;index_x=self.index_x
        if y is None:
            y=self.y;index_y=self.index_y

       
        # Initialize control points for the B-spline surface
        try :
             bSplineCofficents=upliftParamters.reshape([self.nx_knots+self.splineDegree,self.ny_knots+self.splineDegree])
        except:
             raise ValueError ("Need more/less paramates for this kind of surface:(len(x knots) +splineDegree)+ * (len(y knots) +splineDegree) :"
                               + str((self.nx_knots+self.splineDegree)*(self.ny_knots+self.splineDegree)) +"\n"
                               +"but only got: " +str(len(upliftParamters)))
             
        uplift=np.zeros_like(x)
   
        for l in range(len(uplift)):
            uplift[l]=self.BsplineSurfaceAtPoint(x[l],y[l],index_x[l],index_y[l],bSplineCofficents,self.splineDegree,self.kx,self.ky)
             
        uplift=self.CorrectUplift(uplift)
        
  
        
        return uplift
        

        

        
#%%
class bsplineUplift2D(upliftFunction):
    def __init__(self, x,y,kx,ky,padDistance=1e-3,splineDegree=3):
        self.x=x
        self.y=y
        self.kx=kx
        self.ky=ky
        self.padDistance=padDistance
        self.aVector=np.array([[-1,3,-3,1],[3,-6,3,0],[-3,0,3,0],[1,4,1,0]])
        self.aVectorTranspose=np.transpose(self.aVector)
        self.PreapreNodes()

        
    def PreapreNodes(self):
        
        xmin,xmax=self.PadLocationVectorPrecentWise(self.x)
        ymin,ymax=self.PadLocationVectorPrecentWise(self.y)
        
        #self.kx_nots=np.quantile(np.unique(np.sort(xPadded)),np.linspace(0,1,self.kx))
        self.kx_nots=np.linspace(xmin, xmax,self.kx+1)
        self.ky_nots=np.linspace(ymin, ymax,self.ky+1)
        #self.ky_nots=np.quantile(np.unique(np.sort(yPadded)),np.linspace(0,1,self.ky))
        
        self.xi,self.iVector=self.ReturnIndexAndChange(self.kx_nots,self.x)
        self.yj,self.jVector=self.ReturnIndexAndChange(self.ky_nots,self.y)

    def PadLocationVector(self,x):
        x=x.copy()
        indMin=np.argmin(x)
        indMax=np.argmax(x)
        x[indMin]=x[indMin]-self.padDistance*x[indMin]
        x[indMax]=x[indMax]+self.padDistance*x[indMax]
        
        return x
    
    def PadLocationVectorPrecentWise(self,x):

        maxRange=np.ptp(x)*self.padDistance
        #x[np.argmin(x)]=x[np.argmin(x)]-maxRange
        #x[np.argmax(x)]=x[np.argmax(x)]+maxRange
        xmin=np.min(x)-maxRange
        xmax=np.max(x)+maxRange
        
        return xmin,xmax
    

    def ComputeUpliftForDifferentXandY(self,x,y,upliftParamters):
        """ this function can compute the position of x and y for different location than the river pos"""
        
        xIndToRemove=np.where((x<=np.min(self.kx_nots)) | (x>= np.max(self.kx_nots) ))[0]
        yIndToRemove=np.where((y<=np.min(self.ky_nots)) | (y>= np.max(self.ky_nots) ))[0]
        indToRemove=np.append(yIndToRemove, xIndToRemove)
        indToRemove=np.unique(indToRemove)
        mask=np.ones_like(x,dtype=bool)
        mask[indToRemove]=False
        upliftAll=np.zeros_like(x)*np.nan
        
        x=x[mask]
        y=y[mask]
        
        if len(x) != len(y):
            raise TypeError("x and y that are in raneg of nots are not equal in len")
        
        xi,iVector=self.ReturnIndexAndChange(self.kx_nots,x)
        yj,jVector=self.ReturnIndexAndChange(self.ky_nots,y)
        

        uplift=self.Uplift(upliftParamters=upliftParamters,yj=yj,xi=xi,iVector=iVector,jVector=jVector)
        
        upliftAll[mask]=uplift
        
        return upliftAll
        
    def ReturnIndexAndChange(self,k_nots,position_vector):
        """ this function basically divide the data so any x/y point is index as between 0-1 and between the i+1 and i index
        for exampleif knots is [0 1000 2000] and we have two data points 500 and 1600 it will return [0,1] and [0.5,0.6]"""
        
        
        position=pd.DataFrame({'position':position_vector})
        iVector=np.zeros_like(position_vector)-1
        newPosition=np.zeros_like(position_vector)-1
        dx=k_nots[1]-k_nots[0]
        
        for i in range(len(k_nots)-1):
            value=position.loc[(position['position']<k_nots[i+1]) & (position['position']>=k_nots[i])]
            index=value.index
            
            if not value.empty: 
                newPosition[index.values]=np.transpose((value.values-k_nots[i])/dx)
                iVector[index]=np.ones(len(value.index))*i
            
            #reltivePosition=np.where((position_vector<=k_knots[i+1]) & (position_vector>=k_knots[i]) ,position_vector/k_knots[i],reltivePosition)
        #iVector,data_is=np.divmod(newPosition,np.ones_like(newPosition))
        iVector=iVector.astype(int)
        
        if  np.any(newPosition) == -1 or np.any(iVector)== -1:
            raise TypeError("Found -1 values in the iVector and new Position vector , that's a problem!")
        
        return newPosition,iVector
        


    
    def Uplift(self,upliftParamters,yj=None,xi=None,iVector=None,jVector=None):
        #constantUplift=upliftParamters[0]
        #bSplineAmp=upliftParamters[1:]
        bSplineAmp=upliftParamters[:]
        bSplineAmp=self.CorrectParamters(bSplineAmp)
        try :
            bSplineAmp=bSplineAmp.reshape([self.ky+3,self.kx+3])
        except:
            raise ValueError ("Need more paramates for this kind of inversions:" +str((self.ky+3)*(self.kx+3)))
            
        if yj is None:
            yj=self.yj
        if xi is None:
            xi=self.xi
        if iVector is None:
            iVector=self.iVector
        if jVector is None:
            jVector=self.jVector
        
        upliftNew=computeUpliftNumba(bSplineAmp,yj,xi,iVector,jVector)
        upliftNew=self.CorrectUplift(upliftNew)
        
        
        return upliftNew  

    
    def UpliftV2(self,upliftParamters):
        constantUplift=upliftParamters[0]
        bSplineAmp=upliftParamters[1:]
        
        try :
            bSplineAmp=bSplineAmp.reshape([self.ky+3,self.kx+3])
        except:
            raise ValueError ("Need more paramates for this kind of inversions:" +str((self.ky+3)*(self.kx+3))+" plus one for constantUplift")
            
        

        uplift=np.zeros_like(self.x)

        for l in range(len(uplift)):
            vj=self.yj[l];ui=self.xi[l]
            i=self.iVector[l];j=self.jVector[l]
            
            vjVector=np.array([vj**3,vj**2,vj,1])
            vjVector=np.transpose(vjVector)
            uiVector=np.array([ui**3,ui**2,ui,1])
            q=bSplineAmp[j:j+4,i:i+4]
            
            
            uplift[l]=(vjVector @ self.aVector @ q @ self.aVectorTranspose @ uiVector)/36    
        
        uplift=uplift+constantUplift
        
        uplift=self.CorrectUplift(uplift)
        
        return uplift  
    
    def NumOfParamters(self):
        return (self.kx+3)*(self.ky+3)
    
#%%
class bsplineUplift1D_t(bsplineUplift2D):
    def __init__(self, x,kx,padDistance=1e-12,splineDegree=3):
        self.x=x
        self.kx=kx
        self.padDistance=padDistance
        self.aVector=np.array([[-1,3,-3,1],[3,-6,3,0],[-3,0,3,0],[1,4,1,0]])
        self.PreapreNodes()
        self.splineDegree=3
        
    def PreapreNodes(self):
        
        xmin,xmax=self.PadLocationVectorPrecentWise(self.x)

        #self.kx_nots=np.quantile(np.unique(np.sort(xPadded)),np.linspace(0,1,self.kx))
        self.kx_nots=np.linspace(xmin, xmax,self.kx+1)
        #self.ky_nots=np.quantile(np.unique(np.sort(yPadded)),np.linspace(0,1,self.ky))
        
        self.xi,self.iVector=self.ReturnIndexAndChange(self.kx_nots,self.x)
        
    def Uplift(self,upliftParamters,xi=None,iVector=None):
        #constantUplift=upliftParamters[0]
        #bSplineAmp=upliftParamters[1:]
        bSplineAmp=upliftParamters[:]
        
        if len(bSplineAmp) != self.kx+3:
            raise ValueError ("Need more paramates for this kind of inversions:" +str((self.kx+3)) +"\n and only have"+str(len(bSplineAmp))+" params")
            

        if xi is None:
            xi=self.xi
        if iVector is None:
            iVector=self.iVector

        

        upliftNew=np.zeros(len(iVector))

        for l in range(len(iVector)):
            ui=xi[l]
            i=iVector[l]
            
            uiVector=np.array([ui**3,ui**2,ui,1])
            q=bSplineAmp[i:i+4]
            
            
            upliftNew[l]=(np.transpose(uiVector) @ self.aVector @ q )/6    
        
        #upliftNew=upliftNew +constantUplift
        
        upliftNew=self.CorrectUplift(upliftNew)
        
        
        return upliftNew  
    
    def ComputeUpliftForDifferentXandY(self,x,upliftParamters):
        """ this function can compute the position of x and y for different location than the river pos"""
        
        xIndToRemove=np.where((x<=np.min(self.kx_nots)) | (x>= np.max(self.kx_nots) ))[0]
        indToRemove=xIndToRemove
        mask=np.ones_like(x,dtype=bool)
        mask[indToRemove]=False
        upliftAll=np.zeros_like(x)*np.nan
        
        x=x[mask]

        
        xi,iVector=self.ReturnIndexAndChange(self.kx_nots,x)

        uplift=self.Uplift(upliftParamters=upliftParamters,xi=xi,iVector=iVector)
        
        upliftAll[mask]=uplift
        
        return upliftAll
        

#%%
class bsplineUplift2D_CustomNormlization(bsplineUplift2D):
    def CorrectUplift(self, uplift):
        uplift=np.where(uplift<1e-4,1e-4,uplift)
        #uplift=uplift/np.nanmax(uplift)
        return uplift
    
def ConvertUplift_toNoNormlization(uplift_obj):
        return bsplineUplift2D_CustomNormlization(x=uplift_obj.x,y=uplift_obj.y,kx=uplift_obj.kx,ky=uplift_obj.ky,padDistance=uplift_obj.padDistance)
    
    
#%%
@nb.njit(parallel=True)
def computeUpliftNumba(bSplineAmp, yj, xi, iVector, jVector):
    
    # Convert the list of lists to a NumPy array first, then make it contiguous
    aVector = np.ascontiguousarray(np.array([[-1, 3, -3, 1], 
                                             [3, -6, 3, 0], 
                                             [-3, 0, 3, 0], 
                                             [1, 4, 1, 0]], dtype=np.float64))
    
    # Make the transpose contiguous
    aVectorTranspose = np.ascontiguousarray(aVector.T)

    uplift = np.zeros(len(iVector), dtype=np.float64)

    for l in nb.prange(len(iVector)):
        vj, ui, i, j = yj[l], xi[l], iVector[l], jVector[l]

        # Ensure vectors and submatrix are contiguous
        vjVector = np.ascontiguousarray(np.array([vj**3, vj**2, vj, 1], dtype=np.float64))
        uiVector = np.ascontiguousarray(np.array([ui**3, ui**2, ui, 1], dtype=np.float64))
        q = np.ascontiguousarray(bSplineAmp[j:j+4, i:i+4])

        uplift[l] = (vjVector @ aVector @ q @ aVectorTranspose @ uiVector) / 36

    return uplift
#%%
@nb.njit(parallel=True)
def CorrectUpliftWithNumba(uplift):

    # Divide uplift by its maximum value
    uplift_max = np.nanmax(uplift)
    if uplift_max > 0:
        for i in nb.prange(len(uplift)):
            uplift[i] /= uplift_max
            
        # Ensure the uplift values are bounded by 1e-4
        for i in nb.prange(len(uplift)):
            if uplift[i] < 1e-5:
                uplift[i] = 1e-5


    return uplift

        
        
        

        
    
        
    
            
        
        
            
        
        
        
