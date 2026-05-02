#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:49:04 2023

@author: bar
"""
import matplotlib.pyplot as plt
import scabbard as scb
import numpy as np
#import random
import chifrd as frd
from matplotlib.ticker import MaxNLocator
from pyproj import Transformer, Proj
from matplotlib.ticker import MaxNLocator, MultipleLocator
import pandas as pd
import seaborn as sns
#%%
class plotlandscapeAndInversion:
    def __init__(self,inversionFile,fowardFile,maskFile,xFault,yFault,kx,angleToRotate=0,offsetFromFault=0,upliftValuesToIgnore=[0,1,2],K=False,ky=None,MNS=False,u0=0,climateFile=None):
        self.xFault,self.yFault=self.extend_line(np.array(xFault),np.array(yFault))
        self.data=inversionResult=np.load(inversionFile)
        self.mean=inversionResult['meanSoultion']
        self.std=inversionResult['stdSoultion']
        self.basinMask=np.load(maskFile)
        self.offsetFromFault=offsetFromFault
        self.maskUplift=np.ones_like(self.mean[-1,:],dtype=bool);self.maskUplift[upliftValuesToIgnore]=False
        self.upliftValuesToIgnore=upliftValuesToIgnore
        self.u0=u0
        self.factorA0=1
        
        
        
        if MNS is True:
            self.f=frd.LoadDietFowardOnly_m_n_slope(fowardFile)
            self.f_rotated=self.f
        else:
            if ky is not None:
                self.f=frd.LoadFowardModel(filename=fowardFile,spatialErodabilty=K,kx=kx,ky=ky,splineDegree=3)
            else:
                self.f=frd.LoadFowardModel(fowardFile,spatialErodabilty=K,kx=kx)
                
            if angleToRotate != 0:
                self.f_rotated=self.f.RotateFowardCorrd(angleToRotate)
            else:   
                self.f_rotated=self.f
            
            self.x=ComputeDistanceFromFault(self.f_rotated, offsetFromFault)
            
        if climateFile is not None:
            self.climate=np.load(climateFile)
            
    def PlotChiForSoultion(self,indexForSoultion=-1,riverColors=None,slopeColors=None,ax=None):
        PlotChiForSoultion(self.f_rotated, self.mean[indexForSoultion,:],riverColors=riverColors,slopeColors=slopeColors,ax=ax)
        
    def Plot2DUpliftWithinBasins(self,indexForSoultion=-1,ax=None,riverPlot=None,imshowProp=None):
        cb=Plot2DUpliftWithinBasins(self.basinMask,self.mean[indexForSoultion,self.maskUplift],f=self.f,f_rotated=self.f_rotated,ax=ax,
                                    riverPlots=riverPlot,imshowProp=imshowProp)
                                   
        
        return cb
        
    def Plot1DUplift(self,indexForSoultion=-1,meanSoultionColors=None,stdSoultionColors=None,ax=None,continuous=False,numOfSamples=500,norm=1):
        if stdSoultionColors is  None:
            std=None
        else:
            std=self.std[indexForSoultion,:,:]

            
        Plot1DUplift(meanSoultion=self.mean[indexForSoultion,:],std=std,f=self.f_rotated,ax=ax,offsetFromFault=self.offsetFromFault,
                                valuesToIgnore=self.upliftValuesToIgnore,meanSoultionColors=meanSoultionColors,stdSoultionColors=stdSoultionColors,continuous=continuous,numOfSamples=numOfSamples,norm=norm)
        
    def PlotFaultOnMap(self,ax,faultColors):
        ax.plot(self.xFault,self.yFault,**faultColors)
        
    def SetAxisBasedOnBasin(self,ax):
        for (xx,set_lim) in zip([self.f.XXflat[self.basinMask],self.f.YYflat[self.basinMask]],[ax.set_xlim,ax.set_ylim]):
            
            xmin=min(*xx);xmax=max(*xx)
            xrange=np.ptp([xmin,xmax]);
            set_lim([xmin-xrange*0.05,xmax+xrange*0.05])
            
    def SetNumberOfTicks(self,ax,num=4,xaxis=True,yaxis=True):
        if xaxis is True:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=num))
            ax.xaxis.set_minor_locator(MaxNLocator(nbins=10*num))
        if yaxis is True:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=num))
            ax.yaxis.set_minor_locator(MaxNLocator(nbins=10*num))
            
    def ReturnMNS(self):
        
        print ("m:"+str(np.round(self.mean[-1,0],2)))
        print ("n:"+str(np.round(self.mean[-1,1],2)))
        print ("as:"+str(np.round(self.mean[-1,2],2)))
        print ("RMS:"+str(np.round(self.data['RMS'],2)))
        print ("thetaa:"+str(np.round(self.mean[-1,0]/self.mean[-1,1],2)))
        
        
    def ComputeRMS(self):
        return np.round(ComputeRMS(self.f_rotated.Foward(self.mean[-1,:])+self.f_rotated.minElevation-self.f_rotated.selectZ),2)
        
    def ReturnContoiunsX(self):
        return np.linspace(np.min(self.x), np.max(self.x),1000)
    
    def ReturnUpliftBasedOnBrittleLayerOld(self,alpha):
        x=self.ReturnContoiunsX()
        return np.exp(-x/alpha)
    
    def ReturnUpliftBasedOnBrittleLayer(self,alpha):
        x=self.ReturnContoiunsX()
        return np.exp(-x/alpha)*np.cos(x/alpha)
    
    
    def PlotBrittle(self,alpha,ax=None,colors=None):
        if ax is None:
            fig,ax=plt.subplots()
        if colors is None:
            colors={'color':'red','linewidth':2,'linestyle':'dashed','alpha':0.5}
        x=self.ReturnContoiunsX()
        uplift=self.ReturnUpliftBasedOnBrittleLayer(alpha)
        uplift=uplift/np.max(uplift)
        ax.plot(x,uplift,**colors)
        
    def PlotExp(self,alpha,ax=None,colors=None):
    
        if ax is None:
            fig,ax=plt.subplots()
        if colors is None:
            colors={'color':'red','linewidth':2,'linestyle':'dashed','alpha':0.5}
        x=self.ReturnContoiunsX()
        #uplift=self.ReturnUpliftBasedOnBrittleLayer(alpha)
        A=1/np.exp(-x[0] / alpha)
        uplift=A * np.exp(-x / alpha)
        #uplift=uplift/np.max(uplift)
        ax.plot(x,uplift,**colors)
        
        
        
    def PlotExpCos(self,alpha,Umin=None,ax=None,colors=None):
    
        if ax is None:
            fig,ax=plt.subplots()
        if colors is None:
            colors={'color':'red','linewidth':2,'linestyle':'dashed','alpha':0.5}
        x=self.ReturnContoiunsX()
        #uplift=self.ReturnUpliftBasedOnBrittleLayer(alpha)
        if Umin is None:
            A=1/(np.exp(-x[0]/alpha)*np.cos(x[0]/alpha))
            uplift=A*np.exp(-x/alpha)*np.cos(x/alpha)
        else:
            uplift=Umin+(1-Umin)*np.exp(-x/alpha)*np.cos(x/alpha)
        #uplift=uplift/np.max(uplift)
        ax.plot(x,uplift,**colors)
        
        
        
    def ComputeK0(self,u0=None):
        if u0 is None:
            u0=self.u0
        A0=self.f.Amin/self.factorA0
        m=self.mean[-1,0]
        n=self.mean[-1,1]
        a_s=self.mean[-1,2]
        
        return u0/(((A0)**m)*(a_s)**n)
    
    
    def ComputeTravelTime(self,u0):
        k0=self.ComputeK0(u0)
        m=self.mean[-1,0]
        n=self.mean[-1,1]
        travelTime=self.f.ComputeTravelTime(k0,m,n)
        
        return travelTime
    
    
    def PlotTravelTime(self,ax=None,s=1.5):
        if ax is None:
            fig,ax=plt.subplots()
            
        travelTime=self.ComputeTravelTime(self.u0)
        cb=ax.scatter(self.f.Xrivers/1e3,self.f.Yrivers/1e3,c=travelTime/1e6,s=s)
        #fig.colorbar(cb)
        
        return cb
    
    def extend_line(self,x, y):
        # Ensure x and y are numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # Calculate the direction vector and normalize it
        dx, dy = x[1] - x[0], y[1] - y[0]
        length = np.hypot(dx, dy)  # Euclidean distance between the two points
    
        # Normalize the direction vector
        dx, dy = map(lambda d: d / length, [dx, dy])
        distance=(dx**2+dy**2)**0.5
        distance=distance*20000
        # Compute the extended points
        x_new = [x[0] - dx * distance, x[1] + dx * distance]
        y_new = [y[0] - dy * distance, y[1] + dy * distance]
        
        return x_new, y_new
    
    def ComputeSTD(self):
        
        samples = np.random.multivariate_normal(self.mean[-1,:], self.std[-1,:,:], size=500)
        tempUplift=np.zeros([len(samples),np.sum(self.basinMask)],dtype=np.float32)

        
        for i in range(len(samples)):
            print(f"\rComputing {i}      ", end='', flush=True) 
            tempUplift[i,:]=self.f_rotated.UpliftFunction.ComputeUpliftForDifferentXandY(self.f_rotated.XXflat[self.basinMask],self.f_rotated.YYflat[self.basinMask],samples[i,3+self.f_rotated.NumOfKParamters():])
            #tempUpliftInRivers[i,:]=f_rotated.UpliftFunction.Uplift(samples[i,3+f_rotated.NumOfKParamters():])
        print("computing std")
        stdInRivers=np.std(tempUplift,axis=0)


        stdAllDomain=np.ones_like(self.f_rotated.XXflat)*np.nan
        stdAllDomain[self.basinMask]=stdInRivers
            
        stdAllDomain=stdAllDomain.reshape(self.f_rotated.shape)
        
        return stdAllDomain
    
    def PlotPairPlot(self):

        mean=self.mean[-1,:]
        std=self.std[-1,:,:]
        samples = np.random.multivariate_normal(mean, std, 500)
        column_names = ['m', 'n', 'a_s'] + [f'Variable {i+4}' for i in range(len(mean) - 3)]

        df_samples = pd.DataFrame(samples, columns=column_names)
        
        
        pp=sns.pairplot(df_samples, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 50},corner=True)
        
        return pp
    
    def PlotClimate(self,ax=None,plottinRivers=None):
        if ax is None:
            fig,ax=plt.subplots()
            
        extent=GetExtentFromFoward(self.f)
            
        PlotFowardMapWithRivers(ax=ax,plottinRivers=plottinRivers,f=self.f)
        ax.imshow(self.climate,extent=extent,cmap='Blues',alpha=0.45)
        

        
        

#%%
def PlotFowardMapWithRiversSynLandscapes(f,ax=None,plottinRivers=None):
    if ax is None:
        fig,ax=plt.subplots()
    
    if plottinRivers is None:
        plottinRivers={'color':'magenta','s':0.1}
        
    PlotTopoGraphyFromFowardModel(f,ax=ax)
    ax.scatter(f.Xrivers,100e3-f.Yrivers,**plottinRivers)
    
    
def PlotFowardMapWithRivers(f,ax=None,plottinRivers=None):
    if ax is None:
        fig,ax=plt.subplots()
    
    if plottinRivers is None:
        plottinRivers={'color':'magenta','s':0.1}
    PlotTopoGraphyFromFowardModel(f,ax=ax)
    ax.scatter(f.Xrivers,f.Yrivers,**plottinRivers)
    
#%%
def ComputeChiForSoultion(f,soultion):

    
    m=f.Check_m_n(soultion[0])
    n=f.Check_m_n(soultion[1])
    slope=f.Check_slope(soultion[2])
    
    if isinstance(f,frd.dietFoward1DWithK):
        upliftParamters=f.SetKVecotrReturnUpliftParams(soultion)
    elif isinstance(f,frd.BsplineFoward1DOnlyUplift):
        slope=f.Check_slope(soultion[0])
        m=f.Check_m_n(f.m)
        n=f.Check_m_n(f.n)
        upliftParamters=soultion[1:]
    else:
        upliftParamters=soultion[3:]
        

    chi,z=f.GetUpliftParamtersReturnChi(upliftParamters=upliftParamters,m=m,n=n)
    
    return slope,chi,z
    
#%%
def PlotChiForSoultion(f,soultion,ax=None,riverColors=None,slopeColors=None):
    if riverColors is None:
        riverColors={'s':1,'color':'black','alpha':0.4}
    if slopeColors is None:
        slopeColors={'linewidth':0.75,'color':'gray'}
        
        
    slope,chi,z=ComputeChiForSoultion(f,soultion)
    # if ax is None:
    #     fig,ax=plt.subplots()
    
    # m=f.Check_m_n(soultion[0])
    # n=f.Check_m_n(soultion[1])
    # slope=f.Check_slope(soultion[2])
    
    # if isinstance(f,frd.dietFoward1DWithK):
    #     upliftParamters=f.SetKVecotrReturnUpliftParams(soultion)
    # elif isinstance(f,frd.BsplineFoward1DOnlyUplift):
    #     slope=f.Check_slope(soultion[0])
    #     m=f.Check_m_n(f.m)
    #     n=f.Check_m_n(f.n)
    #     upliftParamters=soultion[1:]
    # else:
    #     upliftParamters=soultion[3:]
        

    # chi,z=f.GetUpliftParamtersReturnChi(upliftParamters=upliftParamters,m=m,n=n)
    
    chi=chi/1e3
    z=z/1e3
    
    x=np.linspace(0, np.nanmax(chi))

    ax.scatter(chi,z,**riverColors)
    ax.plot(x,x*slope,**slopeColors) 
    print("RMS:"+ str(np.round(ComputeRMS(slope*chi-z)*1e3,2)))
#%%    
def GetHillShadeFromFoward(f,dx=30,dy=30):
    dx=dx;dy=dy;ny=f.shape[0];nx=f.shape[1]
    grid = scb.RGrid(nx, ny, dx, dy, f.Z, geography = None)
    return grid.hillshade.reshape(f.shape)

#%%
def PlotTopoGraphyFromFowardModel(f, ax=None, extent=None, basinMask=None, fig_width=5):
    if extent is None:
        extent = GetExtentFromFoward(f)

    if ax is None:
        xmin, xmax, ymin, ymax = extent
        fig_height = fig_width * (ymax - ymin) / (xmax - xmin)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    hillShade = GetHillShadeFromFoward(f)

    zmin = np.nanmin(f.Z)
    zmax = np.nanmax(f.Z)

    if basinMask is None:
        Z = f.Z.reshape(f.shape)
    else:
        Z = f.Z.copy()
        Z[~basinMask] = np.nan
        Z = Z.reshape(f.shape)

        hillShade = hillShade.flatten()
        hillShade[~basinMask] = np.nan
        hillShade = hillShade.reshape(f.shape)

    ax.imshow(Z, extent=extent, cmap="gist_earth", vmin=zmin, vmax=zmax)
    ax.imshow(hillShade, extent=extent, cmap="gray", alpha=0.6)

#%%     
def ComputeRMS(vector):
    return np.sqrt(np.mean(vector**2))  
#%%
def PlotUpliftForRivers(upliftParamters,f,f_rotated=None,ax=None):
    if ax is None:
        fig,ax=plt.subplots()
        
    PlotTopoGraphyFromFowardModel(f,ax)
    if f_rotated is None:
        upliftPerPixel=f.UpliftFunction.Uplift(upliftParamters)
    else:
        upliftPerPixel=f_rotated.UpliftFunction.Uplift(upliftParamters)
        
    ax.scatter(f.Xrivers,f.Yrivers,s=0.1,c=upliftPerPixel)
#%%    
def Plot2DUpliftWithinBasins(maskForPixelsToPlot,upliftParamters,f,f_rotated=None,ax=None,extent=None,riverPlots=None,normMin=None,normMax=None,imshowProp=None,plotTopo=True,addColorBar=False,returnUplift=False):
    
    if maskForPixelsToPlot.shape != f.XXflat.shape or maskForPixelsToPlot.dtype != bool:
        raise TypeError("maskForPixelsToPlot needs to be bool array and the same len as the DEM")
    if f_rotated is None:
        f_rotated=f
    x=f_rotated.XXflat[maskForPixelsToPlot]
    uplift=np.ones_like(f.XXflat)*np.nan
    y=None
    
    if isinstance(f_rotated, frd.BsplineFoward2D):
        y=f_rotated.YYflat[maskForPixelsToPlot]
    upliftBeforeMask=ComputeUpliftForXAndY(f=f_rotated,upliftParamters=upliftParamters,x=x,y=y)
    
    uplift[maskForPixelsToPlot]=upliftBeforeMask
        
    uplift=uplift.reshape(f.shape)
    
    
    if ax is None:
        fig,ax=plt.subplots()
        
    if plotTopo:
        PlotTopoGraphyFromFowardModel(f,ax=ax,extent=extent)
        
    if extent is None:
        extent=GetExtentFromFoward(f)
    
    if imshowProp is None:
        imshowProp={'cmap':'viridis','alpha':0.5}
        
        
    if normMax is not None:
        mask=np.where(uplift>normMax,True,False)
        uplift[mask]=np.nan
        
    if normMin is not None:
        mask=np.where(uplift<normMin,True,False)
        uplift[mask]=np.nan    

    cb=ax.imshow(uplift,extent=extent,**imshowProp)
    PlotRiverPosition(f,ax=ax,riverPlots=riverPlots)
    
    if addColorBar:
        cbar_ax = fig.add_axes([0.25, 0.7, 0.5, 0.02])  # adjust values as needed
        cbar = fig.colorbar(cb, cax=cbar_ax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
    
    if returnUplift:
        return uplift,cb        
    else:
        return cb


#%% 
def PlotRiverPosition(f,ax=None,riverPlots=None):
    if riverPlots is None:
        riverPlots={'color':'black','alpha':0.3,'s':0.05}
    ax.scatter(f.XXflat[f.riverMask],f.YYflat[f.riverMask],**riverPlots)
    

#%%
def Plot2DAlongALine(f,x,y,mean,plotAlong=None,std=None,valuesToIgnore=[0,1,2],meanSoultionColors=None,stdSoultionColors=None,ax=None,norm=1):
    mask=np.ones_like(mean,dtype=bool)
    mask[valuesToIgnore]=False
    
    if plotAlong is None:
        plotAlong=x
    
        
    if ax is None:
        fig,ax=plt.subplots()
        
    if std is not None:
        samples=np.random.multivariate_normal(mean,std,size=500)
    else:
        samples=[mean]
        stdSoultionColors=meanSoultionColors
        

        
    for sample_i in samples:
        u=ComputeUpliftForXAndY(f,sample_i[mask],x,y)
        ax.plot(plotAlong,u/norm,**stdSoultionColors)
        
#%%
def GetExtentFromFoward(f):
    return [np.min(f.XXflat),np.max(f.XXflat),np.min(f.YYflat),np.max(f.YYflat)]
#%%    
def Plot1DUplift(meanSoultion,f,ax=None,offsetFromFault=0,std=None,valuesToIgnore=[0,1,2],meanSoultionColors=None,stdSoultionColors=None,continuous=False,numOfSamples=500,norm=1):
    
    if meanSoultionColors is None:
        meanSoultionColors={'color':'blue','s':1}
    if stdSoultionColors is None:
        stdSoultionColors={'color':'gray','alpha':0.2,'s':1}
        
    mask=np.ones_like(meanSoultion,dtype=bool)
    if valuesToIgnore is not None:
        mask[valuesToIgnore]=False
    
    if ax is None:
        fig,ax=plt.subplots()

    
    x=ComputeDistanceFromFault(f,offsetFromFault,continuous=continuous)    
    
    if continuous:
        #if offsetFromFault>0:
        xToCompute=np.linspace(np.min(f.UpliftFunction.x), np.max(f.UpliftFunction.x),500)
        #else:
        #    xToCompute=np.linspace(np.max(f.UpliftFunction.x), np.min(f.UpliftFunction.x),500)
        

        
    if std is not None:
        samples=np.random.multivariate_normal(meanSoultion,std,size=numOfSamples)
        
        
        for sample_i in samples:
            if continuous:
                u=f.UpliftFunction.UpliftForDifferentX(sample_i[mask], xToCompute)
                u=u/np.max(u)
                ax.plot(x,norm*u,**stdSoultionColors)
                
            else:
                u=f.UpliftFunction.Uplift(sample_i[mask])
              
                ax.scatter(np.abs(x),norm*u,**stdSoultionColors)
            
    else:    
        if continuous:
            u=f.UpliftFunction.UpliftForDifferentX(meanSoultion[mask],xToCompute)
            u=u/np.max(u)
            ax.plot(x,norm*u,**meanSoultionColors)
        else:
            ax.scatter(np.abs(x),norm*f.UpliftFunction.Uplift(meanSoultion[mask]),**meanSoultionColors)
            
        

#%%
def ComputeUpliftForXAndY(f,upliftParamters,x,y):
    if isinstance(f, frd.BsplineFoward2D):
        uplift=f.UpliftFunction.ComputeUpliftForDifferentXandY(x=x,y=y,upliftParamters=upliftParamters)
    elif isinstance(f, frd.BsplineFoward1D):
        uplift=f.UpliftFunction.UpliftForDifferentX(x=x,upliftParamters=upliftParamters)
        
    else:
        print(type(f.UpliftFunction))
        print(type(f))
        raise TypeError ("Missing object type that corrspond to something I can do with, give me BsplineFoward2D/1D")
    
    return uplift
#%%
def ComputeDistanceFromFault(f,offsetFromFault,continuous=False):
    return ComputeDistanceFromFaultPerX(f.Xrivers,offsetFromFault,continuous)
    # if offsetFromFault > 0:
    #     x=((f.Xrivers-np.min(f.Xrivers))/1e3+offsetFromFault)
    #     if continuous:
    #         x=np.linspace(np.min(x), np.max(x),500)
    # else:
    #     x=(-f.Xrivers+np.max(f.Xrivers))/1e3+np.abs(offsetFromFault)
    #     if continuous:
    #         x=np.linspace(np.max(x), np.min(x),500)
    #     x=np.abs(x)
        
    # return x

def ComputeDistanceFromFaultPerX(xRiver,offsetFromFault,continuous=False):
    if offsetFromFault > 0:
        x=((xRiver-np.min(xRiver))/1e3+offsetFromFault)
        if continuous:
            x=np.linspace(np.min(x), np.max(x),500)
    else:
        x=(-xRiver+np.max(xRiver))/1e3+np.abs(offsetFromFault)
        if continuous:
            x=np.linspace(np.max(x), np.min(x),500)
        x=np.abs(x)
        
    return x
    
    
def set_latlon_ticks(ax, projLocal):
    # Define the UTM to WGS84 transformer
    transformer = Transformer.from_proj(projLocal, Proj('epsg:4326'), always_xy=True)

    # Get the axis limits
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Get current major x and y tick values (assumed to be in UTM)
    xticks = [tick for tick in ax.get_xticks(minor=False) if xmin <= tick <= xmax]
    yticks = [tick for tick in ax.get_yticks(minor=False) if ymin <= tick <= ymax]

    # Convert the x and y ticks to lat/long
    lon_ticks = [transformer.transform(x, ymin)[0] for x in xticks]
    lat_ticks = [transformer.transform(xmin, y)[1] for y in yticks]

    # Format the tick labels
    lon_labels = [f"{abs(lon):.2f}°" for lon in lon_ticks]
    lat_labels = [f"{abs(lat):.2f}°" for lat in lat_ticks]

    # Add 'W/E' to the rightmost longitude label and 'N/S' to the topmost latitude label
    if lon_ticks:
        lon_labels[-1] += 'W' if lon_ticks[-1] < 0 else 'E'
    if lat_ticks:
        lat_labels[-1] += 'S' if lat_ticks[-1] < 0 else 'N'

    # Set the ticks within the axis limits
    ax.set_xticks(xticks, minor=False)
    ax.set_yticks(yticks, minor=False)

    # Set the new major tick labels
    ax.set_xticklabels(lon_labels, minor=False)
    ax.set_yticklabels(lat_labels, minor=False)


#%% 
def relabel_ticks_utm_to_latlon(ax, proj):
    """
    Relabels the x and y ticks of a given Matplotlib axis from UTM to latitude/longitude.

    Parameters:
    - ax: The Matplotlib axis whose ticks will be relabeled.
    - proj: A `pyproj.Proj` object to convert UTM to lat/long.
    """
    # Get the current x and y tick positions (in UTM coordinates)
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()

    # Get the minimum y and x coordinates from the axis limits (for consistent conversion)
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    # Convert x-ticks (Easting) using ymin for the Northing (since UTM requires both coordinates)
    lon_ticks, _ = proj(x_ticks, np.full_like(x_ticks, ymin), inverse=True)

    # Convert y-ticks (Northing) using xmin for the Easting
    _, lat_ticks = proj(np.full_like(y_ticks, xmin), y_ticks, inverse=True)

    # Replace the x and y tick labels with converted lat/long values
    ax.set_xticklabels([f'{lon:.1f}' for lon in lon_ticks])
    ax.set_yticklabels([f'{lat:.1f}' for lat in lat_ticks])
    
def relplot_ticks_utm_to_latlon(ax, proj, n_ticks=3):
    """
    Relabels the x and y ticks of a given Matplotlib axis from UTM to latitude/longitude,
    using "nice" tick values that are as close as possible to integer multiples.
    Also adds minor ticks at 1/10 of the major tick spacing.

    Parameters:
    - ax: The Matplotlib axis whose ticks will be relabeled.
    - proj: A `pyproj.Proj` object to convert UTM to lat/long.
    - n_ticks: The number of desired ticks for latitude and longitude.
    """
    # Get axis limits in UTM (easting/northing)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Convert the corners to lat/lon
    lon_min_ymin, lat_min_ymin = proj(xmin, ymin, inverse=True)
    lon_min_ymax, lat_min_ymax = proj(xmin, ymax, inverse=True)
    lon_max_ymin, lat_max_ymin = proj(xmax, ymin, inverse=True)

    # Define the lat/lon ranges
    lon_range = [lon_min_ymin, lon_max_ymin]
    lat_range = [lat_min_ymin, lat_min_ymax]

    # Use MaxNLocator to get "nice" latitude and longitude ticks
    lat_locator = MaxNLocator(n_ticks, prune=None)
    lon_locator = MaxNLocator(n_ticks, prune=None)

    lat_vals = lat_locator.tick_values(min(lat_range), max(lat_range))
    lon_vals = lon_locator.tick_values(min(lon_range), max(lon_range))

    # Convert these lat/lon values back to UTM coordinates
    utm_x_vals, _ = proj(lon_vals, np.full_like(lon_vals, lat_min_ymin))
    _, utm_y_vals = proj(np.full_like(lat_vals, lon_min_ymin), lat_vals)

    # Set the ticks to these UTM values
    ax.set_xticks(utm_x_vals)
    ax.set_yticks(utm_y_vals)

    # Set the labels to lat/lon
    ax.set_xticklabels([f'{lon:.1f}' for lon in lon_vals])
    ax.set_yticklabels([f'{lat:.1f}' for lat in lat_vals])

    # Add minor ticks for x and y (every 1/10th of the major ticks)
    ax.xaxis.set_minor_locator(MultipleLocator((utm_x_vals[1] - utm_x_vals[0]) / 10))
    ax.yaxis.set_minor_locator(MultipleLocator((utm_y_vals[1] - utm_y_vals[0]) / 10))
    
    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    
    
def rescale_ticks(ax):
    # Get the current x and y ticks
    x_ticks = ax.get_xticks()
    y_ticks = ax.get_yticks()


    # Set new tick labels by dividing by 1000
    ax.set_xticklabels([f'{tick / 1000:.2f}' for tick in x_ticks])
    ax.set_yticklabels([f'{tick / 1000:.2f}' for tick in y_ticks])
    
    