import numpy as np
import matplotlib.pyplot as plt
import dagger as dag
import scabbard as scb
import pandas as pd
import shapely.geometry
#import globalClimateModel
#import miscFunctions
import pointsInSidePoly
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%
def Rotate(x,y,theta):
    x=np.array(x)
    y=np.array(y)
    theta=np.deg2rad(theta)

    xRotated=x*np.cos(theta)+y*np.sin(theta)
    yRotated=-x*np.sin(theta)+y*np.cos(theta)

    return xRotated,yRotated
#%%
class loadDEMDiet:
    """ currrently I am stcuk and not sure how to export the data - because for the inversion I will need to only use river
    data but for the entrie DEM to show results I will need the whole DEM or part of it"""

    def __init__(self,demFilename,Z0=1000,A0=2e7,minimalSlope=1e-3,ftype=np.float64,precipitation=None):
        self.ftype=ftype
        self.demFilename=demFilename
        self.Z0=np.array(Z0).astype(ftype)
        self.A0=np.array(A0).astype(ftype)

        print ("Loading dem...",flush=True)
        dem = scb.raster2RGrid(self.demFilename)
        dem.compute_graphcon(SFD = True, BCs = None, LM = dag.LMR.priority_flood, preprocess_topo = True,Z0=Z0)
        self.dem=dem
        self.shape=(dem.Y.shape[0],dem.X.shape[0])
        # dem.compute_graph('fill')
        # dem.enforce_minimal_slope(minimalSlope)
        #dag.set_BC_to_remove_seas(dem.con, dem._Z,)
        
        if precipitation is None:
            self.A = dem.graph.accumulate_constant_downstream_SFD(dem.dx * dem.dy).astype(ftype)
        else:
            self.A=self.ComputeDrainageBaseOnVariable(precipitation)
            
        self.dX = dem.con.get_SFD_dx().astype(ftype)
        self.Z = dem._Z.astype(ftype)


        print("getting stack and recs...",flush=True)
        self.recs = dem.con.get_SFD_receivers()
        self.stack = dem.graph.get_SFD_stack()
        
        # dem.d_sources(A0)
        # dem.compute_river_nodes()

        print("Getting rivers...",flush=True)
        self.riverData = pd.DataFrame(dem.quick_river_network(A0))
        self.continentalDivide,self.basinID=dem.quick_basin_extraction(return_basinID = True)
        self.continentalDivide=pd.DataFrame(self.continentalDivide)
        self.RemoveBasinsTouchingBoundries(self.basinID)
        # riversdata is a dict containing the followinf keys: 
        # 'nodes': river node ID in flat DEM referential
        # 'receivers': receiver ID in flat DEM referential
        # 'river_receivers': Receiver ID in the array of river node referential
        # 'basinID': Unique basin Identifyer
        # 'riverID': Unique river identifyer
        # 'A': Drainage Area
        # 'Elevation': Elevation
        # 'dx': local distance to the receivers
        # 'flow_distance': distance from the outlet
        # 'rows': rows of the original dem
        # 'cols': cols of the original dem
        # 'X': Easting coordinate
        # 'Y': Northing coordinate


        # riversData = {**dem.get_rivers_dict(), **dem.get_rivers_rowcolnode()}
        # self.riversData=pd.DataFrame(riversData)

        
    def ComputeDrainageBaseOnVariable(self,variableP):
        if variableP.shape != self.shape:
            raise TypeError ("variable P is not the shape as DEM ")
            
        A=self.dem.graph.accumulate_variable_downstream_SFD(variableP*self.dem.dy*self.dem.dx)
        
        return A
    
    
    def ComputeDraingeBasedOnMetersPerYears(self,metersPerYear):

            
        A=self.dem.graph.accumulate_constant_downstream_SFD(self.dem.dx * self.dem.dy*metersPerYear)
        
        return A
    
    def ReturnDEMWithClimate(self,climatePatternForDEM):
        DEM_with_climate=loadDEMDiet(self.demFilename,Z0=self.Z0,A0=self.A0,precipitation=climatePatternForDEM)
        
        
        return DEM_with_climate
        
            
    def ReturnDEMWithClimateUsingGlobalClimateModel(self,espgCode):
        longs,lats=self.GetArrayCorr()
        rainfallWorld=globalClimateModel.climateModel()
        climatePatternForDEM=rainfallWorld.sample_at_utm(longs,lats,espgCode)
        raindfallLocal=climatePatternForDEM.data.values
        #raindfallLocal=raindfallLocal/np.max(raindfallLocal)
        

        DEM_with_climate=self.ReturnDEMWithClimate(raindfallLocal)
        
        
        return DEM_with_climate,climatePatternForDEM
        
        
        
        
    def RemoveBasinsTouchingBoundries(self,basinIDperPixel):
        pass
        
    def GetArrayCorr(self):
        x=self.dem.X
        y=self.dem.Y
        x,y = np.meshgrid(x,y)

        return x,np.flipud(y)

    def GetFlatCorr(self):
        x=self.dem.X
        y=self.dem.Y
        x,y = np.meshgrid(x,y)

        return x.ravel(),np.flipud(y.ravel())

    def Export(self,filename,mask=None):
        if mask is None:
            mask=self.GenerateMaskForRiverID()

        x,y=self.GetFlatCorr()

        np.savez(filename,origFilanme=self.demFilename, A=self.A[mask], dX=self.dX[mask], Z=self.Z[mask], recs=self.recs
                 ,stack=self.stack,minElevation=self.Z0, minDraingeArea=self.A0,X=x[mask],Y=y[mask])
        
    def ExportByBasins(self,filename,basinIDs=None):
        x,y=self.GetFlatCorr()
        x=x.astype(self.ftype)
        y=y.astype(self.ftype)
        
        if basinIDs is None:
            rivers = self.riverData
            pixelMask=None
        else:
            basinIDs=np.unique(np.array(basinIDs))
            rivers = self.riverData[self.riverData['basinID'].isin(basinIDs)]
            pixelMask=np.isin(self.basinID.ravel(),basinIDs)
        
        mask=self.GenerateMaskForRiverID(rivers.nodes)
        
        
        
        np.savez_compressed(filename,origFilanme=self.demFilename, A=self.A, dX=self.dX, Z=self.Z, recs=self.recs
               ,stack=self.stack,minElevation=self.Z0, minDraingeArea=self.A0,XXflat=x,YYflat=y,riverMask=mask,pixelMask=pixelMask,
               riverNodes=self.riverData['nodes'],shape=[self.dem.ny,self.dem.nx])
        
        
        #return mask


    def GenerateMaskForRiverID(self, riverNodes=None):
        if riverNodes is None:
            riverNodes = self.riverData['nodes']
    
        mask = np.zeros_like(self.A, dtype=bool)
        mask[riverNodes] = True
    
        # verify that the number of True entries matches the number of river nodes
        true_count = mask.sum()
        expected = len(riverNodes)
        if true_count != expected:
            raise ValueError(
                f"Mask has {true_count} True values, but {expected} river nodes were provided"
            )
    
        return mask

    def SelectDemPointsByPolygon(self,polygon):
        print("Starting to check if dem is inside polygon - might take a hot min",flush=True)
        x,y=self.GetFlatCorr()
        insidePoly,_=pointsInSidePoly.CheckIfXandYptsInsidePolygon(x,y, polygon)
        return insidePoly


    def SelectRiversByPolygon(self,polygon,rivers=None):
        
        if rivers is None:
            rivers=self.riversData
        print("Starting to check if river is inside polygon - might take a hot min",flush=True)
        insidePoly,_=pointsInSidePoly.CheckIfXandYptsInsidePolygon(rivers.loc[:,'X'].values,rivers.loc[:,'Y'].values, polygon)
        rivers=rivers.loc[insidePoly]
            
        return rivers

    def PlotDEM(
        self,
        ax=None,
        cmap="gist_earth",
        colorbar=True,
        cbar_size="3%",
        cbar_pad=0.05,
        fig_width=10,
    ):
        """
        Plot DEM with an optional compact colorbar aligned to the axis.

        If no axis is provided, the figure size is chosen to approximately match
        the DEM aspect ratio.
        """

        import matplotlib.pyplot as plt
        

        extents = self.dem.extent()
        xmin, xmax, ymin, ymax = extents

        dem_width = xmax - xmin
        dem_height = ymax - ymin
        aspect_ratio = dem_height / dem_width

        if ax is None:
            fig_height = fig_width * aspect_ratio
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        else:
            fig = ax.figure

        im = ax.imshow(
            self.Z.reshape(self.dem.ny, self.dem.nx),
            cmap=cmap,
            vmin=self.Z0,
            extent=extents,
            origin="upper",
        )

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
            cb = fig.colorbar(im, cax=cax, orientation="vertical")
            cb.set_label("Elevation (m)")

        return im,ax
        
    def PlotDivded(self,ax=None,printNumberOfBasins=500):
        if ax is None:
            fig,ax=plt.subplots()
            
        
        ax.scatter(self.continentalDivide.X,self.continentalDivide.Y,s=0.05,color='white')
        basinSize = self.continentalDivide.groupby('basinID').size().sort_values(ascending=False)
        basinIDToPrint=basinSize[0:printNumberOfBasins].index
        
        for uniqueID_i in basinIDToPrint:
            data=self.continentalDivide.loc[self.continentalDivide.basinID==uniqueID_i]
            ax.text(np.mean(data.X),np.mean(data.Y),str(uniqueID_i),ha='center', va='center')
            
    def PlotDivdedByRivers(self, ax=None, printNumberOfBasins=None):
        if ax is None:
            fig, ax = plt.subplots()

        basinIDToPlot = pd.unique(self.riverData.basinID)

        if printNumberOfBasins is not None:
            basin_sizes = self.continentalDivide[
                self.continentalDivide.basinID.isin(basinIDToPlot)
            ].groupby("basinID").size().sort_values(ascending=False)

            basinIDToPlot = basin_sizes.index[:printNumberOfBasins]

        for basinID in basinIDToPlot:
            data = self.continentalDivide[self.continentalDivide.basinID == basinID]
            if data.empty:
                continue

            ax.text(data.X.mean(), data.Y.mean(), str(basinID),
                    ha="center", va="center", color="magenta")
            ax.scatter(data.X, data.Y, s=0.05, color="white")
            
            
    def ExportBasinByPolygon(self,filename,polygons):
        
        if not isinstance(polygons, list):
            polygons = [polygons]
        
        x,y=self.GetFlatCorr()
        print("Choosing these pixels you're  after, might take a hot second",flush=True)
        pts = np.column_stack((x, y))
        
        basinInPolygons=[]
        for polygon_i in polygons:
            insidePoly=pointsInSidePoly.points_in_polygon_mask_matplotlib(pts, polygon_i)
            basinInPolygon_i=self.basinID.flatten()[insidePoly]
            basinInPolygons.append(basinInPolygon_i)
        basinInPolygons = np.hstack(basinInPolygons)
        basinWithRivers=np.unique(self.riverData.basinID)
        
        basinToExport=np.intersect1d(basinInPolygons,basinWithRivers)
        self.ExportByBasins(filename,basinIDs=basinToExport)
        
        return basinToExport,self.BasinMask(basinToExport)
    
    def BasinMask(self,basinToExport):
        return np.isin(self.basinID.flatten(), basinToExport) #basin to export needs to be a list of basin
    
    def SaveBasinMask(self,filename,basinToExport):
        basinMask=self.BasinMask(basinToExport)
        np.savez_compressed(filename, basinMask=basinMask)
            
    def PlotRivers(self,ax=None):
        if ax is None:
            fig,ax=plt.subplots()
            
        ax.scatter(self.riverData.X,self.riverData.Y,s=0.1,color='blue')
        
    def PlotAll(self,ax=None,numOfBasinsPrint=None):
        if ax is None:
            fig,ax=plt.subplots()
            
        self.PlotDEM(ax=ax)
        self.PlotRivers(ax=ax)
        #self.PlotDivded(ax=ax,printNumberOfBasins=printNumberOfBasins)
        self.PlotDivdedByRivers(ax=ax,printNumberOfBasins=numOfBasinsPrint)
        
        
    def ComputeBasinMaskFromRiverMask(self,riverMask):
        """ this function gets river mask and then return the basin mask. Please notice that for this to work 
        properly it needs A0 and Z0 to set the same as when the river mask was set """
        basinIDs=np.unique(self.basinID.flatten()[riverMask])
        basinMask = np.isin(self.basinID.flatten(), basinIDs)
        
        return basinMask
        
        
            
            
        
        
#%%
class loadDEMDietRemoveBoundaries(loadDEMDiet):
    def RemoveBasinsTouchingBoundries(self,basinIDperPixel,BoundaryWith=10):
        basinIDtouchingBoundry=[]
        for basinID_i in [basinIDperPixel[0:BoundaryWith,:].flatten(),basinIDperPixel[-BoundaryWith:-1,:].flatten(),basinIDperPixel[:,0:BoundaryWith].flatten(),basinIDperPixel[:,-BoundaryWith:-1].flatten()]:
            basinIDtouchingBoundry.extend(basinID_i)
            
        basinIDtouchingBoundry=np.unique(np.array(basinIDtouchingBoundry))
        
        self.riverData = self.riverData[~self.riverData['basinID'].isin(basinIDtouchingBoundry)]
        self.continentalDivide = self.continentalDivide[~self.continentalDivide['basinID'].isin(basinIDtouchingBoundry)]
            
        
    
            

        
