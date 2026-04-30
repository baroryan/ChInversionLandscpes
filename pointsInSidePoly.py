#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:08:49 2024

@author: bar
"""

from numba import jit, njit
import numpy as np
import numba
from matplotlib.path import Path

#import inpoly
#%%

def points_in_polygon_mask_matplotlib(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Vectorized point‐in‐polygon test based on matplotlib.path.Path.contains_points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Array of N [x, y] coordinates to test.
    polygon : np.ndarray, shape (M, 2)
        Array of M vertices defining the polygon. You do *not* need to repeat
        the first vertex at the end—Path will implicitly close the loop.

    Returns
    -------
    mask : np.ndarray, shape (N,)
        Boolean array where mask[i] is True if points[i] lies inside the polygon.

    Notes
    -----
    This uses the C‐backed winding‐number algorithm in matplotlib.path.Path.contains_points
    for very fast batch testing of millions of points.
    """
    pts = np.asarray(points)
    poly = np.asarray(polygon)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be an (N, 2) array; got shape {pts.shape}")
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"polygon must be an (M, 2) array; got shape {poly.shape}")

    path = Path(poly)
    return path.contains_points(pts)

@jit(nopython=True)
def is_inside_sm(polygon, point):
    length = len(polygon)-1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii<length:
        dy  = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy*dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy<0 or dy2<0:
                F = dy*(polygon[jj][0] - polygon[ii][0])/(dy-dy2) + polygon[ii][0]

                if point[0] > F: # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F: # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2==0 and (point[0]==polygon[jj][0] or (dy==0 and (point[0]-polygon[ii][0])*(point[0]-polygon[jj][0])<=0)):
                return 2

        ii = jj
        jj += 1

    #print 'intersections =', intersections
    return intersections & 1  


@njit(parallel=True)
def is_inside_sm_parallel(points, polygon):
    ln = len(points)
    D = np.empty(ln, dtype=numba.boolean) 
    for i in numba.prange(ln):
        D[i] = is_inside_sm(polygon,points[i])
    return D  


def CheckIfPointsAreInsidePolygon(points,polygon):
    """ this is a simple wrapper for checking if points are inside a polygon using the inpoly package
    here I'm just checking that both numpy array of size [N,2] 
    you can find more methods here
    https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python"""
    
    points=np.asarray(points)
    polygon=np.asarray(polygon)
    
    if len(points.shape) != 2 or len(polygon.shape) != 2 or polygon.shape[1] !=2 or points.shape[1] != 2:
        raise TypeError("please make sure both to use 2D arrays! of size [N,2] ")
        
    if polygon.shape[0]<3:
        raise TypeError("polygon needs to include more points")
    
    if isinstance(points,np.ndarray) is False:
        raise TypeError("please make sure both are numpy arrays")
        
    if isinstance(polygon,np.ndarray) is False:
        raise TypeError("please make sure both are numpy arrays")
        
    #try:
    #    pointsInsidePolygon,pointsOnTopOfBoundary=inpoly.inpoly2(points,polygon)
   # except:
    pointsInsidePolygon=is_inside_sm_parallel(points,polygon)
    pointsOnTopOfBoundary=None
    
    return pointsInsidePolygon,pointsOnTopOfBoundary