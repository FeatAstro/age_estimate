#Here we will create a list of extinction curves in order to have the A_V as a function of distance.
#Import all of the needed modules
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
#from tqdm import tqdm

import astropy.coordinates as coord
import astropy.units as u

from scipy.interpolate import LinearNDInterpolator
import h5py
from itertools import product


from astropy.table import Table
from astropy.io import fits

from scipy.stats import gaussian_kde

from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap
from cycler import cycler
import itertools
from astropy.coordinates import GalacticLSR, CartesianDifferential
from scipy.interpolate import griddata

import matplotlib.patheffects as patheffects


start = time.time()





data = fits.getdata('OB_Associations_1kpc_Members.fits', 1)
t = Table(data)
X= np.array(t['X'])
Y= np.array(t['Y'])
Z= np.array(t['Z'])
Vl= np.array(t['Vl'])
Vb= np.array(t['Vb'])
Assoc = np.array(t['ID'])
Name = np.array(t['Name'])
marker = np.array(t['marker'])
color = np.array(t['colour'])

Xc = np.split(X, np.unique(Assoc, return_index=True)[1][1:])
Yc = np.split(Y, np.unique(Assoc, return_index=True)[1][1:])
Zc = np.split(Z, np.unique(Assoc, return_index=True)[1][1:])
Vlc = np.split(Vl, np.unique(Assoc, return_index=True)[1][1:])
Vbc = np.split(Vb, np.unique(Assoc, return_index=True)[1][1:])
Namec = np.split(Name, np.unique(Assoc, return_index=True)[1][1:])
markerc = np.split(marker, np.unique(Assoc, return_index=True)[1][1:])
colorc = np.split(color, np.unique(Assoc, return_index=True)[1][1:])
Asso = np.split(Assoc, np.unique(Assoc, return_index=True)[1][1:])


    


fig, ax1 = plt.subplots(1,1,figsize =(10,10))



for j in range(len(Xc)):
    marker = markerc[j][0]
    color = colorc[j][0]
    Name = Namec[j][0]
    ax1.plot(Xc[j], Yc[j], marker=marker, markersize=6,markeredgecolor='black', color=color, linestyle='None')





    
#ax1.annotate(r'$l = 270^{\circ}$', xy = (-50,-950), fontsize = 16,zorder = 20) 
#ax1.annotate(r'$l = 90^{\circ}$', xy = (-50,900), fontsize = 16, zorder = 20)
#ax1.annotate(r'$l = 180^{\circ}$', xy = (-975,0), fontsize = 16,zorder = 20) 
#ax1.annotate(r'$l = 0^{\circ}$', xy = (840,0), fontsize = 16,zorder = 20)



fits_image_filename = 'mean_and_std_xyz.fits'
hdu_list = fits.open(fits_image_filename)
density_map = hdu_list[1].data

img_top_down = np.nansum(density_map, axis=0)

cb=plt.imshow(img_top_down, origin='lower', cmap='binary', aspect='equal', extent=[-1250, 1250, -1250, 1250],vmin=0,vmax=0.3) # random units
# plt.colorbar(cb)



circle1 = plt.Circle((0, 0), 1000, color='k', fill = False)
ax1.add_patch(circle1)
ax1.plot(0,0, marker = r'$\odot$', markersize = 12, color = 'black')
ax1.set_xlabel('X (pc)', fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax1.set_ylabel('Y (pc)', fontdict={'fontsize': 20, 'fontweight': 'medium'})
ax1.tick_params(axis="x", labelsize=18)
ax1.tick_params(axis="y", labelsize=18)
ax1.set_xticks(np.arange(-1000,1200,200))
ax1.set_yticks(np.arange(-1000,1200,200))
ax1.set_xlim(-1000,1000)
ax1.set_ylim(-1000,1000)


fig.savefig('MapOBAssoc1kpcXY_EdenhoferExtinctionMap_WithNames.png', bbox_inches = 'tight')

