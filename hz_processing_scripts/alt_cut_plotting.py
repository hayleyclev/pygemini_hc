import gemini3d.read as read
from gemini3d.grid.gridmodeldata import model2geogcoords
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# File setup
direc = '/Users/clevenger/Simulations/paper01_event01/' # path to directory containing simulation output files
outdir = '/Users/clevenger/Projects/paper01/events/20230227/gemini_postprocessing/' # path to save plot outputs

# File reading stuff
cfg = read.config(direc) 
xg = read.grid(direc)
parm = "ne" # choose ne, Ti, Te, v1, v2, v2, J1, J2, or J3
dat = read.frame(direc, cfg["time"][2], var=parm) # change [2] for desire index START AT 61

lalt = 256; llon = 256; llat = 256; # resolutions to plot model outputs (higher=better)

print("Sampling in geographic coords...")
galti, gloni, glati, parmgi = model2geogcoords(xg, dat[parm], lalt, llon, llat, wraplon=True)

print("Creating altitude slices...")
altitudes = [110e3, 310e3, 510e3, 710e3]
alt_indices = [np.argmin(abs(galti - alt)) for alt in altitudes]

# Lock latitude and longitude into place
ilon = np.argmin(abs(gloni - 212))
ilat = np.argmin(abs(glati - 65))

# Creating 3D figure space to put 2D maps onto
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
ax.grid(False)

# Normalize plot color/quantities so all of the 2D maps can share the same colorbar
norm = Normalize(vmin=np.nanmin(parmgi), vmax=np.nanmax(parmgi))
mappable = ScalarMappable(norm=norm, cmap='gnuplot2')

print("Making altitude cuts...")
for alt_idx in alt_indices:
    X, Y = np.meshgrid(gloni, glati)  # latitude and longitude
    Z = np.full_like(X, galti[alt_idx])  # altitude
    C = parmgi[alt_idx, :, :].T  # extract data at fixed altitude
    C_normalized = (C - np.nanmin(C)) / (np.nanmax(C) - np.nanmin(C)) # normalize for color mapping
    
    print("Plotting each altitude cut...")
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.gnuplot2(C_normalized), rstride=1, cstride=1, edgecolor='none')

print("Making combined altitude slice figure...")
# Add colorbar + labels
cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('ne (m$^{-3}$)') # or 'Ti (K)', 'Te (K)', 'v$_{1} (m/s)$', 'J_{1} (A/m$^{2}$)', 'ne (m$^{-3}$)'
ax.set_xlabel('Latitude (deg)')
ax.set_ylabel('Longitude (deg)')
ax.set_zlabel('Altitude (m)')
ax.view_init(20, 45) # adjust to move 3d space around to view cuts however you want
ax.set_title('Electron Density at various altitudes') # change for whatever parm

print("Saving figure...")
# Save + show plots
filename = 'ne_alt_cuts.png' # adjust filename for plot being saved
plt.savefig(f'{outdir}/{filename}', dpi=300, bbox_inches='tight')  # adjust dpi for plot resolution

plt.show()
