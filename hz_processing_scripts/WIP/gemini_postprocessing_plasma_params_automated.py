import gemini3d.read as read
from gemini3d.grid.gridmodeldata import model2geogcoords
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# File setup
direc = '/Users/clevenger/Simulations/agu24/fac_precip/'  # path to directory containing simulation output files
outdir = '/Users/clevenger/Simulations/agu24/fac_precip/postprocessing/'  # path to save plot outputs

# File reading stuff
cfg = read.config(direc)
xg = read.grid(direc)
parm = "v1"  # choose from ne, Ti, Te, v1, v2, v3, J1, J2, or J3

lalt = 256
llon = 256
llat = 256  # resolutions to plot model outputs (higher = better)

# User-specified time range
start_index = 60  # Adjust as needed
end_index = 120   # Adjust as needed

# Loop over the desired range of time indices
for time_idx in range(start_index, end_index + 1):
    time_value = cfg["time"][time_idx]  # Get the current time value
    print(f"Processing time index {time_idx} with time {time_value}...")
    
    # Read the data for the current time
    dat = read.frame(direc, time_value, var=parm)
    galti, gloni, glati, parmgi = model2geogcoords(xg, dat[parm], lalt, llon, llat, wraplon=True)

    # Altitude cuts
    altitudes = [110e3, 310e3, 510e3, 710e3]
    alt_indices = [np.argmin(abs(galti - alt)) for alt in altitudes]

    # Lock latitude and longitude into place
    ilon = np.argmin(abs(gloni - 212))
    ilat = np.argmin(abs(glati - 65))

    # Create 3D figure
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)

    # Normalize plot color/quantities
    norm = Normalize(vmin=np.nanmin(parmgi), vmax=np.nanmax(parmgi))
    mappable = ScalarMappable(norm=norm, cmap='bwr')

    # Plot altitude cuts
    for alt_idx in alt_indices:
        X, Y = np.meshgrid(gloni, glati)  # latitude and longitude
        Z = np.full_like(X, galti[alt_idx])  # altitude
        C = parmgi[alt_idx, :, :].T  # extract data at fixed altitude
        C_normalized = (C - np.nanmin(C)) / (np.nanmax(C) - np.nanmin(C)) # normalize for color mapping
        
        print("Plotting each altitude cut...")
        ax.plot_surface(X, Y, Z, facecolors=plt.cm.bwr(C_normalized), rstride=1, cstride=1, edgecolor='none')

    print("Making combined altitude slice figure...")
    # Add colorbar + labels
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('v$_{1} (m/s)$') # or 'Ti (K)', 'Te (K)', 'v$_{1} (m/s)$', 'J_{1} (A/m$^{2}$)', 'ne (m$^{-3}$)'
    ax.set_xlabel('Latitude (deg)')
    ax.set_ylabel('Longitude (deg)')
    ax.set_zlabel('Altitude (m)')
    ax.view_init(20, 45) # adjust to move 3d space around to view cuts however you want
    ax.set_title(f'Line of Sight Velocity at Various Altitudes\nTime: {time_value.strftime("%Y-%m-%d %H:%M:%S")}') # change for whatever parm
    
    # Save the figure
    filename = f'v1_alt_cut_{time_value.strftime("%Y%m%d_%H%M%S")}.png'  # Format time for filename
    plt.savefig(f'{outdir}/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved figure: {outdir}/{filename}")

    plt.close(fig)  # Close the figure to free up memory
