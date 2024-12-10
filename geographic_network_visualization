# This code must be run after the main part of the code which is the simulation is done.
# The goal of the following lines is to map the final graph onto an arbitrary geographical map.
# Version 1.0

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# After updating radii and calculating edge properties

# Collect position values
pos_i_values = [pos[node][0] for node in pos]
pos_j_values = [pos[node][1] for node in pos]

# Find min and max of positions
min_pos_i, max_pos_i = min(pos_i_values), max(pos_i_values)
min_pos_j, max_pos_j = min(pos_j_values), max(pos_j_values)

# Normalize positions
range_pos_i = max_pos_i - min_pos_i
range_pos_j = max_pos_j - min_pos_j
normalized_pos = {
    node: (
        (pos[node][0] - min_pos_i) / range_pos_i,
        (pos[node][1] - min_pos_j) / range_pos_j,
    )
    for node in pos
}

# Define the geographic bounds for France
lon_min, lon_max = -5, 9
lat_min, lat_max = 41, 51

# Scale positions to geographic coordinates
scaled_pos = {
    node: (
        lon_min + normalized_pos[node][0] * (lon_max - lon_min),
        lat_min + normalized_pos[node][1] * (lat_max - lat_min),
    )
    for node in normalized_pos
}

# Set up the map with Cartopy
plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add map features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Normalize edge properties for visualization
max_r = max(radius_list)
edge_opacity = [r / max_r for r in radius_list]
edge_width = [(r / max_r) * 2 for r in radius_list]

# Plot edges
for edge, opacity, width in zip(G.edges(), edge_opacity, edge_width):
    x = [scaled_pos[edge[0]][0], scaled_pos[edge[1]][0]]
    y = [scaled_pos[edge[0]][1], scaled_pos[edge[1]][1]]
    ax.plot(x, y, transform=ccrs.PlateCarree(), color='blue', alpha=opacity, linewidth=width)

# Plot nodes
for node, (lon, lat) in scaled_pos.items():
    ax.plot(lon, lat, transform=ccrs.PlateCarree(), color='red', marker='o', markersize=2)

plt.title(f"Network at t={t}")
plt.show()
