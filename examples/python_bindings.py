#!/usr/bin/env python

from printer_geo import stl, compute, geo

resolution = 0.01
scale = 1 / resolution
radius = 0.25
stepover_percent = 90
stepover = radius * 2 * (stepover_percent / 100) # will probably add some helper functions for this
stepdown = 1 # 1mm
print("Creating tool")
tool = geo.new_endmill(radius, scale)
print("Initializing vulkan")
vk = compute.init_vk()
print("Reading stl")
tris = stl.stl_to_tri("/home/eric/Projects/dropcutter/safe_stl.stl")
geo.move_to_zero(tris)
bounds = geo.get_bounds(tris)
grid = geo.generate_grid(bounds, scale)
columns = geo.generate_columns_chunks(grid, bounds, resolution, scale)
print("Partitioning tris")
partitions = compute.partition_tris(tris, columns, vk)
print("Creating heightmap")
heightmap = geo.generate_heightmap_chunks(grid, partitions, vk)
print("Creating toolpath")
toolpath = geo.generate_toolpath(heightmap, bounds, tool, radius, stepover, scale)
print("Creating layers")
layers = geo.generate_layers(toolpath, bounds, stepdown)
print("Creating gcode")
gcode = geo.generate_gcode(layers, bounds)
f = open("/home/eric/Projects/dropcutter/safe_stl.nc", "w")
f.write(gcode)
f.close()