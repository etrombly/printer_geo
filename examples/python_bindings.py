#!/usr/bin/env python

from printer_geo import stl, compute, geo, geo_vulkan

resolution = 0.01
scale = 1 / resolution
radius = 0.25
stepover_percent = 90
stepover = radius * 2 * (stepover_percent / 100) # will probably add some helper functions for this
stepdown = 1 # 1mm
tool = geo_vulkan.new_endmill(radius, scale);
vk = compute.init_vk()
tris = stl.stl_to_tri("/home/eric/Projects/dropcutter/spider.stl")
geo.move_to_zero(tris)
bounds = geo.get_bounds(tris)
tris = geo_vulkan.to_tri_vk(tris) #convert mesh to vulkan compatible types
grid = geo_vulkan.generate_grid(bounds, scale)
columns = geo_vulkan.generate_columns(grid, bounds, resolution, scale)
partitions = compute.partition_tris(tris, columns, vk)
heightmap = geo_vulkan.generate_heightmap(grid, partitions, vk)
toolpath = geo_vulkan.generate_toolpath(heightmap, bounds, tool, radius, stepover, scale)
layers = geo_vulkan.generate_layers(toolpath, bounds, stepdown)
gcode = geo_vulkan.generate_gcode(layers, bounds)
f = open("/home/eric/Projects/dropcutter/spider.nc", "w")
f.write(gcode)
f.close()