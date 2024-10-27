import numpy as np
from stl import mesh
from geometry_utils import *


UNIT_THICKNESS = 5 # mm
UNIT_WIDTH = 10 # mm
UNIT_HEIGHT = 10 # mm

THICK_LINE_WIDTH = 2 # mm
THICK_LINE_HEIGHT = 1 # mm
THIN_LINE_WIDTH = 1 # mm
THIN_LINE_HEIGHT = .5 # mm
DOT_DIAMETER = 1.5 # mm
DOT_HEIGHT = DOT_DIAMETER / 2 # mm
DOT_SEPARATION = 2.3 # mm


shapes = []

shapes.append(create_cube(np.array([0, 0, -UNIT_THICKNESS]), np.array([UNIT_WIDTH, UNIT_HEIGHT, 0])))
shapes.append(create_hemisphere(np.array([8, 2, 0]), DOT_DIAMETER, DOT_HEIGHT))

path = np.array([
    [UNIT_WIDTH, UNIT_HEIGHT, 0],
    [0, 0, 0],
    [0, UNIT_HEIGHT, 0],
    [UNIT_WIDTH, UNIT_HEIGHT*2, 0],
    [UNIT_WIDTH, UNIT_HEIGHT*3, 0]
])
smoothed_path = fillet_path(path, 10, radius=THICK_LINE_WIDTH/2)
# plot_3d_path(path)
# plot_3d_path(smoothed_path)
shapes.append(create_half_cylinder_path(smoothed_path, THICK_LINE_WIDTH, THICK_LINE_HEIGHT))


# Write the mesh to file "cube.stl"
combined = mesh.Mesh(np.concatenate([shape.data for shape in shapes]))
combined.save('output.stl')
