import numpy as np
from stl import mesh
from geometry_utils import Plate, PlateConfig
from method_utils import *


config = PlateConfig()
config.UNIT_THICKNESS = .4 # mm
config.UNIT_WIDTH = 10 # mm
config.UNIT_HEIGHT = 5 # mm

config.THICK_LINE_WIDTH = 2 # mm
config.THICK_LINE_HEIGHT = 1 # mm
config.THIN_LINE_WIDTH = 1 # mm
config.THIN_LINE_HEIGHT = .5 # mm
config.DOT_DIAMETER = 1.5 # mm
config.DOT_HEIGHT = config.DOT_DIAMETER / 2 # mm
config.DOT_SEPARATION = 2.3 # mm


data = get_method_from_id("16694")
method = Method(data)

plate = Plate(method, config)

# Base plate
shapes.append(create_cube(np.array([-UNIT_WIDTH, -UNIT_HEIGHT * (method.lead_length + 1), -UNIT_THICKNESS]),
                          np.array([UNIT_WIDTH * method.stage, UNIT_HEIGHT, 0])))

# Vertical lines
for i in range(method.stage):
    shapes.append(create_half_cylinder_path(np.array([[i * UNIT_WIDTH, 0, 0],
                                                      [i * UNIT_WIDTH, -method.lead_length * UNIT_HEIGHT, 0]]),
                                            thickness=THIN_LINE_WIDTH,
                                            height=THIN_LINE_HEIGHT))

rows = method.get_first_lead()
path = Method.path_from_method(rows, 2, UNIT_WIDTH, UNIT_HEIGHT)
smoothed_path = fillet_path(path, resolution=10, radius=THICK_LINE_WIDTH/2)
shapes.append(create_half_cylinder_path(smoothed_path, THICK_LINE_WIDTH, THICK_LINE_HEIGHT))
# plot_3d_path(path)
# plot_3d_path(smoothed_path)
# plot_3d_path(resampled_path)

path = Method.path_from_method(rows, 1, UNIT_WIDTH, UNIT_HEIGHT)
resampled_path = resample_path_with_original_points(path, DOT_SEPARATION)
for point in resampled_path:
    shapes.append(create_hemisphere(point, DOT_DIAMETER, DOT_HEIGHT))


# Write the mesh to file "cube.stl"
combined = mesh.Mesh(np.concatenate([shape.data for shape in shapes]))
combined.save('output.stl')
