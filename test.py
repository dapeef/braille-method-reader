import numpy as np
from stl import mesh


UNIT_THICKNESS = 5 # mm
UNIT_WIDTH = 10 # mm
UNIT_HEIGHT = 10 # mm

THICK_LINE_WIDTH = 2 # mm
THICK_LINE_HEIGHT = 1 # mm
THIN_LINE_WIDTH = 1 # mm
THIN_LINE_HEIGHT = .5 # mm


def create_cube(corner1:np.ndarray, corner2:np.ndarray) ->mesh.Mesh:
    # Define the 8 vertices of the cube
    vertices = np.array([
        [corner1[0], corner1[1], corner1[2]],
        [corner2[0], corner1[1], corner1[2]],
        [corner2[0], corner2[1], corner1[2]],
        [corner1[0], corner2[1], corner1[2]],
        [corner1[0], corner1[1], corner2[2]],
        [corner2[0], corner1[1], corner2[2]],
        [corner2[0], corner2[1], corner2[2]],
        [corner1[0], corner2[1], corner2[2]],
    ])

    # Define the 12 triangles composing the cube
    faces = np.array([
        [0,3,1],
        [1,3,2],
        [0,4,7],
        [0,7,3],
        [4,5,6],
        [4,6,7],
        [5,1,2],
        [5,2,6],
        [2,3,6],
        [3,7,6],
        [0,1,5],
        [0,5,4]
    ])

    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[face[j],:]

    return cube


shapes = []

shapes.append(create_cube(np.array([0, 0, -UNIT_THICKNESS]), np.array([UNIT_WIDTH, UNIT_HEIGHT, 0])))

combined = mesh.Mesh(np.concatenate([shape.data for shape in shapes]))

# Write the mesh to file "cube.stl"
combined.save('output.stl')