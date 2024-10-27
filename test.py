import numpy as np
from stl import mesh


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

def create_hemisphere(position: np.ndarray, diameter: float, height: float, theta_resolution: int = 40, phi_resolution: int = 10) -> mesh.Mesh:
    """
    Creates a hemisphere STL mesh.
    
    Args:
        position (np.ndarray): A 2D numpy array (shape: [2]) representing the (x, y) position of the hemisphere's center. The z-coordinate is assumed to be 0.
        diameter (float): The diameter of the hemisphere.
        theta_resolution (int): Number of horizontal subdivisions (around the hemisphere's circumference).
        phi_resolution (int): Number of vertical subdivisions (from top to bottom).
    
    Returns:
        mesh.Mesh: The hemisphere as an STL mesh object.
    """
    # Ensure position is a 2D numpy array with shape [2]
    assert position.shape == (2,), "Position must be a 2D numpy array of shape [2]"
    
    # Calculate radius from diameter
    radius = diameter / 2
    
    # Generate vertices for the hemisphere
    vertices = []
    for i in range(phi_resolution + 1):
        phi = (np.pi / 2) * (i / phi_resolution)  # phi goes from 0 to pi/2 for hemisphere
        for j in range(theta_resolution):
            theta = (2 * np.pi) * (j / theta_resolution)
            x = position[0] + radius * np.cos(theta) * np.sin(phi)
            y = position[1] + radius * np.sin(theta) * np.sin(phi)
            z = height * np.cos(phi)
            vertices.append([x, y, z])
    
    # Add the center point at the base (for closing the bottom of the hemisphere)
    vertices.append([position[0], position[1], 0])
    base_center_index = len(vertices) - 1  # Index of this base center vertex
    
    # Create faces (triangles) for the hemisphere surface
    faces = []
    for i in range(phi_resolution + 1):
        for j in range(theta_resolution):
            # Current vertex indices in the grid
            a = i * theta_resolution + j
            b = a + theta_resolution
            c = (a + 1) % theta_resolution + (i * theta_resolution)
            d = (c + theta_resolution) % (theta_resolution * (phi_resolution + 1))
            
            # Top triangles on the hemisphere surface
            if i < phi_resolution:
                faces.append([a, b, c])
                faces.append([c, b, d])
            else:
                # Bottom triangles closing the base
                faces.append([c, a, base_center_index])

    # Convert to numpy array for vertices and faces
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Initialize the mesh and set up the vectors for each face
    hemisphere = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            hemisphere.vectors[i][j] = vertices[f[j]]
    
    return hemisphere



shapes = []

shapes.append(create_cube(np.array([0, 0, -UNIT_THICKNESS]), np.array([UNIT_WIDTH, UNIT_HEIGHT, 0])))
shapes.append(create_hemisphere(np.array([2, 2]), DOT_DIAMETER, DOT_HEIGHT))


combined = mesh.Mesh(np.concatenate([shape.data for shape in shapes]))

# Write the mesh to file "cube.stl"
combined.save('output.stl')