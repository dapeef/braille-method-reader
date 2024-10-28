import numpy as np
from scipy.interpolate import interp1d
from stl import mesh
import numpy.typing as npt
import matplotlib.pyplot as plt


def normalize(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


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
        position (np.ndarray): A 3D numpy array (shape: [3]) representing the (x, y) position of the hemisphere's center. The z-coordinate is assumed to be 0.
        diameter (float): The diameter of the hemisphere.
        theta_resolution (int): Number of horizontal subdivisions (around the hemisphere's circumference).
        phi_resolution (int): Number of vertical subdivisions (from top to bottom).
    
    Returns:
        mesh.Mesh: The hemisphere as an STL mesh object.
    """
    # Ensure position is a 3D numpy array with shape [3]
    assert position.shape == (3,), "Position must be a 3D numpy array of shape [3]"
    
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
            z = position[2] + height * np.cos(phi)
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

def create_half_cylinder(start: np.ndarray, end: np.ndarray, diameter: float, 
                         theta_resolution: int = 20, length_resolution: int = 10) -> mesh.Mesh:
    """
    Creates a half-cylinder STL mesh oriented with its axis horizontally and its flat face on z=0.
    
    Args:
        start (np.ndarray): 3D numpy array for the starting point of the cylinder axis.
        end (np.ndarray): 3D numpy array for the ending point of the cylinder axis.
        diameter (float): The diameter of the cylinder.
        theta_resolution (int): Number of angular subdivisions around the circumference.
        length_resolution (int): Number of subdivisions along the cylinder's length.
    
    Returns:
        mesh.Mesh: The half-cylinder as an STL mesh object.
    """

    # Calculate radius and direction of the cylinder
    radius = diameter / 2
    axis_vector = end - start
    length = np.linalg.norm(axis_vector)
    axis_direction = axis_vector / length
    
    # Calculate perpendicular vectors for the circular cross-section orientation
    if np.allclose(axis_direction, [0, 0, 1]):
        perp_vector1 = np.array([1, 0, 0])
    else:
        perp_vector1 = np.cross(axis_direction, [0, 0, 1])
        perp_vector1 /= np.linalg.norm(perp_vector1)
    perp_vector2 = np.cross(axis_direction, perp_vector1)

    # Generate vertices for the half-cylinder along its length
    vertices = []
    for i in range(length_resolution + 1):
        t = i / length_resolution
        center_point = start + t * axis_vector
        for j in range(theta_resolution + 1):
            theta = -np.pi * (j / theta_resolution)  # Half-circle (0 to -pi)
            x = center_point + radius * (np.cos(theta) * perp_vector1 + np.sin(theta) * perp_vector2)
            vertices.append(x)

    # Generate vertices for the flat faces at both ends
    base_vertices_start = []
    base_vertices_end = []
    for j in range(theta_resolution + 1):
        theta = np.pi * (j / theta_resolution)
        x_start = start + radius * (np.cos(theta) * perp_vector1 + np.sin(theta) * perp_vector2)
        x_end = end + radius * (np.cos(theta) * perp_vector1 + np.sin(theta) * perp_vector2)
        base_vertices_start.append(x_start)
        base_vertices_end.append(x_end)
    
    # Add the center points at each flat face
    center_start_index = len(vertices)
    vertices.append(start)
    center_end_index = len(vertices)
    vertices.append(end)

    # Create faces for the curved surface
    faces = []
    for i in range(length_resolution):
        for j in range(theta_resolution):
            a = i * (theta_resolution + 1) + j
            b = a + theta_resolution + 1
            c = a + 1
            d = b + 1
            faces.append([a, b, c])
            faces.append([c, b, d])

    # Create faces for the flat semi-circular faces
    for j in range(theta_resolution):
        a = j
        b = j + 1
        faces.append([a, b, center_start_index])  # Start flat face
        a = (length_resolution * (theta_resolution + 1)) + j
        b = a + 1
        faces.append([a, center_end_index, b])  # End flat face

    # Convert to numpy arrays for vertices and faces
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Initialize the mesh and set up the vectors for each face
    half_cylinder = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            half_cylinder.vectors[i][j] = vertices[f[j]]
    
    return half_cylinder

def create_half_cylinder_path(points: np.ndarray, thickness: float, height: float,
                              resolution: int = 20, cap_style: str = "dome") -> mesh.Mesh:
    """
    Creates a half-cylinder STL mesh oriented with its axis horizontally and its flat face on z=0.
    
    Args:
        points (np.ndarray): An array of 3D points defining the path (shape: [N, 3]).
        thickness (float): The width of the semi-cylinder.
        height (float): The height of the semi-cylinder.
        theta_resolution (int): Number of angular subdivisions around the circumference.
        cap_style (string): Type of end to put on the tube. Choose from "dome" or "none".
    
    Returns:
        mesh.Mesh: The half-cylinder as an STL mesh object.
    """
    # Ensure points is a 2D numpy array with shape [N, 3]
    assert points.ndim == 2 and points.shape[1] == 3, "Points must be a 2D numpy array of shape [N, 3]"
    assert cap_style == "dome" or cap_style == "flat", f"Bad cap style: {cap_style}, Choose from \"dome\" or \"none\"."

    radius = thickness / 2

    # Generate vertices for the half-cylinder along its length
    vertices = []
    perp_vector1 = np.array([0, 0, 1])

    for i in range(len(points)):
        if i == 0 or i == len(points) - 1:
            # Calculate radius and direction of the cylinder
            if i == 0:
                axis_vector = points[1] - points[0]
            else:
                axis_vector = points[-1] - points[-2]

            axis_direction = normalize(axis_vector)
            
            # Calculate perpendicular vectors for the circular cross-section orientation
            perp_vector2 = np.cross(axis_direction, perp_vector1)
        
        else:
            in_vector = points[i] - points[i-1]
            in_vector = normalize(in_vector)
            out_vector = points[i+1] - points[i]
            out_vector = normalize(out_vector)

            if np.allclose(in_vector, out_vector):
                perp_vector2 = np.cross(out_vector, perp_vector1)
                
            else:
                perp_vector2 = out_vector - in_vector
                perp_vector2 = normalize(perp_vector2)
                
                # If it's a left-hand turn, the normal vector needs to be flipped to stop the line from turning inside out
                if np.allclose(normalize(np.cross(in_vector, out_vector)), [0, 0, 1]):
                    perp_vector2 *= -1


        for k in range(resolution + 1):
            theta = np.pi/2 - np.pi * (k / resolution)  # Half-circle (0 to -pi)
            x = points[i] + height * np.cos(theta) * perp_vector1 + radius * np.sin(theta) * perp_vector2
            vertices.append(x)
    
    # Create faces for curved surface
    faces = []
    for i in range(len(points) - 1):
        for j in range(resolution):
            a = i * (resolution + 1) + j
            b = a + (resolution + 1)
            c = a + 1
            d = b + 1
            faces.append([a, b, c])
            faces.append([c, b, d])

    # # Add the center points at each flat face
    # center_start_index = len(vertices)
    # vertices.append(points[0])
    # center_end_index = len(vertices)
    # vertices.append(points[1])

    # Create faces for the flat semi-circular faces
    # for j in range(theta_resolution):
    #     a = j
    #     b = j + 1
    #     faces.append([a, b, center_start_index])  # Start flat face
    #     a = (theta_resolution + 1) + j
    #     b = a + 1
    #     faces.append([a, center_end_index, b])  # End flat face

    # Convert to numpy arrays for vertices and faces
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Initialize the mesh and set up the vectors for each face
    half_cylinder = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            half_cylinder.vectors[i][j] = vertices[f[j]]
    
    # Add caps
    if cap_style == "dome":
        start_cap = create_hemisphere(points[ 0], thickness, height, theta_resolution=2*resolution, phi_resolution=int(resolution/2))
        end_cap =   create_hemisphere(points[-1], thickness, height, theta_resolution=2*resolution, phi_resolution=int(resolution/2))

        half_cylinder = mesh.Mesh(np.concatenate([half_cylinder.data, start_cap.data, end_cap.data]))

    return half_cylinder

def fillet_path(
    path: npt.NDArray[np.float64],
    resolution: int,
    radius: float = 1.0
) -> npt.NDArray[np.float64]:
    """
    Smooths corners in a 3D path by replacing corner points with circular fillets.
    
    Args:
        path: Nx3 numpy array containing 3D points defining the path
        fillet_resolution: Number of points to use for each fillet
        radius: Radius of the fillet curves (default: 1.0)
    
    Returns:
        Smoothed path as an Mx3 numpy array where M depends on the number of corners
        and fillet_resolution
    """
    if len(path) < 3:
        return path
    
    def create_fillet(
        p1: npt.NDArray[np.float64],
        corner: npt.NDArray[np.float64],
        p2: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Create points for a circular fillet at a corner."""
        # Get directions from corner to adjacent points
        v1 = normalize(p1 - corner)
        v2 = normalize(p2 - corner)
        
        # Angle between vectors
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        
        # If "corner" is actually the centre of a straight line, then don't need to add a fillet
        if np.allclose(angle, np.pi):
            return np.array([corner])
        
        else:
            # Calculate tangent points
            tan_dist = radius / np.tan(angle / 2)
            t1 = corner + v1 * tan_dist
            t2 = corner + v2 * tan_dist
            
            # Calculate center of fillet circle
            bisector = normalize(v1 + v2)
            center = corner + bisector * (radius / np.sin(angle / 2))
            
            # Create rotation matrix around the axis perpendicular to the plane
            axis = -np.cross(v1, v2)
            axis = normalize(axis)
            
            # Generate points along the fillet
            fillet_points = []
            for i in range(resolution):
                t = i / (resolution - 1)
                # Angle for this point
                current_angle = t * (np.pi - angle)
                
                # Create rotation matrix using Rodrigues' rotation formula
                K = np.array([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ])
                R = (np.eye(3) + np.sin(current_angle) * K + 
                    (1 - np.cos(current_angle)) * np.matmul(K, K))
                
                # Rotate initial vector to create fillet point
                initial_vec = normalize(t1 - center)
                rotated = np.dot(R, initial_vec)
                point = center + rotated * radius
                fillet_points.append(point)
                
            return np.array(fillet_points)
    
    # Generate smoothed path
    smoothed_points = []
    smoothed_points.append(path[0])  # Add first point
    
    for i in range(1, len(path) - 1):
        p1 = path[i - 1]
        corner = path[i]
        p2 = path[i + 1]
        
        # Create fillet for this corner
        fillet = create_fillet(p1, corner, p2)
        smoothed_points.extend(fillet)
    
    smoothed_points.append(path[-1])  # Add last point
    
    return np.array(smoothed_points)

def resample_path(path, interval):
    """
    Resample a 3D path to have points at evenly spaced intervals.
    
    Parameters:
    - path (numpy.ndarray): Original 3D path, shape (N, 3).
    - interval (float): Desired spacing between points in the new path.
    
    Returns:
    - numpy.ndarray: New path with points at evenly spaced intervals.
    """
    # Step 1: Calculate the distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # Cumulative distance along the path
    
    # Step 2: Determine the new evenly spaced distance values
    total_distance = cumulative_distances[-1]
    new_distances = np.arange(0, total_distance, interval)
    
    # Step 3: Interpolate to find new points at these distances
    new_path = np.zeros((len(new_distances), 3))  # Preallocate array for new path points
    for i in range(3):  # Interpolate each coordinate separately
        new_path[:, i] = np.interp(new_distances, cumulative_distances, path[:, i])
    
    return new_path

def resample_path_with_original_points(path, interval):
    """
    Interpolate a 3D path, keeping original points intact and adding interpolated points
    along each segment to achieve nearly uniform spacing.
    
    Parameters:
    - path (numpy.ndarray): Original 3D path, shape (N, 3).
    - interval (float): Desired approximate spacing between interpolated points.
    
    Returns:
    - numpy.ndarray: New path with original and interpolated points.
    """
    new_path = [path[0]]  # Start with the first point in the path
    
    for i in range(len(path) - 1):
        # Get the current and next point
        p1, p2 = path[i], path[i + 1]
        
        # Calculate the distance between p1 and p2
        segment_distance = np.linalg.norm(p2 - p1)
        
        # Determine number of interpolated points for this segment
        num_points = max(int(np.floor(segment_distance / interval)), 1)
        
        # Calculate interpolated points between p1 and p2
        for j in range(1, num_points + 1):
            t = j / num_points  # Interpolation parameter
            interpolated_point = (1 - t) * p1 + t * p2
            new_path.append(interpolated_point)
    
    return np.array(new_path)


def plot_3d_path(
    path: npt.NDArray[np.float64],
    show_points: bool = True,
    point_size: float = 50,
    line_width: float = 1,
    title: str = "3D Path",
    fig_size: tuple[int, int] = (10, 8)
) -> None:
    """
    Visualize a 3D path using matplotlib.
    
    Args:
        path: Nx3 numpy array containing 3D points defining the path
        show_points: Whether to show points along the path (default: True)
        point_size: Size of points if shown (default: 50)
        line_width: Width of the path line (default: 1)
        title: Plot title (default: "3D Path")
        fig_size: Figure size in inches (default: (10, 8))
    """
    # Create figure and 3D axes
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x, y, z = path.T
    
    # Plot the path as a line
    ax.plot(x, y, z, '-', linewidth=line_width, color='blue', zorder=1)
    
    # Optionally show points
    if show_points:
        ax.scatter(x, y, z, s=point_size, color='red', zorder=2)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make the plot more visually appealing
    ax.grid(True)
    
    # Set equal scaling for all axes
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
