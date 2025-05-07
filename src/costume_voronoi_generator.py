import numpy as np
from scipy.spatial import Voronoi
import meshio

# Domain parameters
H, w = 100.0, 100.0
num_seeds = 700  # Adjusted to target ~3505 elements

# Fiber direction data from ImageJ/Fiji (angles in degrees and counts)
angles_deg = np.array([-95.5, -90.5, -85.5, -80.5, -75.5, -70.5, -65.5, -60.5, -55.5, -50.5, -45.5, -40.5, -35.5, -30.5, -25.5, -20.5, -15.5, -10.5, -5.5, -0.5, 4.5, 9.5, 14.5, 19.5, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5, 54.5, 59.5, 64.5, 69.5, 74.5, 79.5, 84.5, 90.5])
counts = np.array([436, 409, 370, 351, 388, 337, 314, 336, 365, 429, 428, 412, 375, 429, 439, 464, 408, 403, 366, 300, 406, 395, 415, 493, 505, 487, 542, 412, 3901, 3742, 3458, 3355, 3180, 828, 880, 777, 798, 363])
total_fibers = counts.sum()  # 3505 fibers

# Sample fiber directions based on the histogram
fiber_angles = []
for angle, count in zip(angles_deg, counts):
    fiber_angles.extend([np.deg2rad(angle)] * count)
fiber_angles = np.array(fiber_angles)

# Generate seed points to align Voronoi edges with fiber directions
np.random.seed(42)
points = []
for theta in fiber_angles[:num_seeds]:
    base = np.random.uniform([0, 0], [w, H])
    offset = 0.5 * np.array([np.cos(theta + np.pi/2), np.sin(theta + np.pi/2)])
    points.append(base + offset)
    points.append(base - offset)

# Add boundary points to ensure nodes at y=0 and y=100
num_boundary_points = 20
bottom_points = np.array([[x, 0.0] for x in np.linspace(0, w, num_boundary_points)])
top_points = np.array([[x, H] for x in np.linspace(0, w, num_boundary_points)])
points = np.vstack([points, bottom_points, top_points])
points = np.array(points)

# Compute Voronoi tessellation
vor = Voronoi(points)

# Extract edges (ridges) as line segments within the domain
all_vertices = vor.vertices
edges = []
y_threshold = 0.9 * H  # Maximum allowed y-difference for edges (90% of domain height)
for ridge in vor.ridge_vertices:
    if -1 in ridge:
        continue  # Skip infinite ridges
    v0, v1 = ridge
    p0, p1 = all_vertices[v0], all_vertices[v1]
    # Check if both points are within the domain
    if (0 <= p0[0] <= w and 0 <= p0[1] <= H) and (0 <= p1[0] <= w and 0 <= p1[1] <= H):
        # Check if the y-difference is within the threshold
        y_diff = abs(p1[1] - p0[1])
        if y_diff <= y_threshold:
            edges.append([v0, v1])

# Add edges to connect boundary points
boundary_indices = list(range(len(points) - 2 * num_boundary_points, len(points)))
for i in range(num_boundary_points - 1):
    edges.append([boundary_indices[i], boundary_indices[i + 1]])
    edges.append([boundary_indices[num_boundary_points + i], boundary_indices[num_boundary_points + i + 1]])
edges = np.array(edges)

# Filter unique vertices used in edges
used_vertices = np.unique(edges)
vertex_map = {old: new for new, old in enumerate(used_vertices)}
filtered_vertices = all_vertices[used_vertices]
edges = np.array([[vertex_map[v0], vertex_map[v1]] for v0, v1 in edges])

# Ensure boundary points are exactly at y=0 and y=100
final_vertices = filtered_vertices.copy()
boundary_vertex_map = {}  # Map to track new indices of boundary points

# Add bottom and top points explicitly to the vertex list
bottom_indices_new = []
top_indices_new = []
for i, idx in enumerate(boundary_indices):
    if idx in used_vertices:
        new_idx = vertex_map[idx]
        if i < num_boundary_points:  # Bottom point
            final_vertices[new_idx] = bottom_points[i]
            bottom_indices_new.append(new_idx)
        else:  # Top point
            final_vertices[new_idx] = top_points[i - num_boundary_points]
            top_indices_new.append(new_idx)
    else:
        if i < num_boundary_points:
            final_vertices = np.vstack([final_vertices, bottom_points[i]])
            bottom_indices_new.append(len(final_vertices) - 1)
        else:
            final_vertices = np.vstack([final_vertices, top_points[i - num_boundary_points]])
            top_indices_new.append(len(final_vertices) - 1)

# Update edges to use the new indices if any boundary points were added
edges_updated = edges.copy()
for i, idx in enumerate(boundary_indices):
    if idx in used_vertices:
        new_idx = vertex_map[idx]
    else:
        new_idx = bottom_indices_new[i] if i < num_boundary_points else top_indices_new[i - num_boundary_points]
    boundary_vertex_map[idx] = new_idx

for e in range(len(edges_updated)):
    v0, v1 = edges_updated[e]
    if v0 in boundary_vertex_map:
        edges_updated[e][0] = boundary_vertex_map[v0]
    if v1 in boundary_vertex_map:
        edges_updated[e][1] = boundary_vertex_map[v1]

# Diagnostic: Check y-coordinates of boundary points after adjustment
bottom_y_after = [final_vertices[i][1] for i in bottom_indices_new]
top_y_after = [final_vertices[i][1] for i in top_indices_new]
print(f"Bottom boundary y-coordinates (after adjustment): {bottom_y_after}")
print(f"Top boundary y-coordinates (after adjustment): {top_y_after}")

# Clamp y-coordinates to ensure they stay within [0, 100]
final_vertices[:, 1] = np.clip(final_vertices[:, 1], 0.0, H)

# Diagnostic: Check y-coordinate range after clamping
print(f"Final vertices y-range after clamping: [{final_vertices[:, 1].min():.2f}, {final_vertices[:, 1].max():.2f}]")

# Tag nodes at the top (y = 100) and bottom (y = 0) boundaries
boundary_tags = np.zeros(len(final_vertices), dtype=int)
tolerance = 1e-3
for i, vertex in enumerate(final_vertices):
    if abs(vertex[1]) < tolerance:
        boundary_tags[i] = 1
    elif abs(vertex[1] - H) < tolerance:
        boundary_tags[i] = 2

# Diagnostic: Check tagged nodes and vertex count
num_bottom = np.sum(boundary_tags == 1)
num_top = np.sum(boundary_tags == 2)
print(f"Number of final vertices: {len(final_vertices)}")
print(f"Number of bottom nodes (tag 1): {num_bottom}")
print(f"Number of top nodes (tag 2): {num_top}")

# Assign all elements to region 1
cell_data = {"region": [np.ones(len(edges_updated), dtype=int)]}

# Add boundary tags as point data
point_data = {"boundary_tags": boundary_tags}

# Prepare mesh for saving
cells = [("line", edges_updated)]
mesh = meshio.Mesh(points=final_vertices, cells=cells, cell_data=cell_data, point_data=point_data)

# Save to XDMF
meshio.write("voronoi_nxt_v8.xdmf", mesh, file_format="xdmf", data_format="HDF")
