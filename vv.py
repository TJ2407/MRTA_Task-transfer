import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import pickle
with open('prepaths.dat', 'rb') as file:
    prep=pickle.load(file)
with open('postpaths.dat', 'rb') as file44:
    postp=pickle.load(file44)
# Load the image
image = cv2.imread('/Users/tejas_sriganesh/Desktop/proj/unnamed.jpg')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range in HSV for red (adjust based on your image)
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])  # Define the range for red color

# Create a mask for red points
red_mask = cv2.inRange(hsv_image, red_lower, red_upper)

# Define color range in HSV for white (adjust based on your image)
white_lower = np.array([0, 0, 200])
white_upper = np.array([255, 30, 255])  # Define the range for white color

# Create a mask for white points
white_mask = cv2.inRange(hsv_image, white_lower, white_upper)

# Combine the masks to get obstacles (any color other than white and red)
obstacle_mask = cv2.bitwise_not(red_mask + white_mask)

# Find coordinates of points in the obstacle category
obstacle_coordinates = np.argwhere(obstacle_mask > 0)

# Find contours in the red mask
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find centroids of individual red clusters
red_centroids = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        red_centroids.append((cX, cY))

# Function to get neighboring cells of a given cell in the grid
def get_neighbors(row, col, num_rows, num_cols):
    neighbors = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row < num_rows - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < num_cols - 1:
        neighbors.append((row, col + 1))
    return neighbors

# Create a grid with each box of 10x10 units over the image
grid_step = 10
grid_color = 'gray'

# Determine the number of grid cells in both dimensions
num_cols = (image.shape[1] + grid_step - 1) // grid_step
num_rows = (image.shape[0] + grid_step - 1) // grid_step

# Initialize the grid colors with white
grid_colors = [['white' for _ in range(num_cols)] for _ in range(num_rows)]

# Mark obstacle cells as blue
for coord in obstacle_coordinates:
    row = coord[0] // grid_step
    col = coord[1] // grid_step
    grid_colors[row][col] = 'blue'

# Make the next nearest neighbors of red points white
for centroid in red_centroids:
    row = centroid[1] // grid_step
    col = centroid[0] // grid_step
    grid_colors[row][col] = 'red'
    
    neighbors = get_neighbors(row, col, num_rows, num_cols)
    for neighbor in neighbors:
        n_row, n_col = neighbor
        if 0 <= n_row < num_rows and 0 <= n_col < num_cols and grid_colors[n_row][n_col] == 'blue':
            grid_colors[n_row][n_col] = 'white'

print(grid_colors)

# Define color mappings for visualization
color_mapping = {
    'white': (255, 255, 255),  # white color in RGB
    'red': (0, 0, 255),       # red color in RGB
    'blue': (0, 0, 0) ,     # blue color in RGB
    'green': (0,100,0),
    'purple':(100,0,100)
}


# Create an image to visualize the map
grid_colors[30][48]='white'
map_image = np.zeros((len(grid_colors), len(grid_colors[0]), 3), dtype=np.uint8)
def plot_nodes_on_map(nodes, color='green'):
    for node in nodes:
        col, row = node
        grid_colors[row][col] = color
nodes_to_plot = prep[0]  # Replace this with your own list of nodes
plot_nodes_on_map(nodes_to_plot, color='green')

for centroid in red_centroids:
    row = centroid[1] // grid_step
    col = centroid[0] // grid_step
    grid_colors[row][col] = 'red'
# Assign colors to grid cells based on the loaded data
for row in range(len(grid_colors)):
    for col in range(len(grid_colors[0])):
        cell_color = color_mapping[grid_colors[row][col]]
        map_image[row, col] = cell_color

# Show the map using Matplotlib
plt.imshow(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
n1=postp[0]
plot_nodes_on_map(n1,color='purple')
for row in range(len(grid_colors)):
    for col in range(len(grid_colors[0])):
        cell_color = color_mapping[grid_colors[row][col]]
        map_image[row, col] = cell_color
plt.imshow(cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
'''
# Save the grid_colors matrix to a binary file using pickle
with open('map.dat', 'wb') as file:
    pickle.dump(grid_colors, file)
'''
# Define a function to convert coordinates to match the graph axes
def convert_to_origin(coord):
    if isinstance(coord, int):
        return (coord * grid_step, (num_rows - 1) * grid_step)
    x, y = coord
    return (x * grid_step, (num_rows - y) * grid_step)

# Define A* pathfinding function with modified boundary checks
def a_star(start, end, grid):
    num_rows = len(grid)
    num_cols = len(grid[0])
    
    def heuristic(node):
        return abs(node[0] - end[0]) + abs(node[1] - end[1])
    
    open_set = PriorityQueue()
    open_set.put((0, start))
    
    came_from = {}
    g_score = {(i, j): float('inf') for i in range(num_rows) for j in range(num_cols)}
    g_score[start] = 0
    
    while not open_set.empty():
        current = open_set.get()[1]
        
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        
        for neighbor in get_neighbors(current[0], current[1], num_rows, num_cols):
            n_row, n_col = neighbor
            if 0 <= n_row < num_rows and 0 <= n_col < num_cols:
                tentative_g_score = g_score[current] + 1  # Assuming each step has a cost of 1
                
                if grid[n_row][n_col] == 'blue':
                    continue  # Skip blue cells
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor)
                    open_set.put((f_score, neighbor))

    return None
red_point_coordinates = []

for row in range(len(grid_colors)):
    for col in range(len(grid_colors[0])):
        if grid_colors[row][col] == 'red':
            # Add the coordinates of the red point to the list
            red_point_coordinates.append((col, row))  # Note the (col, row) order for (x, y)

# Print the red point coordinates
for x, y in red_point_coordinates:
    print(f"Red Point Coordinate: ({x}, {y})")
red_point_paths = []

# Define point numbers for red points
point_numbers = list(range(len(red_point_coordinates)))

# Function to convert point numbers to coordinates
def get_coordinates(point_number):
    return red_point_coordinates[point_number]

# Calculate paths between all possible pairs of red points
for i in range(len(point_numbers)):
    for j in range(i + 1, len(point_numbers)):
        start = get_coordinates(point_numbers[i])
        end = get_coordinates(point_numbers[j])
        path = a_star(start, end, grid_colors)
        if path:
            red_point_paths.append((point_numbers[i], point_numbers[j], path))

# Display point numbers for red points
for point_number, (x, y) in enumerate(red_point_coordinates):
    print(f"Point {point_number}: ({x}, {y})")

# Ask for user input on two point numbers
while True:
    start_point = int(input("Enter the starting point number (0-{}): ".format(len(point_numbers) - 1)))
    end_point = int(input("Enter the ending point number (0-{}): ".format(len(point_numbers) - 1)))

    if start_point < 0 or start_point >= len(point_numbers) or end_point < 0 or end_point >= len(point_numbers):
        print("Invalid point numbers. Please enter valid point numbers.")
    else:
        break

# Find the path between the selected points
selected_path = None
for path in red_point_paths:
    if (path[0] == start_point and path[1] == end_point) or (path[0] == end_point and path[1] == start_point):
        selected_path = path[2]

# Display the selected path
if selected_path:
    print(f"Path from Point {start_point} to Point {end_point}:")
    for point in selected_path:
        x, y = point
        print(f"({x}, {y})")
else:
    print("No path found between the selected points.")
print(len(grid_colors))
print(len(grid_colors[0]))
file.close()
file44.close()