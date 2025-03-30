import cv2
import numpy as np


image_path = 'carr.jpeg'  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not loaded. Check the image path.")
    exit()

height, width, channels = img.shape
print("Image loaded with dimensions:", height, width, channels)

points = np.array([
    (60, 100), (1500, 100), (1500, 700), (60, 700),  # Front face
    (80, 120), (1520, 120), (1520, 720), (80, 720)   # Back face
])

# Compute bounding box dimensions
box_width = points[1][0] - points[0][0]  # Width (X-axis)
box_height = points[2][1] - points[0][1]  # Height (Y-axis)
box_depth = points[4][0] - points[0][0]  # Depth (Z-axis, based on shift)

# Compute the centroid of the bounding box
centroid_x = np.mean(points[:, 0]).astype(int)
centroid_y = np.mean(points[:, 1]).astype(int)
centroid = (centroid_x, centroid_y)

# Print dimensions to the console
print("\nBounding Box Dimensions:")
print(f"Width: {box_width} pixels")
print(f"Height: {box_height} pixels")
print(f"Depth: {box_depth} pixels\n")

# Print dimensions and centroid to the console
print("\nBounding Box Dimensions:")
print(f"Width: {box_width} pixels")
print(f"Height: {box_height} pixels")
print(f"Depth: {box_depth} pixels\n")

print(f"Centroid of the bounding box: {centroid}")

# Print point coordinates
print("Bounding Box Corner Points (X, Y):")
for i, point in enumerate(points):
    face = "Front Face" if i < 4 else "Back Face"
    print(f"Point {i+1} ({face}): {point}")

# Function to draw the 3D bounding box, label points, and mark the centroid
def draw_box(img, points, centroid, color=(0, 255, 0)):
    for i in range(4):
        # Draw front and back face edges
        cv2.line(img, points[i], points[(i+1) % 4], color, 2)
        cv2.line(img, points[i+4], points[(i+1) % 4 + 4], color, 2)
        cv2.line(img, points[i], points[i+4], color, 2)  # Connecting edges

        # Draw and label each point
        cv2.circle(img, points[i], 5, (0, 0, 255), -1)  # Red for front face
        cv2.circle(img, points[i+4], 5, (255, 0, 0), -1)  # Blue for back face
        
        # Label points with coordinates
        cv2.putText(img, f"{points[i]}", (points[i][0] + 10, points[i][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"{points[i+4]}", (points[i+4][0] + 10, points[i+4][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw the centroid
    cv2.circle(img, centroid, 6, (0, 0, 0), -1)  # Black dot at the centroid
    cv2.putText(img, f"Centroid: {centroid}", (centroid[0] + 10, centroid[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Add label "Car" above the bounding box
    label_position = (points[0][0], points[0][1] - 30)  # Slightly above the front-top-left corner
    cv2.putText(img, "Car", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)


draw_box(img, points,centroid)
cv2.imshow("3D Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("annotated_3d_box.jpeg", img)
