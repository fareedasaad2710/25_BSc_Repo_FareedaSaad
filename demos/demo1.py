import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open('coco.names', 'r'
) as f:
    classes = [line.strip() for line in f.readlines()]
    print("Classes loaded:", classes)


# Load the image
image_path = 'demo1_image.jpeg'  # Replace with your image path
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not loaded. Check the image path.")
    exit()

height, width, channels = img.shape
print("Image loaded with dimensions:", height, width, channels)

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize lists for detected class IDs, confidences, and bounding boxes
class_ids = []
confidences = []
boxes = []

# Process the output
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1 and classes[class_id] in ['car', 'truck', 'bus']:  # Adjusted confidence threshold
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-max suppression to eliminate redundant overlapping boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.1)
if len(indexes) == 0:
    print("No vehicles detected.")
else:
    print(f"Detected {len(indexes)} vehicles.")

# Draw bounding boxes around detected vehicles
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(f"Drawing box: {x}, {y}, {w}, {h} with label: {label}")
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Save the image with detected vehicles
output_image_path = 'output.jpg'
cv2.imwrite(output_image_path, img)
print(f"Output image saved to {output_image_path}")

# Display the image with detected vehicles
cv2.imshow('Vehicle Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
