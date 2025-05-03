#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict, deque
import math
import json
import argparse
from tqdm import tqdm

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Helper function to convert data for JSON serialization
def convert_to_json_serializable(obj):
    """Convert NumPy values to standard Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView

def load_kitti_labels(label_file):
    """
    Load KITTI format labels
    
    Args:
        label_file (str): Path to label file
        
    Returns:
        list: List of dictionaries containing ground truth bounding boxes
    """
    gt_boxes = []
    
    if not os.path.exists(label_file):
        return gt_boxes
    
    with open(label_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 15:  # KITTI format should have at least 15 values
                continue
                
            # Parse label
            obj_type = parts[0].lower()
            # Only keep relevant classes (car, pedestrian, cyclist)
            if obj_type not in ['car', 'van', 'truck', 'pedestrian', 'person', 'cyclist', 'bicycle']:
                continue
                
            # Map KITTI types to COCO types for compatibility with YOLOv11
            type_mapping = {
                'car': 'car',
                'van': 'car',
                'truck': 'truck',
                'pedestrian': 'person',
                'person': 'person',
                'cyclist': 'bicycle',
                'bicycle': 'bicycle'
            }
            
            mapped_type = type_mapping.get(obj_type, obj_type)
            
            # Get 2D bbox (KITTI format: left, top, right, bottom)
            bbox = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
            
            # Get 3D dimensions (height, width, length)
            dimensions = [float(parts[8]), float(parts[9]), float(parts[10])]
            
            # Get 3D location (x, y, z)
            location = [float(parts[11]), float(parts[12]), float(parts[13])]
            
            # Get rotation_y
            rotation_y = float(parts[14])
            
            # Create ground truth object
            gt_box = {
                'class_name': mapped_type,
                'bbox_2d': bbox,
                'dimensions': dimensions,  # [height, width, length]
                'location': location,      # [x, y, z]
                'rotation_y': rotation_y
            }
            
            gt_boxes.append(gt_box)
    
    return gt_boxes

def calculate_iou(box1, box2):
    """
    Calculate IoU between two 2D bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there's an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate area of intersection
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    
    return iou

def calculate_precision_recall(detections, ground_truths, iou_threshold=0.5):
    """
    Calculate precision and recall metrics
    
    Args:
        detections (dict): Dictionary of detected boxes by class
        ground_truths (dict): Dictionary of ground truth boxes by class
        iou_threshold (float): IoU threshold for a true positive
        
    Returns:
        dict: Precision and recall metrics
    """
    results = {}
    
    # Process each class separately
    all_classes = set(list(detections.keys()) + list(ground_truths.keys()))
    
    for class_name in all_classes:
        class_dets = detections.get(class_name, [])
        class_gts = ground_truths.get(class_name, [])
        
        # Sort detections by confidence score (descending)
        class_dets = sorted(class_dets, key=lambda x: x.get('score', 0), reverse=True)
        
        # Initialize metrics
        tp = 0  # True positives
        fp = 0  # False positives
        
        # Mark ground truths as matched or not
        gt_matched = [False] * len(class_gts)
        
        # Process each detection
        for det in class_dets:
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_gts):
                if gt_matched[gt_idx]:
                    continue  # Skip already matched ground truths
                
                # Calculate IoU between detection and ground truth
                iou = calculate_iou(det['bbox_2d'], gt['bbox_2d'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if detection is a true positive or false positive
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / len(class_gts) if len(class_gts) > 0 else 0
        
        # Store metrics
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'true_positives': tp,
            'false_positives': fp,
            'ground_truths': len(class_gts),
            'detections': len(class_dets),
            'ap': precision * recall  # Simplified AP calculation
        }
    
    # Calculate mean metrics
    if results:
        num_classes = len(results)
        mean_precision = sum(results[c]['precision'] for c in results) / num_classes
        mean_recall = sum(results[c]['recall'] for c in results) / num_classes
        mean_ap = sum(results[c]['ap'] for c in results) / num_classes
        
        results['mean'] = {
            'precision': mean_precision,
            'recall': mean_recall,
            'ap': mean_ap
        }
    
    return results

def process_images(args):
    """Process images from dataset and evaluate against ground truth"""
    # Configuration
    # Device settings
    device = args.device
    print(f"Using device: {device}")
    
    # Model settings
    yolo_model_size = args.model_size
    depth_model_size = args.depth_model
    
    # Detection settings
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    
    # Initialize models
    print("Initializing models...")
    detector = ObjectDetector(
        model_size=yolo_model_size,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        device=device
    )
    
    depth_estimator = DepthEstimator(
        model_size=depth_model_size,
        device=device
    )
    
    bbox3d_estimator = BBox3DEstimator()
    
    # Get dataset path
    image_dir = args.image_dir
    label_dir = args.label_dir
    
    # Verify directories
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        return
    
    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found: {label_dir}")
        return
    
    # Find image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if args.num_images > 0:
        image_files = image_files[:args.num_images]
    
    if not image_files:
        print(f"Error: No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Prepare for evaluation
    all_detections = {}
    all_ground_truths = {}
    
    # Process each image
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(image_dir, img_file)
        
        # Get corresponding label file
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error reading image: {img_path}")
            continue
        
        # Get dimensions
        height, width = image.shape[:2]
        
        # Step 1: Object Detection
        detection_frame, detections = detector.detect(image.copy(), track=False)
        
        # Step 2: Depth Estimation
        depth_map = depth_estimator.estimate_depth(image)
        
        # Step 3: 3D Bounding Box Estimation
        boxes_3d = []
        
        for detection in detections:
            bbox, score, class_id, obj_id = detection
            
            # Get class name
            class_name = detector.get_class_names()[class_id]
            
            # Get depth in the region of the bounding box
            depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
            
            # Create a simplified 3D box representation
            box_3d = {
                'bbox_2d': bbox,
                'depth_value': depth_value,
                'class_name': class_name,
                'score': float(score)
            }
            
            # Add location information for 3D representation
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Normalize coordinates to center of image
            normalized_x = center_x - width / 2
            normalized_y = height / 2 - center_y
            
            # Use depth to scale for pseudo-3D position
            multiplier = 0.01
            location_x = normalized_x * depth_value * multiplier
            location_y = 0  # Assume on ground plane
            location_z = depth_value
            
            # Store location
            box_3d['location'] = [location_x, location_y, location_z]
            
            # Add dimensions (width, height, length)
            width_px = bbox[2] - bbox[0]
            height_px = bbox[3] - bbox[1]
            
            # Scale factor to convert pixels to meters at given depth
            scale_factor = 0.01 * depth_value
            width_m = width_px * scale_factor
            height_m = height_px * scale_factor
            length_m = width_m * 2.5  # Approximation for length
            
            box_3d['dimensions'] = [height_m, width_m, length_m]
            
            # Add orientation (simplified)
            orientation = np.arctan2(normalized_x, depth_value)
            box_3d['orientation'] = orientation
            
            boxes_3d.append(box_3d)
        
        # Load ground truth from KITTI labels
        gt_boxes = load_kitti_labels(label_file)
        
        # Organize detections and ground truths by class
        for box in boxes_3d:
            class_name = box['class_name'].lower()
            if class_name not in all_detections:
                all_detections[class_name] = []
            all_detections[class_name].append(box)
        
        for box in gt_boxes:
            class_name = box['class_name'].lower()
            if class_name not in all_ground_truths:
                all_ground_truths[class_name] = []
            all_ground_truths[class_name].append(box)
        
        # Visualize results if needed
        if args.visualize:
            # Draw detections and ground truth on the image
            vis_img = image.copy()
            
            # Create a separate image for 3D visualization
            vis_3d_img = image.copy()
            
            # Draw detections
            for box in boxes_3d:
                bbox = box['bbox_2d']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                score = box['score']
                class_name = box['class_name']
                
                # Draw 2D bounding box on standard visualization
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f"{class_name}: {score:.2f}"
                cv2.putText(vis_img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw 3D bounding box on the 3D visualization
                # We'll use the bbox3d_estimator to draw the 3D box
                try:
                    # Use the depth, dimensions and orientation from our calculation
                    location = box['location']
                    dimensions = box['dimensions']
                    orientation = box['orientation']
                    
                    # Create a box_3d dict with all required fields for the 3D box drawing function
                    full_box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': box['depth_value'],
                        'class_name': class_name,
                        'location': location,
                        'dimensions': dimensions,
                        'orientation': orientation
                    }
                    
                    # Use the draw_box_3d function to visualize the 3D box
                    vis_3d_img = bbox3d_estimator.draw_box_3d(vis_3d_img, full_box_3d, color=(255, 0, 0))
                    
                    # Add estimated dimensions and distance text
                    height_m, width_m, length_m = dimensions
                    depth = box['depth_value']
                    dims_text = f"Dims: {width_m:.2f}x{length_m:.2f}x{height_m:.2f}m"
                    depth_text = f"Depth: {depth:.2f}m"
                    
                    cv2.putText(vis_3d_img, dims_text, (x1, y2 + 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(vis_3d_img, depth_text, (x1, y2 + 55), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                except Exception as e:
                    print(f"Error drawing 3D box: {e}")
            
            # Draw ground truth
            for box in gt_boxes:
                bbox = box['bbox_2d']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                class_name = box['class_name']
                
                # Draw 2D bounding box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"GT: {class_name}"
                cv2.putText(vis_img, label, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Also draw ground truth 3D boxes if available with all required information
                if 'dimensions' in box and 'location' in box:
                    try:
                        # Use the bbox3d_estimator to draw the ground truth 3D box
                        gt_full_box = {
                            'bbox_2d': bbox,
                            'class_name': class_name,
                            'location': box['location'],
                            'dimensions': box['dimensions'],
                            'orientation': box.get('rotation_y', 0)
                        }
                        
                        # Draw the ground truth 3D box in green
                        vis_3d_img = bbox3d_estimator.draw_box_3d(vis_3d_img, gt_full_box, color=(0, 255, 0))
                    except Exception as e:
                        print(f"Error drawing ground truth 3D box: {e}")
            
            # Save visualizations
            vis_dir = args.visualize_dir
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save standard visualization
            vis_path = os.path.join(vis_dir, f"vis_{base_name}.jpg")
            cv2.imwrite(vis_path, vis_img)
            
            # Save 3D visualization
            vis_3d_path = os.path.join(vis_dir, f"vis3d_{base_name}.jpg")
            cv2.imwrite(vis_3d_path, vis_3d_img)
            
            # Create a combined visualization
            combined_height = max(vis_img.shape[0], vis_3d_img.shape[0])
            combined_width = vis_img.shape[1] + vis_3d_img.shape[1]
            combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Add titles
            title_height = 30
            combined_img_with_title = np.zeros((combined_height + title_height, combined_width, 3), dtype=np.uint8)
            combined_img_with_title[title_height:, :] = combined_img
            
            # Copy images to the combined visualization
            combined_img_with_title[title_height:title_height+vis_img.shape[0], :vis_img.shape[1]] = vis_img
            combined_img_with_title[title_height:title_height+vis_3d_img.shape[0], vis_img.shape[1]:] = vis_3d_img
            
            # Add titles
            cv2.putText(combined_img_with_title, "2D Detection", (vis_img.shape[1]//2 - 70, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_img_with_title, "3D Bounding Boxes", (vis_img.shape[1] + vis_3d_img.shape[1]//2 - 100, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add vertical separation line
            cv2.line(combined_img_with_title, (vis_img.shape[1], 0), (vis_img.shape[1], combined_height + title_height), (255, 255, 255), 2)
            
            # Save combined visualization
            combined_path = os.path.join(vis_dir, f"combined_{base_name}.jpg")
            cv2.imwrite(combined_path, combined_img_with_title)
    
    # Calculate evaluation metrics
    metrics = calculate_precision_recall(all_detections, all_ground_truths, iou_threshold=0.5)
    
    # Save results
    results = {
        'metrics': metrics,
        'config': {
            'model_size': yolo_model_size,
            'depth_model': depth_model_size,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'device': device,
            'num_images': len(image_files)
        }
    }
    
    # Convert to JSON serializable
    results = convert_to_json_serializable(results)
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {args.output}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print("===================")
    
    if 'mean' in metrics:
        print(f"mAP: {metrics['mean']['ap']:.4f}")
        print(f"Mean Precision: {metrics['mean']['precision']:.4f}")
        print(f"Mean Recall: {metrics['mean']['recall']:.4f}")
    
    print("\nClass-wise Results:")
    for class_name, values in metrics.items():
        if class_name != 'mean':
            print(f"{class_name.capitalize()}:")
            print(f"  AP: {values['ap']:.4f}")
            print(f"  Precision: {values['precision']:.4f}")
            print(f"  Recall: {values['recall']:.4f}")
            print(f"  TP/FP: {values['true_positives']}/{values['false_positives']}")
            print(f"  GT: {values['ground_truths']}, Det: {values['detections']}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D detection model on dataset")
    parser.add_argument("--image_dir", type=str, default="image_2", help="Directory containing images")
    parser.add_argument("--label_dir", type=str, default="label_2", help="Directory containing ground truth labels")
    parser.add_argument("--num_images", type=int, default=50, help="Number of images to process (0 for all)")
    parser.add_argument("--model_size", type=str, default="small", help="YOLOv11 model size (nano, small, medium, large, extra)")
    parser.add_argument("--depth_model", type=str, default="small", help="Depth model size (small, base, large)")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--iou_threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cuda, cpu, mps)")
    parser.add_argument("--output", type=str, default="model_evaluation.json", help="Output JSON file")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--visualize_dir", type=str, default="visualizations", help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    process_images(args)

if __name__ == "__main__":
    main() 