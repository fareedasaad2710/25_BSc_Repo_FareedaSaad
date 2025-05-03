def extract_box_depth_features(boxes_2d, depth_map):
    features = []
    for box in boxes_2d:
        x1, y1, x2, y2 = map(int, box)
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            mean_depth = 0
        else:
            mean_depth = np.mean(crop)
        features.append([x1, y1, x2, y2, mean_depth])
    return torch.tensor(features, dtype=torch.float32)
