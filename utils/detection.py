import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import os

@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    """
    Loads and caches the YOLOv8 object detection model.
    """
    # Create model folder if not exists
    os.makedirs("model", exist_ok=True)
    # Try to load. If it's not present, ultralytics will auto-download it
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        # Fallback to loading standard model from current path
        return YOLO("yolov8n.pt")

def process_frame(frame, model, conf_threshold=0.3, enable_blur=False, enable_grid=False, enable_heatmap=False, heatmap_buffer=None, grid_cols=3, grid_rows=3):
    """
    Processes a single frame: runs YOLOv8 person detection, applies selected overlays (boxes, face blurring, grid counts, heatmaps).
    Returns:
        processed_frame (np.ndarray): The annotated image frame.
        people_count (int): Total number of detected people.
        grid_counts (dict): Number of people in each grid cell.
    """
    h, w, c = frame.shape
    results = model(frame, verbose=False)
    
    people_count = 0
    detections = []
    
    # Extract boxes
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # class 0 is person
            if cls == 0 and conf >= conf_threshold:
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Clip coordinates to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                detections.append((x1, y1, x2, y2, conf))
                
    # 1. Update Heatmap Buffer if enabled
    if enable_heatmap and heatmap_buffer is not None:
        # Decay existing heatmap buffer
        heatmap_buffer *= 0.93
        
        # Add heat in detected people's bounding boxes
        for (x1, y1, x2, y2, _) in detections:
            # Add stronger heat in the center of the box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bw, bh = x2 - x1, y2 - y1
            
            # Simple Gaussian-like heat injection
            y_indices, x_indices = np.ogrid[0:h, 0:w]
            mask = ((x_indices - cx) ** 2 / (bw/2) ** 2 + (y_indices - cy) ** 2 / (bh/2) ** 2) <= 1.0
            heatmap_buffer[mask] += 30.0
            
        # Clip heatmap values to 0-255
        heatmap_buffer[:] = np.clip(heatmap_buffer, 0, 255)
        
    # 2. Process Grid-Based Regional Counting
    grid_counts = {}
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    
    # Initialize counts for all grid coordinates
    for r in range(grid_rows):
        for col in range(grid_cols):
            grid_counts[f"R{r+1}C{col+1}"] = 0
            
    # Assign people to grid cells based on box center
    for (x1, y1, x2, y2, _) in detections:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        grid_col = int(cx / cell_w)
        grid_row = int(cy / cell_h)
        
        # Clamp bounds
        grid_col = min(grid_cols - 1, max(0, grid_col))
        grid_row = min(grid_rows - 1, max(0, grid_row))
        
        grid_counts[f"R{grid_row+1}C{grid_col+1}"] += 1
        
    # --- RENDER OVERLAYS ON FRAME ---
    out_frame = frame.copy()
    
    # A. Render Heatmap Overlay first (so boxes draw on top of it)
    if enable_heatmap and heatmap_buffer is not None:
        heatmap_img = heatmap_buffer.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        
        # Merge heatmap with output frame
        out_frame = cv2.addWeighted(out_frame, 0.65, heatmap_color, 0.35, 0)
        
    # B. Render Grid Overlay
    if enable_grid:
        grid_overlay = out_frame.copy()
        
        # Draw grid lines and counts
        for r in range(grid_rows):
            y = int((r + 1) * cell_h)
            if r < grid_rows - 1:
                cv2.line(out_frame, (0, y), (w, y), (150, 150, 150), 1, cv2.LINE_AA)
                
        for col in range(grid_cols):
            x = int((col + 1) * cell_w)
            if col < grid_cols - 1:
                cv2.line(out_frame, (x, 0), (x, h), (150, 150, 150), 1, cv2.LINE_AA)
                
        # Highlight cells by density and print regional counts
        for r in range(grid_rows):
            for col in range(grid_cols):
                cell_id = f"R{r+1}C{col+1}"
                count = grid_counts[cell_id]
                
                # Determine colors based on cell count
                if count == 0:
                    overlay_color = None
                elif count <= 2:
                    overlay_color = (0, 255, 0) # Green (Low)
                elif count <= 5:
                    overlay_color = (0, 255, 255) # Yellow (Medium)
                else:
                    overlay_color = (0, 0, 255) # Red (High)
                    
                x_start = int(col * cell_w)
                y_start = int(r * cell_h)
                x_end = int((col + 1) * cell_w)
                y_end = int((r + 1) * cell_h)
                
                if overlay_color:
                    # Draw a semi-transparent cell overlay
                    cv2.rectangle(grid_overlay, (x_start, y_start), (x_end, y_end), overlay_color, -1)
                    
                # Put small regional label text
                label = f"{cell_id}: {count}"
                cv2.putText(out_frame, label, (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
        cv2.addWeighted(grid_overlay, 0.15, out_frame, 0.85, 0, out_frame)
        
    # C. Render Bounding Boxes and Privacy Blur
    for (x1, y1, x2, y2, conf) in detections:
        # 1. Apply Face Blur if enabled
        if enable_blur:
            # Approximate the head region (top 20% of the bounding box)
            head_h = int((y2 - y1) * 0.20)
            head_w = int((x2 - x1) * 0.60)
            
            hx1 = x1 + int((x2 - x1 - head_w) / 2)
            hx2 = hx1 + head_w
            hy1 = y1
            hy2 = y1 + head_h
            
            # Ensure coordinates are within image boundaries
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(w, hx2), min(h, hy2)
            
            # Apply strong Gaussian blur
            if hx2 > hx1 and hy2 > hy1:
                head_roi = out_frame[hy1:hy2, hx1:hx2]
                # Filter size must be odd
                ksize = max(15, int(head_w | 1))
                ksize = ksize if ksize % 2 == 1 else ksize + 1
                blurred_head = cv2.GaussianBlur(head_roi, (ksize, ksize), 30)
                out_frame[hy1:hy2, hx1:hx2] = blurred_head
                
        # 2. Draw Premium Styled Bounding Box
        box_color = (46, 204, 113) # Nice Emerald Green
        if people_count > 10:
            box_color = (231, 76, 60) # Alizarin Red if overall limit is likely crossed
        elif people_count > 5:
            box_color = (241, 196, 15) # Sunflower Yellow
            
        # Draw semi-transparent filled box overlay for glassmorphism box effect
        box_overlay = out_frame.copy()
        cv2.rectangle(box_overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(box_overlay, 0.12, out_frame, 0.88, 0, out_frame)
        
        # Bounding box border
        cv2.rectangle(out_frame, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)
        
        # Small corner brackets to look like a high-tech camera target
        len_corner = min(15, int((x2 - x1) * 0.15))
        # Top-Left corner
        cv2.line(out_frame, (x1, y1), (x1 + len_corner, y1), box_color, 2)
        cv2.line(out_frame, (x1, y1), (x1, y1 + len_corner), box_color, 2)
        # Top-Right corner
        cv2.line(out_frame, (x2, y1), (x2 - len_corner, y1), box_color, 2)
        cv2.line(out_frame, (x2, y1), (x2, y1 + len_corner), box_color, 2)
        # Bottom-Left corner
        cv2.line(out_frame, (x1, y2), (x1 + len_corner, y2), box_color, 2)
        cv2.line(out_frame, (x1, y2), (x1, y2 - len_corner), box_color, 2)
        # Bottom-Right corner
        cv2.line(out_frame, (x2, y2), (x2 - len_corner, y2), box_color, 2)
        cv2.line(out_frame, (x2, y2), (x2, y2 - len_corner), box_color, 2)
        
        # Add labels
        label = f"Person: {conf:.2f}"
        (lbl_w, lbl_h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(out_frame, (x1, y1 - lbl_h - 6), (x1 + lbl_w + 6, y1), box_color, -1)
        cv2.putText(out_frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
    return out_frame, people_count, grid_counts
