import json, cv2
import numpy as np
from pathlib import Path

def dots_to_braille_char(dots):
    """
    Convert 6-dot pattern to Braille character representation
    Returns a string representation of the dot pattern
    """
    # Convert dots list to a string pattern for easy lookup
    pattern = ''.join(map(str, dots))
    
    # Common Braille patterns (you can expand this dictionary)
    braille_patterns = {
        '100000': 'a',     # dot 1
        '110000': 'b',     # dots 1,2
        '100100': 'c',     # dots 1,4
        '100110': 'd',     # dots 1,4,5
        '100010': 'e',     # dots 1,5
        '110100': 'f',     # dots 1,2,4
        '110110': 'g',     # dots 1,2,4,5
        '110010': 'h',     # dots 1,2,5
        '010100': 'i',     # dots 2,4
        '010110': 'j',     # dots 2,4,5
        '101000': 'k',     # dots 1,3
        '111000': 'l',     # dots 1,2,3
        '101100': 'm',     # dots 1,3,4
        '101110': 'n',     # dots 1,3,4,5
        '101010': 'o',     # dots 1,3,5
        '111100': 'p',     # dots 1,2,3,4
        '111110': 'q',     # dots 1,2,3,4,5
        '111010': 'r',     # dots 1,2,3,5
        '011100': 's',     # dots 2,3,4
        '011110': 't',     # dots 2,3,4,5
        '101001': 'u',     # dots 1,3,6
        '111001': 'v',     # dots 1,2,3,6
        '010111': 'w',     # dots 2,4,5,6
        '101101': 'x',     # dots 1,3,4,6
        '101111': 'y',     # dots 1,3,4,5,6
        '101011': 'z',     # dots 1,3,5,6
        '000000': 'space', # no dots (space)
        # Add more patterns as needed
    }
    
    # Return the character if found, otherwise return the pattern itself
    return braille_patterns.get(pattern, f'pattern_{pattern}')

def create_braille_categories():
    """
    Create comprehensive Braille categories
    You can expand this based on your dataset's actual patterns
    """
    categories = []
    
    # Generate all possible 6-dot combinations (2^6 = 64 possible patterns)
    for i in range(64):
        # Convert number to 6-bit binary representation
        dots = [(i >> j) & 1 for j in range(6)]
        pattern = ''.join(map(str, dots))
        char_name = dots_to_braille_char(dots)
        
        categories.append({
            "id": i + 1,
            "name": char_name,
            "supercategory": "braille_character",
            "dot_pattern": pattern
        })
    
    return categories

def get_category_id_from_dots(dots):
    """
    Convert 6-dot pattern to category ID (1-64)
    """
    # Convert dots to binary number
    binary_value = 0
    for i, dot in enumerate(dots):
        binary_value |= (dot << i)
    
    return binary_value + 1  # Add 1 since category IDs start from 1

def calculate_bbox_with_padding(x1, y1, x2, y2, padding_ratio=0.15):
    """
    Calculate bounding box with padding to ensure the character is properly enclosed
    
    Args:
        x1, y1, x2, y2: Original corner coordinates
        padding_ratio: Ratio of padding to add (default 15%)
    
    Returns:
        (x, y, width, height) in COCO format
    """
    # Original width and height
    orig_w = x2 - x1
    orig_h = y2 - y1
    
    # Calculate padding
    pad_w = orig_w * padding_ratio
    pad_h = orig_h * padding_ratio
    
    # Apply padding
    padded_x1 = max(0, x1 - pad_w)
    padded_y1 = max(0, y1 - pad_h)
    padded_x2 = x2 + pad_w
    padded_y2 = y2 + pad_h
    
    # Convert to COCO format (x, y, width, height)
    final_x = padded_x1
    final_y = padded_y1
    final_w = padded_x2 - padded_x1
    final_h = padded_y2 - padded_y1
    
    return final_x, final_y, final_w, final_h

def debug_bbox_calculation(img_path, txt_path, max_debug_cells=5):
    """
    Debug function to visualize bbox calculations
    """
    try:
        with open(txt_path, 'r') as f:
            lines = f.read().strip().splitlines()
        
        if len(lines) < 3:
            return
            
        angle = float(lines[0].strip())
        verts = list(map(int, lines[1].split()))
        horzs = list(map(int, lines[2].split()))
        cells = lines[3:] if len(lines) > 3 else []
        
        print(f"\nDebug for {img_path.name}:")
        print(f"  Angle: {angle}")
        print(f"  Vertical lines: {len(verts)} positions - {verts[:10]}..." if len(verts) > 10 else f"  Vertical lines: {verts}")
        print(f"  Horizontal lines: {len(horzs)} positions - {horzs[:10]}..." if len(horzs) > 10 else f"  Horizontal lines: {horzs}")
        print(f"  Total cells: {len(cells)}")
        
        # Debug first few cells
        debug_count = 0
        for cell_line in cells:
            if debug_count >= max_debug_cells:
                break
                
            if not cell_line.strip():
                continue
                
            try:
                parts = cell_line.split()
                if len(parts) < 8:
                    continue
                    
                r, c = int(parts[0]), int(parts[1])
                dots = [int(x) for x in parts[2:8]]
                
                # Calculate indices CAREFULLY
                col_idx = (c-1) * 2  # Each cell spans 2 vertical lines
                row_idx = (r-1) * 3  # Each cell spans 3 horizontal lines
                
                print(f"  Cell {debug_count+1}: r={r}, c={c}")
                print(f"    Dots: {dots} -> {dots_to_braille_char(dots)}")
                print(f"    Col indices: {col_idx}, {col_idx+1} (need < {len(verts)})")
                print(f"    Row indices: {row_idx}, {row_idx+2} (need < {len(horzs)})")
                
                if col_idx+1 >= len(verts) or row_idx+2 >= len(horzs):
                    print(f"    ERROR: Index out of bounds!")
                    continue
                
                x1, x2 = verts[col_idx], verts[col_idx+1]
                y1, y3 = horzs[row_idx], horzs[row_idx+2]
                
                print(f"    Original bbox: x1={x1}, y1={y1}, x2={x2}, y2={y3}")
                print(f"    Original size: {x2-x1} x {y3-y1}")
                
                # Calculate with padding
                final_x, final_y, final_w, final_h = calculate_bbox_with_padding(x1, y1, x2, y3)
                print(f"    Final bbox: x={final_x:.1f}, y={final_y:.1f}, w={final_w:.1f}, h={final_h:.1f}")
                
                debug_count += 1
                
            except Exception as e:
                print(f"    Error processing cell: {e}")
                continue
                
    except Exception as e:
        print(f"Debug error for {img_path.name}: {e}")

def make_coco_with_braille_classes(root, output_json, padding_ratio=0.15, debug_mode=True):
    root = Path(root)
    images, annotations = [], []
    
    # Create all possible Braille categories
    categories = create_braille_categories()
    print(f"Created {len(categories)} Braille character categories")
    
    img_id = ann_id = 1
    processed_files = skipped_files = 0
    
    # Track unique patterns found in dataset
    patterns_found = set()
    bbox_stats = {"min_w": float('inf'), "max_w": 0, "min_h": float('inf'), "max_h": 0}

    for img_path in root.glob("*.jpg"):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        # Read image size
        img = cv2.imread(str(img_path))
        if img is None: 
            print(f"Bad image {img_path.name} – skipped")
            skipped_files += 1
            continue
        h, w = img.shape[:2]
        print(f"Image {img_path.name}: {w} x {h}")

        # Parse Braille annotation format
        try:
            with open(txt_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print(f"Empty annotation file {txt_path.name} – skipped")
                    skipped_files += 1
                    continue
                    
                lines = content.splitlines()
                
            if len(lines) < 3:
                print(f"Insufficient lines in {txt_path.name} – skipped")
                skipped_files += 1
                continue
                
            angle = float(lines[0].strip())
            verts = list(map(int, lines[1].split()))
            horzs = list(map(int, lines[2].split()))
            cells = lines[3:] if len(lines) > 3 else []
            
            if not verts or not horzs:
                print(f"Invalid grid data in {txt_path.name} – skipped")
                skipped_files += 1
                continue
            
            # Debug first image if in debug mode
            if debug_mode and processed_files == 0:
                debug_bbox_calculation(img_path, txt_path)
                
            print(f"Processing {img_path.name}: {len(cells)} cells, angle={angle}")
            print(f"  Grid: {len(verts)} vertical lines, {len(horzs)} horizontal lines")
            
        except Exception as e:
            print(f"Error reading {txt_path.name}: {e} – skipped")
            skipped_files += 1
            continue

        # Add image info
        images.append({"id":img_id,"file_name":img_path.name,"width":w,"height":h})
        
        # Process Braille cells with character classification
        cell_count = valid_cells = 0
        for cell_line in cells:
            if not cell_line.strip():
                continue
                
            cell_count += 1
            try:
                parts = cell_line.split()
                if len(parts) < 8:
                    print(f"  Cell {cell_count}: Insufficient parts ({len(parts)})")
                    continue
                    
                r, c = int(parts[0]), int(parts[1])
                dots = [int(x) for x in parts[2:8]]  # Extract the 6 dot values
                
                # Calculate bounding box indices MORE CAREFULLY
                # DSBI format: each cell is defined by grid intersections
                col_idx = (c-1) * 2      # Each cell spans 2 vertical grid lines
                row_idx = (r-1) * 3      # Each cell spans 3 horizontal grid lines
                
                # Check bounds BEFORE accessing
                if col_idx+1 >= len(verts):
                    print(f"  Cell {cell_count}: Column index {col_idx+1} >= {len(verts)} (out of bounds)")
                    continue
                if row_idx+2 >= len(horzs):
                    print(f"  Cell {cell_count}: Row index {row_idx+2} >= {len(horzs)} (out of bounds)")
                    continue
                    
                # Get corner coordinates
                x1, x2 = verts[col_idx], verts[col_idx+1]
                y1, y3 = horzs[row_idx], horzs[row_idx+2]
                
                # Ensure positive dimensions
                if x2 <= x1 or y3 <= y1:
                    print(f"  Cell {cell_count}: Invalid box dimensions ({x1},{y1},{x2},{y3})")
                    continue

                # Calculate bbox with padding to ensure character is properly enclosed
                final_x, final_y, final_w, final_h = calculate_bbox_with_padding(
                    x1, y1, x2, y3, padding_ratio
                )
                
                # Clamp to image boundaries
                final_x = max(0, final_x)
                final_y = max(0, final_y)
                final_w = min(final_w, w - final_x)
                final_h = min(final_h, h - final_y)
                
                if final_w <= 0 or final_h <= 0:
                    print(f"  Cell {cell_count}: Final box has non-positive dimensions")
                    continue

                # Update bbox statistics
                bbox_stats["min_w"] = min(bbox_stats["min_w"], final_w)
                bbox_stats["max_w"] = max(bbox_stats["max_w"], final_w)
                bbox_stats["min_h"] = min(bbox_stats["min_h"], final_h)
                bbox_stats["max_h"] = max(bbox_stats["max_h"], final_h)

                # Get category ID based on dot pattern
                category_id = get_category_id_from_dots(dots)
                char_name = dots_to_braille_char(dots)
                patterns_found.add(''.join(map(str, dots)))

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [final_x, final_y, final_w, final_h],
                    "area": final_w * final_h,
                    "iscrowd": 0,
                    "segmentation": [],
                    "braille_character": char_name,
                    "dot_pattern": dots,
                    "original_bbox": [x1, y1, x2-x1, y3-y1],  # For debugging
                    "grid_position": {"row": r, "col": c}
                })
                ann_id += 1
                valid_cells += 1
                
            except Exception as e:
                print(f"  Error processing cell {cell_count} '{cell_line}': {e}")
                continue
        
        print(f"  Added {valid_cells}/{cell_count} valid annotations")
        processed_files += 1
        img_id += 1

    # Save results
    with open(output_json, "w") as f:
        json.dump({"images":images,"annotations":annotations,"categories":categories}, f, indent=2)
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully processed: {processed_files} files")
    print(f"Skipped files: {skipped_files} files")
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Unique Braille patterns found: {len(patterns_found)}")
    print(f"Total possible categories: {len(categories)}")
    print(f"Padding ratio used: {padding_ratio}")
    print(f"Saved to: {output_json}")
    
    # Bounding box statistics
    if annotations:
        print(f"\nBounding Box Statistics:")
        print(f"  Width range: {bbox_stats['min_w']:.1f} - {bbox_stats['max_w']:.1f}")
        print(f"  Height range: {bbox_stats['min_h']:.1f} - {bbox_stats['max_h']:.1f}")
        
        # Calculate average bbox size
        avg_w = sum(ann['bbox'][2] for ann in annotations) / len(annotations)
        avg_h = sum(ann['bbox'][3] for ann in annotations) / len(annotations)
        print(f"  Average size: {avg_w:.1f} x {avg_h:.1f}")
    
    # Show some statistics about patterns found
    print(f"\nMost common patterns in your dataset:")
    from collections import Counter
    pattern_counts = Counter()
    for ann in annotations:
        pattern_counts[''.join(map(str, ann['dot_pattern']))] += 1
    
    for pattern, count in pattern_counts.most_common(10):
        char = dots_to_braille_char([int(x) for x in pattern])
        print(f"  {pattern} ({char}): {count} occurrences")

def visualize_bboxes(root, annotation_file, output_dir="debug_visualizations", max_images=5):
    """
    Create visualizations to debug bounding box placement
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Group annotations by image
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Create category lookup
    id_to_cat = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    count = 0
    for img_info in coco_data['images']:
        if count >= max_images:
            break
            
        img_path = Path(root) / img_info['file_name']
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        img_id = img_info['id']
        anns = img_to_anns.get(img_id, [])
        
        print(f"Visualizing {img_info['file_name']} with {len(anns)} annotations")
        
        # Draw bounding boxes
        for ann in anns:
            x, y, w, h = ann['bbox']
            char_name = ann['braille_character']
            
            # Draw original bbox in red
            if 'original_bbox' in ann:
                ox, oy, ow, oh = ann['original_bbox']
                cv2.rectangle(img, (int(ox), int(oy)), (int(ox+ow), int(oy+oh)), (0, 0, 255), 1)
            
            # Draw final bbox in green
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(img, char_name, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"debug_{img_info['file_name']}")
        cv2.imwrite(output_path, img)
        count += 1

# Usage with improved settings
if __name__ == "__main__":
    root_path = r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\braille.coco\valid"
    output_file = "_annotations.coco.json"
    
    # Convert with larger padding to ensure characters are properly enclosed
    make_coco_with_braille_classes(
        root=root_path,
        output_json=output_file,
        padding_ratio=0.3,  # 20% padding - more generous
        debug_mode=True
    )
    
    # Create debug visualizations
    print("\nCreating debug visualizations...")
    visualize_bboxes(root_path, output_file, max_images=3)
    print("Check the 'debug_visualizations' folder to see bbox placement!")