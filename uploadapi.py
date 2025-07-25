import json, cv2
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

def make_coco_with_braille_classes(root, output_json):
    root = Path(root)
    images, annotations = [], []
    
    # Create all possible Braille categories
    categories = create_braille_categories()
    print(f"Created {len(categories)} Braille character categories")
    
    img_id = ann_id = 1
    processed_files = skipped_files = 0
    
    # Track unique patterns found in dataset
    patterns_found = set()

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
                
            print(f"Processing {img_path.name}: {len(cells)} cells, angle={angle}")
            
        except Exception as e:
            print(f"Error reading {txt_path.name}: {e} – skipped")
            skipped_files += 1
            continue

        # Add image info
        images.append({"id":img_id,"file_name":img_path.name,"width":w,"height":h})
        
        # Process Braille cells with character classification
        cell_count = 0
        for cell_line in cells:
            if not cell_line.strip():
                continue
                
            try:
                parts = cell_line.split()
                if len(parts) < 8:
                    continue
                    
                r, c = int(parts[0]), int(parts[1])
                dots = [int(x) for x in parts[2:8]]  # Extract the 6 dot values
                
                # Calculate bounding box
                col_idx = (c-1)*2
                row_idx = (r-1)*3
                
                if (col_idx+1 >= len(verts) or row_idx+2 >= len(horzs)):
                    continue
                    
                x1, x2 = verts[col_idx], verts[col_idx+1]
                y1, y3 = horzs[row_idx], horzs[row_idx+2]
                bbox_w, bbox_h = x2-x1, y3-y1
                
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                # Get category ID based on dot pattern
                category_id = get_category_id_from_dots(dots)
                char_name = dots_to_braille_char(dots)
                patterns_found.add(''.join(map(str, dots)))

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "iscrowd": 0,
                    "segmentation": [],
                    "braille_character": char_name,
                    "dot_pattern": dots
                })
                ann_id += 1
                cell_count += 1
                
            except Exception as e:
                print(f"  Error processing cell '{cell_line}': {e}")
                continue
        
        print(f"  Added {cell_count} valid annotations")
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
    print(f"Saved to: {output_json}")
    
    # Show some statistics about patterns found
    print(f"\nMost common patterns in your dataset:")
    from collections import Counter
    pattern_counts = Counter()
    for ann in annotations:
        pattern_counts[''.join(map(str, ann['dot_pattern']))] += 1
    
    for pattern, count in pattern_counts.most_common(10):
        char = dots_to_braille_char([int(x) for x in pattern])
        print(f"  {pattern} ({char}): {count} occurrences")

# Usage
make_coco_with_braille_classes(
    root=r"C:\Users\ASUS\OneDrive\Documents\fcos_scratch\DSBI-master\train",
    output_json="_annotations_with_classes.coco.json"
)
