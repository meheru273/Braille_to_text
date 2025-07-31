"""
Enhanced Braille Dataset Cleaner - Remove Orphaned Classes & Annotations
========================================================================
This script removes unwanted orphaned classes (like 'braille-characters-qfd4-91Zj')
and keeps only the 26 letter classes (a-z) with proper sequential ID mapping.

Key improvements:
- Better error handling
- More robust class detection
- Proper ID remapping
- Verification of results
- Handles both 'valid' and 'val' folder naming
- FIXED: Image validation and cleanup
"""

import torch
import subprocess
import sys
import os
import pathlib
import json
import shutil
import stat
import time
from collections import Counter, defaultdict
from datetime import datetime

EXPECTED_CLASSES = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
    'pattern_000001', 'pattern_000010', 'pattern_000011', 'pattern_000100', 
    'pattern_000101', 'pattern_000110', 'pattern_000111', 'pattern_001000', 
    'pattern_001001', 'pattern_001010', 'pattern_001011', 'pattern_001100', 
    'pattern_001101', 'pattern_001110', 'pattern_001111', 'pattern_010000', 
    'pattern_010001', 'pattern_010010', 'pattern_010011', 'pattern_010101', 
    'pattern_011000', 'pattern_011001', 'pattern_011010', 'pattern_011011', 
    'pattern_011101', 'pattern_011111', 'pattern_100001', 'pattern_100011', 
    'pattern_100101', 'pattern_100111', 'pattern_110001', 'pattern_110011', 
    'pattern_110101', 'pattern_110111', 'pattern_111011', 'pattern_111101', 
    'pattern_111111', 'space'
]

def install_package(package):
    """Install package with error handling"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f" Installed {package}")
    except subprocess.CalledProcessError as e:
        print(f" Failed to install {package}: {e}")
        raise

def setup_environment():
    """Install required packages"""
    print(" Setting up environment...")
    required_packages = ["roboflow", "pycocotools", "requests"]
    
    for pkg in required_packages:
        try:
            __import__(pkg.replace('-', '_'))
            print(f" {pkg} already installed")
        except ImportError:
            print(f" Installing {pkg}...")
            install_package(pkg)

def download_original_dataset():
    """Download the original dataset"""
    print(" Downloading original dataset...")
    
    try:
        

        from roboflow import Roboflow
        rf = Roboflow(api_key="NRmMU6uU07XILRg52e7n")
        project = rf.workspace("braille-image").project("braille-to-text-custom-kvzne")
        version = project.version(1)
        dataset = version.download("coco")
                
                        
        print(f" Dataset downloaded successfully")
        return dataset
    except Exception as e:
        print(f" Failed to download dataset: {e}")
        raise

def analyze_current_dataset(dataset_path):
    """Analyze current dataset to understand the class structure"""
    print(f"\n Analyzing current dataset: {dataset_path}")
    
    dataset_path = pathlib.Path(dataset_path)
    analysis = {}
    
    # Check for different split naming conventions
    possible_splits = ['train', 'valid', 'val', 'test']
    found_splits = [split for split in possible_splits if (dataset_path / split).exists()]
    
    print(f" Found splits: {found_splits}")
    
    for split in found_splits:
        split_path = dataset_path / split
        annotation_file = split_path / '_annotations.coco.json'
        
        if not annotation_file.exists():
            print(f" No annotation file found in {split}")
            continue
            
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Analyze classes
            valid_classes = []
            invalid_classes = []
            
            for cat in data['categories']:
                if cat['name'] in EXPECTED_CLASSES:
                    valid_classes.append(cat)
                else:
                    invalid_classes.append(cat)
            
            # Count annotations per category
            ann_counts = defaultdict(int)
            orphaned_anns = 0
            
            for ann in data['annotations']:
                cat_id = ann['category_id']
                # Find category name
                cat_name = None
                for cat in data['categories']:
                    if cat['id'] == cat_id:
                        cat_name = cat['name']
                        break
                
                if cat_name:
                    ann_counts[cat_name] += 1
                else:
                    orphaned_anns += 1
            
            analysis[split] = {
                'total_categories': len(data['categories']),
                'valid_classes': valid_classes,
                'invalid_classes': invalid_classes,
                'total_annotations': len(data['annotations']),
                'annotation_counts': dict(ann_counts),
                'orphaned_annotations': orphaned_anns,
                'total_images': len(data['images'])
            }
            
            print(f"\n {split.upper()} ANALYSIS:")
            print(f"   Total categories: {len(data['categories'])}")
            print(f"   Valid classes (a-z): {len(valid_classes)}")
            print(f"   Invalid classes: {len(invalid_classes)}")
            if invalid_classes:
                print(f"     Invalid class names: {[c['name'] for c in invalid_classes]}")
            print(f"   Total annotations: {len(data['annotations'])}")
            print(f"   Orphaned annotations: {orphaned_anns}")
            print(f"   Total images: {len(data['images'])}")
            
        except Exception as e:
            print(f" Error analyzing {split}: {e}")
            continue
    
    return analysis

def clean_dataset_annotations(dataset_path):
    """
    Clean dataset by removing invalid classes and orphaned annotations
    """
    print(f"\n Cleaning dataset annotations...")
    
    dataset_path = pathlib.Path(dataset_path)
    stats = {
        'removed_classes': 0,
        'removed_annotations': 0,
        'final_classes': 0,
        'splits_processed': 0
    }
    
    # Find all splits
    possible_splits = ['train', 'valid', 'val', 'test']
    found_splits = [split for split in possible_splits if (dataset_path / split).exists()]
    
    for split in found_splits:
        split_path = dataset_path / split
        annotation_file = split_path / '_annotations.coco.json'
        
        if not annotation_file.exists():
            print(f"  No annotation file in {split}, skipping")
            continue
            
        print(f"\n Processing {split} split...")
        
        try:
            # Load annotations
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            original_categories = len(data['categories'])
            original_annotations = len(data['annotations'])
            
            # Identify valid and invalid classes
            valid_categories = []
            invalid_category_ids = set()
            
            for cat in data['categories']:
                if cat['name'] in EXPECTED_CLASSES:
                    valid_categories.append(cat)
                else:
                    invalid_category_ids.add(cat['id'])
                    print(f"       Will remove class: '{cat['name']}' (ID: {cat['id']})")
            
            # Create new sequential ID mapping for valid classes
            # Sort by name to ensure consistent a=0, b=1, etc. (0-based indexing for model)
            valid_categories.sort(key=lambda x: x['name'])
            old_to_new_id = {}
            new_categories = []
            
            for new_id, cat in enumerate(valid_categories):
                old_to_new_id[cat['id']] = new_id
                new_categories.append({
                    'id': new_id,
                    'name': cat['name'],
                    'supercategory': cat.get('supercategory', 'letter')
                })
                print(f"     Mapping: '{cat['name']}' {cat['id']} -> {new_id}")
            
            # Filter and update annotations
            valid_annotations = []
            removed_annotations = 0
            
            for ann in data['annotations']:
                if ann['category_id'] in old_to_new_id:
                    # Update category ID to new sequential ID
                    ann['category_id'] = old_to_new_id[ann['category_id']]
                    valid_annotations.append(ann)
                else:
                    removed_annotations += 1
            
            # Update data structure
            data['categories'] = new_categories
            data['annotations'] = valid_annotations
            
            # Save cleaned annotations
            with open(annotation_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update stats
            stats['removed_classes'] += (original_categories - len(new_categories))
            stats['removed_annotations'] += removed_annotations
            stats['final_classes'] = len(new_categories)
            stats['splits_processed'] += 1
            
            print(f"     Removed {original_categories - len(new_categories)} invalid classes")
            print(f"     Removed {removed_annotations} orphaned annotations")
            print(f"     Final classes: {len(new_categories)}")
            print(f"     Final annotations: {len(valid_annotations)}")
            
        except Exception as e:
            print(f" Error processing {split}: {e}")
            continue
    
    return stats

def validate_and_clean_images(dataset_path):
    """
    Validate that all referenced images exist and remove orphaned image entries
    """
    print(f"\n Validating and cleaning image references...")
    
    dataset_path = pathlib.Path(dataset_path)
    stats = {
        'missing_images': 0,
        'removed_image_entries': 0,
        'removed_annotations': 0,
        'splits_processed': 0
    }
    
    # Find all splits
    possible_splits = ['train', 'valid', 'val', 'test']
    found_splits = [split for split in possible_splits if (dataset_path / split).exists()]
    
    for split in found_splits:
        split_path = dataset_path / split
        annotation_file = split_path / '_annotations.coco.json'
        
        if not annotation_file.exists():
            print(f"  No annotation file in {split}, skipping")
            continue
            
        print(f"\n Processing {split} split...")
        
        try:
            # Load annotations
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            original_images = len(data['images'])
            original_annotations = len(data['annotations'])
            
            # Check which images actually exist
            valid_images = []
            valid_image_ids = set()
            missing_images = []
            
            for img in data['images']:
                image_path = split_path / img['file_name']
                if image_path.exists():
                    valid_images.append(img)
                    valid_image_ids.add(img['id'])
                else:
                    missing_images.append(img['file_name'])
                    print(f"    ❌ Missing image: {img['file_name']}")
            
            # Remove annotations for missing images
            valid_annotations = []
            removed_annotations = 0
            
            for ann in data['annotations']:
                if ann['image_id'] in valid_image_ids:
                    valid_annotations.append(ann)
                else:
                    removed_annotations += 1
            
            # Update data structure
            data['images'] = valid_images
            data['annotations'] = valid_annotations
            
            # Save cleaned annotations
            with open(annotation_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update stats
            stats['missing_images'] += len(missing_images)
            stats['removed_image_entries'] += (original_images - len(valid_images))
            stats['removed_annotations'] += removed_annotations
            stats['splits_processed'] += 1
            
            print(f"     Results:")
            print(f"      Original images: {original_images}")
            print(f"      Valid images: {len(valid_images)}")
            print(f"      Missing images: {len(missing_images)}")
            print(f"      Removed annotations: {removed_annotations}")
            print(f"      Final annotations: {len(valid_annotations)}")
            
            if missing_images:
                print(f"      Missing image files:")
                for img in missing_images[:5]:  # Show first 5
                    print(f"        - {img}")
                if len(missing_images) > 5:
                    print(f"        ... and {len(missing_images) - 5} more")
            
        except Exception as e:
            print(f"❌ Error processing {split}: {e}")
            continue
    
    return stats

def enhanced_clean_dataset_annotations(dataset_path):
    """
    Enhanced cleaning that handles both classes AND missing images
    """
    print(f"\n Enhanced dataset cleaning...")
    
    # First, clean the class structure (your existing code)
    class_stats = clean_dataset_annotations(dataset_path)
    
    # Then, validate and clean image references (NEW)
    image_stats = validate_and_clean_images(dataset_path)
    
    # Combine stats
    combined_stats = {
        **class_stats,
        'missing_images': image_stats['missing_images'],
        'removed_image_entries': image_stats['removed_image_entries'],
        'removed_annotations_for_images': image_stats['removed_annotations']
    }
    
    return combined_stats

def verify_cleaned_dataset(dataset_path):
    """Verify the cleaned dataset has exactly 26 classes (a-z) with proper mapping"""
    print(f"\n Verifying cleaned dataset...")
    
    dataset_path = pathlib.Path(dataset_path)
    verification_results = {}
    expected_classes = set(EXPECTED_CLASSES)
    
    # Find splits
    possible_splits = ['train', 'valid', 'val', 'test']
    found_splits = [split for split in possible_splits if (dataset_path / split).exists()]
    
    all_good = True
    
    for split in found_splits:
        split_path = dataset_path / split
        annotation_file = split_path / '_annotations.coco.json'
        
        if not annotation_file.exists():
            continue
            
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Get class names and IDs
            found_classes = {cat['name']: cat['id'] for cat in data['categories']}
            found_class_names = set(found_classes.keys())
            
            # Check if we have exactly a-z
            missing_classes = expected_classes - found_class_names
            extra_classes = found_class_names - expected_classes
            
            # Check ID mapping (should be sequential 0-25 for 0-based indexing)
            expected_mapping = {chr(ord('a') + i): i for i in range(26)}
            mapping_correct = found_classes == expected_mapping
            
            # Count annotations per class
            class_counts = Counter()
            for ann in data['annotations']:
                cat_name = next((cat['name'] for cat in data['categories'] 
                               if cat['id'] == ann['category_id']), 'UNKNOWN')
                class_counts[cat_name] += 1
            
            verification_results[split] = {
                'classes': len(data['categories']),
                'annotations': len(data['annotations']),
                'images': len(data['images']),
                'class_distribution': dict(class_counts),
                'found_classes': found_class_names,
                'missing_classes': missing_classes,
                'extra_classes': extra_classes,
                'mapping_correct': mapping_correct,
                'class_mapping': found_classes
            }
            
            print(f"\n {split.upper()} VERIFICATION:")
            print(f"   Classes: {len(data['categories'])}/26 {'' if len(data['categories']) == 26 else '❌'}")
            
            if missing_classes:
                print(f"    Missing letters: {sorted(missing_classes)}")
                all_good = False
            if extra_classes:
                print(f"    Extra classes: {sorted(extra_classes)}")
                all_good = False
            if not missing_classes and not extra_classes:
                print(f"   All 26 letters (a-z) present ")
            
            print(f"   ID mapping correct: {'' if mapping_correct else '❌'}")
            if not mapping_correct:
                print(f"   Expected: a=0, b=1, ..., z=25")
                print(f"   Found: {dict(sorted(found_classes.items()))}")
                all_good = False
            
            print(f"   Annotations: {len(data['annotations'])}")
            print(f"   Images: {len(data['images'])}")
            
            # Show class distribution summary
            if class_counts:
                min_count = min(class_counts.values())
                max_count = max(class_counts.values())
                avg_count = sum(class_counts.values()) / len(class_counts)
                print(f"   Annotation distribution: min={min_count}, max={max_count}, avg={avg_count:.1f}")
            
        except Exception as e:
            print(f" Error verifying {split}: {e}")
            all_good = False
            continue
    
    return verification_results, all_good

def main():
    """Main function with enhanced cleaning"""
    print(" ENHANCED BRAILLE DATASET CLEANER")
    print("=" * 60)
    
    try:
        # Step 0: Setup environment
        setup_environment()
        
        # Step 1: Download original dataset
        print("\n STEP 1: Downloading original dataset...")
        dataset = download_original_dataset()
        dataset_path = pathlib.Path(dataset.location)
        print(f"Downloaded to: {dataset_path}")
        
        # Step 2: Analyze current dataset
        print("\n STEP 2: Analyzing current dataset...")
        analysis = analyze_current_dataset(dataset_path)
        
        # Step 3: Enhanced cleaning (classes + images)
        print("\n STEP 3: Enhanced dataset cleaning...")
        stats = enhanced_clean_dataset_annotations(dataset_path)
        
        # Step 4: Verify cleaned dataset
        print("\n STEP 4: Verifying cleaned dataset...")
        verification_results, all_good = verify_cleaned_dataset(dataset_path)
        
        # Final summary
        print("\n" + "=" * 60)
        if all_good and stats['final_classes'] == 26:
            print(" CLEANING COMPLETED SUCCESSFULLY!")
        else:
            print("  CLEANING COMPLETED WITH WARNINGS!")
        print("=" * 60)
        
        print(f" Cleaned dataset location: {dataset_path}")
        print(f" Summary:")
        print(f"   Splits processed: {stats['splits_processed']}")
        print(f"   Classes removed: {stats['removed_classes']}")
        print(f"   Annotations removed (invalid classes): {stats['removed_annotations']}")
        print(f"   Missing images found: {stats.get('missing_images', 0)}")
        print(f"   Image entries removed: {stats.get('removed_image_entries', 0)}")
        print(f"   Annotations removed (missing images): {stats.get('removed_annotations_for_images', 0)}")
        print(f"   Final classes: {stats['final_classes']}")
        
        if stats['final_classes'] == 26 and stats.get('missing_images', 0) == 0:
            print(" SUCCESS: Dataset is now clean and ready for training!")
            print("   - Exactly 26 classes (a-z) with sequential IDs (0-25)")
            print("   - All referenced images exist")
        else:
            if stats['final_classes'] != 26:
                print(f" WARNING: Expected 26 classes, got {stats['final_classes']}")
            if stats.get('missing_images', 0) > 0:
                print(f" WARNING: Found {stats['missing_images']} missing image files")
        
        print(f"\n Your cleaned dataset is ready for training!")
        print(f" Dataset path: {dataset_path}")
        
        return True
        
    except Exception as e:
        print(f" Error during cleaning process: {e}")
        import traceback
        traceback.print_exc()
        return False

# THIS IS THE MISSING PIECE - ACTUALLY RUN THE SCRIPT!
if __name__ == "__main__":
    print(" Starting Enhanced Braille Dataset Cleaner...")
    success = main()
    if success:
        print("\n All done! Your dataset is clean and ready to use.")
    else:
        print("\n Process failed. Check the error messages above.")
    
    # Keep the window open on Windows
    input("\nPress Enter to exit...")