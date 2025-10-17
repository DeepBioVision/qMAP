import os
import argparse
import glob
import numpy as np
import cv2
import json
from pathlib import Path
import pandas as pd

def load_ground_truth_mask(mask_path):
    """Load ground truth mask from TIF file"""
    try:
        # Try PIL first as it handles TIF files better
        from PIL import Image
        img = Image.open(mask_path)
        mask = np.array(img)
        
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel if RGB
        
        # Ensure it's the right data type
        mask = mask.astype(np.uint32)
        
        return mask
    except Exception as e:
        # Fallback to OpenCV
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            return mask.astype(np.uint32)
        except Exception as e2:
            raise ValueError(f"Error loading mask {mask_path}: {e2}")

def create_prediction_mask_from_json(json_path, image_shape):
    """Create prediction mask from HoverNet JSON output"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros(image_shape, dtype=np.uint32)
    
    if 'nuc' in data:
        for inst_id, inst_info in data['nuc'].items():
            inst_id = int(inst_id)
            if 'contour' in inst_info:
                contour = np.array(inst_info['contour'], dtype=np.int32)
                try:
                    # Fix OpenCV compatibility issue
                    contour_fixed = contour.astype(np.int32)
                    cv2.fillPoly(mask, [contour_fixed], inst_id)
                except Exception as e:
                    # Fallback: use PIL to create mask
                    try:
                        from PIL import Image, ImageDraw
                        pil_mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
                        draw = ImageDraw.Draw(pil_mask)
                        # Convert contour to PIL format (list of tuples)
                        contour_pil = [tuple(point) for point in contour]
                        draw.polygon(contour_pil, fill=inst_id)
                        # Convert back to numpy and add to mask
                        pil_array = np.array(pil_mask)
                        mask[pil_array > 0] = inst_id
                    except Exception as e2:
                        print(f"Warning: Could not create mask for nucleus {inst_id}: {e2}")
                        continue
    
    return mask

def compute_dice_score(true_mask, pred_mask):
    """Compute Dice score between ground truth and prediction masks"""
    # Convert to binary masks (any non-zero pixel is foreground)
    true_binary = (true_mask > 0).astype(np.uint8)
    pred_binary = (pred_mask > 0).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(true_binary, pred_binary).sum()
    union = true_binary.sum() + pred_binary.sum()
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / union
    return dice

def compute_instance_dice_score(true_mask, pred_mask):
    """Compute instance-level Dice score using HoverNet's method"""
    try:
        from metrics.stats_utils import get_dice_1, remap_label
        
        # Ensure contiguous labeling
        true_remapped = remap_label(true_mask, by_size=False)
        pred_remapped = remap_label(pred_mask, by_size=False)
        
        # Compute Dice score
        dice = get_dice_1(true_remapped, pred_remapped)
        return dice
    except Exception as e:
        print(f"Warning: Instance Dice computation failed: {e}")
        return np.nan

def evaluate_dataset(dataset_name, pred_dir, gt_dir):
    """Evaluate a dataset and return results"""
    print(f"\n=== Evaluating {dataset_name} ===")
    
    # Find all prediction JSON files
    json_files = glob.glob(os.path.join(pred_dir, "json", "*.json"))
    json_files.sort()
    
    results = []
    
    for json_file in json_files:
        basename = os.path.basename(json_file).replace('.json', '')
        
        # Find corresponding ground truth mask
        gt_file = os.path.join(gt_dir, f"{basename}.tif")
        
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth not found for {basename}")
            continue
        
        try:
            # Load ground truth
            gt_mask = load_ground_truth_mask(gt_file)
            
            # Create prediction mask from JSON
            pred_mask = create_prediction_mask_from_json(json_file, gt_mask.shape)
            
            # Debug information
            gt_unique = np.unique(gt_mask)
            pred_unique = np.unique(pred_mask)
            gt_nuclei_count = len(gt_unique) - 1 if 0 in gt_unique else len(gt_unique)
            pred_nuclei_count = len(pred_unique) - 1 if 0 in pred_unique else len(pred_unique)
            
            # Compute Dice scores
            binary_dice = compute_dice_score(gt_mask, pred_mask)
            instance_dice = compute_instance_dice_score(gt_mask, pred_mask)
            
            results.append({
                'image': basename,
                'binary_dice': binary_dice,
                'instance_dice': instance_dice,
                'gt_nuclei': gt_nuclei_count,
                'pred_nuclei': pred_nuclei_count
            })
            
            print(f"{basename}: Binary Dice={binary_dice:.4f}, Instance Dice={instance_dice:.4f}, "
                  f"GT nuclei={gt_nuclei_count}, Pred nuclei={pred_nuclei_count}")
            
            # Debug: Print mask info for first few images
            if len(results) <= 3:
                print(f"  Debug - GT unique values: {gt_unique[:10]}{'...' if len(gt_unique) > 10 else ''}")
                print(f"  Debug - Pred unique values: {pred_unique[:10]}{'...' if len(pred_unique) > 10 else ''}")
                print(f"  Debug - GT shape: {gt_mask.shape}, Pred shape: {pred_mask.shape}")
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            continue
    
    return results

def main(pred_suffix: str = ""):
    print("Computing Dice scores for HoverNet predictions...")
    print("DEBUG: Starting main function")

    # Allow CLI to override env var
    if not pred_suffix:
        pred_suffix = os.getenv("PRED_SUFFIX", "").strip()
    suffix_part = f"_{pred_suffix}" if pred_suffix else ""
    
    print(f"Using prediction suffix: '{pred_suffix}' -> suffix_part: '{suffix_part}'")

    all_results = []
    
    # Evaluate CryoNuSeg dataset
    cryonuseg_pred = f"testset/CryoNuSeg/tissue images/hovernet_out{suffix_part}"
    cryonuseg_gt = "testset/CryoNuSeg/Annotator 1 (biologist)/label masks"
    
    print(f"CryoNuSeg pred path: {cryonuseg_pred}")
    print(f"CryoNuSeg pred exists: {os.path.exists(cryonuseg_pred)}")
    
    if os.path.exists(cryonuseg_pred) and os.path.exists(cryonuseg_gt):
        cryonuseg_results = evaluate_dataset("CryoNuSeg", cryonuseg_pred, cryonuseg_gt)
        all_results.extend([(r, "CryoNuSeg") for r in cryonuseg_results])
    
    # Evaluate MoNuSegTestData dataset
    monuseg_pred = f"testset/MoNuSegTestData/hovernet_out{suffix_part}"
    monuseg_gt = "testset/MoNuSegTestData"  # Ground truth might be in XML format
    
    if os.path.exists(monuseg_pred):
        print(f"\n=== MoNuSegTestData ===")
        print("Note: MoNuSegTestData uses XML annotations, not TIF masks.")
        print("Skipping Dice computation for this dataset.")
    
    # Evaluate NuInsSeg dataset (sample a few tissue types)
    nuinsseg_base = "testset/NuInsSeg"
    if os.path.exists(nuinsseg_base):
        print(f"\n=== NuInsSeg (Sample) ===")
        print("Evaluating sample tissue types from NuInsSeg...")
        
        # Sample a few tissue types for evaluation
        sample_tissues = ["human bladder", "human brain", "human liver"]
        
        for tissue in sample_tissues:
            tissue_pred = os.path.join(nuinsseg_base, tissue, "tissue images", f"hovernet_out{suffix_part}")
            tissue_gt = os.path.join(nuinsseg_base, tissue, "label masks")
            
            if os.path.exists(tissue_pred) and os.path.exists(tissue_gt):
                tissue_results = evaluate_dataset(f"NuInsSeg-{tissue}", tissue_pred, tissue_gt)
                all_results.extend([(r, f"NuInsSeg-{tissue}") for r in tissue_results])
    
    # Compute overall statistics
    if all_results:
        print(f"\n{'='*60}")
        print("OVERALL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        # Separate by dataset
        datasets = {}
        for result, dataset in all_results:
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(result)
        
        # Print results for each dataset
        for dataset, results in datasets.items():
            binary_dices = [r['binary_dice'] for r in results]
            instance_dices = [r['instance_dice'] for r in results]
            
            print(f"\n{dataset}:")
            print(f"  Images: {len(results)}")
            print(f"  Binary Dice - Mean: {np.mean(binary_dices):.4f} ± {np.std(binary_dices):.4f}")
            print(f"  Binary Dice - Min: {np.min(binary_dices):.4f}, Max: {np.max(binary_dices):.4f}")
            print(f"  Instance Dice - Mean: {np.mean(instance_dices):.4f} ± {np.std(instance_dices):.4f}")
            print(f"  Instance Dice - Min: {np.min(instance_dices):.4f}, Max: {np.max(instance_dices):.4f}")
        
        # Overall statistics
        all_binary_dices = [r['binary_dice'] for r, _ in all_results]
        all_instance_dices = [r['instance_dice'] for r, _ in all_results]
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total Images: {len(all_results)}")
        print(f"  Overall Binary Dice - Mean: {np.mean(all_binary_dices):.4f} ± {np.std(all_binary_dices):.4f}")
        print(f"  Overall Instance Dice - Mean: {np.mean(all_instance_dices):.4f} ± {np.std(all_instance_dices):.4f}")
        
        # Save detailed results to CSV
        detailed_results = []
        for result, dataset in all_results:
            detailed_results.append({
                'dataset': dataset,
                'image': result['image'],
                'binary_dice': result['binary_dice'],
                'instance_dice': result['instance_dice'],
                'gt_nuclei': result['gt_nuclei'],
                'pred_nuclei': result['pred_nuclei']
            })
        
        df = pd.DataFrame(detailed_results)
        df.to_csv('dice_scores_results.csv', index=False)
        print(f"\nDetailed results saved to: dice_scores_results.csv")
        
    else:
        print("No results found. Please check that prediction and ground truth directories exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_suffix", type=str, default="", help="Suffix of hovernet_out folders to evaluate (e.g., 'skin, original, cryonuseg, monuseg, nuinsseg')")
    args = parser.parse_args()
    main(args.pred_suffix)
