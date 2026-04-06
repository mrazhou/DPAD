import os
import json
import glob
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    parser.add_argument("--test_data_path", type=str, default="Ricky06662/ReasonSeg_test", help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir, test_data):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, f"infer_json/{test_data}/output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # for accumulating all data
    total_intersection = 0
    total_union = 0
    all_ious = []
    
    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
            intersection = item['intersection']
            union = item['union']
            
            # calculate IoU of each item
            iou = intersection / union if union > 0 else 0
            all_ious.append({
                'image_id': item['image_id'],
                'iou': iou
            })
            
            # accumulate total intersection and union
            total_intersection += intersection
            total_union += union
    
    # calculate gIoU
    gIoU = np.mean([item['iou'] for item in all_ious])
    # calculate cIoU
    cIoU = total_intersection / total_union if total_union > 0 else 0


    new_data = {test_data: {
            "gIoU": gIoU,
            "cIoU": cIoU
        }}
    existing_data = {}
    # save to eval.json
    res_path = os.path.join(output_dir, "eval.json")
    if os.path.exists(res_path):
        with open(res_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    existing_data.update(new_data)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    # print the results
    print(f"[Test data] --> \t\t{test_data}: gIoU {gIoU:.4f} / cIoU {cIoU:.4f}")
    print(existing_data)
    # print(f"gIoU (average of per image IoU): {gIoU:.4f}")
    # print(f"cIoU (total_intersection / total_union): {cIoU:.4f}")
    

if __name__ == "__main__":
    args = parse_args()
    args.test_data = args.test_data_path.replace("Ricky06662/", "")
    calculate_metrics(args.output_dir, args.test_data)
