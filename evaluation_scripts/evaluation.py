import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import pdb
import os
import re
import numpy as np

from evaluation_scripts.vis_mask import visualize_image_and_masks_v1_enhanced

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'  # 匹配最简单的JSON对象
    json_match = re.search(json_pattern, output_text)
    # pdb.set_trace()
    if json_match:
        data = json.loads(json_match.group(0))
        # 查找bbox键
        bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
        # pdb.set_trace()
        if bbox_key and len(data[bbox_key]) == 4:
            content_bbox = data[bbox_key]
            content_bbox = [round(int(content_bbox[0])*x_factor), round(int(content_bbox[1])*y_factor), round(int(content_bbox[2])*x_factor), round(int(content_bbox[3])*y_factor)]
        # 查找points键
        points_keys = [key for key in data.keys() if 'points' in key.lower()][:2]  # 获取前两个points键
        if len(points_keys) == 2:
            point1 = data[points_keys[0]]
            point2 = data[points_keys[1]]
            point1 = [round(int(point1[0])*x_factor), round(int(point1[1])*y_factor)]
            point2 = [round(int(point2[0])*x_factor), round(int(point2[1])*y_factor)]
            points = [point1, point2]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = ""
    if think_match:
        think_text = think_match.group(1)
    
    return content_bbox, points, think_text

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection, union

def main():
    args = parse_args()
    test_data = args.test_data_path.replace("Ricky06662/", "")
    mask_output_path = args.output_path + '/infer_mask/' + test_data
    os.makedirs(mask_output_path, exist_ok=True)
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    resize_size = 840
    # dataset = load_from_disk(args.test_data_path)['test']
    dataset = load_dataset(args.test_data_path, split='test')
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    # pdb.set_trace()
    dataset = dataset.select(range(start_idx, end_idx))
    
    # # base
    # QUESTION_TEMPLATE = \
    #     "Please find '{Question}' with bbox and points." \
    #     "Compare the difference between objects and find the most closely matched one." \
    #     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
    #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
    #     "i.e., <think> thinking process here </think>" \
    #     "<answer>{Answer}</answer>"\
    
    print("args.output_path:", args.output_path)
    
    # Prefix = "<d_think>...</d_think>"
    Prefix = ""

    QUESTION_TEMPLATE = \
        "Please find '{Question}' with bbox and points." \
        "Compare the difference between objects and find the most closely matched one." \
        "Output the thinking process in <think> </think>, final answer in <answer> </answer> and a caption for the bbox content in <caption> </caption> tags." \
        "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>" \
        "<caption>a red apple</caption>"

    # QUESTION_TEMPLATE = \
    #     "Please find '{Question}' with bbox and points." \
    #     "Compare the difference between objects and find the most closely matched one." \
    #     "Output the thinking process in <think> </think> and final answer in <answer> </answer>." \
    #     "Output captions for the bbox and the whole picture respectively in <caption> <caption> tags." \
    #     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format." \
    #     "i.e., <think> thinking process here </think>" \
    #     "<answer>{Answer}</answer>" \
    #     "<caption>{Caption}</caption>"
    
    messages = []
    id_list = []
    for item in dataset:
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": item["image"].resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=item["text"].lower().strip("."), 
                                                      Answer='{"bbox": [10,100,200,210], "points_1": [30,110], "points_2": [35,180]}',
                                                      Caption='{"local": "a red apple", "global": "There are many apples on the tree"}')
                }
            ]
        }]
        messages.append(message)
        id_list.append({
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "image": item["image"],
            "mask": item["mask"],
            "img_height": item["img_height"],
            "img_width": item["img_width"]
        })

    all_outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
        batch_id_list = id_list[i:i + args.batch_size]
        
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + Prefix for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # pdb.set_trace()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                query = batch_messages[id_idx][0]['content'][1]['text']
                output_txt = batch_output_text[id_idx]
                try:
                    print(f"-----OUTPUT:", output_txt)
                    # _output_txt = output_txt.split("judge>")[0] + output_txt.split("judge>")[2]
                    bbox, points, think = extract_bbox_points_think(
                                            # _output_txt, 
                                            output_txt, 
                                            batch_id_list[id_idx]["img_width"]/resize_size, 
                                            batch_id_list[id_idx]["img_height"]/resize_size
                                        )
                    segmentation_model.set_image(batch_id_list[id_idx]["image"])
                    masks, scores, _ = segmentation_model.predict(
                        point_coords=points,
                        point_labels=[1,1],
                        box=bbox
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                    mask = masks[0].astype(bool)
                    gt_mask = np.array(batch_id_list[id_idx]["mask"])
                    # pdb.set_trace()
                    try:
                        intersection, union = compute_iou(mask, gt_mask)
                    except Exception as e:
                        # skip this because the image or mask is not correct
                        continue
                    
                    iou = intersection / union if union > 0 else 0
                    if iou < 0.5:
                        visualize_image_and_masks_v1_enhanced(
                            batch_id_list[id_idx]["image"],
                            mask,
                            gt_mask,
                            output_path=f"{mask_output_path}/{batch_id_list[id_idx]['image_id']}.png",
                            bbox = bbox,
                            point_coords = np.array(points),
                            point_labels = np.array([1,1]),
                            display_text = query.split("'")[1]
                        )
                except Exception as e:
                    # add penalty in this situation
                    think = f"[Error]... {e}"
                    intersection = 0
                    union = np.array(batch_id_list[id_idx]["mask"]).sum()
                
                all_outputs.append({
                    "image_id": batch_id_list[id_idx]["image_id"],
                    "ann_id": batch_id_list[id_idx]["ann_id"],
                    "query": query,
                    "output_txt": output_txt,
                    "think": think,
                    "intersection": int(intersection),
                    "union": int(union)
                })
        print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
        
        # clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    
    # Modify the output file name, add idx
    json_output_path = os.path.join(args.output_path, f"infer_json/{test_data}")
    os.makedirs(json_output_path, exist_ok=True)
    output_file = os.path.join(json_output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
