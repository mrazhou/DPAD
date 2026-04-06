import torch
import math
from PIL import Image
from typing import Union, List, Tuple
import re
import json

# 为确保代码可运行，我们先定义一个 ImageObject 的类型别名
# 在实际使用中，这通常是 from PIL import Image
ImageObject = Image.Image

# --- 前置依赖：您提供的 process_image 函数 ---
def process_image(image: ImageObject, max_pixels: int, min_pixels: int) -> ImageObject:
    """
    对图像进行预处理，确保其像素总数在指定范围内，并转换为RGB格式。
    """
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        # Pillow 9.1.0 之后，Image.NEAREST 更名为 Image.Resampling.NEAREST
        try:
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)
        except AttributeError: # 兼容旧版 Pillow
            image = image.resize((width, height), resample=Image.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        try:
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)
        except AttributeError:
            image = image.resize((width, height), resample=Image.NEAREST)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

# --- 核心实现 ---

# 提前加载模型和预处理器以提高效率，避免在函数调用时重复加载。
# 这部分代码通常放在脚本的全局范围或一个类中。
try:
    from transformers import CLIPProcessor, CLIPModel
    
    # 使用通用的 OpenAI CLIP 模型
    model_id = "openai/clip-vit-base-patch32"
    # model_id = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    print(f"CLIP model [{model_id}] loaded successfully on device: {device}")

except ImportError:
    print("Please install transformers and torch: pip install transformers torch")
    model, processor, device = None, None, None
except Exception as e:
    print(f"Could not load CLIP model. Ensure you have an internet connection. Error: {e}")
    model, processor, device = None, None, None


def get_clip_similarity_for_bbox(
    image: ImageObject, 
    bbox_str: str | None, 
    predict_str: str
) -> float:
    """
    计算图像中指定边界框（bounding box）区域与文本描述（caption）之间的CLIP相似度。

    Args:
        image (PIL.Image.Image): 输入的PIL图像对象。
        bbox (List[int]): 包含四个整数的列表 [x1, y1, x2, y2]，
                           分别代表边界框左上角的x, y坐标和右下角的x, y坐标。
        caption (str): 用于比较的文本描述。

    Returns:
        float: 返回一个浮点数，代表计算出的CLIP相似度得分（余弦相似度）。
               如果模型未加载或出现错误，返回 0.0。
    """
    if not all([model, processor, device]):
        print("CLIP model or processor not available. Returning 0.0")
        return 0.0
    
    try:
        if bbox_str is not None:
            bbox_str = bbox_str.strip()
            gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
            gt_match = re.search(gt_box_pattern, bbox_str)
            if gt_match:
                gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)),
                        int(gt_match.group(3)), int(gt_match.group(4))]

            # 1. 根据边界框裁剪图像
            cropped_image = image.crop(tuple(gt_bbox))
            ptype = 'global'
        else:
            cropped_image = image
            ptype = 'local'
    except Exception as e:
        # print(f"Error cropping image with bbox {gt_bbox}: {e}")
        return 0.0
    
    try:
        cap_pattern = r'<caption>(.*?)</caption>'
        cap_match = re.search(cap_pattern, predict_str, re.DOTALL)
        if cap_match:
            caption = cap_match.group(1).strip()
            # caption = json.loads(caption)[ptype]
            # caption = json.loads(caption)
        else:
            return 0.0
    except Exception as e:
        # print(f"Error extracting caption from predict_str {predict_str}: {e}")
        return 0.0
    

    # 2. 预处理裁剪后的图像和文本
    # processor 会自动处理图像的大小调整、归一化以及文本的分词
    inputs = processor(
        text=[caption], 
        images=[cropped_image], 
        return_tensors="pt", 
        padding=True
    ).to(device)
    
    # 如果文本token长度大于77，则返回0.0
    if len(inputs.input_ids[0]) > 77:
        return 0.0

    # 3. 使用CLIP模型获取图像和文本的特征嵌入
    with torch.no_grad(): # 无需计算梯度，节省资源
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    # 4. 对特征嵌入进行归一化
    image_embeds_normalized = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds_normalized = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # 5. 计算归一化后的特征之间的余弦相似度（点积）
    # CLIP 的输出 logits 实际上是 similarity * logit_scale，我们这里直接用点积得到纯粹的相似度
    similarity_score = torch.matmul(text_embeds_normalized, image_embeds_normalized.T).squeeze().item()

    return similarity_score


def get_embedding(
    input_data: Union[Image.Image, str]
) -> Union[torch.Tensor, None]:
    """
    函数一：获取单个输入（图片或文本）的CLIP特征嵌入。

    Args:
        input_data: PIL图像对象或文本字符串。

    Returns:
        torch.Tensor: 返回一个一维的特征嵌入张量，如果出错则返回None。
    """
    try:
        if isinstance(input_data, Image.Image):
            inputs = processor(images=[input_data], return_tensors="pt").to(device)
            embedding = model.get_image_features(**inputs)
        elif isinstance(input_data, str):
            token_ids = processor(text=input_data, return_tensors="pt").input_ids
            if token_ids.shape[1] > 77:
                print(f"Warning: Text input's token length ({token_ids.shape[1]}) > 77.")
                return None
            inputs = {"input_ids": token_ids.to(device)}
            embedding = model.get_text_features(**inputs)
        else:
            raise TypeError("Input must be a PIL Image or a string.")
        
        # 返回的 embedding 形状是 [1, embed_dim]，我们将其降维为 [embed_dim]
        return embedding.squeeze(0)

    except Exception as e:
        print(f"An error occurred during embedding generation: {e}")
        return None

def get_embedding_similarity(
    embed1: torch.Tensor,
    embed2: torch.Tensor
) -> float:
    """
    函数二：计算两个特征嵌入之间的余弦相似度。

    Args:
        embed1 (torch.Tensor): 第一个特征嵌入 (一维张量)。
        embed2 (torch.Tensor): 第二个特征嵌入 (一维张量)。

    Returns:
        float: 两个嵌入之间的余弦相似度 (0-1范围)。
    """
    if embed1 is None or embed2 is None:
        return 0.0
        
    try:
        # 归一化
        embed1_normalized = embed1 / embed1.norm(p=2, dim=-1, keepdim=True)
        embed2_normalized = embed2 / embed2.norm(p=2, dim=-1, keepdim=True)
        
        # 点积计算余弦相似度
        return torch.dot(embed1_normalized, embed2_normalized).item()
    
    except Exception as e:
        print(f"An error occurred during similarity calculation: {e}")
        return 0.0

def parser_str(predict_str: str, image):
        answer = r'<answer>(.*?)</answer>'  
        answer_match = re.search(answer, predict_str)
        if answer_match:
            data = json.loads(answer_match.group(1))
            bbox_key = 'bbox'
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = data[bbox_key]

            # 1. 根据边界框裁剪图像
            cropped_image = image.crop(tuple(content_bbox))

        cap_pattern = r'<caption>(.*?)</caption>'
        cap_match = re.search(cap_pattern, predict_str, re.DOTALL)
        caption_str = cap_match.group(1).strip()
        captions = json.loads(caption_str)
        
        return cropped_image, captions

def gaussian_kernel_similarity(v1: float, v2: float, sigma: float = 0.2) -> float:
    """
    方案一：计算高斯核函数（RBF）相似度 (最推荐)。

    奖励值呈钟形曲线分布，当两个数值相等时为1，随着差值增大平滑地向0衰减。
    
    Args:
        v1 (float): 第一个数值。
        v2 (float): 第二个数值。
        sigma (float): 带宽超参数，控制曲线的“胖瘦”，即对差值的敏感度。
                        值越小，函数越“尖锐”，对差值越敏感。
                        默认值0.2是一个比较常用的选择。

    Returns:
        float: 反映两个数值相近程度的奖励值 (范围 0-1).
    """
    # 避免sigma为0导致除零错误
    if sigma == 0:
        return 1.0 if v1 == v2 else 0.0
        
    squared_distance = (v1 - v2) ** 2
    mean_v = (v1 + v2) / 2
    return math.exp(-squared_distance / (2 * sigma ** 2))
    # return mean_v * math.exp(-squared_distance / (2 * sigma ** 2))
    

# --- 示例用法 ---
if __name__ == '__main__':
    # 确保模型已成功加载
    if model and processor:
        # 1. 创建一个示例图像 (例如，一个300x300的红色方块中有一个蓝色的内部方块)
        # 这只是一个简单的例子，您可以替换为您自己的图像加载逻辑
        try:
            # 创建一个底色为红色的图像
            dummy_image = Image.new('RGB', (300, 300), color = 'red')
            
            # 在中间画一个蓝色的方块
            for x in range(100, 200):
                for y in range(100, 200):
                    dummy_image.putpixel((x, y), (0, 0, 255)) # 蓝色
            
            print("Generated a dummy image for testing.")
            # dummy_image.save("dummy_test_image.png") # 可选：保存图像以查看

            # 2. 定义边界框和描述文本
            # 边界框正好框住中间的蓝色方块
            blue_box = [100, 100, 200, 200] 
            # 边界框只框住一部分红色区域
            red_box = [10, 10, 50, 50]
            
            caption_blue = "a blue square"
            caption_red = "a red background"
            
            # 3. 计算相似度
            
            # 测试1: 蓝色方块区域与 "a blue square" 的相似度 (预期会很高)
            similarity_blue = get_clip_similarity_for_bbox(dummy_image, blue_box, caption_blue)
            print(f"Similarity between the blue box region and '{caption_blue}': {similarity_blue:.4f}")
            
            # 测试2: 红色区域与 "a red background" 的相似度 (预期会很高)
            similarity_red = get_clip_similarity_for_bbox(dummy_image, red_box, caption_red)
            print(f"Similarity between the red box region and '{caption_red}': {similarity_red:.4f}")

            # 测试3: 蓝色方块区域与 "a red background" 的相似度 (预期会很低)
            similarity_mismatch = get_clip_similarity_for_bbox(dummy_image, blue_box, caption_red)
            print(f"Similarity between the blue box region and '{caption_red}': {similarity_mismatch:.4f}")

        except Exception as e:
            print(f"An error occurred during the example run: {e}")