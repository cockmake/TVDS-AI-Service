# 负责ai业务相关的处理
from typing import List, Dict
import random
import torch
from torchvision.ops import nms
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from utils import split_vehicle_img

model_name = "yoloe-v8l-seg"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLOE(f"./models/{model_name}.pt").to(device)
print(f"{model_name}模型加载完成")


def component_detection_infer(source, visual_prompts: List[Dict]):
    split_vehicle = split_vehicle_img(source)
    scale_ratio = split_vehicle['scale_ratio']
    img_list = split_vehicle['img_list']
    r = {}
    for visual_prompt in visual_prompts:
        # 对于每个视觉提示（每个零部件），进行检测
        component_id = visual_prompt['component_id']
        refer_image = visual_prompt['refer_image']
        prompt = visual_prompt['prompt']
        detection_conf = visual_prompt['detection_conf']
        detection_iou = visual_prompt['detection_iou']
        abnormality_desc = visual_prompt['abnormality_desc']

        boxes_to_nms, scores_to_nms = [], []
        for image_info in img_list:
            # 对于每个分割后的图像，进行检测
            image = image_info['img']
            w_offset = image_info['w_offset']
            results = model.predict(
                source=image,
                refer_image=refer_image,
                visual_prompts=prompt,
                predictor=YOLOEVPSegPredictor,
                imgsz=640,
                conf=detection_conf,
                iou=detection_iou
            )
            result = results[0]
            pos = torch.clone(result.boxes.xyxy)
            pos[:, [0, 2]] += w_offset
            boxes_to_nms.append(pos)
            scores_to_nms.append(result.boxes.conf)
        boxes_to_nms = torch.concatenate(boxes_to_nms, dim=0)
        if boxes_to_nms.shape[0] == 0:
            # 没有检测到指定的零部件
            continue
        scores_to_nms = torch.concatenate(scores_to_nms, dim=0)
        # 进行NMS
        nms_indices = nms(
            boxes_to_nms,
            scores_to_nms,
            iou_threshold=detection_iou
        )
        boxes = boxes_to_nms[nms_indices]
        confidences = scores_to_nms[nms_indices]
        # 进行后处理
        boxes = (boxes / scale_ratio).round().int()
        # 稍微调整一下边界框的坐标
        boxes[:, [0, 1]] -= 20
        boxes[:, [2, 3]] += 20
        # 限制边界框的坐标在图像范围内
        torch.clamp_(boxes[:, 0], min=0)
        torch.clamp_(boxes[:, 1], min=0)
        torch.clamp_(boxes[:, 2], max=source.shape[1])
        torch.clamp_(boxes[:, 3], max=source.shape[0])
        r[component_id] = {
            'boxes': boxes.cpu().numpy().tolist(),
            'confidences': confidences.cpu().numpy().tolist(),
            'abnormalityResults': [''] * len(boxes),
            'isAbnormal': [random.choice([True, False]) for _ in range(len(boxes))]
        }
        # 进行异常检测
        # <!todo>
    return r


def main():
    pass


if __name__ == '__main__':
    main()
