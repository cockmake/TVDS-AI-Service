# 负责ai业务相关的处理
from typing import List, Dict
import random
import torch
from torchvision.ops import nms, batched_nms
from ultralytics import YOLOE, YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

from utils import split_vehicle_img

device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "yoloe-v8l-seg"
# model_name = "yoloe-11l-seg"
# model = YOLOE(f"./models/{model_name}.pt").to(device)

model_name = "yolo11x.pt"
model = YOLO(f"./models/{model_name}").to(device)
print(f"{model_name}模型加载完成")


def component_detection_infer_V1(source, visual_prompts: List[Dict]):
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
                iou=detection_iou,
                verbose=False
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
            'isAbnormal': [random.choice([False, False]) for _ in range(len(boxes))]
        }
        # 进行异常检测
        # <!todo>
    return r


def component_detection_infer_V2(source, component_info: List):
    r = {}
    split_vehicle = split_vehicle_img(source)
    scale_ratio = split_vehicle['scale_ratio']
    image_split = split_vehicle['img_list']
    offset_list = [image_info['w_offset'] for image_info in image_split]
    img_list = [image_info['img'] for image_info in image_split]
    box_list, cls_list, conf_list = [], [], []
    result_generator = []
    for img in img_list:
        result = model.predict(
            source=img,
            # save=True,
            conf=0.4,
            iou=0.25,
            verbose=False
        )
        result_generator.append(result[0])
    for i, result in enumerate(result_generator):
        # 偏移量
        if len(result.boxes.cls) == 0:
            continue
        offset = offset_list[i]
        result_boxes_xyxy = result.boxes.xyxy.clone()
        result_boxes_xyxy[:, [0, 2]] += offset
        box_list.append(result_boxes_xyxy)
        cls_list.append(result.boxes.cls.clone())
        conf_list.append(result.boxes.conf.clone())
    if len(box_list) != 0:
        boxes = torch.cat(box_list, dim=0)
        idxs = torch.cat(cls_list, dim=0)
        scores = torch.cat(conf_list, dim=0)
        # Apply NMS
        keep_indices = batched_nms(boxes, scores, idxs, iou_threshold=0.25)
        boxes = boxes[keep_indices] / scale_ratio
        scores = scores[keep_indices]
        labels = idxs[keep_indices]
        # 进行后处理
        boxes = boxes.round().int()
        labels = labels.int()
        boxes[:, [0, 1]] -= 20
        boxes[:, [2, 3]] += 20
        # 限制边界框的坐标在图像范围内
        torch.clamp_(boxes[:, 0], min=0)
        torch.clamp_(boxes[:, 1], min=0)
        torch.clamp_(boxes[:, 2], max=source.shape[1])
        torch.clamp_(boxes[:, 3], max=source.shape[0])
        # 将检测结果写入到对应的零部件中
        unique_labels = labels.unique()
        for u_label in unique_labels:
            component_id = component_info[u_label.item()]['id']
            component_mask = (labels == u_label)
            r[component_id] = {
                'boxes': boxes[component_mask].cpu().numpy().tolist(),
                'confidences': scores[component_mask].cpu().numpy().tolist(),
                'abnormalityResults': [''] * component_mask.sum().item(),
                'isAbnormal': [random.choice([False, False]) for _ in range(component_mask.sum().item())]
            }
    # 进行异常检测
    # <!todo>
    return r


def main():
    pass


if __name__ == '__main__':
    main()
