import time
from copy import deepcopy

import numpy as np
import torch
import cv2 as cv
import random
from ultralytics import YOLOE
from matplotlib import pyplot as plt
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from utils import concatenate_images, plt_show_cv2_img, split_vehicle_img

# Initialize a YOLOE model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLOE("./models/yoloe-v8l-seg.pt").to(device)
print("模型加载完成")


def get_visual_prompt_input():
    # Set visual prompt
    bboxes1 = np.array(
        [
            [100, 87, 200, 331],
            [251, 90, 337, 328]
        ],
        dtype=np.float32
    )
    refer_image1 = cv.imread("assets/template-1.png")

    bboxes2 = np.array(
        [
            [35, 117, 120, 283],
            [240, 127, 325, 276]
        ],
        dtype=np.float32
    )
    refer_image2 = cv.imread("assets/template-2.png")

    bboxes3 = np.array(
        [
            [286, 283, 366, 438],
            [1, 101, 105, 414]
        ],
        dtype=np.float32
    )
    refer_image3 = cv.imread("assets/template-3.png")

    bboxes4 = np.array(
        [
            [316, 452, 213, 267],
            [90, 444, 2, 263]
        ],
        dtype=np.float32
    )
    refer_image4 = cv.imread("assets/template-4.png")

    bboxes5 = np.array(
        [
            [23, 242, 117, 437],
            [281, 94, 383, 369]
        ],
        dtype=np.float32
    )
    refer_image5 = cv.imread("assets/template-5.png")

    bboxes6 = np.array(
        [
            [22, 251, 109, 429],
            [247, 259, 333, 429]
        ],
        dtype=np.float32
    )
    refer_image6 = cv.imread("assets/template-6.png")

    annotated_images = [
        {
            'img': refer_image1,
            'bboxes': bboxes1,
        },
        {
            'img': refer_image2,
            'bboxes': bboxes2,
        },
        {
            'img': refer_image3,
            'bboxes': bboxes3,
        },
        {
            'img': refer_image4,
            'bboxes': bboxes4,
        },
        {
            'img': refer_image5,
            'bboxes': bboxes5,
        },
        {
            'img': refer_image6,
            'bboxes': bboxes6,
        }
    ]

    concatenated_image, all_bboxes = concatenate_images(annotated_images)
    # 绘制
    concatenated_image_copy = deepcopy(concatenated_image)
    for box in all_bboxes.bboxes:
        cv.rectangle(concatenated_image_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    # plt_show_cv2_img(concatenated_image_copy, "Template Image")
    cv.imwrite("results/Template Image.png", concatenated_image_copy)
    # 生成视觉提示
    visual_prompt = {
        'bboxes': all_bboxes.bboxes,
        'cls': np.array([0] * len(all_bboxes))
    }
    return concatenated_image, visual_prompt


def main():
    refer_image, visuals = get_visual_prompt_input()
    print("视觉提示生成完成")
    # source = cv.imread(f"assets/test-a.png")
    img = cv.imread("assets/20230831010_2_1.jpg")
    split_result = split_vehicle_img(img)
    start = time.time()
    for i in range(20):
        for item in split_result['img_list']:
            source = item['img']
            results = model.predict(
                source=source,
                refer_image=refer_image,
                visual_prompts=visuals,
                predictor=YOLOEVPSegPredictor,
                imgsz=640,
                conf=0.08,
                iou=0.1,
                save=False,
                project="runs",  # 存储目录
            )
    print(f"{time.time() - start}秒")
    # apply result to image
    # for result in results:
    #     for box in result.boxes:
    #         xyxy = box.xyxy[0].cpu().numpy()
    #         conf = box.conf[0].cpu().numpy()
    #         cv.rectangle(source, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    #         cv.putText(source, f"Conf: {conf:.2f}", (int(xyxy[0]), int(xyxy[1] - 10)), cv.FONT_HERSHEY_PLAIN, 1,
    #                    (0, 255, 0), 2)
    # cv.imwrite(f"results/test-a-result.png", source)


if __name__ == '__main__':
    main()
