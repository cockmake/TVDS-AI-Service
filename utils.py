from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from ultralytics.utils.instance import Bboxes

def merge_adjacent_boxes(boxes, scores, labels, w_distance_threshold, h_distance_threshold):
    if boxes.numel() == 0:
        return boxes, scores, labels

    num_boxes = boxes.shape[0]

    # 使用并查集 (DSU) 进行高效聚类
    parent = list(range(num_boxes))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    # 遍历所有框对，判断是否相邻并进行合并
    for i in range(num_boxes):
        box1 = boxes[i]
        for j in range(i + 1, num_boxes):
            box2 = boxes[j]

            gap_x = max(0.0, max(box1[0], box2[0]) - min(box1[2], box2[2]))
            gap_y = max(0.0, max(box1[1], box2[1]) - min(box1[3], box2[3]))

            if gap_x < w_distance_threshold and gap_y < h_distance_threshold:
                union(i, j)

    # 按根节点对框进行分组
    clusters = {}
    for i in range(num_boxes):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # 合并每个聚类中的框
    merged_boxes_list = []
    merged_scores_list = []
    merged_labels_list = []

    for root in clusters:
        indices = torch.tensor(clusters[root], device=boxes.device, dtype=torch.long)

        cluster_boxes = boxes[indices]
        cluster_scores = scores[indices]
        cluster_labels = labels[indices]

        # 创建一个能完全包围簇内所有框的新框
        min_x1 = torch.min(cluster_boxes[:, 0])
        min_y1 = torch.min(cluster_boxes[:, 1])
        max_x2 = torch.max(cluster_boxes[:, 2])
        max_y2 = torch.max(cluster_boxes[:, 3])
        merged_box = torch.tensor([min_x1, min_y1, max_x2, max_y2], device=boxes.device)

        # 新框的类别和置信度继承自簇内置信度最高的那个框
        max_score_idx = torch.argmax(cluster_scores)
        merged_score = cluster_scores[max_score_idx]
        merged_label = cluster_labels[max_score_idx]

        merged_boxes_list.append(merged_box)
        merged_scores_list.append(merged_score)
        merged_labels_list.append(merged_label)

    if not merged_boxes_list:
        return (torch.empty((0, 4), device=boxes.device),
                torch.empty(0, device=scores.device),
                torch.empty(0, device=labels.device))

    return torch.stack(merged_boxes_list), torch.stack(merged_scores_list), torch.stack(merged_labels_list)

def plt_show_cv2_img(img: np.ndarray, title: str = ""):
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def img_preprocess(
        img,
        target_size,
):
    """
    Pre-processes the input image.

    Args:
        target_size: int | tuple: target size for the image.
        img (Numpy.ndarray): image about to be processed.

    Returns:
        img_process (Numpy.ndarray): image preprocessed for inference.
        ratio (tuple): width, height ratios in letterbox.
        pad_w (float): width padding in letterbox.
        pad_h (float): height padding in letterbox.
    """
    shape = img.shape[:2]
    new_shape = target_size
    if isinstance(target_size, int):
        new_shape = (target_size, target_size)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img_process = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))
    return img_process, ratio, (pad_w, pad_h)


def box_preprocess(
        bounding_box: np.ndarray,
        ratio,
        pad_w,
        pad_h
):
    bboxes = Bboxes(bounding_box, format="xyxy")
    bboxes.mul(ratio)
    bboxes.add((pad_w, pad_h, pad_w, pad_h))
    return bboxes


def block_preprocess_thread(
        concatenated_image,
        all_bboxes,
        img,
        bboxes,
        h_offset,
        w_offset,
        image_height,
        image_width,
):
    # 调整图像大小
    img_resized, ratio, (pad_w, pad_h) = img_preprocess(img, (image_height, image_width))
    # 将图像放入拼接图像中
    concatenated_image[h_offset:h_offset + image_height, w_offset:w_offset + image_width] = img_resized
    # 调整边界框
    bboxes = box_preprocess(bboxes, ratio, pad_w, pad_h)
    bboxes.add((w_offset, h_offset, w_offset, h_offset))
    # 将调整后的边界框添加到列表中
    all_bboxes.append(bboxes)


def concatenate_images(
        annotated_images: list[dict],
        max_height=640,
        max_width=640,
        thread_num=0,
):
    """
    该函数将所有的annotated_images拼接成一张图像，拼接后的图片和对于拼接的图片的bounding box也会被返回
    Args:
        thread_num: 0表示不使用多线程，> 0表示使用多线程
        max_height: 图像拼接后的最大高度
        max_width: 图像拼接后的最大宽度
        annotated_images: list[dict:{img, bboxes}]
    """
    print("拼接图像中...")
    # 计算有多少行和多少列可以放置
    num_images = len(annotated_images)
    num_rows = num_cols = 1
    while num_rows * num_cols < num_images:
        if num_rows == num_cols:
            num_cols += 1
        else:
            num_rows += 1
    print(f"拼接图像的行数: {num_rows}, 列数: {num_cols}")
    # 计算每个图像的宽度和高度
    image_width = max_width // num_cols
    image_height = max_height // num_rows
    # 创建一个空白图像
    concatenated_image = np.full((image_height * num_rows, image_width * num_cols, 3), 114, dtype=np.uint8)
    # 创建一个空的列表来存储边界框
    all_bboxes = []
    # 遍历每个图像
    if thread_num > 0:
        # 多线程合成（其实不用影响也不大）
        # 创建线程池
        print("使用多线程拼接图像...")
        with ThreadPoolExecutor(max_workers=thread_num) as executor:
            futures = []
            for i, annotated_image in enumerate(annotated_images):
                img = annotated_image["img"]
                bboxes = annotated_image["bboxes"]
                # 计算当前图像在拼接图像中的位置
                row = i // num_cols
                col = i % num_cols
                w_offset = col * image_width
                h_offset = row * image_height
                # 提交任务到线程池
                future = executor.submit(
                    block_preprocess_thread,
                    concatenated_image,
                    all_bboxes,
                    img,
                    bboxes,
                    h_offset,
                    w_offset,
                    image_height,
                    image_width
                )
                futures.append(future)
            # 等待所有线程完成
            for future in futures:
                future.result()
    else:
        print("使用单线程拼接图像...")
        for i, annotated_image in enumerate(annotated_images):
            img = annotated_image["img"]
            bboxes = annotated_image["bboxes"]
            # 计算当前图像在拼接图像中的位置
            row = i // num_cols
            col = i % num_cols
            w_offset = col * image_width
            h_offset = row * image_height
            block_preprocess_thread(
                concatenated_image, all_bboxes, img, bboxes, h_offset, w_offset, image_height, image_width
            )
    # 将所有边界框拼接在一起
    all_bboxes = Bboxes.concatenate(all_bboxes)
    # 返回拼接后的图像和边界框
    print("拼接完成")
    return concatenated_image, all_bboxes


def apply_bboxes(
        concatenated_image,
        all_bboxes,
        color=(0, 255, 0),
        thickness=2,
        in_place=True,
):
    """
    在图像上绘制边界框
    Args:
        concatenated_image: 拼接后的图像
        all_bboxes: 拼接后的边界框
        color: 边界框颜色
        thickness: 边界框线条粗细
        in_place: 是否在原图上绘制
    """
    # 复制图像
    if not in_place:
        concatenated_copy = deepcopy(concatenated_image)
        image = concatenated_copy
    else:
        image = concatenated_image
    # 遍历每个边界框
    for box in all_bboxes.bboxes:
        # 获取边界框的坐标
        [x1, y1, x2, y2] = box.astype(np.int32)
        # 绘制边界框
        cv.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def split_vehicle_img(
        img: np.ndarray,
        max_width: int = 1024,
        step_ratio: float = 0.15,
) -> Dict:
    """
    Args:
        img: 原来的大图
        max_width: 分割后每个小图的最大宽度，如果原图的高度大于max_width，则将高度缩放到max_width
        step_ratio: 分割步长比例，表示每次的分割宽度是上一个分割宽度的step_ratio倍
    Returns:
        Dict: 分割后的小图列表还有该图像的起始位置和缩放信息
    """
    # 计算分割宽度
    assert len(img.shape) == 3, "图像必须是三通道的"
    h, w, _ = img.shape
    # 如果图片的高度大于max_width，则将高缩放到max_width，宽度按比例缩放，并返回缩放比例系数
    if h > max_width:
        scale_ratio = max_width / h
        img = cv.resize(img, (int(w * scale_ratio), int(h * scale_ratio)))
        h, w, _ = img.shape
    else:
        scale_ratio = 1.0
    return {
        'scale_ratio': scale_ratio,
        'img_list': [
            {
                'img': img[:, i:i + max_width],
                'w_offset': i
            }
            for i in range(0, w, int(max_width * step_ratio))
        ]
    }
