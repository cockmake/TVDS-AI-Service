import io

from fastapi import APIRouter, Body, Path, Query, Header, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi import Depends
import os
import numpy as np
import aiohttp
import cv2 as cv
from data_access.minio_client import get_object
from data_access import minio_client
from utils import concatenate_images, plt_show_cv2_img, apply_bboxes
from settings import MINIO_LOCAL_CACHE_DIR

component_template_image_router = APIRouter(tags=['Component Template Image'])


@component_template_image_router.post('/visual-prompt/preview')
async def get_component_template_image_preview(
        component_template_image: dict = Body(embed=False),
):
    assert len(component_template_image) == 1, '预览图像有且仅有一个零部件'
    key = list(component_template_image.keys())[0]
    bucket_name = component_template_image[key]['bucketName']
    template_image = component_template_image[key]['templateImage'][:12]
    # 'templateImage': [{'imagePath': '1915454965191065601/1915788577791279104.jpg', 'boxes': [[x1, y1, x2, y2], ...]}]
    annotated_images = []
    async with aiohttp.ClientSession() as session:
        for image_info in template_image:
            image_path = image_info['imagePath']
            image_bytes = await get_object(bucket_name, image_path, session, MINIO_LOCAL_CACHE_DIR)
            annotated_images.append({
                'img': cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_COLOR),
                'bboxes': np.array([
                    [box[0], box[1], box[2], box[3]] for box in image_info['boxes']
                ])
            })
    concatenated_image, all_bboxes = concatenate_images(annotated_images, max_height=1024, max_width=1024)
    apply_bboxes(concatenated_image, all_bboxes)
    # 保存预览图像
    # cv.imwrite('preview.jpg', concatenated_image)
    # 返回图像的字节流
    _, buffer = cv.imencode(".png", concatenated_image)
    mage_bytes_io = io.BytesIO(buffer.tobytes())
    return StreamingResponse(
        mage_bytes_io,
        media_type='image/png'
    )
