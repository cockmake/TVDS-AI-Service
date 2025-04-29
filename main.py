import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services import component_template_image_router
from dependencies import app_lifespan

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(lifespan=app_lifespan, version="1.0.0", root_path="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(component_template_image_router, prefix="/component-template-image")
