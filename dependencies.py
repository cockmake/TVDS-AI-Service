import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from data_access import rabbit_async_client
from data_access.rabbitmq_async_client import rabbitmq_component_location_infer_V1, rabbitmq_component_location_infer_V2
from settings import PRODUCER_COMPONENT_LOCATION_QUEUE_NAME


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    print("初始化进程")
    logging.info("Initializing RabbitMQ client and consumer...")
    consume_background_tasks = []
    async with rabbit_async_client:
        # 位置推理
        consume_background_tasks.append(
            asyncio.create_task(
                rabbit_async_client.consume(
                    PRODUCER_COMPONENT_LOCATION_QUEUE_NAME,
                    rabbitmq_component_location_infer_V2
                )
            )
        )
        # 异常推理
        # <!todo>
        yield
        print("关闭进程")
        logging.info("Shutting down RabbitMQ client and consumer...")
        for task in consume_background_tasks:
            task.cancel()
