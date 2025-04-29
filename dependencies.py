from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from data_access.rabbitmq_async_client import rabbit_async_client
import asyncio
from ai import rabbitmq_component_defection_infer, rabbitmq_component_location_infer


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    print("初始化进程")
    logging.info("Initializing RabbitMQ client and consumer...")
    consume_background_tasks = []
    async with rabbit_async_client:
        # await rabbit_async_client.consume("producer.component.location.queue", message_handler)
        # 异常推理
        # consume_background_tasks.append(
        #     asyncio.create_task(
        #         rabbit_async_client.consume(
        #             "producer.component.location.queue",
        #             rabbitmq_component_defection_infer
        #         )
        #     )
        # )
        # 位置推理
        consume_background_tasks.append(
            asyncio.create_task(
                rabbit_async_client.consume(
                    "producer.component.location.queue",
                    rabbitmq_component_location_infer
                )
            )
        )
        yield
        print("关闭进程")
        logging.info("Shutting down RabbitMQ client and consumer...")
        for task in consume_background_tasks:
            task.cancel()
