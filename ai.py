import json
import logging

from aio_pika.abc import AbstractIncomingMessage


# 加载模型

# 处理消息队列
async def rabbitmq_component_location_infer(message: AbstractIncomingMessage):
    async with message.process():
        print('component_location_infer')
        # 处理消息
        data = json.loads(message.body.decode())
        logging.info(f"Received message: {data}")


async def rabbitmq_component_defection_infer(message: AbstractIncomingMessage):
    async with message.process():
        print('component_defection_infer')
        # 处理消息
        data = json.loads(message.body.decode())
        logging.info(f"Received message: {data}")
