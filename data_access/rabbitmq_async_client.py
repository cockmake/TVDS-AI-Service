import asyncio
import json
import logging
import os
from typing import Optional, Union, Dict, Callable, Any, Awaitable

import aio_pika
import aiohttp
import cv2 as cv
import numpy as np
import snowflake
from ai import component_detection_infer
from aio_pika.abc import AbstractIncomingMessage
from data_access.minio_client import get_object, put_object
from settings import MINIO_LOCAL_CACHE_DIR, SNOWFLAKE_INSTANCE, CONSUMER_COMPONENT_LOCATION_KEY, \
    COMPONENT_LOCATION_EXCHANGE_NAME
from utils import concatenate_images
from settings import MINIO_PUT_CACHE, MINIO_GET_CACHE


class AsyncRabbitMQError(Exception):
    """自定义 RabbitMQ 客户端异常"""
    pass


class AsyncRabbitMQClient:
    def __init__(
            self,
            prefetch_count: int = 1,
            heartbeat: int = 10,
            # reconnect_interval: int = 10,
    ):
        self.host: Optional[str] = os.environ.get('RABBITMQ_HOST')
        port_str: Optional[str] = os.environ.get('RABBITMQ_PORT')
        self.username: Optional[str] = os.environ.get('RABBITMQ_USERNAME')
        self.password: Optional[str] = os.environ.get('RABBITMQ_PASSWORD')

        if not all([self.host, port_str, self.username, self.password]):
            raise AsyncRabbitMQError(
                "RabbitMQ connection details missing in environment variables (RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD)"
            )
        try:
            self.port: int = int(port_str)
        except ValueError:
            raise AsyncRabbitMQError(f"Invalid RABBITMQ_PORT: '{port_str}'. Must be an integer.")

        self.prefetch_count: int = prefetch_count
        self.heartbeat: int = heartbeat
        # self.reconnect_interval: int = reconnect_interval
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.RobustChannel] = None

    async def connect(self):
        """连接到 RabbitMQ 服务器"""
        try:
            self.connection = await aio_pika.connect_robust(
                host=self.host,
                port=self.port,
                login=self.username,
                password=self.password,
                heartbeat=self.heartbeat,
                # reconnect_interval=self.reconnect_interval
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.prefetch_count)
            logging.info("Connected to RabbitMQ server.")
        except Exception as e:
            raise AsyncRabbitMQError(f"Failed to connect to RabbitMQ: {e}")
        return self

    async def __aenter__(self):
        """异步上下文管理器进入时连接到 RabbitMQ"""
        if not self.connection or self.connection.is_closed:
            await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出时关闭连接"""
        await self.close()
        if exc_type:
            logging.error(f"Exception occurred: {exc_val}")

    async def close(self):
        """异步关闭通道和连接"""
        try:
            if self.channel and not self.channel.is_closed:
                await self.channel.close()
                logging.info("RabbitMQ channel closed.")
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
                logging.info("RabbitMQ connection closed.")
        except Exception as e:
            logging.error(f"Error closing RabbitMQ connection/channel: {e}")
            # 记录错误，但允许程序继续

    async def consume(
            self,
            queue_name: str,
            callback: Callable[[AbstractIncomingMessage], Awaitable[Any]],
            no_ack: bool = False,
    ):
        """开始消费指定队列的消息"""
        if not self.channel:
            raise AsyncRabbitMQError("Channel is not available.")
        try:
            logging.info(f"Starting consuming messages from queue '{queue_name}'...")
            # 开始消费，consume 会在后台运行
            # callback 必须是一个 async 函数
            queue = await self.channel.get_queue(queue_name)
            await queue.consume(callback, no_ack=no_ack)
            logging.info(f"Consumer set up for queue '{queue_name}'. Waiting for messages.")
            # 返回一个 Future，允许调用者等待（如果需要，但通常在生命周期中不需要）
            # 或者可以直接返回，让其在事件循环后台运行
        except Exception as e:
            logging.error(f"Error starting consumer for queue '{queue_name}': {e}")
            raise AsyncRabbitMQError(f"Error starting consumer: {e}") from e

    async def publish(
            self,
            exchange_name: str,
            routing_key: str,
            message: Union[str, bytes, Dict[str, Any]]
    ):
        if not self.channel:
            raise AsyncRabbitMQError("Channel is not available.")
        if isinstance(message, (dict, list)):
            message_body = json.dumps(message).encode('utf-8')
            content_type = 'application/json'
        elif isinstance(message, str):
            message_body = message.encode('utf-8')
            content_type = 'text/plain'
        elif isinstance(message, bytes):
            message_body = message
            content_type = 'application/octet-stream'
        else:
            raise TypeError("Message must be dict, list, str, or bytes")

        try:
            logging.info(f"Publishing message to exchange '{exchange_name}' with routing key '{routing_key}'...")
            exchange = await self.channel.get_exchange(exchange_name)
            await exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    content_type=content_type
                ),
                routing_key=routing_key
            )
            logging.info(f"Message published to exchange '{exchange_name}' with routing key '{routing_key}'.")
        except Exception as e:
            logging.error(f"Error publishing message: {e}")
            raise AsyncRabbitMQError(f"Error publishing message: {e}") from e


rabbit_async_client = AsyncRabbitMQClient()


# 处理消息队列
async def rabbitmq_component_location_infer(message: AbstractIncomingMessage):
    async with message.process():
        print('component_location_infer')
        # 处理消息
        data = json.loads(message.body.decode())
        logging.info(f"Received message: {json.dumps(data, indent=2, ensure_ascii=False)}")
        task_id = data['taskId']
        detection_result_bucket = data['detectionResultBucket']
        template_image_list = data['templateImageList']
        async with aiohttp.ClientSession() as session:
            component_visual_prompt = []
            print("开始读取模板图像")
            for component_id, component_info in template_image_list.items():
                template_image_info = component_info['templateImage']
                annotated_images = []
                for image_info in template_image_info:
                    image_bytes = await get_object(
                        component_info['bucketName'],
                        image_info['imagePath'],
                        session,
                        MINIO_LOCAL_CACHE_DIR,
                        is_cache=MINIO_GET_CACHE
                    )
                    annotated_images.append({
                        'img': cv.imdecode(np.frombuffer(image_bytes, np.uint8), cv.IMREAD_COLOR),
                        'bboxes': np.array([
                            [box[0], box[1], box[2], box[3]] for box in image_info['boxes']
                        ])
                    })
                concatenated_image, all_bboxes = concatenate_images(annotated_images)
                visuals = {
                    'bboxes': all_bboxes.bboxes,
                    'cls': np.array([0] * len(all_bboxes))
                }
                component_visual_prompt.append({
                    'component_id': component_id,
                    'refer_image': concatenated_image,
                    'prompt': visuals,
                    'detection_conf': component_info['detectionConf'],
                    'detection_iou': component_info['detectionIou'],
                    'abnormality_desc': component_info['abnormalityDesc'],
                })
            print("读取模板图像完成")
            print("开始读取车辆图像")
            railway_vehicle_bucket = data['railwayVehicleBucket']
            vehicle_image_path_list = data['vehicleImagePaths']
            component_location_result_list = []
            vehicle_image_list = []
            for i, vehicle_image_path in enumerate(vehicle_image_path_list):
                print(f"读取车辆方位图像：{i}")
                image_byte = await get_object(
                    railway_vehicle_bucket,
                    vehicle_image_path,
                    session,
                    MINIO_LOCAL_CACHE_DIR,
                    is_cache=MINIO_GET_CACHE
                )
                vehicle_image = cv.imdecode(np.frombuffer(image_byte, np.uint8), cv.IMREAD_COLOR)
                print(f"车辆方位图像读取完成：{i}")
                print(f"开始检测车辆方位图像：{i}")
                component_location_result = component_detection_infer(vehicle_image, component_visual_prompt)
                print(f"车辆方位图像检测完成：{i}")
                component_location_result_list.append(component_location_result)
                vehicle_image_list.append(vehicle_image)
    print("检测结果：\n", component_location_result_list)
    snowflake_generator = snowflake.SnowflakeGenerator(SNOWFLAKE_INSTANCE)
    for i, component_location_result in enumerate(component_location_result_list):
        vehicle_image = vehicle_image_list[i]
        for component_id, detection_result in component_location_result.items():
            # 生成唯一的 ID
            images_path = []
            for i, (box, conf) in enumerate(zip(detection_result['boxes'], detection_result['confidences'])):
                x1, y1, x2, y2 = box
                # 生成唯一 ID
                unique_id = next(snowflake_generator)
                # 将处理结果上传到 MinIO
                image_path = f"{task_id}/{component_id}/{unique_id}.jpg"
                component_image = vehicle_image[y1:y2, x1:x2]
                _, image_buffer = cv.imencode('.jpg', component_image)
                await put_object(
                    detection_result_bucket,
                    image_path,
                    image_buffer.tobytes(),
                    is_cache=MINIO_PUT_CACHE
                )
                images_path.append(image_path)
            detection_result['imagePaths'] = images_path
    producer_message = {
        'taskId': task_id,
        'componentLocationResults': component_location_result_list,
    }
    print("最终检测结果：\n", producer_message)
    print("开始发布消息")
    await rabbit_async_client.publish(
        COMPONENT_LOCATION_EXCHANGE_NAME,
        CONSUMER_COMPONENT_LOCATION_KEY,
        producer_message
    )


async def rabbitmq_component_defection_infer(message: AbstractIncomingMessage):
    async with message.process():
        print('component_defection_infer')
        # 处理消息
        data = json.loads(message.body.decode())
        logging.info(f"Received message: {data}")


async def main():
    client = AsyncRabbitMQClient()
    # 这里可以添加更多的操作，例如发布消息、消费消息等
    async with client:
        # 发布消息示例
        await client.publish(
            exchange_name="component.location.exchange",
            routing_key="consumer.component.location.key",
            message={"location": "China", "status": "active"}
        )

        # 定义一个简单的消息处理函数
        async def message_handler(message: AbstractIncomingMessage):
            async with message.process():
                # 处理消息
                data = json.loads(message.body.decode())
                logging.info(f"Received message: {data}")

        # 消费消息
        await client.consume("producer.component.location.queue", message_handler)
        # 保持运行，直到手动停止
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping consumer...")


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()
