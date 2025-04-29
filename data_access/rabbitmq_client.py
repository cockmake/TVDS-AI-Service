import os
import pika
import json
from typing import Optional, Union, Dict, List
import logging


class RabbitMQError(Exception):
    """自定义 RabbitMQ 客户端异常"""
    pass


class RabbitMQClient:
    def __init__(
            self,
            prefetch_count: int = 1,
            heartbeat: int = 30,
    ):

        self.host: Optional[str] = os.environ.get('RABBITMQ_HOST')
        port_str: Optional[str] = os.environ.get('RABBITMQ_PORT')
        self.username: Optional[str] = os.environ.get('RABBITMQ_USERNAME')
        self.password: Optional[str] = os.environ.get('RABBITMQ_PASSWORD')

        if not all([self.host, port_str, self.username, self.password]):
            raise RabbitMQError(
                "RabbitMQ connection details missing in environment variables (RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_USERNAME, RABBITMQ_PASSWORD)"
            )
        try:
            self.port: int = int(port_str)
        except ValueError:
            raise RabbitMQError(f"Invalid RABBITMQ_PORT: '{port_str}'. Must be an integer.")

        self.prefetch_count: int = prefetch_count
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            heartbeat=heartbeat,
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.basic_qos(prefetch_count=self.prefetch_count)

    def close(self):
        """关闭通道和连接"""
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
                logging.info("RabbitMQ channel closed.")
            if self.connection and self.connection.is_open:
                self.connection.close()
                logging.info("RabbitMQ connection closed.")
        except Exception as e:
            logging.error(f"Error closing RabbitMQ connection/channel: {e}")
            # 即使关闭出错，也继续执行，但记录错误

    def __enter__(self):
        """进入上下文管理器时返回自身"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器时关闭连接"""
        self.close()

        if exc_type:
            logging.error(f"Exception occurred: {exc_val}")

    def publish(self, exchange: str, routing_key: str, body: Union[Dict, List, str, bytes]):
        """发布消息"""
        if not self.channel or not self.channel.is_open:
            raise RabbitMQError("Cannot publish message, channel is not open.")

        if isinstance(body, (dict, list)):
            message_body = json.dumps(body).encode('utf-8')
            content_type = 'application/json'
        elif isinstance(body, str):
            message_body = body.encode('utf-8')
            content_type = 'text/plain'
        elif isinstance(body, bytes):
            message_body = body
            content_type = 'application/octet-stream'  # 或者根据实际情况设置
        else:
            raise TypeError("Body must be dict, list, str, or bytes")

        try:
            properties = pika.BasicProperties(content_type=content_type, delivery_mode=2)  # delivery_mode=2 使消息持久化
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=message_body,
                properties=properties
            )
            logging.debug(f"Message published to exchange '{exchange}', routing key '{routing_key}'")
        except Exception as e:
            logging.error(f"Failed to publish message: {e}")
            raise RabbitMQError(f"Failed to publish message: {e}") from e

    def consume(
            self,
            queue: str,
            callback,
            auto_ack: bool = False
    ):
        if not self.channel or not self.channel.is_open:
            raise RabbitMQError("Cannot consume messages, channel is not open.")
        try:
            self.channel.basic_consume(queue=queue, on_message_callback=callback, auto_ack=auto_ack)
            logging.info(
                f"Starting consuming messages from queue '{queue}' (auto_ack={auto_ack}). Press CTRL+C to exit.")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logging.info("Consumer stopped by user.")
            self.close()
        except Exception as e:
            logging.error(f"Error during consumption: {e}")
            self.close()  # 出现错误时尝试关闭连接
            raise RabbitMQError(f"Error during consumption: {e}") from e


rabbitmq_client = RabbitMQClient()


def main():
    def callback(ch, method, properties, body):
        # 处理消息
        print(f"Received message: {body.decode()}")
        # 发送确认
        ch.basic_ack(delivery_tag=method.delivery_tag)

    # 测试
    with rabbitmq_client:
        # 发布消息
        rabbitmq_client.publish(
            exchange='component.location.exchange',
            routing_key='consumer.component.location.key',
            body={"key": "value"}
        )
        try:
            # 消费消息
            rabbitmq_client.consume(queue='producer.component.location.queue', callback=callback)
        except RabbitMQError as e:
            logging.error(f"RabbitMQ error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
