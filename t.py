import asyncio
import json
import logging

from aio_pika.abc import AbstractIncomingMessage

from data_access.rabbitmq_async_client import rabbit_async_client



async def main():
    # 创建后台监听任务列表

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping consumer...")


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.close()
