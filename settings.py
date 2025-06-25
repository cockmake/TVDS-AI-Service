# uvicorn main:app --host=0.0.0.0 --port=3456 --log-level=info --workers=3


# export MINIO_ENDPOINT=192.168.188.1:9000
# export MINIO_ENDPOINT=127.0.0.1:9000
# export MINIO_ACCESS_KEY=root
# export MINIO_SECRET_KEY=csu@2024


# export RABBITMQ_HOST=192.168.188.1
# export RABBITMQ_HOST=127.0.0.1
# export RABBITMQ_PORT=5674
# export RABBITMQ_USERNAME=root
# export RABBITMQ_PASSWORD=csu@2024

MINIO_GET_CACHE = False
MINIO_PUT_CACHE = False
COMPONENT_LOCATION_EXCHANGE_NAME = "component.location.exchange"
CONSUMER_COMPONENT_LOCATION_KEY = "consumer.component.location.key"
PRODUCER_COMPONENT_LOCATION_QUEUE_NAME = "producer.component.location.queue"

MINIO_LOCAL_CACHE_DIR = "minio-local-cache"

SNOWFLAKE_INSTANCE = 0
