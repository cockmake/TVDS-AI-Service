import io
import os

import aiofiles
from miniopy_async import Minio

minio_endpoint = os.getenv('MINIO_ENDPOINT')
minio_access_key = os.getenv('MINIO_ACCESS_KEY')
minio_secret_key = os.getenv('MINIO_SECRET_KEY')

assert minio_endpoint is not None and minio_access_key is not None and minio_secret_key is not None, \
    "Please set MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY environment variables."
# Initialize Async MinIO client
minio_client = Minio(
    minio_endpoint,
    access_key=minio_access_key,
    secret_key=minio_secret_key,
    secure=False
)


async def get_object(bucket_name, object_name, session, minio_cache_dir="./minio-local-cache", is_cache=True):
    """
    Get an object from a MinIO bucket.
    if local file exists, return it
    :param bucket_name: Name of the bucket
    :param object_name: Name of the object
    :param session: Session object for async operations
    :param minio_cache_dir: Directory to cache the object locally
    :return: The object byte data
    """
    local_path = os.path.join(minio_cache_dir, bucket_name, object_name)
    if is_cache and os.path.exists(local_path):
        print("本地磁盘读取")
        async with aiofiles.open(local_path, 'rb') as f:
            data = await f.read()
        return data
    else:
        minio_object = await minio_client.get_object(bucket_name, object_name, session)
        minio_byte = await minio_object.read()
        if minio_byte is None:
            raise ValueError("Object not found or empty.")
        # 保存到本地
        if is_cache:
            dir_path = os.path.dirname(local_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(minio_byte)
        return minio_byte


async def put_object(bucket_name, object_name, data: bytes, minio_cache_dir="./minio-local-cache", is_cache=True):
    # 上传并缓存对象
    # 上传成功后缓存到本地
    # 本地存在就覆盖
    if not isinstance(data, bytes):
        raise ValueError("Data must be in bytes format.")
    data_io = io.BytesIO(data)
    object_write_result = await minio_client.put_object(bucket_name, object_name, data_io, len(data))
    if object_write_result is not None and is_cache:
        local_path = os.path.join(minio_cache_dir, bucket_name, object_name)
        dir_path = os.path.dirname(local_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        async with aiofiles.open(local_path, 'wb') as f:
            await f.write(data)
