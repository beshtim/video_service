import logging
from json import loads, load

from app.enum import EnvironmentVariables as EnvVariables

from kafka import KafkaConsumer
from minio import Minio

def main():
    try:
        # To consume latest messages and auto-commit offsets
        consumer = KafkaConsumer(
            EnvVariables.KAFKA_TOPIC_NAME.get_env(),
            bootstrap_servers=f'{EnvVariables.KAFKA_SERVER.get_env()}:{EnvVariables.KAFKA_PORT.get_env()}',
            value_deserializer=lambda x: loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=False,
        )

        minio = Minio(
            endpoint=f'{EnvVariables.MINIO_HOST.get_env()}:{EnvVariables.MINIO_PORT.get_env()}', 
            access_key=EnvVariables.MINIO_USER.get_env(), 
            secret_key=EnvVariables.MINIO_PASSWORD.get_env(), 
            secure=False
            )

        for message in consumer:
            
            
            if "image_name" in message.value.keys():
                
                bucket = 'events'
                im_name = message.value['image_name']

                try:
                    response = minio.get_object(bucket, im_name)
                    im_bytes = response.read()
                    response.close()
                    response.release_conn()
                except Exception as e:
                    print(f"Error with getting image from minio {e}")
                
            else:
                print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                                message.offset, message.key, message.value))

    except Exception as e:
        logging.info('Connection successful', e)
