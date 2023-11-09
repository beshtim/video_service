import logging
from json import loads

from app.enum import EnvironmentVariables as EnvVariables

from kafka import KafkaConsumer
from minio import Minio

import telebot
bot = telebot.TeleBot(EnvVariables.TELETOKEN.get_env())

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
            bot.send_message(145590903, "%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                                 message.offset, message.key, message.value))
            
            bucket = 'test'
            im_name = message.value['description']

            try:
                response = minio.get_object(bucket, im_name)
                im_bytes = response.read()
            finally:
                response.close()
                response.release_conn()

            bot.send_photo(145590903, photo=im_bytes)
            
            print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                                 message.offset, message.key, message.value))

    except Exception as e:
        logging.info('Connection successful', e)
