## How to use

### Using Docker Compose 
You will need Docker installed to follow the next steps. To create and run the image use the following command:

```bash
> docker-compose up --build
```

The configuration will create a cluster with 3 containers:

- Consumer container
- Publisher container(yolo container)
- kafka container
- kafdrop container
- zookeeper container
- minio comtainer

The Publisher container will create a simple RESTful API application that sends data to Kafka and minio. It will take a few seconds to come up, then will be accessible at `http://localhost:8000`.

The Consumer container is a script that aims to wait and receive messages from Kafka.

The kafdrop container will provide acess to  web UI for viewing Kafka topics and browsing consumer groups that can be accessed at `http://localhost:19000`.

Minio S3 web UI for checking buckets and data can be accessed at `http://localhost:9001`

### FastAPI Swagger

The swagger, an automatic interactive API documentation, will be accessible at `http://localhost:8000/docs`


## Project Structure
Below is a project structure created:
```cmd
.
├── README.md
├── docker-compose.yml
└── services
    ├── consumer
    │   └── <simple app>
    ├── publisher
    │   └── <simple app>
    └── yoloAPI
        ├── app
        │   ├── core
        │   │   ├── gateways/ # init connections
        │   │   └── models/ # schemas
        │   │
        │   ├── dependencies/ # checks existence of instances 
        │   │
        │   ├── routers/ # API rotes
        │   │   ├── detect.py
        │   │   ├── dragndrop.py
        │   │   ├── index.py
        │   │   ├── minio.py
        │   │   └── publisher.py
        │   │
        │   ├── templates/ # html stuff
        │   │   └── <*.html>
        │   │
        │   ├── enum.py # EnvironmentVariables
        │   └── main.py # main :)
        │
        ├── Dockerfile
        └── requirements.txt
```

## Environment Variables
Listed below are the environment variables needed to run the application. They can be included in docker-compose or to run locally, it's necessary to create an `.env` file in the root of the Publisher and Consumer service folders.

- Publisher:
```bash
KAFKA_TOPIC_NAME=
KAFKA_SERVER=
KAFKA_PORT=
```

- Consumer:
```bash
KAFKA_TOPIC_NAME=
KAFKA_SERVER=
KAFKA_PORT=
```

- YOLO
```bash
KAFKA_SERVER=
KAFKA_TOPIC_NAME=
KAFKA_PORT=
MINIO_HOST=
MINIO_PORT=
MINIO_USER=
MINIO_PASSWORD=
```


## Help and Resources
You can read more about the tools documentation:

- [aiokafka](https://aiokafka.readthedocs.io/en/stable/ka)
- [Docker](https://docs.docker.com/get-started/overview/)
- [FastAPI](https://fastapi.tiangolo.com)
- [Kafdrop](https://github.com/obsidiandynamics/kafdrop)
- [Kafka](https://kafka.apache.org)
- [Kafka-python](https://kafka-python.readthedocs.io/en/master/)
- [Minio-python](https://min.io/docs/minio/linux/developers/python/API.html)
