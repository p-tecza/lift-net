# lift-net

docker build -t bert-trainer .
docker run --gpus all -v ${PWD}:/app bert-trainer


docker compose up 