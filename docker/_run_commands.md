# Budowanie obrazu kontenera do nauki sieci :
docker build -t net_trainer:latest .

# Uczenie sieci LSTM:
docker-compose -f lstm-net.yml up --build

# Uczenie sieci opartej o BERT:
docker-compose -f bert-net.yml up --build