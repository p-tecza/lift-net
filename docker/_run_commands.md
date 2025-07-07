# Budowanie obrazu kontenera do nauki sieci:
docker build -t net_trainer:latest .

# Budowanie obrazu kontenera appki webowej:
docker build -f Dockerfile.app -t dcnds-net-app:latest ../web_app

# Uczenie sieci LSTM:
docker-compose -f lstm-net.yml up --build

# Uczenie sieci opartej o BERT:
docker-compose -f bert-net.yml up --build

# Uruchomienie aplikacji webowej umożliwiającej korzystanie z sieci:
docker-compose -f app-compose.yml up --build