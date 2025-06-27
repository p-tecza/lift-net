FROM tensorflow/tensorflow:2.15.0-gpu

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y git && apt-get clean

# Instalacja bibliotek Python
RUN pip install --no-cache-dir \
    transformers \
    pandas \
    scikit-learn \
    keras \
    numpy \
    nltk 

# Ustaw katalog roboczy
WORKDIR /app