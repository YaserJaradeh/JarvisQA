FROM python:3

RUN apt-get install -y git
RUN mkdir -p /app/
RUN cd /app/ && \
    git clone https://github.com/YaserJaradeh/JarvisQA.git && \
    cd JarvisQA && \
    pip install -r requirements.txt

WORKDIR /app/JarvisQA/

CMD python tpdl2020_eval.py