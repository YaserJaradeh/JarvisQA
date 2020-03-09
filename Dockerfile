FROM python:3

RUN apt-get install -y git
RUN mkdir -p /app/JarvisQA
ADD . /app/JarvisQA
RUN cd /app/JarvisQA && \
    pip install -r requirements.txt

WORKDIR /app/JarvisQA

CMD ["/bin/bash"]