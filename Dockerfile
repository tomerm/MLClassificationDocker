FROM tiangolo/meinheld-gunicorn:python3.6

LABEL maintainer="Alex Shensis"

RUN apt-get update && apt-get install -y openjdk-8-jdk

#RUN find /usr/lib/jvm/java-8-openjdk-amd64

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

#RUN pip install flask
#RUN pip install jsonschema
RUN pip install torch torchvision
ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN python -m nltk.downloader stopwords

COPY ./app /app
