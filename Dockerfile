FROM python:3.9.6

ADD requirements.txt /

RUN pip install -r /requirements.txt

ADD camel_agent.py /
ADD vector_store.py / 
ADD chats / 

ENV PYTHONUNBUFFERED=1

CMD [ "python", "./camel_agent.py" ]