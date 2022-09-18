FROM python:3.7

RUN mkdir /ivy
WORKDIR /ivy

COPY requirements.txt /ivy/
RUN pip install -r requirements.txt

COPY . /ivy/

CMD python -m main
