FROM python:3.9

WORKDIR /app

ADD ./req.txt /app

RUN pip install -r req.txt

ADD ./model /app

CMD ["python", "run_training_process.py"]