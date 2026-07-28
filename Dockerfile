FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt app/requirements.txt

RUN pip install -r app/requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app/queue-management.py"]