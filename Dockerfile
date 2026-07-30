FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt app/requirements.txt

RUN pip install -r app/requirements.txt --break-system-packages && \
    pip uninstall -y --break-system-packages opencv-python opencv-python-headless && \
    pip install --break-system-packages opencv-python-headless==5.0.0.93

COPY . .

EXPOSE 5000

CMD ["python", "handler.py"]