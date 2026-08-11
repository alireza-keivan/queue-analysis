FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt app/requirements.txt

RUN pip install -r app/requirements.txt --break-system-packages && \
    pip uninstall -y --break-system-packages opencv-python opencv-python-headless && \
    pip install --break-system-packages opencv-python-headless==5.0.0.93

# Bake the model weights into the image. They are not in the repo, so without
# this every cold worker downloads them from Ultralytics on its first job -
# paid for at GPU rate, on the critical path, and dependent on an external
# host being up. Must match MODEL in app/queue.yaml, and lands in /app so the
# relative path in that config resolves at runtime.
RUN python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')"

COPY . .

EXPOSE 5000

CMD ["python", "handler.py"]