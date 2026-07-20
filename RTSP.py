import cv2
import threading
from typing import Tuple
import os
class RTSPCameraStream:
    def __init__(self, user: str, password: str, ip: str, port: int, stream_path: str):
        self.url = os.getenv("URL")
        self.cap = cv2.VideoCapture(self.url)
        self.is_running = False
        self.lock = threading.Lock()
        self.frame: Tuple[bool,cv2.Mat] | None
    def start(self) -> None:
        self.is_running = True
        thread = threading.Thread(target=self._update_frame, args=())
        thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame
            return False, None
        
    def main():
        user = os.getenv("USER")
        password = os.getenv("PASSWORD")
        ip = os.getenv("IP")
        port = os.getenv("PORT")
        stream_path = os.getenv("STREAM_PATH")

        camera = RTSPCameraStream(user,password, ip, port, stream_path)
        camera.start()

        try:
            while True:
                ret, frame = camera.read()
                if ret:
                    cv2.imshow("RTSP CAMERA STREAM:", frame)
                    if cv2.waitKey(1) & 0XFF == ord("q"):
                        break
                    else:
                        print("Failed To Get Frame")
                    
        finally:
            camera.stop()
            cv2.destroyAllWindows()
   
    if __name__ == "__main__":
        main()