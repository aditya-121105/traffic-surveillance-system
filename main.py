import cv2
import time
import os
import torch
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


class TrafficSurveillanceSystem:

    def __init__(self, video_path="input/tf_video.mp4"):
        # ---------------- SETUP ----------------
        os.makedirs("output", exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using compute device: {self.device}")

        self.model = YOLO("yolov8m.pt")
        self.model.to(self.device)

        self.cap = cv2.VideoCapture(video_path)

        # ---------------- GLOBALS ----------------
        self.roi_points = []

        self.track_history = defaultdict(list)
        self.inside_roi_ids = set()
        self.counted_ids = set()

        self.counts_total = {
            "heavy": 0,
            "passenger": 0,
            "two_wheeler": 0
        }

        self.roi_polygon = None
        self.out = None
        self.prev_time = time.time()

    # ---------------- DRAW ROI ----------------
    def draw_polygon(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))

    # ---------------- ROI SELECTION ----------------
    def select_roi(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (1500, 800))

        cv2.namedWindow("Draw ROI")
        cv2.setMouseCallback("Draw ROI", self.draw_polygon)

        print("👉 Click multiple points to draw ROI, press ENTER")

        while True:
            temp = frame.copy()

            for p in self.roi_points:
                cv2.circle(temp, p, 5, (0, 0, 255), -1)

            if len(self.roi_points) > 1:
                cv2.polylines(temp, [np.array(self.roi_points)], False, (255, 0, 0), 2)

            cv2.imshow("Draw ROI", temp)

            if cv2.waitKey(1) == 13:
                break

        cv2.destroyWindow("Draw ROI")
        self.roi_polygon = np.array(self.roi_points)

    # ---------------- VIDEO WRITER ----------------
    def setup_writer(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 20

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = f"output/result_{int(time.time())}.avi"

        self.out = cv2.VideoWriter(output_path, fourcc, fps, (1500, 800))

        if not self.out.isOpened():
            print("❌ VideoWriter failed")

        print(f"✅ Saving video to: {output_path}")

    # ---------------- PROCESS FRAME ----------------
    def process(self):
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 2 != 0:
                continue

            frame = cv2.resize(frame, (1500, 800))
            height, width, _ = frame.shape

            # Draw ROI
            cv2.polylines(frame, [self.roi_polygon], True, (255, 0, 0), 2)

            results = self.model.track(
                frame,
                persist=True,
                conf=0.3,
                imgsz=1280,
                iou=0.5,
                verbose=False
            )

            current_ids_in_frame = set()
            current_counts = {"heavy": 0, "passenger": 0, "two_wheeler": 0}

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for box, track_id, cls in zip(boxes, track_ids, classes):

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    cls = int(cls)
                    area = (x2 - x1) * (y2 - y1)

                    # Class mapping
                    if cls in [5, 7] and area > 15000:
                        label = "Heavy"
                        key = "heavy"
                        color = (0, 0, 255)

                    elif cls == 2:
                        label = "Passenger"
                        key = "passenger"
                        color = (255, 0, 0)

                    elif cls == 3 and area > 1000:
                        label = "TwoWheeler"
                        key = "two_wheeler"
                        color = (0, 255, 255)

                    else:
                        continue

                    inside = cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False) >= 0

                    if inside:
                        current_ids_in_frame.add(track_id)
                        self.inside_roi_ids.add(track_id)
                        current_counts[key] += 1

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label} ID:{int(track_id)}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    self.track_history[track_id].append(key)

            # EXIT COUNT
            exited_ids = self.inside_roi_ids - current_ids_in_frame

            for tid in exited_ids:
                if tid not in self.counted_ids:
                    self.counted_ids.add(tid)

                    if len(self.track_history[tid]) > 0:
                        last_class = self.track_history[tid][-1]
                        self.counts_total[last_class] += 1

                self.inside_roi_ids.remove(tid)

            # DISPLAY CURRENT
            cv2.putText(frame, "Current Vehicles:", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            y = 90
            for k, v in current_counts.items():
                cv2.putText(frame, f"{k}: {v}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                y += 50

            # DISPLAY TOTAL
            cv2.putText(frame, "Total Vehicles Passed:", (1000, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            y = 90
            for k, v in self.counts_total.items():
                cv2.putText(frame, f"{k}: {v}", (1000, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                y += 50

            # FPS
            curr_time = time.time()
            fps_display = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            cv2.putText(frame, f"FPS: {int(fps_display)}", (10, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # SAVE + SHOW
            self.out.write(frame)
            cv2.imshow("Traffic Surveillance System", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # ---------------- RUN ----------------
    def run(self):
        self.select_roi()
        self.setup_writer()
        self.process()

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    import sys
    video_source = sys.argv[1] if len(sys.argv) > 1 else "input/tf_video.mp4"
    system = TrafficSurveillanceSystem(video_source)
    system.run()