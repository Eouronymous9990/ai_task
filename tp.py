import tkinter as tk
from tkinter import ttk, font
import threading
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


class TechVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tech Vision Tools - Dark Mode")
        self.root.geometry("800x600")
        self.root.configure(bg='#121212')

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure('.', background='#121212', foreground='white')
        self.style.configure('TFrame', background='#121212')
        self.style.configure('TLabel', background='#121212', foreground='#00ffaa', font=('Helvetica', 14, 'bold'))
        self.style.configure('TButton', background='#252525', foreground='white',
                             font=('Helvetica', 12), borderwidth=1)
        self.style.map('TButton',
                       background=[('active', '#353535'), ('pressed', '#454545')],
                       foreground=[('active', '#00ffaa')])

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        title_font = font.Font(family='Helvetica', size=18, weight='bold')
        ttk.Label(self.main_frame, text="TECH VISION TOOLS",
                  font=title_font, anchor='center').pack(pady=(0, 20))

        self.btn_frame = ttk.Frame(self.main_frame)
        self.btn_frame.pack(expand=True, fill=tk.BOTH)

        options = [
            ("🎨 Hand Drawing", self.run_hand_drawing),
            ("✋ Hand Tracking", self.run_hand_tracking),
            ("🧍 Pose Estimation", self.run_pose_estimation),
            ("🖼️ Segmentation", self.run_segmentation),
            ("🔍 Object Detection", self.run_object_detection)
        ]

        for text, cmd in options:
            btn = ttk.Button(self.btn_frame, text=text, command=cmd, style='TButton')
            btn.pack(fill=tk.X, pady=8, ipady=10)

        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=(20, 0))

        self.status_var = tk.StringVar(value="Ready to select tool")
        ttk.Label(self.status_frame, textvariable=self.status_var,
                  font=('Helvetica', 11), foreground='#00aaff').pack(side=tk.LEFT)

        exit_btn = ttk.Button(self.main_frame, text="Exit", command=self.on_close,
                              style='TButton')
        exit_btn.pack(pady=(20, 0), ipady=5)

        self.running = False
        self.current_process = None

    def on_close(self):
        if self.running:
            self.stop_process()
        self.root.destroy()

    def stop_process(self):
        self.running = False
        if self.current_process and self.current_process.is_alive():
            self.current_process.join(timeout=1)
        cv2.destroyAllWindows()

    def run_in_thread(self, target):
        self.stop_process()
        self.running = True
        self.current_process = threading.Thread(target=target)
        self.current_process.daemon = True
        self.current_process.start()

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update()

    def run_hand_drawing(self):
        self.run_in_thread(self._hand_drawing)

    def _hand_drawing(self):
        self.update_status("Running Hand Drawing - Press 'Q' to close")

        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            mp_draw = mp.solutions.drawing_utils

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.update_status("Error: Could not open camera")
                return

            canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

            colors = [
                (0, 0, 0),  # أسود
                (0, 0, 255),  # أحمر
                (0, 255, 0),  # أخضر
                (255, 0, 0),  # أزرق
                (0, 255, 255),  # أصفر
                (255, 0, 255),  # أرجواني
                (255, 255, 0),  # سماوي
                (128, 0, 0),  # أزرق داكن
                (0, 128, 0),  # أخضر داكن
                (0, 0, 128),  # أحمر داكن
                (128, 128, 128),  # رمادي
                (255, 255, 255)  # أبيض (ممحاة)
            ]

            color_palette_height = 60
            color_palette = np.ones((color_palette_height, 640, 3), dtype=np.uint8) * 255

            color_width = 640 // len(colors)
            for i, color in enumerate(colors):
                color_palette[:, i * color_width:(i + 1) * color_width] = color[::-1]

            for i in range(1, len(colors)):
                cv2.line(color_palette, (i * color_width, 0), (i * color_width, color_palette_height), (0, 0, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color_names = ["Black", "Red", "Green", "Blue", "Yellow",
                           "Purple", "Cyan", "D.Blue", "D.Green", "D.Red",
                           "Gray", "Eraser"]

            for i, name in enumerate(color_names):
                text_size = cv2.getTextSize(name, font, 0.4, 1)[0]
                text_x = i * color_width + (color_width - text_size[0]) // 2
                text_y = color_palette_height // 2 + text_size[1] // 2
                cv2.putText(color_palette, name, (text_x, text_y),
                            font, 0.4, (0, 0, 0) if colors[i][0] > 127 else (255, 255, 255), 1)

            prev_x, prev_y = 0, 0
            drawing = False
            current_color = (0, 0, 255)
            brush_size = 5
            hand_overlay = np.zeros((480, 640, 3), dtype=np.uint8)

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status("Error: Failed to capture frame")
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_overlay.fill(0)

                results = hands.process(frame_rgb)
                finger_x, finger_y = 0, 0

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            hand_overlay,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                            connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )

                        index_finger_tip = hand_landmarks.landmark[8]
                        finger_x = int(index_finger_tip.x * w)
                        finger_y = int(index_finger_tip.y * h)

                        if finger_y < color_palette_height:
                            color_index = min(finger_x // color_width, len(colors) - 1)
                            current_color = colors[color_index]
                            drawing = False
                        else:
                            if prev_x == 0 and prev_y == 0:
                                prev_x, prev_y = finger_x, finger_y

                            if drawing:
                                cv2.line(canvas, (prev_x, prev_y - color_palette_height),
                                         (finger_x, finger_y - color_palette_height),
                                         current_color[::-1], brush_size)

                            prev_x, prev_y = finger_x, finger_y

                        cv2.circle(hand_overlay, (finger_x, finger_y), brush_size + 5, (0, 255, 255), -1)
                        cv2.circle(hand_overlay, (finger_x, finger_y), brush_size, (0, 0, 0), -1)

                display = np.zeros((480 + color_palette_height, 640, 3), dtype=np.uint8)
                display[:color_palette_height, :] = color_palette
                display[color_palette_height:, :] = canvas

                mask = cv2.cvtColor(hand_overlay, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                hand_region = display[color_palette_height:, :]
                hand_region[mask > 0] = hand_overlay[mask > 0]

                cv2.rectangle(display, (5, 5), (30, 30), current_color[::-1], -1)
                cv2.rectangle(display, (5, 5), (30, 30), (0, 0, 0), 2)

                cv2.putText(display, f"Size: {brush_size}", (50, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                cv2.imshow("Hand Drawing - Keys: D=Draw, C=Clear, Q=Quit", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    drawing = not drawing
                elif key == ord('c'):
                    canvas[:] = 255
                elif key == ord('+'):
                    brush_size = min(brush_size + 2, 30)
                elif key == ord('-'):
                    brush_size = max(brush_size - 2, 1)
                elif key == ord('q'):
                    break

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.update_status("Ready to select tool")

    def run_hand_tracking(self):
        self.run_in_thread(self._hand_tracking)

    def _hand_tracking(self):
        self.update_status("Running Hand Tracking - Press 'Q' to close")

        try:
            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils

            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.update_status("Error: Could not open camera")
                return

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status("Error: Failed to capture frame")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3),
                            connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )

                cv2.imshow("Hand Tracking", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.update_status("Ready to select tool")

    def run_pose_estimation(self):
        self.run_in_thread(self._pose_estimation)

    def _pose_estimation(self):
        self.update_status("Running Pose Estimation - Press 'Q' to close")

        try:
            model = YOLO('yolov8n-pose.pt')
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.update_status("Error: Could not open camera")
                return

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status("Error: Failed to capture frame")
                    break

                results = model(frame, stream=True, verbose=False)
                for r in results:
                    annotated_frame = r.plot()
                    cv2.imshow("Pose Estimation", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.update_status("Ready to select tool")

    def run_segmentation(self):
        self.run_in_thread(self._segmentation)

    def _segmentation(self):
        self.update_status("Running Segmentation - Press 'Q' to close")

        try:
            model = YOLO('yolov8n-seg.pt')
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.update_status("Error: Could not open camera")
                return

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status("Error: Failed to capture frame")
                    break

                results = model(frame, stream=True, verbose=False)
                for r in results:
                    annotated_frame = r.plot()
                    cv2.imshow("Segmentation", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.update_status("Ready to select tool")

    def run_object_detection(self):
        self.run_in_thread(self._object_detection)

    def _object_detection(self):
        self.update_status("Running Object Detection - Press 'Q' to close")

        try:
            model = YOLO('yolov8n.pt')
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.update_status("Error: Could not open camera")
                return

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.update_status("Error: Failed to capture frame")
                    break

                results = model(frame, stream=True)
                for r in results:
                    annotated_frame = r.plot()
                    cv2.imshow("Object Detection", annotated_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        except Exception as e:
            self.update_status(f"Error: {str(e)}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            self.update_status("Ready to select tool")


if __name__ == "__main__":
    root = tk.Tk()
    app = TechVisionApp(root)
    root.mainloop()