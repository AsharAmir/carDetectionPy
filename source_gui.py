import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
import os
from PIL import Image, ImageTk
from collections import defaultdict, deque


# --- Configuration ---
class Config:
    TARGET_WIDTH = 24  # Real-world width in meters
    TARGET_HEIGHT = 249  # Real-world height in meters
    DETECTION_CONFIDENCE = 0.5  # Minimum confidence threshold
    NMS_THRESHOLD = 0.45  # Non-maximum suppression threshold
    SPEED_MEMORY = 30  # Number of frames to store for speed calculation
    MAX_SPEED_KMH = 180  # Maximum realistic speed in km/h
    VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
    SPEED_SMOOTHING_FACTOR = 0.7  # EMA smoothing factor (0-1)
    DISPLAY_SCALE = 1.0  # Scale factor for display
    MIN_FRAMES_FOR_SPEED = 5  # Minimum frames needed for reliable speed calculation
    SPEED_CALCULATION_METHOD = (
        "displacement"  # Options: "displacement", "frame-to-frame"
    )


# --- ROI Selection ---
ROI_POINTS = []  # Store ROI points from user input
show_markers = True  # Global variable to control marker visibility


# --- Vehicle Class to store tracking information ---
class Vehicle:
    def __init__(self, tracker_id, class_id, side=None, counter=0):
        self.tracker_id = tracker_id
        self.class_id = class_id
        self.positions = deque(maxlen=Config.SPEED_MEMORY)
        self.timestamps = deque(maxlen=Config.SPEED_MEMORY)
        self.speed_history = deque(maxlen=10)
        self.smoothed_speed = 0
        self.in_roi = False
        self.last_seen = time.time()
        self.side = side  # New attribute for side
        self.counter = counter  # New attribute for counter
        # Generate a random color as a tuple for easier use with OpenCV
        self.color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        self.frame_count = 0  # Count frames for this vehicle

    def update_position(self, position, timestamp, in_roi):
        self.positions.append(position)
        self.timestamps.append(timestamp)
        self.in_roi = in_roi
        self.last_seen = time.time()
        self.frame_count += 1

    def calculate_speed(self, target_height, fps):
        """Calculate vehicle speed based on position history"""
        # Need minimum number of frames for reliable calculation
        if len(self.positions) < Config.MIN_FRAMES_FOR_SPEED or not self.in_roi:
            return 0.0

        if Config.SPEED_CALCULATION_METHOD == "displacement":
            # Use first and last positions for displacement method
            y_positions = [pos[1] for pos in self.positions]
            distance_m = (
                abs(y_positions[-1] - y_positions[0])
                / target_height
                * Config.TARGET_WIDTH
            )
            time_s = (len(self.positions) - 1) / fps

        else:  # frame-to-frame method
            # Calculate average speed from frame-to-frame movements
            distances = []
            for i in range(1, len(self.positions)):
                y_diff = abs(self.positions[i][1] - self.positions[i - 1][1])
                distance_m = y_diff / target_height * Config.TARGET_WIDTH
                distances.append(distance_m)

            avg_distance = sum(distances) / len(distances) if distances else 0
            distance_m = avg_distance * len(distances)
            time_s = len(distances) / fps

        # Calculate speed in km/h
        if time_s > 0:
            # 3.6 is the conversion factor from m/s to km/h
            speed_kmh = (distance_m / time_s) * 3.6

            # Apply realistic constraints
            speed_kmh = max(5.0, min(speed_kmh, Config.MAX_SPEED_KMH))  # Minimum 5 km/h

            # Apply exponential moving average for smoothing
            if not self.speed_history:
                self.smoothed_speed = speed_kmh
            else:
                self.smoothed_speed = (
                    Config.SPEED_SMOOTHING_FACTOR * speed_kmh
                    + (1 - Config.SPEED_SMOOTHING_FACTOR) * self.smoothed_speed
                )

            self.speed_history.append(self.smoothed_speed)
            return self.smoothed_speed

        return 0.0


# --- ViewTransformer Class ---
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        if len(source) < 4 or len(target) < 4:
            raise ValueError("At least 4 points are required for transformation.")
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.array([])
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# --- GUI Application ---
class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Counter App")
        self.root.geometry("1024x768")  # Larger default window size

        # Create main container
        self.main_container = tk.Frame(root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Settings frame at the top
        self.settings_frame = tk.LabelFrame(
            self.main_container, text="Settings", padx=5, pady=5
        )
        self.settings_frame.pack(fill=tk.X, pady=5)

        # Frame size controls
        self.size_frame = tk.Frame(self.settings_frame)
        self.size_frame.pack(fill=tk.X, pady=2)

        tk.Label(self.size_frame, text="Frame Size:").pack(side=tk.LEFT, padx=5)
        self.frame_width_var = tk.StringVar(value="640")
        self.frame_width_entry = tk.Entry(
            self.size_frame, textvariable=self.frame_width_var, width=6
        )
        self.frame_width_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(self.size_frame, text="x").pack(side=tk.LEFT, padx=2)

        self.frame_height_var = tk.StringVar(value="480")
        self.frame_height_entry = tk.Entry(
            self.size_frame, textvariable=self.frame_height_var, width=6
        )
        self.frame_height_entry.pack(side=tk.LEFT, padx=2)

        # Apply button for frame size
        self.btn_apply_size = tk.Button(
            self.size_frame, text="Apply Size", command=self.apply_frame_size
        )
        self.btn_apply_size.pack(side=tk.LEFT, padx=5)

        # UDP settings
        self.udp_frame = tk.Frame(self.settings_frame)
        self.udp_frame.pack(fill=tk.X, pady=2)

        tk.Label(self.udp_frame, text="UDP Address:").pack(side=tk.LEFT, padx=5)
        self.udp_address_var = tk.StringVar(value="127.0.0.1:5000")
        self.udp_address_entry = tk.Entry(
            self.udp_frame, textvariable=self.udp_address_var, width=20
        )
        self.udp_address_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Video canvas
        self.canvas = tk.Canvas(self.main_container, width=640, height=480, bg="black")
        self.canvas.pack(pady=5)

        # Control buttons frame
        self.button_frame = tk.Frame(self.main_container)
        self.button_frame.pack(fill=tk.X, pady=5)

        # Buttons with improved layout
        self.btn_open = tk.Button(
            self.button_frame, text="Open Video", command=self.open_video
        )
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_toggle_markers = tk.Button(
            self.button_frame, text="Hide Labels", command=self.toggle_markers
        )
        self.btn_toggle_markers.pack(side=tk.LEFT, padx=5)

        self.btn_start = tk.Button(
            self.button_frame, text="Start", command=self.start_processing
        )
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = tk.Button(
            self.button_frame, text="Stop", command=self.stop_processing
        )
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.btn_save = tk.Button(
            self.button_frame, text="Save Video", command=self.save_video
        )
        self.btn_save.pack(side=tk.LEFT, padx=5)

        # Variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.show_markers = True
        self.roi_polygon = None
        self.view_transformer = None
        self.speed_estimator = None
        self.fps = 30
        self.labels_visible = True  # Flag to track label visibility

        # Initialize YOLO model
        self.model = YOLO("yolov8x.pt")
        self.tracker = sv.ByteTrack()
        self.vehicles = {}
        self.left_count = 0
        self.right_count = 0

    def open_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)

            # Set frame size based on user input
            width = int(self.frame_width_var.get())
            height = int(self.frame_height_var.get())
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.is_playing = True
            self.select_roi()
            self.process_video()

    def toggle_markers(self):
        self.labels_visible = not self.labels_visible
        self.btn_toggle_markers.config(
            text="Hide Labels" if self.labels_visible else "Show Labels"
        )

    def start_processing(self):
        if self.cap is not None:
            self.is_playing = True
            self.process_video()

    def stop_processing(self):
        self.is_playing = False

    def select_roi(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not read the first frame.")
            return

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(ROI_POINTS) < 4:
                    ROI_POINTS.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if ROI_POINTS:
                    ROI_POINTS.pop()

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", on_mouse)

        while True:
            temp_frame = frame.copy()
            if self.show_markers:
                for point in ROI_POINTS:
                    cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
                if len(ROI_POINTS) > 1:
                    for i in range(len(ROI_POINTS)):
                        pt1 = ROI_POINTS[i]
                        pt2 = ROI_POINTS[(i + 1) % len(ROI_POINTS)]
                        cv2.line(temp_frame, pt1, pt2, (0, 255, 0), 2)

            cv2.putText(
                temp_frame,
                f"Select 4 points. {len(ROI_POINTS)}/4 selected. Left-click to add, right-click to remove. Press Enter when done.",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Select ROI", temp_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if len(ROI_POINTS) >= 4:
                    break
                else:
                    messagebox.showerror("Error", "You must select at least 4 points.")
            elif key == 27:  # Escape key
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        self.roi_polygon = np.array(ROI_POINTS, dtype=np.int32)

        # Create perspective transform
        SOURCE = np.array(ROI_POINTS[:4], dtype=np.float32)
        TARGET = np.array(
            [
                [0, 0],
                [Config.TARGET_WIDTH, 0],
                [Config.TARGET_WIDTH, Config.TARGET_HEIGHT],
                [0, Config.TARGET_HEIGHT],
            ],
            dtype=np.float32,
        )
        self.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    def process_video(self):
        if not self.is_playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.is_playing = False
            return

        # Process the frame
        annotated_frame = self.process_frame(frame)

        # Convert the frame to a format Tkinter can display
        img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)

        # Update the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

        # Repeat
        self.root.after(10, self.process_video)

    def process_frame(self, frame, include_labels=True):
        # Run object detection
        result = self.model(frame, imgsz=640, conf=Config.DETECTION_CONFIDENCE)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections to include only vehicles
        mask = np.zeros(len(detections), dtype=bool)
        for i, (confidence, class_id) in enumerate(
            zip(detections.confidence, detections.class_id)
        ):
            if (
                class_id in Config.VEHICLE_CLASSES
                and confidence >= Config.DETECTION_CONFIDENCE
            ):
                mask[i] = True
        detections = detections[mask]

        # Update tracking
        detections = self.tracker.update_with_detections(detections=detections)

        # Get bottom center points of detections
        original_points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )

        # Transform points for speed calculation
        transformed_points = self.view_transformer.transform_points(
            points=original_points
        )

        # Update vehicle data and calculate speeds
        speeds = []
        for i, (tracker_id, original_point, class_id) in enumerate(
            zip(detections.tracker_id, original_points, detections.class_id)
        ):
            # Check if point is inside ROI
            in_roi = (
                cv2.pointPolygonTest(
                    self.roi_polygon, (original_point[0], original_point[1]), False
                )
                >= 0
            )

            # Determine side based on x-coordinate relative to the center of the ROI
            roi_center_x = np.mean([point[0] for point in self.roi_polygon])
            side = "L" if original_point[0] < roi_center_x else "R"

            # Create or update vehicle object
            if tracker_id not in self.vehicles:
                counter = self.left_count + 1 if side == "L" else self.right_count + 1
                self.vehicles[tracker_id] = Vehicle(tracker_id, class_id, side, counter)
                if side == "L":
                    self.left_count += 1
                else:
                    self.right_count += 1
            else:
                self.vehicles[tracker_id].update_position(
                    transformed_points[i], time.time(), in_roi
                )

            # Calculate speed for vehicles in ROI
            if in_roi:
                speed = self.vehicles[tracker_id].calculate_speed(
                    Config.TARGET_HEIGHT, self.fps
                )
                speeds.append(speed)
            else:
                speeds.append(0.0)  # No speed if not in ROI

        # Annotate the frame
        annotated_frame = frame.copy()
        labels = []
        for i in range(len(detections.tracker_id)):
            tracker_id = detections.tracker_id[i]  # Get the individual tracker_id
            vehicle = self.vehicles.get(tracker_id)
            if vehicle and vehicle.in_roi:
                labels.append(f"{vehicle.side}{vehicle.counter} {speeds[i]:.1f} km/h")
            else:
                labels.append("")  # Empty label for vehicles not in ROI

        # Ensure the number of labels matches the number of detections
        if len(labels) != len(detections):
            labels = [""] * len(detections)  # Fallback to empty labels if mismatch

        # Annotate the frame with vehicle labels if visible
        if self.labels_visible and include_labels:
            for i, label in enumerate(labels):
                if label:  # Only annotate if there is a label
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(original_points[i][0]), int(original_points[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,  # Reduced from 0.5
                        (255, 255, 255),
                        1,  # Reduced from 2
                    )

            # Vehicle count with smaller text
            cv2.putText(
                annotated_frame,
                f"Left: {self.left_count}, Right: {self.right_count}",  # Shortened text
                (10, 20),  # Moved up from (20, 40)
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,  # Reduced from 0.6
                (255, 255, 255),
                1,  # Reduced from 2
            )

        # Send data over UDP if address is provided
        if self.udp_address_var.get():
            try:
                self.send_data_udp(
                    {
                        "left_count": self.left_count,
                        "right_count": self.right_count,
                        "vehicles": [
                            {
                                "id": v.tracker_id,
                                "side": v.side,
                                "speed": v.smoothed_speed,
                                "in_roi": v.in_roi,
                            }
                            for v in self.vehicles.values()
                        ],
                    }
                )
            except Exception as e:
                print(f"UDP sending error: {e}")

        return annotated_frame

    def send_data_udp(self, data):
        """Send traffic data over UDP"""
        import socket
        import json

        try:
            # Convert all NumPy int32 to native Python int
            data = {
                "left_count": int(data["left_count"]),
                "right_count": int(data["right_count"]),
                "vehicles": [
                    {
                        "id": int(v["id"]),
                        "side": v["side"],
                        "speed": float(v["speed"]),
                        "in_roi": v["in_roi"],
                    }
                    for v in data["vehicles"]
                ],
            }

            host, port = self.udp_address_var.get().split(":")
            port = int(port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message = json.dumps(data).encode("utf-8")
            sock.sendto(message, (host, port))
        except Exception as e:
            print(f"UDP error: {e}")
        finally:
            sock.close()

    def create_view_transformer(self):
        if len(ROI_POINTS) >= 4:
            source = np.array(ROI_POINTS[:4], dtype=np.float32)
            target = np.array(
                [
                    [0, 0],
                    [Config.TARGET_WIDTH, 0],
                    [Config.TARGET_WIDTH, Config.TARGET_HEIGHT],
                    [0, Config.TARGET_HEIGHT],
                ],
                dtype=np.float32,
            )
            self.view_transformer = ViewTransformer(source=source, target=target)
        else:
            raise ValueError(
                "At least 4 ROI points are required to create the view transformer."
            )

    def apply_frame_size(self):
        try:
            width = int(self.frame_width_var.get())
            height = int(self.frame_height_var.get())

            # Update canvas size
            self.canvas.config(width=width, height=height)

            # If video is loaded, update its properties
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Adjust window size if needed
            self.root.update_idletasks()

        except ValueError:
            messagebox.showerror(
                "Error", "Please enter valid numbers for width and height"
            )

    def save_video(self):
        if self.cap is None:
            messagebox.showerror("Error", "No video is currently loaded.")
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            title="Save Video As",
        )
        if not output_path:
            return

        include_labels = messagebox.askyesno(
            "Include Labels", "Do you want to include labels in the video?"
        )

        try:
            # Get frame size
            width = int(self.frame_width_var.get())
            height = int(self.frame_height_var.get())

            # Create VideoWriter with high quality settings
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for compatibility
            writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (width, height), isColor=True
            )

            if not writer.isOpened():
                raise Exception("Could not open video writer")

            # Show progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Saving Video")
            progress_window.geometry("300x100")

            progress_label = tk.Label(
                progress_window, text="Saving video... Please wait."
            )
            progress_label.pack(pady=10)

            progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
            progress_bar.pack(fill=tk.X, padx=10)
            progress_bar.start()

            # Reset video to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process frame with or without labels
                annotated_frame = self.process_frame(frame, include_labels)

                # Ensure frame is the correct size
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))

                # Write frame
                writer.write(annotated_frame)
                frame_count += 1

                # Update progress periodically
                if frame_count % 30 == 0:
                    progress_label.config(text=f"Processed {frame_count} frames...")
                    progress_window.update()

            # Clean up
            writer.release()
            progress_window.destroy()

            # Try to ensure the video file is properly closed
            cv2.destroyAllWindows()

            messagebox.showinfo(
                "Success", f"Video saved successfully!\nTotal frames: {frame_count}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save video: {str(e)}")
            if "writer" in locals():
                writer.release()


# --- Main Function ---
if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.mainloop()
