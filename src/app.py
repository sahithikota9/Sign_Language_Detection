import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time

from hand_detector import HandDetector
from model import SignLanguageModel

class SignSenseApp:
    def __init__(self, root):
        """
        Initialize the Sign Sense application
        
        Args:
            root (tk.Tk): Tkinter root window
        """
        self.root = root
        self.root.title("HIT-5 - Sign Language Recognition")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_path = os.path.join(self.models_dir, "sign_language_model.h5")
        self.encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
        
        self.detector = HandDetector(detection_confidence=0.7)
        self.model = None
        self.load_model()
        
        self.prediction = None
        self.confidence = 0
        self.last_predictions = []  # Store recent predictions for smoothing
        self.prediction_history = []  # Store prediction history for display
        self.max_history = 10
        
        # Camera settings
        self.camera_id = 0
        self.cap = None
        self.open_camera()
        
        # UI settings
        self.confidence_threshold = 0.7
        self.show_landmarks = True
        self.mirror_view = True
        
        # Create UI elements
        self.create_widgets()
        
        # Start video loop
        self.update_frame()
        
        # Set up closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        """Load the trained sign language recognition model"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
                self.model = SignLanguageModel(self.model_path)
                self.model.load_model(self.model_path, self.encoder_path)
                print("Model loaded successfully")
            else:
                print("Model files not found. Please train the model first.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def open_camera(self):
        """Open the camera for video capture"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return False
        
        return True
    
    def create_widgets(self):
        """Create the UI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left frame for video and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)
        
        # Video frame
        self.video_frame = ttk.Frame(left_frame, width=640, height=480)
        self.video_frame.pack(pady=10)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Controls frame
        controls_frame = ttk.LabelFrame(left_frame, text="Controls")
        controls_frame.pack(fill="x", pady=10, padx=5)
        
        # Camera selection
        ttk.Label(controls_frame, text="Camera:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(controls_frame, textvariable=self.camera_var, values=["0", "1", "2"], width=5)
        camera_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        camera_combo.bind("<<ComboboxSelected>>", self.change_camera)
        
        # Confidence threshold
        ttk.Label(controls_frame, text="Confidence Threshold:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.threshold_var = tk.DoubleVar(value=self.confidence_threshold)
        threshold_scale = ttk.Scale(controls_frame, from_=0.1, to=0.9, variable=self.threshold_var, 
                                   orient="horizontal", length=100, command=self.update_threshold)
        threshold_scale.grid(row=0, column=3, padx=5, pady=5)
        self.threshold_label = ttk.Label(controls_frame, text=f"{self.confidence_threshold:.1f}")
        self.threshold_label.grid(row=0, column=4, padx=5, pady=5)
        
        # Show landmarks checkbox
        self.landmarks_var = tk.BooleanVar(value=self.show_landmarks)
        landmarks_check = ttk.Checkbutton(controls_frame, text="Show Hand Landmarks", 
                                         variable=self.landmarks_var, command=self.toggle_landmarks)
        landmarks_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Mirror view checkbox
        self.mirror_var = tk.BooleanVar(value=self.mirror_view)
        mirror_check = ttk.Checkbutton(controls_frame, text="Mirror View", 
                                      variable=self.mirror_var, command=self.toggle_mirror)
        mirror_check.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="w")
        
        # Right frame for prediction display
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side="right", fill="both", padx=10)
        
        # Current prediction display
        prediction_frame = ttk.LabelFrame(right_frame, text="Current Prediction")
        prediction_frame.pack(fill="x", pady=10)
        
        self.prediction_label = ttk.Label(prediction_frame, text="?", font=("Arial", 120))
        self.prediction_label.pack(pady=20)
        
        self.confidence_label = ttk.Label(prediction_frame, text="Confidence: 0%")
        self.confidence_label.pack(pady=5)
        
        # Prediction history
        history_frame = ttk.LabelFrame(right_frame, text="Prediction History")
        history_frame.pack(fill="both", expand=True, pady=10)
        
        self.history_text = tk.Text(history_frame, height=10, width=20, state="disabled")
        self.history_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
        # Check if model is loaded
        if self.model is None:
            self.status_var.set("Warning: No model loaded. Please train a model first.")
            messagebox.showwarning("No Model", "No trained model found. Please run train_model.py first.")
    
    def update_frame(self):
        """Update the video frame and process hand detection"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Mirror the frame if enabled
                if self.mirror_view:
                    frame = cv2.flip(frame, 1)
                
                # Detect hands
                frame, hands = self.detector.find_hands(frame, draw=self.show_landmarks)
                
                # Process for prediction if hands are detected and model is loaded
                if hands and self.model is not None:
                    # Extract features
                    features = self.detector.extract_features(hands[0]["landmarks"])
                    
                    if features is not None:
                        # Get prediction
                        letter, conf = self.model.predict(features)
                        
                        # Add to recent predictions for smoothing
                        self.last_predictions.append((letter, conf))
                        if len(self.last_predictions) > 5:  # Keep only last 5 predictions
                            self.last_predictions.pop(0)
                        
                        # Get most common prediction from recent history
                        if len(self.last_predictions) >= 3:
                            # Count occurrences of each letter
                            letter_counts = {}
                            for l, c in self.last_predictions:
                                if c >= self.confidence_threshold:  # Only count high confidence predictions
                                    letter_counts[l] = letter_counts.get(l, 0) + 1
                            
                            # Get the most common letter
                            if letter_counts:
                                most_common = max(letter_counts.items(), key=lambda x: x[1])
                                if most_common[1] >= 3:  # If it appears at least 3 times
                                    letter = most_common[0]
                                    # Get average confidence for this letter
                                    conf = np.mean([c for l, c in self.last_predictions if l == letter])
                        
                        # Update prediction if confidence is above threshold
                        if conf >= self.confidence_threshold:
                            if self.prediction != letter:
                                self.prediction = letter
                                self.add_to_history(letter, conf)
                            self.confidence = conf
                        
                        # Display prediction on frame
                        cv2.putText(frame, f"{letter} ({conf:.2f})", (10, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update the video label
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Update prediction display
                self.update_prediction_display()
        
        # Schedule the next update
        self.root.after(10, self.update_frame)
    
    def update_prediction_display(self):
        """Update the prediction display"""
        if self.prediction:
            self.prediction_label.config(text=self.prediction)
            self.confidence_label.config(text=f"Confidence: {self.confidence:.2f}")
        else:
            self.prediction_label.config(text="?")
            self.confidence_label.config(text="Confidence: 0.00")
    
    def add_to_history(self, letter, confidence):
        """
        Add a prediction to the history
        
        Args:
            letter (str): Predicted letter
            confidence (float): Prediction confidence
        """
        # Add to history list
        timestamp = time.strftime("%H:%M:%S")
        self.prediction_history.append((timestamp, letter, confidence))
        
        # Limit history size
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # Update history display
        self.history_text.config(state="normal")
        self.history_text.delete(1.0, tk.END)
        
        for ts, l, c in reversed(self.prediction_history):
            self.history_text.insert(tk.END, f"{ts}: {l} ({c:.2f})\n")
        
        self.history_text.config(state="disabled")
    
    def change_camera(self, event=None):
        """Change the camera source"""
        try:
            new_id = int(self.camera_var.get())
            if new_id != self.camera_id:
                self.camera_id = new_id
                self.open_camera()
        except ValueError:
            pass
    
    def update_threshold(self, event=None):
        """Update the confidence threshold"""
        self.confidence_threshold = self.threshold_var.get()
        self.threshold_label.config(text=f"{self.confidence_threshold:.1f}")
    
    def toggle_landmarks(self):
        """Toggle showing hand landmarks"""
        self.show_landmarks = self.landmarks_var.get()
    
    def toggle_mirror(self):
        """Toggle mirror view"""
        self.mirror_view = self.mirror_var.get()
    
    def on_closing(self):
        """Clean up resources when closing the application"""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignSenseApp(root)
    root.mainloop()