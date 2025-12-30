import os
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox, Label, Button
from PIL import Image, ImageTk
import pickle

from hand_detector import HandDetector

class DataCollectionApp:
    def __init__(self, root):
        """
        Initialize the data collection application
        
        Args:
            root (tk.Tk): Tkinter root window
        """
        self.root = root
        self.root.title("Sign Sense - Data Collection")
        self.root.geometry("800x600")
        
        # Initialize variables
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.current_letter = "A"
        self.sample_count = 0
        self.max_samples = 100
        self.collecting = False
        self.countdown = 0
        
        # Initialize hand detector
        self.detector = HandDetector(detection_confidence=0.7)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            self.root.destroy()
            return
        
        # Create UI elements
        self.create_widgets()
        
        # Start video loop
        self.update_frame()
        
        # Set up closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create the UI elements"""
        # Frame for video feed
        self.video_frame = tk.Frame(self.root, width=640, height=480)
        self.video_frame.pack(pady=10)
        
        self.video_label = Label(self.video_frame)
        self.video_label.pack()
        
        # Frame for controls
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        # Letter selection
        self.letter_label = Label(self.control_frame, text=f"Current Letter: {self.current_letter}", font=("Arial", 16))
        self.letter_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Sample count
        self.count_label = Label(self.control_frame, text=f"Samples: {self.sample_count}/{self.max_samples}", font=("Arial", 16))
        self.count_label.grid(row=0, column=1, padx=10, pady=10)
        
        # Buttons
        self.prev_btn = Button(self.control_frame, text="Previous Letter", command=self.prev_letter)
        self.prev_btn.grid(row=1, column=0, padx=10, pady=10)
        
        self.next_btn = Button(self.control_frame, text="Next Letter", command=self.next_letter)
        self.next_btn.grid(row=1, column=1, padx=10, pady=10)
        
        self.collect_btn = Button(self.control_frame, text="Start Collecting", command=self.toggle_collection)
        self.collect_btn.grid(row=1, column=2, padx=10, pady=10)
        
        # Status message
        self.status_label = Label(self.root, text="Position your hand in the frame and click 'Start Collecting'", font=("Arial", 12))
        self.status_label.pack(pady=10)
    
    def update_frame(self):
        """Update the video frame and process hand detection"""
        ret, frame = self.cap.read()
        if ret:
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            frame, hands = self.detector.find_hands(frame)
            
            # If collecting data and hands are detected
            if self.collecting and hands and self.countdown <= 0:
                # Extract features
                features = self.detector.extract_features(hands[0]["landmarks"])
                
                if features is not None:
                    # Save features
                    self.save_sample(features)
                    
                    # Update sample count
                    self.sample_count += 1
                    self.count_label.config(text=f"Samples: {self.sample_count}/{self.max_samples}")
                    
                    # Check if we've collected enough samples
                    if self.sample_count >= self.max_samples:
                        self.collecting = False
                        self.collect_btn.config(text="Start Collecting")
                        messagebox.showinfo("Complete", f"Collected {self.max_samples} samples for letter {self.current_letter}")
            
            # Display countdown if active
            if self.countdown > 0:
                cv2.putText(frame, str(self.countdown), (frame.shape[1]//2, frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                
                # Update countdown every second
                current_time = time.time()
                if current_time - self.last_countdown_time >= 1:
                    self.countdown -= 1
                    self.last_countdown_time = current_time
            
            # Display current letter
            cv2.putText(frame, f"Letter: {self.current_letter}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to RGB for tkinter
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the video label
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        
        # Schedule the next update
        self.root.after(10, self.update_frame)
    
    def save_sample(self, features):
        """
        Save a sample of hand features for the current letter
        
        Args:
            features (numpy.ndarray): Feature vector from hand landmarks
        """
        # Create directory for the letter if it doesn't exist
        letter_dir = os.path.join(self.data_dir, self.current_letter)
        os.makedirs(letter_dir, exist_ok=True)
        
        # Save the features
        sample_path = os.path.join(letter_dir, f"sample_{self.sample_count}.pkl")
        with open(sample_path, 'wb') as f:
            pickle.dump(features, f)
    
    def toggle_collection(self):
        """Toggle data collection on/off"""
        if self.collecting:
            self.collecting = False
            self.collect_btn.config(text="Start Collecting")
            self.status_label.config(text="Collection paused")
        else:
            # Reset sample count if changing letter
            self.sample_count = 0
            self.count_label.config(text=f"Samples: {self.sample_count}/{self.max_samples}")
            
            # Start countdown
            self.countdown = 3
            self.last_countdown_time = time.time()
            self.status_label.config(text="Get ready! Collection will start after countdown")
            
            # Start collecting after countdown
            self.collecting = True
            self.collect_btn.config(text="Stop Collecting")
    
    def next_letter(self):
        """Switch to the next letter in the alphabet"""
        # Stop collection if active
        if self.collecting:
            self.toggle_collection()
        
        # Get the next letter
        current_ord = ord(self.current_letter)
        if current_ord < ord('Z'):
            self.current_letter = chr(current_ord + 1)
        
        # Update UI
        self.letter_label.config(text=f"Current Letter: {self.current_letter}")
        self.sample_count = 0
        self.count_label.config(text=f"Samples: {self.sample_count}/{self.max_samples}")
    
    def prev_letter(self):
        """Switch to the previous letter in the alphabet"""
        # Stop collection if active
        if self.collecting:
            self.toggle_collection()
        
        # Get the previous letter
        current_ord = ord(self.current_letter)
        if current_ord > ord('A'):
            self.current_letter = chr(current_ord - 1)
        
        # Update UI
        self.letter_label.config(text=f"Current Letter: {self.current_letter}")
        self.sample_count = 0
        self.count_label.config(text=f"Samples: {self.sample_count}/{self.max_samples}")
    
    def on_closing(self):
        """Clean up resources when closing the application"""
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectionApp(root)
    root.mainloop()