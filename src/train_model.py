import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox, Label, Button, Frame, StringVar, IntVar, Scale, HORIZONTAL

from model import SignLanguageModel

class ModelTrainingApp:
    def __init__(self, root):
        """
        Initialize the model training application
        
        Args:
            root (tk.Tk): Tkinter root window
        """
        self.root = root
        self.root.title("Sign Sense - Model Training")
        self.root.geometry("600x500")
        
        # Initialize variables
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.epochs = IntVar(value=50)
        self.batch_size = IntVar(value=32)
        self.validation_split = IntVar(value=20)  # 20%
        self.status_text = StringVar(value="Ready to train")
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        """Create the UI elements"""
        # Title
        title_label = Label(self.root, text="Sign Language Recognition Model Training", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Training parameters frame
        params_frame = Frame(self.root)
        params_frame.pack(pady=10, fill="x", padx=20)
        
        # Epochs
        Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="w", pady=5)
        Scale(params_frame, from_=10, to=200, orient=HORIZONTAL, variable=self.epochs, 
              length=300).grid(row=0, column=1, sticky="ew", pady=5)
        
        # Batch size
        Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", pady=5)
        Scale(params_frame, from_=8, to=128, orient=HORIZONTAL, variable=self.batch_size, 
              length=300).grid(row=1, column=1, sticky="ew", pady=5)
        
        # Validation split
        Label(params_frame, text="Validation Split (%):").grid(row=2, column=0, sticky="w", pady=5)
        Scale(params_frame, from_=10, to=40, orient=HORIZONTAL, variable=self.validation_split, 
              length=300).grid(row=2, column=1, sticky="ew", pady=5)
        
        # Buttons frame
        button_frame = Frame(self.root)
        button_frame.pack(pady=20)
        
        # Load data button
        self.load_btn = Button(button_frame, text="Load Training Data", command=self.load_data, width=20)
        self.load_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Train model button
        self.train_btn = Button(button_frame, text="Train Model", command=self.train_model, width=20, state="disabled")
        self.train_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Status label
        status_label = Label(self.root, textvariable=self.status_text, font=("Arial", 12))
        status_label.pack(pady=10)
        
        # Results frame
        self.results_frame = Frame(self.root)
        self.results_frame.pack(pady=10, fill="both", expand=True, padx=20)
        
        # Initial status message
        self.status_text.set("Click 'Load Training Data' to begin")
    
    def load_data(self):
        """Load and prepare the training data"""
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_dir):
                messagebox.showerror("Error", "Data directory not found. Please collect data first.")
                return
            
            # Initialize lists for features and labels
            features = []
            labels = []
            
            # Set status
            self.status_text.set("Loading data...")
            self.root.update()
            
            # Loop through each letter directory
            letter_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            
            if not letter_dirs:
                messagebox.showerror("Error", "No data found. Please collect data first.")
                self.status_text.set("No data found")
                return
            
            for letter in letter_dirs:
                letter_dir = os.path.join(self.data_dir, letter)
                
                # Get all sample files
                sample_files = [f for f in os.listdir(letter_dir) if f.endswith('.pkl')]
                
                for sample_file in sample_files:
                    # Load the sample
                    with open(os.path.join(letter_dir, sample_file), 'rb') as f:
                        feature = pickle.load(f)
                    
                    # Add to lists
                    features.append(feature)
                    labels.append(letter)
            
            # Convert to numpy arrays
            self.X = np.array(features)
            self.y = np.array(labels)
            
            # Split into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
            )
            
            # Update status
            self.status_text.set(f"Data loaded: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
            
            # Enable train button
            self.train_btn.config(state="normal")
            
            # Show data summary
            self.show_data_summary()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_text.set("Error loading data")
    
    def show_data_summary(self):
        """Display a summary of the loaded data"""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Count samples per letter
        unique_labels, counts = np.unique(self.y, return_counts=True)
        
        # Display summary
        summary_text = "Data Summary:\n\n"
        for label, count in zip(unique_labels, counts):
            summary_text += f"Letter {label}: {count} samples\n"
        
        summary_label = Label(self.results_frame, text=summary_text, justify="left", font=("Arial", 10))
        summary_label.pack(pady=10, anchor="w")
    
    def train_model(self):
        """Train the sign language recognition model"""
        try:
            # Disable buttons during training
            self.load_btn.config(state="disabled")
            self.train_btn.config(state="disabled")
            
            # Update status
            self.status_text.set("Initializing model...")
            self.root.update()
            
            # Initialize model
            model = SignLanguageModel()
            
            # Get training parameters
            epochs = self.epochs.get()
            batch_size = self.batch_size.get()
            validation_split = self.validation_split.get() / 100.0
            
            # Update status
            self.status_text.set(f"Training model with {epochs} epochs, batch size {batch_size}...")
            self.root.update()
            
            # Train the model
            history = model.train(
                self.X_train, self.y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Evaluate on test set
            y_test_encoded = model.label_encoder.transform(self.y_test)
            y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded)
            test_loss, test_acc = model.model.evaluate(self.X_test, y_test_categorical)
            
            # Save the model
            model_path = os.path.join(self.models_dir, "sign_language_model.h5")
            encoder_path = os.path.join(self.models_dir, "label_encoder.pkl")
            model.save_model(model_path, encoder_path)
            
            # Update status
            self.status_text.set(f"Model trained and saved! Test accuracy: {test_acc:.2f}")
            
            # Show training results
            self.show_training_results(history, test_acc)
            
            # Re-enable buttons
            self.load_btn.config(state="normal")
            self.train_btn.config(state="normal")
            
            # Show success message
            messagebox.showinfo("Success", f"Model trained successfully with {test_acc:.2f} test accuracy and saved to {model_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_text.set("Error training model")
            
            # Re-enable buttons
            self.load_btn.config(state="normal")
            self.train_btn.config(state="normal")
    
    def show_training_results(self, history, test_acc):
        """
        Display training results
        
        Args:
            history: Training history object
            test_acc (float): Test accuracy
        """
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Create a summary text
        summary_text = f"Training Results:\n\n"
        summary_text += f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n"
        summary_text += f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n"
        summary_text += f"Test accuracy: {test_acc:.4f}\n\n"
        summary_text += f"Model saved to: {os.path.join(self.models_dir, 'sign_language_model.h5')}"
        
        summary_label = Label(self.results_frame, text=summary_text, justify="left", font=("Arial", 10))
        summary_label.pack(pady=10, anchor="w")

if __name__ == "__main__":
    # Import tensorflow here to avoid loading it before it's needed
    import tensorflow as tf
    
    root = tk.Tk()
    app = ModelTrainingApp(root)
    root.mainloop()