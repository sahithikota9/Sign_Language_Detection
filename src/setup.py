import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import platform

# Check Python version
python_version = sys.version_info
if python_version.major == 3 and python_version.minor > 10:
    print(f"Warning: You are using Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    print("This project is recommended to be used with Python 3.8-3.10")
    print("Some dependencies may not work correctly with your Python version.")
    
    # Show warning dialog if running the GUI
    if __name__ == "__main__":
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(
            "Python Version Warning",
            f"You are using Python {python_version.major}.{python_version.minor}.{python_version.micro}\n\n"
            "This project is recommended to be used with Python 3.8-3.10.\n"
            "Some dependencies may not work correctly with your Python version.\n\n"
            "Consider creating a virtual environment with a compatible Python version."
        )
        root.destroy()

class SetupApp:
    def __init__(self, root):
        """
        Initialize the setup application
        
        Args:
            root (tk.Tk): Tkinter root window
        """
        self.root = root
        self.root.title("Sign Sense - Setup")
        self.root.geometry("600x400")
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        """Create the UI elements"""
        # Title
        title_label = ttk.Label(self.root, text="Sign Sense Setup", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        # Instructions
        instructions = ttk.Label(self.root, text="Welcome to Sign Sense! Follow these steps to get started:", 
                               font=("Arial", 12))
        instructions.pack(pady=10)
        
        # Steps frame
        steps_frame = ttk.Frame(self.root)
        steps_frame.pack(pady=10, fill="both", expand=True, padx=20)
        
        # Step 1: Install dependencies
        step1_frame = ttk.LabelFrame(steps_frame, text="Step 1: Install Dependencies")
        step1_frame.pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="Install required Python packages").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.install_btn = ttk.Button(step1_frame, text="Install", command=self.install_dependencies)
        self.install_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Step 2: Collect data
        step2_frame = ttk.LabelFrame(steps_frame, text="Step 2: Collect Training Data")
        step2_frame.pack(fill="x", pady=5)
        
        ttk.Label(step2_frame, text="Collect sign language alphabet samples").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.collect_btn = ttk.Button(step2_frame, text="Launch", command=self.launch_data_collection)
        self.collect_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Step 3: Train model
        step3_frame = ttk.LabelFrame(steps_frame, text="Step 3: Train Recognition Model")
        step3_frame.pack(fill="x", pady=5)
        
        ttk.Label(step3_frame, text="Train the sign language recognition model").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.train_btn = ttk.Button(step3_frame, text="Launch", command=self.launch_training)
        self.train_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Step 4: Launch application
        step4_frame = ttk.LabelFrame(steps_frame, text="Step 4: Launch Sign Sense")
        step4_frame.pack(fill="x", pady=5)
        
        ttk.Label(step4_frame, text="Start the main application").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.app_btn = ttk.Button(step4_frame, text="Launch", command=self.launch_app)
        self.app_btn.grid(row=0, column=1, padx=10, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to start")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
    
    def install_dependencies(self):
        """Install required dependencies"""
        self.status_var.set("Installing dependencies...")
        self.root.update()
        
        try:
            # Check if requirements.txt exists
            if not os.path.exists("requirements.txt"):
                messagebox.showerror("Error", "requirements.txt not found")
                self.status_var.set("Error: requirements.txt not found")
                return
            
            # Install dependencies
            process = subprocess.Popen([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                messagebox.showerror("Error", f"Failed to install dependencies: {stderr.decode()}")
                self.status_var.set("Error installing dependencies")
                return
            
            messagebox.showinfo("Success", "Dependencies installed successfully")
            self.status_var.set("Dependencies installed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def launch_data_collection(self):
        """Launch the data collection application"""
        self.status_var.set("Launching data collection...")
        self.root.update()
        
        try:
            # Check if collect_data.py exists
            if not os.path.exists("collect_data.py"):
                messagebox.showerror("Error", "collect_data.py not found")
                self.status_var.set("Error: collect_data.py not found")
                return
            
            # Launch data collection
            subprocess.Popen([sys.executable, "collect_data.py"])
            self.status_var.set("Data collection launched")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def launch_training(self):
        """Launch the model training application"""
        self.status_var.set("Launching model training...")
        self.root.update()
        
        try:
            # Check if train_model.py exists
            if not os.path.exists("train_model.py"):
                messagebox.showerror("Error", "train_model.py not found")
                self.status_var.set("Error: train_model.py not found")
                return
            
            # Launch training
            subprocess.Popen([sys.executable, "train_model.py"])
            self.status_var.set("Model training launched")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def launch_app(self):
        """Launch the main application"""
        self.status_var.set("Launching Sign Sense...")
        self.root.update()
        
        try:
            # Check if app.py exists
            if not os.path.exists("app.py"):
                messagebox.showerror("Error", "app.py not found")
                self.status_var.set("Error: app.py not found")
                return
            
            # Launch app
            subprocess.Popen([sys.executable, "app.py"])
            self.status_var.set("Sign Sense launched")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SetupApp(root)
    root.mainloop()