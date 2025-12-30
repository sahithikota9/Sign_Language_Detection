import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

class SignLanguageModel:
    def __init__(self, model_path=None):
        """
        Initialize the sign language recognition model
        
        Args:
            model_path (str, optional): Path to a pre-trained model
        """
        self.model = None
        self.label_encoder = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self, input_shape, num_classes):
        """
        Build a neural network model for sign language recognition
        
        Args:
            input_shape (tuple): Shape of input features
            num_classes (int): Number of classes to predict
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the model on sign language data
        
        Args:
            X (numpy.ndarray): Feature vectors
            y (numpy.ndarray): Labels
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object with training metrics
        """
        if not self.label_encoder:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        # Convert to one-hot encoding
        y_categorical = to_categorical(y_encoded)
        
        # Build model if not already built
        if not self.model:
            self.build_model(X.shape[1], len(self.label_encoder.classes_))
        
        # Train the model
        history = self.model.fit(
            X, y_categorical,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Predict sign language from feature vector
        
        Args:
            X (numpy.ndarray): Feature vector
            
        Returns:
            str: Predicted label
            float: Confidence score
        """
        if not self.model or not self.label_encoder:
            raise ValueError("Model not trained or loaded")
        
        # Reshape if single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Get prediction probabilities
        pred_proba = self.model.predict(X)[0]
        
        # Get the predicted class index and confidence
        pred_idx = np.argmax(pred_proba)
        confidence = pred_proba[pred_idx]
        
        # Convert index to label
        predicted_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return predicted_label, confidence
    
    def save_model(self, model_path, encoder_path=None):
        """
        Save the trained model and label encoder
        
        Args:
            model_path (str): Path to save the model
            encoder_path (str, optional): Path to save the label encoder
        """
        if not self.model:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        self.model.save(model_path)
        
        # Save the label encoder
        if self.label_encoder and encoder_path:
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
    
    def load_model(self, model_path, encoder_path=None):
        """
        Load a trained model and label encoder
        
        Args:
            model_path (str): Path to the saved model
            encoder_path (str, optional): Path to the saved label encoder
        """
        # Load the model
        self.model = load_model(model_path)
        
        # Load the label encoder if provided
        if encoder_path and os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        
        return self.model