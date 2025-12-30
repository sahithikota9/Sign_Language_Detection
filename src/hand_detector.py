import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe
        
        Args:
            static_mode (bool): If True, detection runs on every frame
            max_hands (int): Maximum number of hands to detect
            detection_confidence (float): Minimum confidence for hand detection
            tracking_confidence (float): Minimum confidence for hand tracking
        """
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """
        Detect hands in an image and optionally draw landmarks
        
        Args:
            img (numpy.ndarray): Input image (BGR format)
            draw (bool): Whether to draw hand landmarks on the image
            
        Returns:
            numpy.ndarray: Image with or without drawings
            list: List of detected hands with landmark information
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image
        self.results = self.hands.process(img_rgb)
        
        # Initialize list to store hand information
        all_hands = []
        
        h, w, c = img.shape
        
        # Check if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                hand_info = {}
                
                # Get hand type (left or right)
                if self.results.multi_handedness:
                    hand_info["type"] = self.results.multi_handedness[hand_idx].classification[0].label
                
                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
                
                hand_info["landmarks"] = landmarks
                all_hands.append(hand_info)
                
                # Draw landmarks if requested
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        return img, all_hands
    
    def extract_features(self, landmarks):
        """
        Extract features from hand landmarks for classification
        
        Args:
            landmarks (list): List of landmark coordinates
            
        Returns:
            numpy.ndarray: Feature vector for classification
        """
        if not landmarks:
            return None
        
        # Extract features (normalized coordinates relative to wrist)
        features = []
        wrist = landmarks[0]  # Wrist is the first landmark
        
        for lm in landmarks:
            # Calculate relative position to wrist and normalize
            dx = lm[0] - wrist[0]
            dy = lm[1] - wrist[1]
            
            # Add to feature vector
            features.extend([dx, dy])
        
        return np.array(features, dtype=np.float32)