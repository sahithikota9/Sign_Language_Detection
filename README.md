python 3.9.13 version
   ```
   py -3.9 -m venv venv
   venv\Scripts\activate


3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

> **Note**: If you're using Python 3.13 or newer, you will encounter compatibility issues with some dependencies. Please follow the environment setup guide to create a compatible environment.

## Usage

1. Run the main application:
   ```
   python app.py
   ```

2. The application will open with your webcam feed.
3. Position your hand in the frame and form sign language alphabets.
4. The application will display the recognized letter in real-time.

## Training Your Own Model

If you want to train your own model:

1. Run the data collection script:
   ```
   python collect_data.py
   ```

2. Follow the on-screen instructions to collect samples for each alphabet.

3. Train the model:
   ```
   python train_model.py
   ```

4. The new model will be saved in the `models` directory.

## Project Structure

- `app.py`: Main application with Tkinter GUI
- `hand_detector.py`: Hand tracking and feature extraction using MediaPipe
- `model.py`: Sign language recognition model
- `collect_data.py`: Script for collecting training data
- `train_model.py`: Script for training the model
- `models/`: Directory for storing trained models
- `data/`: Directory for storing training data

## License

MIT

## Acknowledgments

- MediaPipe for hand tracking
- OpenCV for computer vision capabilities