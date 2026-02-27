# Premium Guest Face-Recognition Entry

An automated lounge entry system using Python, OpenCV, and DeepFace.

## Features
- **Hands-Free Access Control**: Real-time facial recognition for seamless entry.
- **Member Database**: SQLite database for storing member profiles and embeddings.
- **Admin Tools**: Easy-to-use CLI for adding and removing members.
- **Visual Feedback**: Real-time bounding boxes and access status.

## Prerequisites
- Python 3.10+
- Webcam
- Required Libraries: `opencv-python`, `deepface`, `numpy`, `sqlite3`

## Setup
1. Clone this repository or copy the project files.
2. Install dependencies (if not already installed):
   ```bash
   pip install opencv-python deepface numpy
   ```

## Usage

### 1. Register a Member
To add a new member to the system, run:
```bash
python admin.py add "John Doe"
```
A camera window will open. Look at the camera and press **'c'** to capture.

### 2. List or Remove Members
To see all registered members:
```bash
python admin.py list
```
To remove a member:
```bash
python admin.py delete "John Doe"
```

### 3. Start the Entry System
To launch the real-time recognition and access control system:
```bash
python main.py
```
- A window titled **"Premium Lounge Entry"** will appear.
- Recognized members will see a green box and **"ACCESS GRANTED"**.
- Unknown guests will see a red box and **"Access Denied"**.
- Press **'q'** to exit.

## Technical Details
- **Model**: VGG-Face (Default)
- **Matching Metric**: Cosine Similarity (Threshold: 0.68)
- **Face Detection**: OpenCV (Cascade Classifier / DNN)

## Security Disclaimer
- This is a prototype system.
- For production use, consider adding **Liveness Detection** to prevent spoofing with photos.
- Ensure compliance with local biometric data privacy laws (e.g., GDPR).
