import time
import database
import utils

# Optional heavy dependencies: import if available, otherwise handle gracefully
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    from deepface import DeepFace  # type: ignore
    HAS_DEEPFACE = True
except Exception:
    DeepFace = None
    HAS_DEEPFACE = False

# Configuration
THRESHOLD = 0.68 # Adjust based on experiments
PROCESS_EVERY_N_FRAMES = 5  # To improve performance
MODEL_NAME = "VGG-Face"

def start_recognition():
    print("Initializing Face Recognition Entry System...")
    
    # Ensure the database is initialized (creates table if missing)
    database.init_db()

    # Load members from database
    members = database.get_all_members()
    if not members:
        print("Warning: No members in database. Use admin.py add <name> first.")
    
    print(f"Loaded {len(members)} premium members.")
    # Check required heavy dependencies
    if not HAS_CV2:
        print("Error: OpenCV (cv2) is not installed. Install with: pip install opencv-python")
        return

    if not HAS_DEEPFACE:
        print("Error: deepface is not installed. Install with: pip install deepface")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    frame_count = 0
    results = [] # Store latest detection results (bounding box and name)
    
    print("System active. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        display_frame = frame.copy()
        
        # Performance optimization: only process every N frames
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = []
            try:
                # Detect faces and extract embeddings
                # DeepFace.represent can detect multiple faces and return their embeddings and bounding boxes
                # It uses 'enforce_detection=False' to avoid crashing when no face is present
                extractions = DeepFace.represent(img_path=frame, 
                                                model_name=MODEL_NAME, 
                                                enforce_detection=False,
                                                detector_backend='opencv') # Or 'retinaface' for better but slower

                # DeepFace.represent may return a dict for single face, normalize to list
                if isinstance(extractions, dict):
                    extractions = [extractions]

                for face in extractions:
                    # Skip if face wasn't detected (face_confidence = 0)
                    if face.get('face_confidence', 0) < 0.5:
                        continue
                        
                    embedding = face["embedding"]
                    region = face["facial_area"] # x, y, w, h
                    
                    best_match = None
                    max_similarity = -1
                    
                    # Compare with database
                    for member in members:
                        similarity = utils.cosine_similarity(embedding, member["embedding"])
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = member["name"]
                    
                    # Decision logic
                    if max_similarity >= THRESHOLD:
                        results.append({
                            "name": best_match,
                            "similarity": max_similarity,
                            "region": region,
                            "granted": True
                        })
                    else:
                        results.append({
                            "name": "Guest / Unknown",
                            "similarity": max_similarity,
                            "region": region,
                            "granted": False
                        })
            except Exception as e:
                # Log error or ignore frame
                # print(f"Error: {e}")
                pass
        
        # Draw results on frame
        for res in results:
            r = res["region"]
            color = (0, 255, 0) if res["granted"] else (0, 0, 255)
            label = f"{res['name']} ({res['similarity']:.2f})" if res["granted"] else "Access Denied"
            
            # Draw rectangle
            cv2.rectangle(display_frame, (r['x'], r['y']), (r['x'] + r['w'], r['y'] + r['h']), color, 2)
            
            # Draw label
            cv2.putText(display_frame, label, (r['x'], r['y'] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
            # Grant access visual message
            if res["granted"]:
                cv2.putText(display_frame, "ACCESS GRANTED", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        cv2.imshow("Premium Lounge Entry", display_frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_recognition()
