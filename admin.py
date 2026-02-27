import cv2
import sys
import os
import json
import sqlite3
import numpy as np
from deepface import DeepFace
import database

# Use the same database initialization
database.init_db()

def capture_face(name):
    print(f"Starting camera to capture face for '{name}'...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Look at the camera. Press 'c' to capture or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the frame
        cv2.imshow("Capture Face - Press 'c' to capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Save temporary image for DeepFace
            temp_path = f"temp_{name}.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                # Extract face embedding
                print(f"Extracting embedding for '{name}'...")
                # DeepFace.represent returns a list of results (one per face)
                results = DeepFace.represent(img_path=temp_path, model_name='VGG-Face', enforce_detection=True)
                
                if results and len(results) > 0:
                    embedding = results[0]["embedding"]
                    # Add to database
                    database.add_member(name, embedding)
                    print(f"Successfully added member '{name}' to database.")
                    os.remove(temp_path)
                    break
                else:
                    print("No face detected. Please try again.")
                    os.remove(temp_path)
            except Exception as e:
                print(f"Error during extraction: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        elif key == ord('q'):
            print("Capture cancelled.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

def list_members():
    members = database.get_all_members()
    print("--- Current Members ---")
    for idx, member in enumerate(members):
        print(f"{idx + 1}. {member['name']}")
    print("-----------------------")

def delete_member(name):
    # For simplicity, we can add a delete function in database.py
    # or just use a placeholder here for now
    conn = sqlite3.connect(database.DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM members WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    print(f"Member '{name}' deleted if existed.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python admin.py [add|list|delete] [name]")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "add":
        if len(sys.argv) < 3:
            print("Usage: python admin.py add <name>")
            return
        capture_face(sys.argv[2])
    elif cmd == "list":
        list_members()
    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: python admin.py delete <name>")
            return
        delete_member(sys.argv[2])
    else:
        print("Unknown command.")

if __name__ == "__main__":
    main()
