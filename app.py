import cv2
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
import os
import json
import numpy as np

# try importing DeepFace; if unavailable warn later
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    DeepFace = None
    HAS_DEEPFACE = False

import sqlite3

# --- Database Logic ---
DB_NAME = "members.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            membership_type TEXT DEFAULT 'Premium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def add_member(name, embedding):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    embedding_json = json.dumps(embedding)
    cursor.execute('INSERT INTO members (name, embedding) VALUES (?, ?)', (name, embedding_json))
    conn.commit()
    conn.close()

def get_members():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT name, embedding FROM members')
    rows = cursor.fetchall()
    conn.close()
    return [{"name": r[0], "embedding": json.loads(r[1])} for r in rows]

# --- Main Application ---
class PremiumEntryApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Premium Lounge Face-Recognition Entry")
        self.window.geometry("1100x700")
        self.window.configure(bg="#2c3e50")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # State
        self.running = False
        self.cap = None
        self.members = []
        self.process_every_n_frames = 5
        self.frame_count = 0
        self.last_results = []
        self.threshold = 0.68
        # recognition thread guard
        self.processing = False
        
        init_db()
        self.setup_ui()
        self.load_members()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.window, bg="#1a252f", height=80)
        header.pack(fill=tk.X)
        lbl_title = tk.Label(header, text="PREMIUM LOUNGE ACCESS CONTROL", font=("Helvetica", 24, "bold"), 
                             bg="#1a252f", fg="#ecf0f1", pady=20)
        lbl_title.pack()

        # Main Layout
        main_frame = tk.Frame(self.window, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left: Video Feed
        self.video_frame = tk.Label(main_frame, bg="#34495e", bd=2, relief="groove")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right: Controls & Logs
        right_panel = tk.Frame(main_frame, bg="#2c3e50", width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

        # Buttons
        btn_frame = tk.Frame(right_panel, bg="#2c3e50")
        btn_frame.pack(fill=tk.X)

        self.btn_toggle = tk.Button(btn_frame, text="START SYSTEM", command=self.toggle_system,
                                   font=("Helvetica", 14, "bold"), bg="#27ae60", fg="white", 
                                   padx=20, pady=10, relief="flat", cursor="hand2")
        self.btn_toggle.pack(fill=tk.X, pady=5)

        btn_register = tk.Button(btn_frame, text="REGISTER MEMBER", command=self.register_member,
                                     font=("Helvetica", 12), bg="#2980b9", fg="white", 
                                     padx=20, pady=8, relief="flat", cursor="hand2")
        btn_register.pack(fill=tk.X, pady=5)

        btn_refresh = tk.Button(btn_frame, text="REFRESH DATABASE", command=self.load_members,
                                     font=("Helvetica", 12), bg="#7f8c8d", fg="white", 
                                     padx=20, pady=8, relief="flat", cursor="hand2")
        btn_refresh.pack(fill=tk.X, pady=5)

        # Logs
        tk.Label(right_panel, text="ACCESS LOGS", font=("Helvetica", 12, "bold"), 
                 bg="#2c3e50", fg="#bdc3c7").pack(pady=(20, 5), anchor="w")
        
        self.log_list = tk.Listbox(right_panel, bg="#1a252f", fg="#2ecc71", font=("Consolas", 10),
                                  borderwidth=0, highlightthickness=0)
        self.log_list.pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        self.status_var = tk.StringVar(value="System Offline")
        status_bar = tk.Label(self.window, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, 
                              anchor=tk.W, bg="#34495e", fg="#ecf0f1", font=("Helvetica", 10))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_members(self):
        self.members = get_members()
        self.log(f"Loaded {len(self.members)} members.")

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_list.insert(0, f"[{timestamp}] {message}")
        if self.log_list.size() > 50:
            self.log_list.delete(50, tk.END)

    def toggle_system(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not access camera.")
                return
            self.running = True
            self.btn_toggle.config(text="STOP SYSTEM", bg="#e74c3c")
            self.status_var.set("System Active - Monitoring...")
            self.log("System started.")
            self.update_frame()
        else:
            self.running = False
            self.btn_toggle.config(text="START SYSTEM", bg="#27ae60")
            if self.cap:
                self.cap.release()
            self.video_frame.config(image='')
            self.status_var.set("System Offline")
            self.log("System stopped.")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # Mirror effect
            display_frame = frame.copy()
            
            # Recognition Logic (offload heavy work to thread)
            if self.frame_count % self.process_every_n_frames == 0 and not self.processing:
                if HAS_DEEPFACE:
                    # copy frame for thread to avoid mutation issues
                    threading.Thread(target=self.process_recognition, args=(frame.copy(),), daemon=True).start()
                else:
                    # skip recognition when dependency missing
                    self.last_results = []
            
            # Drawing
            for res in self.last_results:
                r = res["region"]
                color = (0, 255, 0) if res["granted"] else (0, 0, 255)
                # OpenCV uses BGR, PIL uses RGB
                cv2.rectangle(display_frame, (r['x'], r['y']), (r['x'] + r['w'], r['y'] + r['h']), color, 2)
                
                label = f"{res['name']} ({res['similarity']:.2f})" if res["granted"] else "Unknown"
                cv2.putText(display_frame, label, (r['x'], r['y'] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if res["granted"]:
                    cv2.putText(display_frame, "ACCESS GRANTED", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Convert to Tkinter Image
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # Resize to fit frame while keeping aspect ratio
            img.thumbnail((700, 500), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)
            
            self.frame_count += 1
            
        self.window.after(10, self.update_frame)

    def process_recognition(self, frame):
        # executed in background thread to avoid blocking UI
        self.processing = True
        try:
            extractions = DeepFace.represent(img_path=frame, model_name='VGG-Face', 
                                            enforce_detection=False, detector_backend='opencv')
            
            new_results = []
            for face in extractions:
                if face.get('face_confidence', 0) < 0.6:
                    continue
                
                embedding = face["embedding"]
                region = face["facial_area"]
                
                best_match = None
                max_similarity = -1
                
                for member in self.members:
                    # Dot product of normalized vectors
                    v1 = np.array(embedding)
                    v2 = np.array(member["embedding"])
                    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match = member["name"]
                
                granted = max_similarity >= self.threshold
                new_results.append({
                    "name": best_match if granted else "Guest",
                    "similarity": max_similarity,
                    "region": region,
                    "granted": granted
                })
                
                if granted and (not self.last_results or not any(r["name"] == best_match for r in self.last_results)):
                    # schedule log update on main thread
                    self.window.after(0, self.log, f"Access Granted: {best_match} ({max_similarity:.2f})")
            
            # update results on main thread
            self.window.after(0, setattr, self, 'last_results', new_results)
        except Exception as e:
            # log error for debugging
            self.window.after(0, self.log, f"Recognition error: {e}")
        finally:
            self.processing = False

    def register_member(self):
        if not HAS_DEEPFACE:
            messagebox.showerror("Feature Unavailable", 
                                "Face registration requires deepface, which is not available on Python 3.14.\n\n"
                                "To enable face registration:\n"
                                "1. Install Python 3.11 or 3.12\n"
                                "2. Create a virtual environment with that Python\n"
                                "3. Install deepface: pip install deepface\n\n"
                                "Alternatively, use admin.py to register members from command line.")
            return
        
        name = simpledialog.askstring("Register", "Enter Member Name:")
        if not name: return
        
        messagebox.showinfo("Capture", f"Prepare to capture face for {name}. Look at the camera.")
        
        # Stop monitoring if active
        was_running = self.running
        if was_running: self.toggle_system()
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            try:
                self.status_var.set("Processing New Member...")
                # first pass enforce detection to avoid false positives
                results = DeepFace.represent(img_path=frame, model_name='VGG-Face', enforce_detection=True)
            except Exception as e:
                err = str(e)
                # common failure: face not detected, retry with relaxed settings
                if "Face could not be detected" in err or "enforce_detection" in err:
                    try:
                        self.log("Primary detection failed, retrying without enforcement.")
                        results = DeepFace.represent(img_path=frame, model_name='VGG-Face', enforce_detection=False)
                    except Exception as e2:
                        messagebox.showerror("Error", f"Registration failed: {e2}")
                        results = None
                else:
                    messagebox.showerror("Error", f"Registration failed: {err}")
                    results = None

            if results:
                add_member(name, results[0]["embedding"])
                messagebox.showinfo("Success", f"Member {name} registered successfully!")
                self.load_members()
            else:
                messagebox.showerror("Error", "No face could be extracted from the frame. Make sure your face is visible and try again.")
        
        self.status_var.set("System Offline")
        if was_running: self.toggle_system()
if __name__ == "__main__":
    root = tk.Tk()
    app = PremiumEntryApp(root)
    root.mainloop()
