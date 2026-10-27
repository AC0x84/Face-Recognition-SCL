import os
import pickle
import shutil
import time
import urllib.request
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

DB_DIR = "known_faces"          # known_faces/<Student Name>/photo.jpg
CACHE_FILE = "encodings_sface.pkl"
MODEL_DIR = "models"
THRESHOLD = 0.40                # cosine similarity: higher = stricter
PROCESS_EVERY = 2               # recognize every N frames
FRAME_W = 640

MODELS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = {}
    for fn, url in MODELS.items():
        path = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(path) or os.path.getsize(path) < 50000:
            print(f"Downloading {fn} ... (one time only)")
            urllib.request.urlretrieve(url, path)
        paths[fn] = path
    return paths


class FaceApp:
    def __init__(self, root):
        self.root = root
        root.title("School Face Recognizer")
        os.makedirs(DB_DIR, exist_ok=True)

        paths = ensure_models()
        self.detector = cv2.FaceDetectorYN.create(
            paths["yunet.onnx"], "", (320, 320), 0.8, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(paths["sface.onnx"], "")

        self.known_feats = []
        self.known_names = []
        self.results = []          # [(x, y, w, h, label)]
        self.last_frame = None
        self.frame_count = 0

        self.video_label = tk.Label(root)
        self.video_label.pack()

        bar = tk.Frame(root)
        bar.pack(fill="x", pady=6)
        tk.Button(bar, text="Upload Image(s)", command=self.upload_images).pack(side="left", padx=5)
        tk.Button(bar, text="Add Face From Camera", command=self.add_from_camera).pack(side="left", padx=5)
        tk.Button(bar, text="Reload Database", command=self.load_database).pack(side="left", padx=5)
        self.status = tk.Label(bar, text="", anchor="w")
        self.status.pack(side="left", padx=10)

        self.load_database()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera", "Could not open the camera.")
            root.destroy()
            return

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update()

    # ---------- face helpers ----------
    def detect_faces(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return [] if faces is None else list(faces)

    def get_feature(self, img, face):
        aligned = self.recognizer.alignCrop(img, face)
        feat = self.recognizer.feature(aligned).flatten()
        return feat / np.linalg.norm(feat)

    def encode_image(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > 1000:
            f = 1000 / w
            img = cv2.resize(img, (int(w * f), int(h * f)))
        faces = self.detect_faces(img)
        if not faces:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])   # biggest face
        return self.get_feature(img, face)

    # ---------- database ----------
    def load_database(self):
        cache = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)
            except Exception:
                cache = {}

        new_cache, feats, names = {}, [], []
        for person in sorted(os.listdir(DB_DIR)):
            folder = os.path.join(DB_DIR, person)
            if not os.path.isdir(folder):
                continue
            for fn in os.listdir(folder):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                path = os.path.join(folder, fn)
                mtime = os.path.getmtime(path)
                if path in cache and cache[path][0] == mtime:
                    feat = cache[path][1]
                else:
                    feat = self.encode_image(path)
                new_cache[path] = (mtime, feat)
                if feat is not None:
                    feats.append(feat)
                    names.append(person)

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(new_cache, f)

        self.known_feats = np.array(feats) if feats else np.empty((0, 128))
        self.known_names = names
        self.status.config(text=f"{len(set(names))} people, {len(names)} photos loaded")

    # ---------- adding people ----------
    def upload_images(self):
        paths = filedialog.askopenfilenames(
            title="Choose photos of ONE student",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not paths:
            return
        name = simpledialog.askstring("Name", "Student name (and anything else to show):")
        if not name:
            return
        name = name.strip()
        folder = os.path.join(DB_DIR, name)
        os.makedirs(folder, exist_ok=True)

        added = 0
        for p in paths:
            dest = os.path.join(folder, f"{int(time.time() * 1000)}_{os.path.basename(p)}")
            shutil.copy(p, dest)
            if self.encode_image(dest) is not None:
                added += 1
            else:
                os.remove(dest)   # no face found
        self.load_database()
        messagebox.showinfo("Done", f"Added {added} of {len(paths)} photos for {name}.")

    def add_from_camera(self):
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()
        faces = self.detect_faces(frame)
        if len(faces) != 1:
            messagebox.showwarning("Add face", f"I see {len(faces)} faces. Show exactly one face.")
            return
        name = simpledialog.askstring("Name", "Who is this?")
        if not name:
            return
        name = name.strip()
        folder = os.path.join(DB_DIR, name)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, f"cam_{int(time.time() * 1000)}.jpg"), frame)
        self.load_database()

    # ---------- recognition ----------
    def recognize(self, frame):
        results = []
        for face in self.detect_faces(frame):
            x, y, w, h = [int(v) for v in face[:4]]
            label = "Unknown"
            if len(self.known_feats):
                feat = self.get_feature(frame, face)
                sims = self.known_feats @ feat          # cosine similarity
                i = int(np.argmax(sims))
                if sims[i] >= THRESHOLD:
                    label = self.known_names[i]
            results.append((x, y, w, h, label))
        self.results = results

    def update(self):
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if w != FRAME_W:
                frame = cv2.resize(frame, (FRAME_W, int(h * FRAME_W / w)))
            self.last_frame = frame.copy()

            self.frame_count += 1
            if self.frame_count % PROCESS_EVERY == 0:
                self.recognize(frame)

            for (x, y, fw, fh, label) in self.results:
                color = (0, 200, 0) if label != "Unknown" else (0, 0, 220)
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                cv2.rectangle(frame, (x, y + fh), (x + fw, y + fh + 28), color, cv2.FILLED)
                cv2.putText(frame, label, (x + 6, y + fh + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=self.photo)
        self.root.after(15, self.update)

    def on_close(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    FaceApp(root)
    root.mainloop()
