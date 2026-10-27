# School Face Recognizer

A Python app that opens your webcam, recognizes faces in real time, and writes each person's name next to their face. You teach it by uploading photos or by adding faces straight from the camera.

## Features

- Live camera face detection and recognition
- Name label drawn next to each recognized face ("Unknown" for faces it doesn't know)
- **Upload Image(s)** button to add a person from photos
- **Add Face From Camera** button to teach it someone on the spot
- Works fully offline, with no cloud needed
- Built with OpenCV (YuNet detector + SFace recognizer), NumPy, Pillow and Tkinter

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/school-face-recognizer.git
cd school-face-recognizer
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
sudo apt install python3-tk python3-pil.imagetk   # Linux only, for the window
```

## Usage

```bash
python app.py
```

On first run it downloads two small model files into `models/` (one time only).

1. Click **Upload Image(s)**, choose 3 to 5 clear photos of one person, and type their name.
2. Or show one face to the camera and click **Add Face From Camera**.
3. Anyone you've added is now labeled automatically.

Tips: use photos with good lighting at different angles. The `THRESHOLD` value at the top of `app.py` controls strictness (raise it if people get mixed up, lower it if known people show as Unknown).

## Project structure

```
app.py             main application
requirements.txt   Python dependencies
known_faces/       one folder per person (created automatically, not uploaded to GitHub)
models/            downloaded AI models (not uploaded to GitHub)
```

## Privacy

This project processes face data. Only add people who have given permission (and for minors, permission from a parent or the school). The `known_faces/` folder is deliberately excluded from Git, so never upload it to a public place.

## Roadmap

- [ ] Attendance logging to CSV
- [ ] Web version
- [ ] Alerts for unknown faces
- [ ] Syncing the database across devices

## Author

Azariya (AC0X-84)
