# 🖐️ Hand Gesture Controlled Subway Surfers

Control the **Subway Surfers** game using your **hand gestures in real-time via webcam**! This project uses **MediaPipe, scikit-learn, and OpenCV** to recognize gestures and simulate keyboard presses for controlling actions like moving left/right, jumping, rolling, using the hoverboard, and pausing the game.

---

## 🎮 Demo

[Click to watch demo video](https://github.com/brij26/Gesture-Controlled-Subway-Surfer-Game/blob/main/output.mp4)

---

## 🚀 Features

* Real-time gesture detection using webcam
* Machine learning model for gesture classification
* Gesture-to-keyboard mapping using `pyautogui`
* Works with Subway Surfers (Web or PC version)
* Easy to train your own gestures and extend functionality

---

## ✋ Recognized Gestures

| Gesture    | Action         | Key Pressed     |
| ---------- | -------------- | --------------- |
| go\_left   | Move Left      | ⬅️ `a`          |
| go\_right  | Move Right     | ➡️ `d`          |
| jump       | Jump           | ⬆️ `w`          |
| roll       | Roll           | ⬇️ `s`          |
| hoverboard | Use Hoverboard | `space` (twice) |
| pause      | Pause Game     | `p`             |
| relax      | No Action      | —               |

---

## 🧠 How It Works

**Data Collection:** Record gestures using MediaPipe and store them as `.npy` or `.csv`.

**Model Training:** Train a classifier like Random Forest, LSTM, or Transformer to learn gesture patterns.

**Live Prediction:** The webcam captures your hand, predicts the gesture, and maps it to a keyboard action in real time.

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* scikit-learn
* pyautogui
* NumPy
* joblib

---

## 📦 Installation

```bash
git clone https://github.com/brij26/Gesture-Controlled-Subway-Surfer-Game.git
cd gesture-controlled-subway-surfers

# Create a conda environment
conda create -p venv python=3.10 -y
conda activate ./venv

# If conda is not available, install Miniconda/Anaconda first

# Install dependencies
pip install -r requirements.txt
```

---

Enjoy controlling **Subway Surfers** with your hand gestures! 🚀🖐️🎮
