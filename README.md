```markdown
# 🖐️ Gesture Controlled Subway Surfer Game

Control **Subway Surfers** using **hand gestures in real-time** via webcam!

This project uses **MediaPipe, scikit-learn, OpenCV**, and machine learning models to recognize hand gestures and simulate keyboard presses for game actions such as moving left/right, jumping, rolling, using hoverboard, and pausing the game.

---

## 🎮 Demo

[▶️ Watch Demo Video](https://github.com/brij26/Gesture-Controlled-Subway-Surfer-Game/blob/main/output.mp4)

---

## 🚀 Features

✅ Real-time gesture detection using webcam  
✅ Machine learning models for gesture classification (RandomForest, LSTM, Transformer)  
✅ Gesture-to-keyboard mapping using `pyautogui`  
✅ Works with Subway Surfers (Web or PC version)  
✅ Easily extendable for training your own gestures and actions

---

## ✋ Recognized Gestures

| Gesture     | Action          | Key Pressed       |
|-------------|-----------------|-------------------|
| `go_left`   | Move Left       | ⬅️ `a`            |
| `go_right`  | Move Right      | ➡️ `d`            |
| `jump`      | Jump            | ⬆️ `w`            |
| `roll`      | Roll            | ⬇️ `s`            |
| `hoverboard`| Use Hoverboard  | `space` (twice)   |
| `pause`     | Pause Game      | `p`               |
| `relax`     | No Action       | —                 |

---

## 🧠 How It Works

1. **Data Collection**  
   Collect hand gesture data using MediaPipe, saved as `.npy` files under `gesture_data/`.
   
2. **Model Training**  
   Train a machine learning model (`RandomForest`, `LSTM`, or `Transformer`) using `models/` scripts on the gesture data.

3. **Live Prediction**  
   Webcam captures your hand → predicts gesture → maps it to keyboard action to control Subway Surfers in real-time.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- scikit-learn
- pyautogui
- NumPy
- joblib

---

## 📂 Directory Structure

```

prashantsaxe-gesture-controlled-subway-surfer-game/
├── README.md
├── requirements.txt
├── data\_collection/
├── detect\_only\_hand/
├── find\_trails\_of\_hand\_points/
├── gesture\_data/
├── models/
└── results/

````

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/prashantsaxe-gesture-controlled-subway-surfer-game.git
cd prashantsaxe-gesture-controlled-subway-surfer-game

# Create environment (recommended)
conda create -p venv python=3.10 -y
conda activate ./venv

# If conda is not available, install Miniconda or Anaconda first

# Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Running

* **Collect Gesture Data:**

  ```bash
  python data_collection/data_collection_jump.py
  # Replace with the corresponding gesture script you want to record
  ```
* **Train Model:**

  ```bash
  python models/randomforest.py
  # or
  python models/train_LSTM.py
  # or
  python models/train_transformer.py
  ```
* **Run Live Gesture Control:**

  ```bash
  python results/result.py
  ```

---

## 📈 Contributions

Feel free to open issues or pull requests for improvements, new gesture integrations, or adding better models.

---

## 📜 License

This project is open-source for learning and personal projects. Feel free to modify and experiment.

---

Enjoy controlling Subway Surfers with your gestures! 🚀✋🕹️

```

---

### How to use:

✅ Simply **copy and paste** directly into your `prashantsaxe-gesture-controlled-subway-surfer-game/README.md`.  
✅ Replace `<your-username>` with your GitHub username in the `git clone` URL before pushing.  
✅ It is cleanly structured, readable, SEO-friendly, and showcases your project professionally on GitHub.  

If you need badges (stars, license, python version) or a minimal GIF preview for the top, let me know for instant addition before you push.
```
