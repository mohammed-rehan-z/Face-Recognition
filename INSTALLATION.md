# ⚙️ Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download dlib Landmark Model

Download:

`shape_predictor_68_face_landmarks.dat`

Place it in the project root directory.

---

## 5. Run Project

```bash
python smart_new.py
```