# 🌟 Features Explained

## 👀 Focus Detection

The system uses Eye Aspect Ratio (EAR) to determine whether the student is focused.

- Focused → image captured
- Distracted → image ignored

---

## 🧠 Face Recognition

Uses the `face_recognition` library to:

- Detect faces
- Generate facial encodings
- Compare faces in real-time

---

## ✅ Attendance Automation

Attendance is automatically:

- Marked once per student
- Timestamped
- Stored in log files

---

## 📄 Attendance Summary

After each session:

- Total students expected
- Total present
- Present student list

are saved into a summary file.