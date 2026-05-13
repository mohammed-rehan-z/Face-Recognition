import cv2
import os
import dlib
import numpy as np
import pickle
import face_recognition
from imutils import face_utils
from datetime import datetime
import shutil

# === CONFIG ===
EAR_THRESHOLD = 0.25
ENCODINGS_FILE = "encodings.pkl"
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.txt"
ATTENDANCE_DIR = "attendance_logs"
TOLERANCE = 0.45

# === INIT MODELS ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# === UTILITY FUNCTIONS ===
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def delete_student_data():
    student = input("Enter the student name to delete their data: ").strip()
    path = os.path.join(DATASET_DIR, student)
    if os.path.exists(path):
        confirm = input(f"Are you sure you want to delete {student}'s dataset? (y/n): ").lower()
        if confirm == "y":
            shutil.rmtree(path)
            print(f"[🗑️] Deleted dataset for {student}.")
        else:
            print("[❌] Deletion cancelled.")
    else:
        print("[⚠️] No such student found in dataset.")

def generate_encodings():
    print("[INFO] Generating face encodings...")
    known_encodings = []
    known_names = []

    for student in os.listdir(DATASET_DIR):
        student_path = os.path.join(DATASET_DIR, student)
        if not os.path.isdir(student_path):
            continue
        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(student)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

    print("[✅] Encodings saved.")

def capture_faces():
    name = input("Enter student name: ").strip()
    dataset_path = os.path.join(DATASET_DIR, name)
    os.makedirs(dataset_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print("[INFO] Starting face capture with focus detection...")

    while count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        status = "No Face Detected"
        color = (255, 255, 255)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[36:42]
            rightEye = shape[42:48]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            if ear > EAR_THRESHOLD:
                status = "Focused"
                color = (0, 255, 0)
            else:
                status = "Distracted"
                color = (0, 0, 255)

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, color, 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, color, 1)

            if status == "Focused":
                img_path = os.path.join(dataset_path, f"{count}.jpg")
                cv2.imwrite(img_path, frame)
                count += 1
                print(f"[INFO] Captured image {count}/20")

        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Focus & Capture", frame)

        if cv2.waitKey(100) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✅] Dataset capture complete.")
    generate_encodings()

def recognize_attendance():
    if not os.path.exists(ENCODINGS_FILE):
        print("[⚠️] No encodings found. Please run capture first.")
        return

    try:
        total_expected = int(input("Enter total number of students expected today: ").strip())
    except ValueError:
        print("[❌] Invalid number.")
        return

    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    attendance_path = os.path.join(ATTENDANCE_DIR, ATTENDANCE_FILE)

    print("[INFO] Starting attendance recognition...")
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    recorded = set()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for encoding, (top, right, bottom, left) in zip(encodings, locations):
            distances = face_recognition.face_distance(known_encodings, encoding)
            min_distance = min(distances)
            index = np.argmin(distances)
            name = known_names[index] if min_distance < TOLERANCE else "Unknown"

            if name != "Unknown" and name not in recorded:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_path, "a") as file:
                    file.write(f"{name} - {timestamp}\n")
                recorded.add(name)
                print(f"[MARKED] {name} - {timestamp}")

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Marked: {len(recorded)}/{total_expected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Attendance Recognition", frame)
        key = cv2.waitKey(1)
        if key == 27 or len(recorded) >= total_expected:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✅] Attendance session ended.")

    # Save summary
    summary_file = os.path.join(
        ATTENDANCE_DIR,
        f"attendance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(summary_file, "w") as f:
        f.write(f"Attendance Summary - {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"Total expected: {total_expected}\n")
        f.write(f"Total marked: {len(recorded)}\n\n")
        f.write("Students marked present:\n")
        for name in sorted(recorded):
            f.write(f"- {name}\n")

    print(f"[📄] Attendance summary saved as: {summary_file}")

# === MENU ===
def main_menu():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)

    while True:
        print("\n🎓 Smart Attendance System")
        print("1. 📸 Capture new student data")
        print("2. ✅ Recognize & mark attendance")
        print("3. 🗑️  Delete student dataset")
        print("0. ❌ Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            capture_faces()
        elif choice == "2":
            recognize_attendance()
        elif choice == "3":
            delete_student_data()
        elif choice == "0":
            print("Goodbye! 👋")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main_menu()
