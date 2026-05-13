# 🛠 Troubleshooting

## Webcam Not Opening

### Solution

- Check webcam permissions
- Close other webcam applications
- Restart system

---

## dlib Error

### Solution

Install CMake first:

```bash
pip install cmake
```

Then reinstall dlib:

```bash
pip install dlib
```

---

## No Face Detected

### Solution

- Improve lighting
- Face camera directly
- Remove obstructions

---

## Attendance Not Marking

### Solution

- Re-capture student dataset
- Generate fresh encodings
- Ensure face visibility

---

## Encoding File Missing

### Solution

Run student capture first to create:

```text
encodings.pkl
```