# TruthLens — Deepfake Image Detection Website

A local Flask website using the supplied trained CNN model to classify uploaded images as `fake` or `real`.

## Features
- Login and registration pages
- Passwords stored as Werkzeug password hashes in a local SQLite database (`users.db`)
- Login/logout and protected detector page/API
- Image preview, upload, prediction, confidence scores
- Model trained on the supplied sample dataset (5 fake + 5 real images)

## Run on Windows
1. Extract the ZIP and open the `deepfake_detection_website` folder in VS Code.
2. In the terminal, install dependencies:
   ```powershell
   py -m pip install -r requirements.txt
   ```
3. Start the website:
   ```powershell
   py app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser. You will be sent to the login page.
5. Choose **Create an account**, register, and then use the detector.

## Important
- User accounts are local to this installation. This is a student demo, not production-ready authentication.
- Set a stable `FLASK_SECRET_KEY` environment variable before deploying or sharing the site outside your own computer.
- The model was trained with only 10 images, so predictions are experimental and not reliable proof of authenticity.
- Uploaded images are analyzed in memory and not saved by the prediction route.

## JavaScript features
- `static/app.js`: image upload preview, drag-and-drop, file-size/type checks, asynchronous prediction request, result rendering and loading state.
- `static/auth.js`: password show/hide and client-side registration validation.
