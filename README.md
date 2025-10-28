# 👨‍🎓 Student Face Recognition System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-STREAMLIT-APP-URL.streamlit.app/)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)
![License](https://img.shields.io/badge/license-MIT-green)

A Streamlit web application for student face recognition and database management, powered by the high-accuracy InsightFace `buffalo_l` model.

---

## 🚀 Live Demo

You can access the live application here:

**[➡️ Launch Student Recognition App](https://YOUR-STREAMLIT-APP-URL.streamlit.app/)**

---

## 📸 Application Preview

| Recognition Page | Database Management Page |
| :---: | :---: |
| <img src="image_9953da.png" alt="Recognition Page" width="100%"> | <img src="image_995434.png" alt="Database Management Page" width="100%"> |

*(Note: You'll need to upload these images to your repo and fix the paths for them to display)*

---

## ✨ Features

* **👨‍🎓 Student Recognition:** Upload a class photo, and the system identifies all known students, drawing bounding boxes and displaying their names and confidence scores.
* **➕ Database Management:** An interactive page to:
    * **Add New Students:** Upload multiple photos for a new student.
    * **View Database:** See a list of all students currently in the `known_faces` directory.
    * **Rebuild Database:** Re-process all images to update the face embeddings file.
* **⚡ High-Accuracy:** Utilizes the state-of-the-art `buffalo_l` model from InsightFace for reliable detection and embedding generation.

---

## 🛠️ Tech Stack & Workflow

This app uses a two-part system built entirely within Streamlit:

1.  **Database Generation (`Manage Database` Page):**
    * The `build_database` function uses **InsightFace** (`FaceAnalysis`) to scan the `known_faces/` directory.
    * It detects faces, extracts a 512-dimension embedding vector for each, and saves them (along with their names) into a single `database/face_embeddings.pkl` file.

2.  **Recognition (`Student Recognition` Page):**
    * A user uploads a new image.
    * InsightFace detects all faces in this new image.
    * For each detected face, it generates an embedding.
    * This new embedding is compared against all embeddings in the `.pkl` database using **Cosine Similarity**.
    * If a match is found above the `SIMILARITY_THRESHOLD` (set to 0.5), the student's name is drawn on the image.

---

## 🖥️ Local Setup & Installation

To run this application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO.git](https://github.com/YOUR-USERNAME/YOUR-REPO.git)
    cd YOUR-REPO
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

---

## ⚠️ Important Deployment Note

This application is deployed on **Streamlit Cloud**, which uses an **ephemeral filesystem**.

> **What this means:** Any new students added or any database rebuilds performed on the "Manage Database" page of the *live web app* are **TEMPORARY**. All changes will be **erased** as soon as the app restarts or goes to sleep.

### How to Permanently Update the Database:
1.  Run the application **locally** (see steps above).
2.  Use the "Manage Database" page to add new students and rebuild the database. This will update your *local* `known_faces/` and `database/face_embeddings.pkl` files.
3.  **Commit and push** these changes to your GitHub repository.

    ```bash
    git add known_faces/ database/face_embeddings.pkl
    git commit -m "Updated student database"
    git push origin main
    ```
4.  Streamlit Cloud will automatically detect the push and redeploy your app with the permanent changes.

---

## 📄 License

This project is licensed under the MIT License.
