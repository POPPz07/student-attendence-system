# app.py

import streamlit as st
import os
import cv2
import pickle
import insightface
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Recognition System",
    page_icon="👨‍🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PATHS AND CONFIGURATION ---
KNOWN_FACES_DIR = "known_faces"
DATABASE_FILE = "database/face_embeddings.pkl"
MODEL_NAME = 'buffalo_l'
SIMILARITY_THRESHOLD = 0.49

# --- HELPER FUNCTIONS (Refactored from your scripts) ---

@st.cache_resource
def load_face_analysis_model():
    """Loads the insightface model and caches it."""
    app = insightface.app.FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=-1, det_size=(640, 640)) # Use -1 for CPU, 0 for GPU
    return app

def build_database(model):
    """Scans the known_faces directory, generates embeddings, and saves them."""
    if not os.path.exists(KNOWN_FACES_DIR):
        st.error(f"Directory not found: '{KNOWN_FACES_DIR}'. Please create it.")
        return False

    known_faces_data = {"names": [], "embeddings": []}
    
    person_folders = [name for name in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))]
    
    if not person_folders:
        st.warning("The 'known_faces' directory is empty. Please add student folders and images.")
        return False

    progress_bar = st.progress(0, text="Initializing database build...")
    total_people = len(person_folders)
    
    for i, name in enumerate(person_folders):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        progress_text = f"Processing student: {name} ({i+1}/{total_people})"
        progress_bar.progress((i) / total_people, text=progress_text)

        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            img = cv2.imread(image_path)
            if img is None: continue

            faces = model.get(img)
            if not faces: continue
            
            embedding = faces[0].normed_embedding
            known_faces_data["names"].append(name)
            known_faces_data["embeddings"].append(embedding)

    progress_bar.progress(1.0, text="Finalizing and saving database...")

    os.makedirs(os.path.dirname(DATABASE_FILE), exist_ok=True)
    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(known_faces_data, f)
        
    progress_bar.empty()
    return True

def recognize_faces(model, uploaded_image):
    """Performs face recognition on an uploaded image."""
    if not os.path.exists(DATABASE_FILE):
        st.error("Face database not found. Please build it from the 'Manage Database' page.")
        return None

    with open(DATABASE_FILE, "rb") as f:
        known_faces_data = pickle.load(f)
    known_embeddings = np.array(known_faces_data["embeddings"])
    known_names = known_faces_data["names"]

    # Convert PIL Image to OpenCV format
    img = np.array(uploaded_image.convert('RGB'))
    
    unknown_faces = model.get(img)

    if not unknown_faces:
        st.warning("No faces detected in the uploaded image.")
        return img

    for face in unknown_faces:
        unknown_embedding = face.normed_embedding.reshape(1, -1)
        similarities = cosine_similarity(unknown_embedding, known_embeddings)[0]
        
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[best_match_index]
        
        bbox = face.bbox.astype(int)
        
        if best_match_score > SIMILARITY_THRESHOLD:
            name = known_names[best_match_index]
            color = (0, 255, 0) # Green
        else:
            name = "Unknown"
            color = (0, 0, 255) # Red
            
        label = f"{name}: {best_match_score:.2f}"
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (bbox[0], bbox[1] - h - 10), (bbox[0] + w, bbox[1] - 5), color, -1)
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Convert back for display

# --- STREAMLIT UI ---

# Load the model once
face_model = load_face_analysis_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["👨‍🎓 Student Recognition", "➕ Manage Database"])

if page == "👨‍🎓 Student Recognition":
    st.title("Student Face Recognition")
    st.write("Upload a class photo and the system will identify the known students.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Recognize Students", use_container_width=True):
            with st.spinner("Processing image... This may take a moment."):
                pil_image = Image.open(uploaded_file)
                result_image = recognize_faces(face_model, pil_image)

            if result_image is not None:
                with col2:
                    st.image(result_image, caption="Recognition Result", use_column_width=True, channels="BGR")

elif page == "➕ Manage Database":
    st.title("Manage Student Database")
    st.write("Add new students and their photos to the recognition system.")

    # --- Section to Add New Students ---
    st.subheader("Add a New Student")
    
    new_student_name = st.text_input("Enter the student's name (this will be the folder name):")
    uploaded_photos = st.file_uploader("Upload photos of the student (at least one clear photo):", 
                                       accept_multiple_files=True, type=["jpg", "png", "jpeg"])

    if st.button("➕ Add Student to Database", use_container_width=True):
        if new_student_name and uploaded_photos:
            student_dir = os.path.join(KNOWN_FACES_DIR, new_student_name)
            os.makedirs(student_dir, exist_ok=True)
            
            for i, photo in enumerate(uploaded_photos):
                try:
                    img = Image.open(photo)
                    img.save(os.path.join(student_dir, f"{new_student_name}_{i+1}.png"))
                except Exception as e:
                    st.error(f"Error saving file {photo.name}: {e}")
            
            st.success(f"Successfully added {len(uploaded_photos)} photos for '{new_student_name}'.")
            st.info("Please rebuild the database now to include the new student.")
        else:
            st.error("Please provide a name and at least one photo.")

    st.divider()

    # --- Section to Rebuild Database ---
    st.subheader("Rebuild Face Database")
    st.warning("This process scans all images in the 'known_faces' directory. Run this after adding or changing student photos.", icon="⚠️")
    
    if st.button("🔄 Rebuild Database", use_container_width=True):
        with st.spinner("Rebuilding database... This might take a while depending on the number of images."):
            if build_database(face_model):
                st.success("Face database rebuilt successfully!")
            else:
                st.error("Database build failed. Check the console for errors or add student data.")

    # --- Section to View Current Students ---
    st.divider()
    with st.expander("📂 View Current Students in Database"):
        if os.path.exists(KNOWN_FACES_DIR):
            student_list = [name for name in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))]
            if student_list:
                for student in student_list:
                    st.markdown(f"- {student}")
            else:
                st.write("No students found in the 'known_faces' directory.")
        else:
            st.write("The 'known_faces' directory has not been created yet.")