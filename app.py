# app.py

import streamlit as st
import os
import cv2
import pickle
import insightface
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import shutil  # <-- Added for removing directories

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
SIMILARITY_THRESHOLD = 0.49 # Tuned based on your last request

# --- INJECT CUSTOM CSS FOR A "FANCY" LOOK ---
def local_css(file_name):
    """ Helper to load custom CSS """
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback CSS if style.css is not found
        st.markdown("""
        <style>
        /* --- Card-like Containers --- */
        .card {
            background-color: #262730; /* Streamlit's dark theme container color */
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .card:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        }

        /* --- Custom Titles --- */
        .title-text {
            font-size: 2.5em;
            font-weight: bold;
            color: #FAFAFA; /* Streamlit's default light text */
            padding-bottom: 10px;
        }
        
        .header-text {
            font-size: 1.75em;
            font-weight: bold;
            color: #FAFAFA;
            padding-bottom: 10px;
        }
        
        /* --- Make buttons pop more --- */
        .stButton > button {
            border-radius: 8px;
            border: 2px solid #FF4B4B; /* Streamlit's primary color */
            color: #FF4B4B;
            background-color: transparent;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            border-color: #FFFFFF;
            color: #FFFFFF;
            background-color: #FF4B4B;
        }
        
        /* --- Style for the "Delete" button to be more dangerous --- */
        .stButton > button[kind="primary"] {
            border: 2px solid #FF4B4B;
            background-color: #FF4B4B;
            color: #FFFFFF;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #FF6B6B;
            border-color: #FF6B6B;
        }
        </style>
        """, unsafe_allow_html=True)

# Apply the custom CSS
local_css("style.css") # You can optionally move the CSS to a file named style.css


# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_face_analysis_model():
    """Loads the insightface model and caches it."""
    app = insightface.app.FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=-1, det_size=(640, 640)) # Use -1 for CPU
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

    try:
        with open(DATABASE_FILE, "rb") as f:
            known_faces_data = pickle.load(f)
        known_embeddings = np.array(known_faces_data["embeddings"])
        known_names = known_faces_data["names"]
    except (pickle.UnpicklingError, EOFError):
        st.error("Database file is corrupt or empty. Please rebuild the database.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the database: {e}")
        return None

    if len(known_names) == 0:
        st.error("Database is empty. Please add students and rebuild the database.")
        return None

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

def remove_student(student_name):
    """Removes a student's directory and all their photos."""
    student_dir = os.path.join(KNOWN_FACES_DIR, student_name)
    if os.path.exists(student_dir):
        try:
            shutil.rmtree(student_dir) # Recursively delete the folder and all its contents
            return True
        except Exception as e:
            st.error(f"Error deleting student folder: {e}")
            return False
    else:
        st.error("Student directory not found. Cannot remove.")
        return False

# --- STREAMLIT UI ---

# Load the model once
face_model = load_face_analysis_model()

# --- Sidebar Navigation ---
st.sidebar.markdown('<div class="title-text" style="font-size: 1.75em; text-align: center; padding-bottom: 20px;">👨‍🎓 FaceRec Pro</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Main Navigation", ["🏠 Home - Recognition", "🗃️ Manage Database", "ℹ️ About"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.info("This app uses InsightFace for high-accuracy face recognition. Manage your student database or run recognition on a class photo.")


# --- Page 1: Home - Student Recognition ---
if page == "🏠 Home - Recognition":
    st.markdown('<div class="title-text">Student Face Recognition</div>', unsafe_allow_html=True)
    st.write("Upload a class photo and the system will identify all known students based on your database.")

    uploaded_file = st.file_uploader("Upload an image for recognition", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="header-text">Original Upload</div>', unsafe_allow_html=True)
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Recognize Students", use_container_width=True, type="primary"):
            with st.spinner("Processing image... This may take a moment."):
                pil_image = Image.open(uploaded_file)
                result_image = recognize_faces(face_model, pil_image)

            if result_image is not None:
                with col2:
                    st.markdown('<div class="header-text">Recognition Result</div>', unsafe_allow_html=True)
                    st.image(result_image, caption="Recognition Result", use_container_width=True, channels="BGR")
    
    st.markdown("---")
    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        1.  **Detection:** The system first scans the image to find all faces using the `buffalo_l` detection model.
        2.  **Embedding:** For each detected face, it generates a 512-dimension "face embedding" (a unique mathematical signature).
        3.  **Comparison:** It compares this new embedding to all the pre-calculated embeddings in the `face_embeddings.pkl` database using Cosine Similarity.
        4.  **Identification:** If a comparison score is higher than the set threshold (currently `0.49`), the name from the database is drawn on the face.
        """)

# --- Page 2: Manage Database ---
elif page == "🗃️ Manage Database":
    st.markdown('<div class="title-text">Manage Student Database</div>', unsafe_allow_html=True)
    st.write("Add new students, remove existing ones, or rebuild the entire face embedding database.")
    
    st.markdown("---")
    
    # --- Split layout into two columns for better organization ---
    col1, col2 = st.columns([1.5, 1]) # Make first column wider
    
    with col1:
        # --- Card 1: Add or Remove Students ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="header-text">Add a New Student</div>', unsafe_allow_html=True)
        
        new_student_name = st.text_input("Enter the student's name (this will be the folder name):")
        uploaded_photos = st.file_uploader("Upload photos (at least one clear photo):", 
                                           accept_multiple_files=True, type=["jpg", "png", "jpeg"])

        if st.button("➕ Add Student to Folder", use_container_width=True):
            if new_student_name and uploaded_photos:
                student_dir = os.path.join(KNOWN_FACES_DIR, new_student_name)
                os.makedirs(student_dir, exist_ok=True)
                
                saved_count = 0
                for i, photo in enumerate(uploaded_photos):
                    try:
                        img = Image.open(photo)
                        # Create a standardized file name
                        img.save(os.path.join(student_dir, f"{new_student_name.lower().replace(' ', '_')}_{i+1}.png"))
                        saved_count += 1
                    except Exception as e:
                        st.error(f"Error saving file {photo.name}: {e}")
                
                st.success(f"Successfully added {saved_count} photos for '{new_student_name}'.")
                st.info("Remember to rebuild the database to include these new images!")
            else:
                st.error("Please provide a name and at least one photo.")
        
        st.divider()

        # --- NEW: Remove Student Section ---
        st.markdown('<div class="header-text">Remove an Existing Student</div>', unsafe_allow_html=True)
        
        if os.path.exists(KNOWN_FACES_DIR):
            student_list = [name for name in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))]
            if not student_list:
                st.info("The 'known_faces' directory is empty. No students to remove.")
            else:
                student_to_remove = st.selectbox("Select student to remove:", options=[""] + student_list)

                if st.button("🔥 Delete Student Permanently", use_container_width=True, type="primary"):
                    if student_to_remove:
                        with st.spinner(f"Removing '{student_to_remove}'..."):
                            if remove_student(student_to_remove):
                                st.success(f"Successfully removed '{student_to_remove}'.")
                                st.warning("IMPORTANT: You must rebuild the database to finalize this change.", icon="⚠️")
                                st.experimental_rerun() # Force rerun to update the selectbox
                    else:
                        st.error("Please select a student to remove.")
        else:
            st.info("The 'known_faces' directory has not been created yet.")
            
        st.markdown('</div>', unsafe_allow_html=True) # End card

    with col2:
        # --- Card 2: Database Maintenance ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="header-text">Database Maintenance</div>', unsafe_allow_html=True)
        
        st.warning("You **must** rebuild the database after adding or removing any students for changes to take effect.", icon="⚠️")
        
        if st.button("🔄 Rebuild Database", use_container_width=True):
            with st.spinner("Rebuilding database... This may take a while."):
                if build_database(face_model):
                    st.success("Face database rebuilt successfully!")
                else:
                    st.error("Database build failed. Check logs or add student data.")
        
        st.divider()

        # --- Section to View Current Students ---
        with st.expander("📂 View Current Students in Folder"):
            if os.path.exists(KNOWN_FACES_DIR):
                student_list = [name for name in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))]
                if student_list:
                    for student in sorted(student_list):
                        st.markdown(f"- `{student}`")
                else:
                    st.write("No students found in the 'known_faces' directory.")
            else:
                st.write("The 'known_faces' directory has not been created yet.")
        
        st.markdown('</div>', unsafe_allow_html=True) # End card


# --- Page 3: About ---
elif page == "ℹ️ About":
    st.markdown('<div class="title-text">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">Core Technology</div>', unsafe_allow_html=True)
    st.markdown(f"""
    This application is an end-to-end face recognition system built in Python and Streamlit.
    
    -   **Face Detection & Recognition:** `insightface`
    -   **Backend Model:** `buffalo_l` (A high-accuracy, lightweight model)
    -   **Comparison Algorithm:** `Cosine Similarity`
    -   **Similarity Threshold:** `{SIMILARITY_THRESHOLD}` (Scores above this are considered a match)
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">Recognition Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    1.  **Database Build:** When you 'Rebuild Database', the system scans all images in the `known_faces` directory. For each face, it generates a 512-dimension mathematical vector (an "embedding") and stores it in the `database/face_embeddings.pkl` file.
    2.  **Live Recognition:** When you upload a new photo, the system detects all faces and generates new embeddings for them.
    3.  **Matching:** Each new embedding is compared against *every* embedding in the database. The system finds the "closest" match and, if the similarity score is above the threshold, identifies the person.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">Deployment Warning ⚠️</div>', unsafe_allow_html=True)
    st.error("""
    **This app runs on an ephemeral filesystem.**
    
    This means any students added or removed via this web interface are **TEMPORARY**. The app's container will be reset (deleting all changes) if it restarts or goes to sleep.
    
    **To make permanent changes:**
    1.  Run this app on your **local computer**.
    2.  Add/Remove students and rebuild the database locally.
    3.  Push the updated `known_faces/` folder and `database/face_embeddings.pkl` file to your GitHub repository.
    """)
    st.markdown('</div>', unsafe_allow_html=True)