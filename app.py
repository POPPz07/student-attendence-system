# app.py

import streamlit as st
import os
import cv2
import pickle
import insightface
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import shutil

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

# --- INJECT CUSTOM CSS FOR A "FANCY" LOOK ---
def local_css(file_name):
    """ Helper to load custom CSS """
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Advanced CSS with light theme and professional styling
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
        
        /* --- Global Styles --- */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* --- Light Background --- */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
            background-attachment: fixed;
        }
        
        /* --- Hero Header with Gradient Text --- */
        .hero-title {
            font-size: 3.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 10px;
            animation: fadeInDown 0.8s ease-out;
            letter-spacing: -1px;
        }
        
        .hero-subtitle {
            text-align: center;
            color: #5a6c7d;
            font-size: 1.2em;
            margin-bottom: 30px;
            animation: fadeInUp 0.8s ease-out;
        }
        
        /* --- Premium Card Containers --- */
        .card {
            background: #ffffff;
            border-radius: 16px;
            padding: 35px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.06);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(0, 0, 0, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(74, 144, 226, 0.05), transparent);
            transition: left 0.7s;
        }
        
        .card:hover::before {
            left: 100%;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(74, 144, 226, 0.15), 0 4px 8px rgba(0, 0, 0, 0.08);
        }
        
        /* --- Gradient Card Variant --- */
        .card-gradient {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
            border-radius: 16px;
            padding: 35px;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(74, 144, 226, 0.12);
            border: 1px solid rgba(74, 144, 226, 0.15);
            transition: all 0.3s ease;
        }
        
        .card-gradient:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(74, 144, 226, 0.2);
        }
        
        /* --- Section Headers --- */
        .title-text {
            font-size: 2.8em;
            font-weight: 800;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            padding-bottom: 15px;
            margin-bottom: 20px;
            animation: fadeInLeft 0.6s ease-out;
        }
        
        .header-text {
            font-size: 1.9em;
            font-weight: 700;
            color: #2c3e50;
            padding-bottom: 12px;
            border-bottom: 3px solid transparent;
            border-image: linear-gradient(90deg, #4A90E2, #357ABD) 1;
            margin-bottom: 20px;
            display: inline-block;
        }
        
        .section-header {
            font-size: 1.5em;
            font-weight: 600;
            color: #34495e;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* --- Premium Buttons --- */
        .stButton > button {
            border-radius: 12px;
            border: 2px solid transparent;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            color: #FFFFFF;
            font-weight: 600;
            font-size: 1.05em;
            padding: 12px 28px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(74, 144, 226, 0.4);
            background: linear-gradient(135deg, #5ba3f5 0%, #4682c4 100%);
        }
        
        .stButton > button:active {
            transform: translateY(0px);
        }
        
        /* --- Danger Button (Delete) --- */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            border: 2px solid transparent;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }
        
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 25px rgba(231, 76, 60, 0.4);
            transform: translateY(-2px);
            background: linear-gradient(135deg, #ec6254 0%, #d44638 100%);
        }
        
        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8f9fc 100%);
            border-right: 1px solid rgba(74, 144, 226, 0.15);
        }
        
        [data-testid="stSidebar"] .title-text {
            font-size: 2em;
            color: #2c3e50;
        }
        
        /* --- Radio Buttons --- */
        .stRadio > label {
            font-size: 1.1em;
            font-weight: 600;
            color: #34495e;
        }
        
        .stRadio > div {
            gap: 15px;
        }
        
        .stRadio > div > label {
            background: rgba(74, 144, 226, 0.08);
            padding: 12px 20px;
            border-radius: 12px;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            cursor: pointer;
            color: #2c3e50;
        }
        
        .stRadio > div > label:hover {
            background: rgba(74, 144, 226, 0.15);
            border-color: rgba(74, 144, 226, 0.3);
            transform: translateX(5px);
        }
        
        /* --- File Uploader --- */
        [data-testid="stFileUploader"] {
            background: rgba(74, 144, 226, 0.03);
            border: 2px dashed rgba(74, 144, 226, 0.3);
            border-radius: 15px;
            padding: 25px;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(74, 144, 226, 0.5);
            background: rgba(74, 144, 226, 0.08);
        }
        
        /* --- Text Input --- */
        .stTextInput > div > div > input {
            background: #ffffff;
            border: 2px solid rgba(74, 144, 226, 0.2);
            border-radius: 12px;
            color: #2c3e50;
            padding: 12px 16px;
            font-size: 1.05em;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4A90E2;
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
            background: #ffffff;
        }
        
        /* --- Select Box --- */
        .stSelectbox > div > div > div {
            background: #ffffff;
            border: 2px solid rgba(74, 144, 226, 0.2);
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .stSelectbox > div > div > div:hover {
            border-color: rgba(74, 144, 226, 0.4);
        }
        
        /* --- Expander --- */
        .streamlit-expanderHeader {
            background: rgba(74, 144, 226, 0.08);
            border-radius: 12px;
            border: 1px solid rgba(74, 144, 226, 0.15);
            font-weight: 600;
            font-size: 1.05em;
            transition: all 0.3s ease;
            color: #2c3e50;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(74, 144, 226, 0.12);
            border-color: rgba(74, 144, 226, 0.3);
        }
        
        /* --- Info/Warning/Error Boxes --- */
        .stInfo, .stWarning, .stError, .stSuccess {
            border-radius: 12px;
            border-left: 4px solid;
            padding: 15px 20px;
            background: #ffffff;
        }
        
        /* --- Progress Bar --- */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
            border-radius: 10px;
        }
        
        /* --- Divider --- */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(74, 144, 226, 0.3), transparent);
            margin: 30px 0;
        }
        
        /* --- Animations --- */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* --- Image Styling --- */
        [data-testid="stImage"] {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        [data-testid="stImage"]:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 32px rgba(74, 144, 226, 0.2);
        }
        
        /* --- Badge Styling --- */
        .badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            color: white;
            font-weight: 600;
            font-size: 0.9em;
            margin: 5px;
            box-shadow: 0 2px 10px rgba(74, 144, 226, 0.25);
        }
        
        /* --- Stats Card --- */
        .stat-card {
            background: linear-gradient(135deg, rgba(74, 144, 226, 0.08) 0%, rgba(53, 122, 189, 0.08) 100%);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(74, 144, 226, 0.2);
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(74, 144, 226, 0.2);
        }
        
        .stat-number {
            font-size: 2.5em;
            font-weight: 800;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .stat-label {
            color: #5a6c7d;
            font-size: 1em;
            font-weight: 600;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)

# Apply the custom CSS
local_css("style.css")


# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_face_analysis_model():
    """Loads the insightface model and caches it."""
    app = insightface.app.FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=-1, det_size=(640, 640))
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
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)
            
        label = f"{name}: {best_match_score:.2f}"
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (bbox[0], bbox[1] - h - 10), (bbox[0] + w, bbox[1] - 5), color, -1)
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def remove_student(student_name):
    """Removes a student's directory and all their photos."""
    student_dir = os.path.join(KNOWN_FACES_DIR, student_name)
    if os.path.exists(student_dir):
        try:
            shutil.rmtree(student_dir)
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
st.sidebar.markdown('<div class="hero-title" style="font-size: 2.2em; text-align: center; padding: 20px 0;">👨‍🎓 Veriface</div>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="text-align: center; color: #5a6c7d; font-size: 0.95em; margin-bottom: 25px;">Advanced Recognition System</p>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home - Recognition", "🗃️ Manage Database", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="card" style="padding: 20px; margin: 15px 0;">', unsafe_allow_html=True)
st.sidebar.markdown("### 🎯 Quick Info")
st.sidebar.info("This app uses **InsightFace** with the Buffalo_L model for high-accuracy face recognition. Manage your student database or run recognition on class photos.", icon="💡")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Database stats in sidebar
if os.path.exists(DATABASE_FILE):
    try:
        with open(DATABASE_FILE, "rb") as f:
            known_faces_data = pickle.load(f)
        total_embeddings = len(known_faces_data["names"])
        unique_students = len(set(known_faces_data["names"]))
        
        st.sidebar.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div class="stat-number">{unique_students}</div>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div class="stat-label">Students in Database</div>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div class="stat-number">{total_embeddings}</div>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div class="stat-label">Total Face Embeddings</div>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    except:
        pass


# --- Page 1: Home - Recognition ---
if page == "🏠 Home - Recognition":
    # Hero Section
    st.markdown('<div class="hero-title">Student Face Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">🎓 Upload a class photo and instantly identify all registered students</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content container
    with st.container():
        st.markdown('<div class="card-gradient">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📸 Drop your image here or click to browse", 
            type=["jpg", "png", "jpeg"],
            label_visibility="visible"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Display original image
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📤 Original Upload</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recognition button
        if st.button("🔍 Recognize Students", use_container_width=True, type="primary"):
            with st.spinner("🔄 Processing image... Detecting faces and matching identities..."):
                pil_image = Image.open(uploaded_file)
                result_image = recognize_faces(face_model, pil_image)

            if result_image is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">✅ Recognition Result</div>', unsafe_allow_html=True)
                st.image(result_image, use_container_width=True, channels="BGR")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Info section
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.expander("💡 How does this work?", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔍 Detection Phase")
                st.markdown("""
                The system scans the uploaded image using the **buffalo_l** detection model to locate all faces with high precision.
                """)
                
                st.markdown("#### 🧬 Embedding Generation")
                st.markdown("""
                Each detected face is converted into a **512-dimensional vector** (embedding) that represents unique facial features.
                """)
            
            with col2:
                st.markdown("#### 📊 Similarity Comparison")
                st.markdown("""
                The new embeddings are compared against the database using **Cosine Similarity** to find the closest matches.
                """)
                
                st.markdown("#### ✨ Identification")
                st.markdown(f"""
                If the similarity score exceeds **{SIMILARITY_THRESHOLD}**, the system identifies the person and labels them in the image.
                """)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Page 2: Manage Database ---
elif page == "🗃️ Manage Database":
    st.markdown('<div class="hero-title">Manage Student Database</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">🎯 Add students, remove entries, or rebuild the recognition database</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1.6, 1], gap="large")
    
    with col1:
        # Add Student Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="header-text">➕ Add New Student</div>', unsafe_allow_html=True)
        
        new_student_name = st.text_input(
            "Student Name",
            placeholder="e.g., John Doe",
            help="This will be used as the folder name and display name"
        )
        
        uploaded_photos = st.file_uploader(
            "📷 Upload Student Photos (Multiple images recommended)", 
            accept_multiple_files=True, 
            type=["jpg", "png", "jpeg"],
            help="Upload at least one clear, front-facing photo"
        )

        if st.button("✨ Add Student to Database", use_container_width=True):
            if new_student_name and uploaded_photos:
                student_dir = os.path.join(KNOWN_FACES_DIR, new_student_name)
                os.makedirs(student_dir, exist_ok=True)
                
                saved_count = 0
                progress_bar = st.progress(0)
                for i, photo in enumerate(uploaded_photos):
                    try:
                        img = Image.open(photo)
                        img.save(os.path.join(student_dir, f"{new_student_name.lower().replace(' ', '_')}_{i+1}.png"))
                        saved_count += 1
                        progress_bar.progress((i + 1) / len(uploaded_photos))
                    except Exception as e:
                        st.error(f"Error saving file {photo.name}: {e}")
                
                progress_bar.empty()
                st.success(f"🎉 Successfully added **{saved_count}** photos for '{new_student_name}'!")
                st.info("⚠️ Remember to **rebuild the database** to include these new images!", icon="💾")
            else:
                st.error("❌ Please provide a student name and at least one photo.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Remove Student Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="header-text">🗑️ Remove Existing Student</div>', unsafe_allow_html=True)
        
        if os.path.exists(KNOWN_FACES_DIR):
            student_list = [name for name in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))]
            if not student_list:
                st.info("📂 The 'known_faces' directory is empty. No students to remove.", icon="ℹ️")
            else:
                student_to_remove = st.selectbox(
                    "Select student to remove:",
                    options=[""] + sorted(student_list),
                    help="Choose a student to permanently delete from the database"
                )

                if st.button("🔥 Delete Student Permanently", use_container_width=True, type="primary"):
                    if student_to_remove:
                        with st.spinner(f"🗑️ Removing '{student_to_remove}'..."):
                            if remove_student(student_to_remove):
                                st.success(f"✅ Successfully removed '{student_to_remove}'.")
                                st.warning("⚠️ **IMPORTANT:** You must rebuild the database to finalize this change!", icon="🔄")
                                st.experimental_rerun()
                    else:
                        st.error("❌ Please select a student to remove.")
        else:
            st.info("📂 The 'known_faces' directory has not been created yet.", icon="ℹ️")
            
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Database Maintenance Section
        st.markdown('<div class="card-gradient">', unsafe_allow_html=True)
        st.markdown('<div class="header-text">🔧 Database Maintenance</div>', unsafe_allow_html=True)
        
        st.warning("⚠️ You **must** rebuild the database after adding or removing any students for changes to take effect.", icon="🔄")
        
        if st.button("🔄 Rebuild Database", use_container_width=True):
            with st.spinner("⚙️ Rebuilding database... This may take a while."):
                if build_database(face_model):
                    st.success("🎉 Face database rebuilt successfully!")
                    st.balloons()
                else:
                    st.error("❌ Database build failed. Check logs or add student data.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # View Current Students Section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="header-text">👥 Current Students</div>', unsafe_allow_html=True)
        
        if os.path.exists(KNOWN_FACES_DIR):
            student_list = [name for name in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, name))]
            if student_list:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{len(student_list)}</div><div class="stat-label">Registered Students</div></div>', unsafe_allow_html=True)
                
                with st.expander("📋 View Full Student List", expanded=True):
                    for i, student in enumerate(sorted(student_list), 1):
                        st.markdown(f'<span class="badge">{i}. {student}</span>', unsafe_allow_html=True)
            else:
                st.info("📂 No students found in the 'known_faces' directory.", icon="ℹ️")
        else:
            st.info("📂 The 'known_faces' directory has not been created yet.", icon="ℹ️")
        
        st.markdown('</div>', unsafe_allow_html=True)


# --- Page 3: About ---
elif page == "ℹ️ About":
    st.markdown('<div class="hero-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">🚀 Advanced AI-powered face recognition system</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown('<div class="card-gradient">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">⚙️ Core Technology Stack</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 AI/ML Components
        - **Face Detection & Recognition:** `insightface`
        - **Backend Model:** `buffalo_l` (High-accuracy, lightweight)
        - **Comparison Algorithm:** `Cosine Similarity`
        - **Framework:** `Streamlit` + `OpenCV`
        """)
    
    with col2:
        st.markdown(f"""
        #### 🎯 Configuration
        - **Embedding Dimensions:** `512`
        - **Detection Size:** `640x640`
        - **Processing:** CPU-optimized
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Recognition Workflow
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">🔄 Recognition Workflow</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📥 Phase 1: Database Build
        
        1. **Image Collection** - System scans all images in the `known_faces/` directory
        2. **Face Detection** - Locates faces in each image using buffalo_l
        3. **Embedding Generation** - Creates 512-D vectors for each face
        4. **Storage** - Saves all embeddings to `face_embeddings.pkl`
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Phase 2: Live Recognition
        
        1. **Upload Processing** - Detect faces in the new photo
        2. **Feature Extraction** - Generate embeddings for detected faces
        3. **Similarity Matching** - Compare against database using cosine similarity
        4. **Identification** - Label faces with names if score exceeds threshold
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Technical Details
    st.markdown('<div class="card-gradient">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">🧠 How Face Recognition Works</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Face recognition is achieved through deep learning models that transform facial images into mathematical representations:
    
    - **Neural Network Architecture**: The buffalo_l model uses a ResNet-based architecture trained on millions of faces
    - **Embedding Space**: Each face is mapped to a 512-dimensional vector where similar faces are close together
    - **Cosine Similarity**: Measures the angle between two vectors, ranging from -1 (opposite) to 1 (identical)
    - **Threshold Selection**: The threshold balances between false positives and false negatives
    
    **Why 512 dimensions?** This provides enough capacity to distinguish billions of unique faces while remaining computationally efficient.
    """.format(threshold=SIMILARITY_THRESHOLD))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Deployment Warning
    st.markdown('<div class="card" style="border: 2px solid rgba(231, 76, 60, 0.3);">', unsafe_allow_html=True)
    st.markdown('<div class="header-text" style="color: #e74c3c;">⚠️ Important Deployment Notice</div>', unsafe_allow_html=True)
    
    st.error("""
    ### 🔴 Ephemeral Filesystem Warning
    
    **This app runs on an ephemeral (temporary) filesystem.**
    
    Any students added or removed via this web interface are **TEMPORARY**. The app's container will be reset upon restart, deleting all changes.
    
    ### ✅ To make permanent changes:
    
    1. **Run locally** - Clone and run this app on your personal computer
    2. **Make changes** - Add/remove students and rebuild the database locally
    3. **Push to repository** - Upload the updated `known_faces/` folder and `database/face_embeddings.pkl` to your Git repository
    4. **Redeploy** - Your changes will persist in the deployed version
    
    ### 💾 For production use:
    
    Consider integrating with cloud storage (AWS S3, Google Cloud Storage) or a persistent database to maintain student data across restarts.
    """, icon="🚨")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Credits & Resources
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="header-text">🙏 Credits & Resources</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📚 Libraries Used
        - **InsightFace** - Face detection and recognition
        - **OpenCV** - Image processing
        - **Streamlit** - Web application framework
        - **scikit-learn** - Similarity computations
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 Useful Links
        - [InsightFace Documentation](https://github.com/deepinsight/insightface)
        - [Streamlit Docs](https://docs.streamlit.io)
        - [Face Recognition Papers](https://paperswithcode.com/task/face-recognition)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)