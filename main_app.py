import streamlit as st
from PIL import Image
import io

# Import our "engine" script
try:
    import interpret_cam
except ImportError:
    st.error("Fatal Error: `interpret_cam.py` not found. Please ensure both scripts are in the same folder.")
    st.stop()

# --- 1. Set up the page configuration ---
st.set_page_config(
    page_title="AgriVision Analyzer",
    layout="wide"
)

# Injects CSS to create the "green notification bar"
st.markdown("""
<style>
    /* This creates the green "Detected Plant Disease" bar */
    .disease-bar {
        background-color: #1E4E4D; 
        color: #EAEAEA; 
        padding: 12px;
        border-radius: 8px; /* Rounded corners */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Define our model list ---
# Get the model names from our engine's config
MODEL_LIST = list(interpret_cam.MODEL_CONFIG.keys()) 
DEFAULT_MODEL = MODEL_LIST[0] # This is "ResNet50 (Champion)"


# --- 3. Initialize Session State (The App's "Memory") ---
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'active_model' not in st.session_state:
    st.session_state.active_model = None
if 'cam_image' not in st.session_state:
    st.session_state.cam_image = None
if 'pred_label' not in st.session_state:
    st.session_state.pred_label = None

def clear_session_state():
    """
    Called by the 'New Upload' button to reset the app.
    """
    st.session_state.uploaded_image = None
    st.session_state.active_model = None
    st.session_state.cam_image = None
    st.session_state.pred_label = None

def run_analysis(model_name):
    """
    A helper function to run the analysis and update session state.
    """
    with st.spinner(f"Analyzing with {model_name}..."):
        cam_image, pred_label = interpret_cam.generate_cam_visualization(
            model_name, 
            st.session_state.uploaded_image
        )
    
    # Save results to memory
    st.session_state.cam_image = cam_image
    st.session_state.pred_label = pred_label
    st.session_state.active_model = model_name

# --- 4. Build the User Interface (UI) ---
st.title("🍃 AgriVision Plant Disease Analyzer")

col1, col2 = st.columns([0.6, 0.4]) # 60% for image, 40% for controls

# --- Column 1: Image Upload and Display Card ---
with col1:
    # --- THIS IS THE NEW "CARD" CONTAINER from your wireframe ---
    with st.container(border=True): 
        if st.session_state.uploaded_image is None:
            # --- STATE 1: HOME ---
            st.subheader("Upload Photo")
            uploaded_file = st.file_uploader(
                "Upload a plant leaf image...", 
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed", # Hides the label
                on_change=clear_session_state 
            )
            
            if uploaded_file is not None:
                st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                
                # --- AUTOMATIC DEFAULT RUN ---
                run_analysis(DEFAULT_MODEL) # Run our Champion model
                st.rerun() # Re-run the script to show the "Output State"
        
        else:
            # --- STATE 2: OUTPUT ---
            st.subheader("Analysis Result") 
            
            # --- 2. "Detected Plant Disease" bar MOVED HERE ---
            # --- Use our custom CSS green bar ---
            st.markdown(
                f'<div class="disease-bar">Detected Plant Disease: {st.session_state.pred_label}</div>', 
                unsafe_allow_html=True
            )
            
            st.divider() # Visual separator
            
            # --- 3. Image with FIXED WIDTH ---
            st.image(
                st.session_state.cam_image, 
                caption=f"Model Analysis: {st.session_state.active_model}",
                width=650 # <-- Set a fixed width (e.g., 650px)
                          # This fixes the deprecation warning AND the large size
            )
            # -------------------------------------------------

# --- Column 2: Controls Card ---
with col2:
    # --- THIS IS THE NEW "CARD" CONTAINER from your wireframe ---
    with st.container(border=True):
        st.subheader("Controls")
        
        if st.session_state.uploaded_image is None:
            st.info("Please upload an image to begin.")
        else:
            # --- STATE 2: OUTPUT CONTROLS ---
            st.write("**Compare Models:**")
            
            selected_model = st.session_state.active_model
            
            # Create 4 buttons. When a new one is clicked, re-run analysis.
            if st.button(MODEL_LIST[0], use_container_width=True, type="primary" if selected_model == MODEL_LIST[0] else "secondary"):
                if selected_model != MODEL_LIST[0]:
                    run_analysis(MODEL_LIST[0])
                    st.rerun()

            if st.button(MODEL_LIST[1], use_container_width=True, type="primary" if selected_model == MODEL_LIST[1] else "secondary"):
                if selected_model != MODEL_LIST[1]:
                    run_analysis(MODEL_LIST[1])
                    st.rerun()
                    
            if st.button(MODEL_LIST[2], use_container_width=True, type="primary" if selected_model == MODEL_LIST[2] else "secondary"):
                if selected_model != MODEL_LIST[2]:
                    run_analysis(MODEL_LIST[2])
                    st.rerun()
                    
            if st.button(MODEL_LIST[3], use_container_width=True, type="primary" if selected_model == MODEL_LIST[3] else "secondary"):
                if selected_model != MODEL_LIST[3]:
                    run_analysis(MODEL_LIST[3])
                    st.rerun()

            # "New Upload" button (matches your wireframe)
            st.divider()
            st.button("New Upload", on_click=clear_session_state, use_container_width=True)