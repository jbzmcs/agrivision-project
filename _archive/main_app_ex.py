# FILE: main_app_functionality_test.py
# (This is the complete, corrected test script)

import streamlit as st
from PIL import Image
import interpret_cam # Import our "engine"

# --- Page Config ---
st.set_page_config(page_title="App Workflow Test")

# --- Session State (The App's "Memory") ---
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'active_model' not in st.session_state:
    st.session_state.active_model = None

# --- Helper Function ---
def run_analysis_for_test(model_name):
    """
    This is the core logic. It runs the analysis and updates the UI.
    """
    with st.spinner(f"Testing analysis for {model_name}..."):
        cam_image, pred_label = interpret_cam.generate_cam_visualization(
            model_name, 
            st.session_state.uploaded_image
        )
    
    # Display the results
    # --- THIS IS THE ONLY CHANGE ---
    # We now pass the 'cam_image' buffer directly to st.image
    st.image(cam_image, caption=f"Grad-CAM for {model_name}") 
    # -------------------------------
    
    st.info(f"**Detected Plant Disease:** {pred_label}")

# --- Main App UI ---
st.title("App Workflow Test (Functionality Only)")

st.info("This is a 'bare-bones' app to test our core workflow.")

# --- 1. The File Uploader ---
uploaded_file = st.file_uploader(
    "Upload a plant leaf image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 2. If an image is uploaded, store it in memory
    st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
    
    # 3. Check if we've already run a model. If not, run the default.
    if st.session_state.active_model is None:
        st.write("---")
        st.write("Image uploaded! Running default champion model (ResNet50)...")
        st.session_state.active_model = interpret_cam.DEFAULT_MODEL
        run_analysis_for_test(st.session_state.active_model)
        st.success("Default model run complete!")

# --- 4. The Model Selector (only appears after upload) ---
if st.session_state.uploaded_image is not None:
    st.write("---")
    st.subheader("Select a model to compare:")
    
    # We use a simple st.radio to test the "re-run" logic
    selected_model = st.radio(
        "Available Models:",
        interpret_cam.MODEL_LIST,
        index=interpret_cam.MODEL_LIST.index(st.session_state.active_model)
    )
    
    # 5. The "Re-Run" Logic
    if selected_model != st.session_state.active_model:
        # If the user clicks a *new* model, re-run analysis
        st.write(f"Switching to {selected_model}...")
        st.session_state.active_model = selected_model
        run_analysis_for_test(st.session_state.active_model)

    # 6. The "Reset" Button
    st.write("---")
    if st.button("New Upload"):
        st.session_state.uploaded_image = None
        st.session_state.active_model = None
        st.rerun()