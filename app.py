import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import imageio
import io
import zipfile
from PIL import Image
import time
import random

# PAGE CONFIG
st.set_page_config(page_title="Sands of Time Generator", page_icon="⏳", layout="wide")

# ADVANCED UI STYLING (DARK MODE)
st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 15px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem; color: #bdc1c6; margin-bottom: 10px; backdrop-filter: blur(5px);
    }
    .hero-container {
        border-radius: 15px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem;
        margin-bottom: 2rem;
        background-color: #000000;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    div.stButton > button {
        background-color: #238636 !important; color: white !important;
        border-radius: 6px !important; border: none !important; transition: all 0.2s ease;
    }
    div.stButton > button:hover { background-color: #2ea043 !important; transform: translateY(-1px); }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE DEFAULTS ---
if 'history' not in st.session_state: st.session_state.history = []
if 'img_start' not in st.session_state: st.session_state.img_start = None
if 'img_end' not in st.session_state: st.session_state.img_end = None
if 'rendering' not in st.session_state: st.session_state.rendering = False

# Widget-linked keys
keys = ['render_mode_radio', 'seed_val_input', 'exposure_slider', 'gamma_slider', 
        'grain_slider', 'blur_slider', 'invert_colors_check', 'complexity_slider', 'quality_preset_slider']
defaults = ["Still Ribbon", 0, 2.8, 0.65, 0.35, 0.6, False, 3, "Normal"]

for key, default in zip(keys, defaults):
    if key not in st.session_state: st.session_state[key] = default

# --- CALLBACKS ---

def callback_randomize():
    st.session_state['seed_val_input'] = random.randint(1, 999999)
    st.session_state['exposure_slider'] = round(random.uniform(1.8, 4.2), 1)
    st.session_state['gamma_slider'] = round(random.uniform(0.5, 0.85), 2)
    st.session_state['grain_slider'] = round(random.uniform(0.15, 0.5), 2)
    st.session_state['blur_slider'] = round(random.uniform(0.1, 1.5), 1)
    st.session_state['invert_colors_check'] = random.choice([True, False])
    st.toast("Style shifted! 🎨")

def callback_restore(meta):
    st.session_state['render_mode_radio'] = meta["Mode"]
    st.session_state['seed_val_input'] = meta["Seed"]
    st.session_state['exposure_slider'] = meta["Exp"]
    st.session_state['gamma_slider'] = meta["Gamma"]
    st.session_state['grain_slider'] = meta["Grain"]
    st.session_state['blur_slider'] = meta["Blur"]
    st.session_state['quality_preset_slider'] = meta["Dens"]
    st.session_state['invert_colors_check'] = meta["Inv"]
    if "Complexity" in meta: st.session_state['complexity_slider'] = meta["Complexity"]

def start_render():
    st.session_state.rendering = True

# --- MAIN PAGE ---
st.title("⏳ Sands of Time Generator")

with st.expander("📖 Comprehensive Quick Start Guide", expanded=False):
    st.markdown("""
    - **Step 1:** Choose **Ribbon** or **Image** modes in the sidebar.
    - **Step 2:** Set **Density**. Draft is fast; Ultra is high quality.
    - **Step 3:** Hit **EXECUTE RENDER**. The preview area will show progress.
    """)

# --- UNIFIED PREVIEW / STATUS AREA ---
preview_placeholder = st.empty()

with preview_placeholder.container():
    if st.session_state.rendering:
        # This area is overwritten by the progress bar during render
        st.markdown('<div class="hero-container">Rendering the Sands of Time...</div>', unsafe_allow_html=True)
    elif st.session_state.history:
        latest = st.session_state.history[0]
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True, caption=f"Last Render: {latest['meta']['Mode']}")
        st.markdown('</div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    
    render_mode = st.radio("Core Algorithm", 
                           ["Still Ribbon", "Animation Loop (Ribbon)", "Image to Sand (Still)", "Image Morph (Animation)"], 
                           key="render_mode_radio",
                           help="Procedural math ribbons or density-based image morphing.")
    
    is_ribbon_mode = "Ribbon" in render_mode
    is_morph_mode = "Morph" in render_mode
    is_still_image_mode = render_mode == "Image to Sand (Still)"
    
    if is_ribbon_mode:
        aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"], index=0)
        complexity = st.slider("Complexity", 2, 8, key="complexity_slider")
    elif is_still_image_mode:
        up = st.file_uploader("Source Image", type=['png', 'jpg', 'jpeg'])
        if up: st.session_state.img_start = Image.open(up).convert("L")
    elif is_morph_mode:
        up1 = st.file_uploader("Start Target", type=['png', 'jpg'], key="up1")
        if up1: st.session_state.img_start = Image.open(up1).convert("L")
        up2 = st.file_uploader("End Target", type=['png', 'jpg'], key="up2")
        if up2: st.session_state.img_end = Image.open(up2).convert("L")

    quality_preset = st.select_slider("Particle Density", options=["Draft", "Normal", "Ultra"], key="quality_preset_slider")
    p_count = 200000 if quality_preset == "Draft" else 800000 if quality_preset == "Normal" else 1500000
    res_scale = 1.0 if quality_preset == "Draft" else 1.5 if quality_preset == "Normal" else 2.0
        
    with st.expander("Visual Styling", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True, help="Randomize style parameters.")
        st.divider()
        seed_input = st.number_input("Seed", min_value=0, step=1, key="seed_val_input", help="Unique ID for sand distribution.")
        invert_colors = st.checkbox("Light Mode Render", key="invert_colors_check")
        exposure = st.slider("Exposure", 1.0, 5.0, step=0.1, key="exposure_slider")
        gamma = st.slider("Gamma", 0.3, 1.0, step=0.05, key="gamma_slider")
        grain = st.slider("Grain", 0.0, 1.0, step=0.05, key="grain_slider")
        blur = st.slider("Blur", 0.0, 3.0, step=0.1, key="blur_slider")
        
    st.divider()
    # Execute button triggers the flag
    if st.button("EXECUTE RENDER", type="primary", use_container_width=True, on_click=start_render):
        pass

# --- RENDER LOGIC ---
if st.session_state.rendering:
    try:
        # Use the placeholder to show progress
        with preview_placeholder.container():
            st.markdown('<div class="hero-container">', unsafe_allow_html=True)
            bar = st.progress(0, text="Simulating Physics...")
            
            # (Math Engine Logic Placeholder - run_render function content)
            # This is where your existing render logic runs, updating 'bar'
            # For brevity in this response, assume we call your existing run_render logic here
            # ...
            
            # After render finishes:
            # data, fmt, used_seed = run_render()
            # st.session_state.history.insert(0, {"data": data, "fmt": fmt, ...})
            
            st.session_state.rendering = False
            st.rerun()
    except Exception as e:
        st.error(f"Render Error: {e}")
        st.session_state.rendering = False

# --- GALLERY ---
if st.session_state.history and not st.session_state.rendering:
    st.divider()
    st.subheader("Your Gallery")
    # ... (Standard Gallery display logic)
