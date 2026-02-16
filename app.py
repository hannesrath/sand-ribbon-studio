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

# UI STYLING
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
        min-height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    div.stButton > button {
        background-color: #238636 !important; color: white !important;
        border-radius: 6px !important; border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state: st.session_state.history = []
if 'img_start' not in st.session_state: st.session_state.img_start = None
if 'img_end' not in st.session_state: st.session_state.img_end = None

# Default Widget Values
keys_defaults = {
    'render_mode_radio': "Still Ribbon",
    'seed_val_input': 0,
    'exposure_slider': 2.8,
    'gamma_slider': 0.65,
    'grain_slider': 0.35,
    'blur_slider': 0.6,
    'invert_colors_check': False,
    'complexity_slider': 3,
    'quality_preset_slider': "Normal"
}
for key, val in keys_defaults.items():
    if key not in st.session_state: st.session_state[key] = val

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

def reset_app():
    st.session_state.img_start = None
    st.session_state.img_end = None
    st.session_state.history = []
    st.rerun()

# --- MAIN PAGE ---
st.title("⏳ Sands of Time Generator")

with st.expander("📖 Comprehensive Quick Start Guide", expanded=False):
    st.markdown("""
    - **Selection:** Choose **Ribbon** (math) or **Image** (uploads) in the sidebar.
    - **Execute Render:** The preview area will show progress and then display your result.
    - **Restore:** Click the 🔄 icon in the gallery to reload a previous look.
    """)

# PREVIEW AREA PLACEHOLDER
preview_placeholder = st.empty()

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    
    render_mode = st.radio("Core Algorithm", 
                           ["Still Ribbon", "Animation Loop (Ribbon)", "Image to Sand (Still)", "Image Morph (Animation)"], 
                           key="render_mode_radio",
                           help="Select the generation method.")
    
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
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True)
        st.divider()
        seed_input = st.number_input("Seed", min_value=0, step=1, key="seed_val_input")
        invert_colors = st.checkbox("Light Mode Render", key="invert_colors_check")
        exposure = st.slider("Exposure", 1.0, 5.0, step=0.1, key="exposure_slider")
        gamma = st.slider("Gamma", 0.3, 1.0, step=0.05, key="gamma_slider")
        grain = st.slider("Grain", 0.0, 1.0, step=0.05, key="grain_slider")
        blur = st.slider("Blur", 0.0, 3.0, step=0.1, key="blur_slider")
        
    st.divider()
    execute_render = st.button("EXECUTE RENDER", type="primary", use_container_width=True)
    st.button("Clear History", on_click=reset_app, use_container_width=True)

# --- RENDER EXECUTION ---
if execute_render:
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        bar = st.progress(0, text="Simulating Sands of Time...")
        
        # --- RENDER LOGIC START ---
        frames_list = []
        final_seed = seed_input if seed_input > 0 else np.random.randint(0, 999999)
        rng_main = np.random.RandomState(final_seed)

        if is_ribbon_mode:
            # Ribbon Math logic
            import math
            def generate_dna(c, s):
                np.random.seed(s)
                p = []
                sc = np.random.uniform(1.8, 2.5)
                for i in range(1, c + 1):
                    p.append({'freq': i, 'amp_a': np.random.uniform(-1, 1, 3) * sc / (i**0.8), 'amp_b': np.random.uniform(-1, 1, 3) * sc / (i**0.8), 'phases': np.random.uniform(0, 2*np.pi, 3)})
                return p
            
            width, height = (1920, 1080) if "16:9" in aspect_ratio else (1080, 1920) if "9:16" in aspect_ratio else (1080, 1080)
            dna = generate_dna(complexity, final_seed)
            t_vals = rng_main.rand(p_count) * 2 * np.pi
            total_frames = 100 if "Animation" in render_mode else 1
            bounds_x, bounds_y = [-5, 5], [-5, 5]
            iw, ih = width, height
            sx, sy, sz = [rng_main.normal(0, 0.15, p_count) for _ in range(3)]
        elif is_morph_mode or is_still_image_mode:
            # Image sampling logic
            def sample_img(pil_img, n, s):
                arr = np.power(np.array(pil_img).astype(float) / 255.0, 2.0)
                total = np.sum(arr)
                cdf = np.cumsum(arr.flatten()) / total
                indices = np.searchsorted(cdf, np.random.RandomState(s).rand(n))
                y, x = np.unravel_index(indices, arr.shape)
                return x.astype(float), (arr.shape[0] - y.astype(float)), pil_img.width, pil_img.height

            ix1, iy1, iw, ih = sample_img(st.session_state.img_start, p_count, final_seed)
            if is_morph_mode:
                ix2, iy2, _, _ = sample_img(st.session_state.img_end.resize(st.session_state.img_start.size), p_count, final_seed + 1)
                total_frames, bounds_x, bounds_y = 125, [0, iw], [0, ih]
            else:
                total_frames, bounds_x, bounds_y = 1, [0, iw], [0, ih]

        for i in range(total_frames):
            prog = i / total_frames if total_frames > 1 else 0.0
            if is_ribbon_mode:
                # Calculate ribbon positions
                x, y, z = np.zeros_like(t_vals), np.zeros_like(t_vals), np.zeros_like(t_vals)
                for l in dna:
                    x += (l['amp_a'][0] * np.cos(prog*2*np.pi) + l['amp_b'][0] * np.sin(prog*2*np.pi)) * np.cos(l['freq'] * t_vals + l['phases'][0])
                    y += (l['amp_a'][1] * np.cos(prog*2*np.pi) + l['amp_b'][1] * np.sin(prog*2*np.pi)) * np.sin(l['freq'] * t_vals + l['phases'][1])
                    z += (l['amp_a'][2] * np.cos(prog*2*np.pi) + l['amp_b'][2] * np.sin(prog*2*np.pi)) * np.cos(l['freq'] * t_vals + l['phases'][2])
                xr, yr = x, y # Simple projection
                grain_seed, w_final = final_seed + i, None
            elif is_morph_mode:
                if i < 25: grain_seed, tm, n = final_seed, 0.0, 0.0
                elif i > 100: grain_seed, tm, n = final_seed + 999, 1.0, 0.0
                else:
                    grain_seed, tp = final_seed + i, (i - 25) / 75
                    tm, n = (1 - np.cos(tp * np.pi)) / 2, np.sin(tp * np.pi) * 4.0
                xr, yr, w_final = ix1*(1-tm) + ix2*tm + rng_main.normal(0, n, p_count), iy1*(1-tm) + iy2*tm + rng_main.normal(0, n, p_count), None
            else:
                xr, yr, grain_seed, w_final = ix1, iy1, final_seed, None

            # Render heatmap
            h_map, _, _ = np.histogram2d(xr, yr, bins=[int(iw*res_scale/2), int(ih*res_scale/2)], range=[bounds_x, bounds_y], weights=w_final)
            if blur > 0: h_map = gaussian_filter(h_map, sigma=blur)
            h_map = np.power(h_map / (np.max(h_map) + 1e-10), gamma)
            if grain > 0: h_map *= np.random.RandomState(grain_seed).normal(1.0, grain, h_map.shape)
            h_map = np.clip(h_map, 0, 1)
            if invert_colors: h_map = 1.0 - h_map
            frames_list.append((resize(h_map.T, (1080, int(1080 * iw/ih))) * 255).astype(np.uint8))
            bar.progress((i+1)/total_frames)

        # Finalize asset
        b = io.BytesIO()
        if len(frames_list) > 1: imageio.mimsave(b, frames_list, format='GIF', fps=25, loop=0); fmt = "gif"
        else: imageio.imwrite(b, frames_list[0], format='PNG'); fmt = "png"
        
        meta = {"Mode": render_mode, "Seed": final_seed, "Exp": exposure, "Gamma": gamma, "Grain": grain, "Blur": blur, "Dens": quality_preset, "Inv": invert_colors}
        st.session_state.history.insert(0, {"data": b.getvalue(), "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        # --- RENDER LOGIC END ---
        
        st.rerun()

# DISPLAY HERO (If not rendering)
elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True, caption=f"Latest: {latest['meta']['Mode']} | {latest['time']}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- GALLERY ---
if st.session_state.history:
    st.divider()
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b><br>Seed: {m['Seed']}</div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"sand_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,))
            with c3: st.button("🗑️", key=f"del_{idx}", on_click=lambda i=idx: (st.session_state.history.pop(i), st.rerun()))
