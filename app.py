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
    /* Global Background and Text */
    .stApp {
        background-color: #0e1117 !important;
        color: #e0e0e0 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Glassmorphism Cards for Gallery */
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem; 
        color: #bdc1c6; 
        margin-bottom: 10px;
        backdrop-filter: blur(5px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #238636 !important; /* Green for Generate */
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    
    div.stButton > button:hover {
        background-color: #2ea043 !important;
        transform: translateY(-1px);
    }
    
    /* Secondary/Reset Buttons */
    [data-testid="stSidebar"] div.stButton > button:nth-child(2) {
        background-color: transparent !important;
        border: 1px solid #30363d !important;
        color: #8b949e !important;
    }

    /* Sliders and Inputs */
    .stSlider [data-baseweb="slider"] {
        margin-bottom: 25px;
    }
    
    /* Tutorial Expander */
    .stExpander {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# SESSION STATE INITIALIZATION
if 'history' not in st.session_state: st.session_state.history = []
if 'img_start' not in st.session_state: st.session_state.img_start = None
if 'img_end' not in st.session_state: st.session_state.img_end = None

# --- UI HELPERS ---

def randomize_settings():
    modes = ["Still Ribbon", "Animation Loop (Ribbon)", "Image to Sand (Still)", "Image Morph (Animation)"]
    st.session_state["render_mode"] = random.choice(modes)
    st.session_state["seed_val"] = random.randint(1, 999999)
    st.session_state["exposure"] = round(random.uniform(1.8, 4.2), 1)
    st.session_state["gamma"] = round(random.uniform(0.5, 0.85), 2)
    st.session_state["grain"] = round(random.uniform(0.15, 0.5), 2)
    st.session_state["blur"] = round(random.uniform(0.2, 1.2), 1)
    st.session_state["quality_preset"] = random.choice(["Draft", "Normal"])
    st.session_state["invert_colors"] = random.choice([True, False])
    if "complexity" in st.session_state:
        st.session_state["complexity"] = random.randint(3, 6)
    st.toast("The sands have shifted! 🎲")

def restore_settings(meta):
    st.session_state["render_mode"] = meta["Mode"]
    st.session_state["seed_val"] = meta["Seed"]
    st.session_state["exposure"] = meta["Exp"]
    st.session_state["gamma"] = meta["Gamma"]
    st.session_state["grain"] = meta["Grain"]
    st.session_state["blur"] = meta["Blur"]
    st.session_state["quality_preset"] = meta["Dens"]
    st.session_state["invert_colors"] = meta["Inv"]
    if "Complexity" in meta: st.session_state["complexity"] = meta["Complexity"]
    st.rerun()

def delete_item(index):
    st.session_state.history.pop(index)
    st.rerun()

def reset_app():
    st.session_state.img_start = None
    st.session_state.img_end = None
    st.session_state.history = []
    st.rerun()

def create_zip_export(history):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        metadata_content = "SANDS OF TIME GENERATOR - EXPORT LOG\n" + "="*40 + "\n\n"
        for i, item in enumerate(history):
            filename = f"sand_{i}.{item['fmt']}"
            zip_file.writestr(filename, item['data'])
            m = item['meta']
            metadata_content += f"File: {filename} | Mode: {m['Mode']} | Seed: {m['Seed']}\n"
        zip_file.writestr("metadata_log.txt", metadata_content)
    return zip_buffer.getvalue()

# --- MAIN PAGE HEADER ---
st.title("⏳ Sands of Time")
st.caption("A cinematic procedural sand art generator.")

with st.expander("📖 Quick Start Guide"):
    st.markdown("""
    - **Selection:** Choose between math-based ribbons or your own images.
    - **Density:** 'Normal' is best for most; 'Ultra' is for final high-res exports.
    - **Look Dev:** Use **Exposure** to bring out the 'light' in the sand particles.
    - **Grain:** Adds filmic texture (Optimized for flicker-free animations).
    """)

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    
    st.button("🎲 Surprise Me!", on_click=randomize_settings, use_container_width=True)
    st.divider()

    render_mode = st.radio("Core Algorithm", 
                           ["Still Ribbon", "Animation Loop (Ribbon)", "Image to Sand (Still)", "Image Morph (Animation)"], 
                           key="render_mode")
    
    is_ribbon_mode = "Ribbon" in render_mode
    is_morph_mode = "Morph" in render_mode
    is_still_image_mode = render_mode == "Image to Sand (Still)"
    
    if is_ribbon_mode:
        aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"], index=0)
        complexity = st.slider("Complexity", 2, 8, value=st.session_state.get("complexity", 3), key="complexity")
    elif is_still_image_mode:
        up = st.file_uploader("Source Image", type=['png', 'jpg', 'jpeg'])
        if up: st.session_state.img_start = Image.open(up).convert("L")
    elif is_morph_mode:
        up1 = st.file_uploader("Start Target", type=['png', 'jpg'], key="up1")
        if up1: st.session_state.img_start = Image.open(up1).convert("L")
        up2 = st.file_uploader("End Target", type=['png', 'jpg'], key="up2")
        if up2: st.session_state.img_end = Image.open(up2).convert("L")

    quality_preset = st.select_slider("Particle Density", options=["Draft", "Normal", "Ultra"], key="quality_preset")
    if quality_preset == "Draft": p_count, res_scale = 200_000, 1.0 
    elif quality_preset == "Normal": p_count, res_scale = 800_000, 1.5
    else: p_count, res_scale = 1_500_000, 2.0 
        
    with st.expander("Visual Styling", expanded=True):
        seed_input = st.number_input("Seed", min_value=0, step=1, key="seed_val")
        invert_colors = st.checkbox("Light Mode Render", key="invert_colors")
        exposure = st.slider("Exposure", 1.0, 5.0, step=0.1, key="exposure")
        gamma = st.slider("Gamma", 0.3, 1.0, step=0.05, key="gamma")
        grain = st.slider("Grain", 0.0, 1.0, step=0.05, key="grain")
        blur = st.slider("Blur", 0.0, 3.0, step=0.1, key="blur")
        
    st.divider()
    generate_btn = st.button("EXECUTE RENDER", type="primary", use_container_width=True)
    st.button("Clear History", on_click=reset_app, use_container_width=True)

# --- CORE MATH ENGINE ---
# (Includes previously implemented grain flicker fix)

def get_resolution(aspect_name):
    if "16:9" in aspect_name: return 1920, 1080
    if "9:16" in aspect_name: return 1080, 1920
    return 1080, 1080

def generate_ribbon_dna(complexity_val, seed):
    np.random.seed(seed)
    params = []
    scale = np.random.uniform(1.8, 2.5)
    for i in range(1, complexity_val + 1):
        decay = 0.8
        amp_a = np.random.uniform(-1, 1, 3) * scale / (i**decay)
        amp_b = np.random.uniform(-1, 1, 3) * scale / (i**decay)
        params.append({'freq': i, 'amp_a': amp_a, 'amp_b': amp_b, 'phases': np.random.uniform(0, 2*np.pi, 3)})
    return {'params': params, 'tilt_x': np.radians(np.random.uniform(20, 80)), 'tilt_y': np.radians(np.random.uniform(0, 360))}

def calc_ribbon_spine(t, dna, prog):
    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    angle = prog * 2 * np.pi
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    for l in dna['params']:
        ax, ay, az = [l['amp_a'][j] * cos_a + l['amp_b'][j] * sin_a for j in range(3)]
        x += ax * np.cos(l['freq'] * t + l['phases'][0])
        y += ay * np.sin(l['freq'] * t + l['phases'][1])
        z += az * np.cos(l['freq'] * t + l['phases'][2])
    return x, y, z

def sample_image_density(pil_img, num_particles, seed):
    img_arr = np.array(pil_img).astype(float)
    if np.max(img_arr) > 0: img_arr = (img_arr / np.max(img_arr)) * 255.0
    img_arr = np.power(img_arr / 255.0, 2.0) 
    total = np.sum(img_arr)
    if total == 0: return np.zeros(num_particles), np.zeros(num_particles), pil_img.width, pil_img.height
    cdf = np.cumsum(img_arr.flatten()) / total
    rng = np.random.RandomState(seed)
    indices = np.searchsorted(cdf, rng.rand(num_particles))
    y, x = np.unravel_index(indices, img_arr.shape)
    return x.astype(float), (img_arr.shape[0] - y.astype(float)), pil_img.width, pil_img.height

def run_render():
    bar = st.progress(0)
    frames_list = []
    final_seed = seed_input if seed_input > 0 else np.random.randint(0, 999999)
    rng_main = np.random.RandomState(final_seed)

    if is_ribbon_mode:
        width, height = get_resolution(aspect_ratio)
        dna = generate_ribbon_dna(complexity, final_seed)
        t_vals = rng_main.rand(p_count) * 2 * np.pi
        total_frames = 100 if "Animation" in render_mode else 1
        bounds_x, bounds_y = [-5, 5], [-5, 5]
        iw, ih = width, height
        thickness = (np.sin(t_vals * 2.0 + rng_main.rand()*6) + 1.2) * 0.5
        sx, sy, sz = [rng_main.normal(0, 0.15, p_count) for _ in range(3)]
    elif is_morph_mode:
        w_s, h_s = st.session_state.img_start.size
        ix1, iy1, iw, ih = sample_image_density(st.session_state.img_start, p_count, final_seed)
        ix2, iy2, _, _ = sample_image_density(st.session_state.img_end.resize((w_s, h_s)), p_count, final_seed + 1)
        total_frames, bounds_x, bounds_y = 125, [0, iw], [0, ih]
    else: 
        ix1, iy1, iw, ih = sample_image_density(st.session_state.img_start, p_count, final_seed)
        total_frames, bounds_x, bounds_y = 1, [0, iw], [0, ih]

    for i in range(total_frames):
        prog = i / total_frames if total_frames > 1 else 0.0
        if is_morph_mode:
            if i < 25: grain_seed, t_m, noise = final_seed, 0.0, 0.0
            elif i > 100: grain_seed, t_m, noise = final_seed + 999, 1.0, 0.0
            else:
                grain_seed, tp = final_seed + i, (i - 25) / 75
                t_m, noise = (1 - np.cos(tp * np.pi)) / 2, np.sin(tp * np.pi) * 4.0
            xr, yr = ix1*(1-t_m) + ix2*t_m + rng_main.normal(0, noise, p_count), iy1*(1-t_m) + iy2*t_m + rng_main.normal(0, noise, p_count)
            w_final = None
        elif is_ribbon_mode:
            grain_seed = final_seed + i
            xs, ys, zs = calc_ribbon_spine(t_vals, dna, prog)
            x, y, z = xs + sx*thickness, ys + sy*thickness, zs + sz*thickness
            tx, ty = dna['tilt_x'], dna['tilt_y']
            yr_r, zr_r = y*np.cos(tx) - z*np.sin(tx), y*np.sin(tx) + z*np.cos(tx)
            xr, yr = x*np.cos(ty) + zr_r*np.sin(ty), yr_r
            w_final = np.exp(-(zr_r - zr_r.min()) / (zr_r.max() - zr_r.min() + 1e-6) * 1.5)
        else:
            grain_seed, xr, yr, w_final = final_seed, ix1, iy1, None

        heatmap, _, _ = np.histogram2d(xr, yr, bins=[int(iw*res_scale/2), int(ih*res_scale/2)], range=[bounds_x, bounds_y], weights=w_final)
        if blur > 0: heatmap = gaussian_filter(heatmap, sigma=blur)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        heatmap = np.power(np.log1p(heatmap * exposure * 10) / np.log1p(exposure * 10), gamma)
        if grain > 0:
            rng_grain = np.random.RandomState(grain_seed)
            heatmap *= rng_grain.normal(1.0, grain, heatmap.shape)
        heatmap = np.clip(heatmap, 0, 1)
        if invert_colors: heatmap = 1.0 - heatmap
        frames_list.append((resize(np.flipud(heatmap.T), (1080, int(1080 * iw/ih))) * 255).astype(np.uint8))
        bar.progress((i+1)/total_frames)

    b = io.BytesIO()
    if len(frames_list) > 1: imageio.mimsave(b, frames_list, format='GIF', fps=25, loop=0); fmt = "gif"
    else: imageio.imwrite(b, frames_list[0], format='PNG'); fmt = "png"
    return b.getvalue(), fmt, final_seed

# MAIN INTERFACE
if generate_btn:
    try:
        data, fmt, used_seed = run_render()
        meta = {"Mode": render_mode, "Seed": used_seed, "Exp": exposure, "Gamma": gamma, "Grain": grain, "Blur": blur, "Dens": quality_preset, "Inv": invert_colors}
        if is_ribbon_mode: meta["Complexity"] = complexity
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()
    except Exception as e: st.error(f"Render Error: {e}")

# GALLERY & EXPORT
if st.session_state.history:
    st.divider()
    g_col1, g_col2 = st.columns([3, 1])
    with g_col1:
        st.subheader("Your Creations")
    with g_col2:
        zip_data = create_zip_export(st.session_state.history)
        st.download_button("📦 DOWNLOAD ALL (ZIP)", data=zip_data, file_name=f"sands_of_time_{int(time.time())}.zip", mime="application/zip", use_container_width=True)

    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            st.markdown(f"""
            <div class="metadata-card">
            <b>{m['Mode']}</b> • {item['time']}<br>
            <span style='color: #8b949e;'>Seed: {m['Seed']} | Exp: {m['Exp']} | Grain: {m['Grain']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1,1,1])
            with c1: st.download_button("💾", item['data'], f"sand_{idx}.{item['fmt']}", key=f"dl_{idx}")
            with c2: 
                if st.button("🔄", key=f"res_{idx}", help="Restore Settings"): restore_settings(m)
            with c3:
                if st.button("🗑️", key=f"del_{idx}", help="Delete"): delete_item(idx)
