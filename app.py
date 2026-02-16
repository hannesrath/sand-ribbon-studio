import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import imageio
import io
from PIL import Image
import time

# PAGE CONFIG
st.set_page_config(page_title="Sand Ribbon Generator", page_icon="⏳", layout="wide")

# UI STYLING
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f5f5f5 !important; }
    .stApp { background-color: #ffffff !important; color: #000000 !important; }
    .stMarkdown, .stText, h1, h2, h3, h4, p, label { color: #000000 !important; }
    div.stButton > button {
        background-color: #000000 !important; border-radius: 4px; border: none; padding: 0.5rem 1rem; color: #ffffff !important;
    }
    div.stButton > button p { color: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

if 'history' not in st.session_state: st.session_state.history = []
if 'img_start' not in st.session_state: st.session_state.img_start = None
if 'img_end' not in st.session_state: st.session_state.img_end = None

def reset_app():
    st.session_state.img_start = None
    st.session_state.img_end = None
    st.rerun()

st.title("⏳ Sand Ribbon Generator")

# SIDEBAR
with st.sidebar:
    st.header("Generator Settings")
    render_mode = st.radio("Output Type", 
                           ["Still Ribbon", "Animation Loop (Ribbon)", "Image to Sand (Still)", "Image Morph (Animation)"], 
                           index=0)
    
    is_ribbon_mode = "Ribbon" in render_mode
    is_morph_mode = "Morph" in render_mode
    is_still_image_mode = render_mode == "Image to Sand (Still)"
    is_any_image_mode = is_morph_mode or is_still_image_mode

    if is_ribbon_mode:
        aspect_ratio = st.selectbox("Aspect Ratio", ["16:9 (Landscape)", "9:16 (Portrait)", "1:1 (Square)"], index=0)
        complexity = st.slider("Shape Complexity", 2, 8, 3)
    elif is_still_image_mode:
        up = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
        if up: st.session_state.img_start = Image.open(up).convert("L")
    elif is_morph_mode:
        col1, col2 = st.columns(2)
        with col1:
            up1 = st.file_uploader("Start Target", type=['png', 'jpg'], key="up1")
            if up1: st.session_state.img_start = Image.open(up1).convert("L")
        with col2:
            up2 = st.file_uploader("End Target", type=['png', 'jpg'], key="up2")
            if up2: st.session_state.img_end = Image.open(up2).convert("L")

    quality_preset = st.select_slider("Density", options=["Draft", "Normal", "Ultra"], value="Normal")
    if quality_preset == "Draft": p_count, res_scale = 200_000, 1.0 
    elif quality_preset == "Normal": p_count, res_scale = 800_000, 1.5
    else: p_count, res_scale = 1_500_000, 2.0 
        
    with st.expander("Look Development", expanded=True):
        seed_val = st.number_input("Manual Seed (0 = Random)", min_value=0, value=0, step=1)
        invert_colors = st.checkbox("Invert Colors", value=False)
        exposure = st.slider("Exposure", 1.0, 5.0, 2.8)
        gamma = st.slider("Gamma", 0.3, 1.0, 0.65)
        grain = st.slider("Grain", 0.0, 1.0, 0.35)
        blur = st.slider("Blur", 0.0, 3.0, 0.6)
        
    col_gen, col_res = st.columns(2)
    with col_gen:
        generate_btn = st.button("Generate", type="primary", use_container_width=True)
    with col_res:
        st.button("Reset", on_click=reset_app, use_container_width=True)

# --- CORE MATH ENGINE ---

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
        ax = l['amp_a'][0] * cos_a + l['amp_b'][0] * sin_a
        ay = l['amp_a'][1] * cos_a + l['amp_b'][1] * sin_a
        az = l['amp_a'][2] * cos_a + l['amp_b'][2] * sin_a
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
    status = st.empty()
    bar = st.progress(0)
    frames_list = []
    
    # Handle Seed
    final_seed = seed_val if seed_val > 0 else np.random.randint(0, 999999)
    rng_main = np.random.RandomState(final_seed)

    if is_ribbon_mode:
        width, height = get_resolution(aspect_ratio)
        dna = generate_ribbon_dna(complexity, final_seed)
        t_vals = rng_main.rand(p_count) * 2 * np.pi
        total_frames = 100 if "Animation" in render_mode else 1
        bounds_x, bounds_y = [-5, 5], [-5, 5]
        iw, ih = width, height
        thickness = (np.sin(t_vals * 2.0 + rng_main.rand()*6) + 1.2) * 0.5
        sx, sy, sz = rng_main.normal(0, 0.15, p_count), rng_main.normal(0, 0.15, p_count), rng_main.normal(0, 0.15, p_count)
    elif is_morph_mode:
        w_s, h_s = st.session_state.img_start.size
        ix1, iy1, iw, ih = sample_image_density(st.session_state.img_start, p_count, final_seed)
        ix2, iy2, _, _ = sample_image_density(st.session_state.img_end.resize((w_s, h_s)), p_count, final_seed + 1)
        total_frames, bounds_x, bounds_y = 125, [0, iw], [0, ih]
    else: # Still Image
        ix1, iy1, iw, ih = sample_image_density(st.session_state.img_start, p_count, final_seed)
        total_frames, bounds_x, bounds_y = 1, [0, iw], [0, ih]

    for i in range(total_frames):
        prog = i / total_frames if total_frames > 1 else 0.0
        if is_ribbon_mode:
            xs, ys, zs = calc_ribbon_spine(t_vals, dna, prog)
            x, y, z = xs + sx*thickness, ys + sy*thickness, zs + sz*thickness
            tx, ty = dna['tilt_x'], dna['tilt_y']
            yr_r = y * np.cos(tx) - z * np.sin(tx)
            zr_r = y * np.sin(tx) + z * np.cos(tx)
            xr, yr = x * np.cos(ty) + zr_r * np.sin(ty), yr_r
            w_final = np.exp(-(zr_r - zr_r.min()) / (zr_r.max() - zr_r.min() + 1e-6) * 1.5)
        elif is_morph_mode:
            if i < 25: t_m, noise = 0.0, 0.0
            elif i > 100: t_m, noise = 1.0, 0.0
            else:
                tp = (i - 25) / 75
                t_m = (1 - np.cos(tp * np.pi)) / 2
                noise = np.sin(tp * np.pi) * 4.0
            xr, yr, w_final = ix1*(1-t_m) + ix2*t_m + rng_main.normal(0, noise, p_count), iy1*(1-t_m) + iy2*t_m + rng_main.normal(0, noise, p_count), None
        else:
            xr, yr, w_final = ix1, iy1, None

        heatmap, _, _ = np.histogram2d(xr, yr, bins=[int(iw*res_scale/2), int(ih*res_scale/2)], range=[bounds_x, bounds_y], weights=w_final)
        if blur > 0: heatmap = gaussian_filter(heatmap, sigma=blur)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        heatmap = np.power(np.log1p(heatmap * exposure * 10) / np.log1p(exposure * 10), gamma)
        if grain > 0: heatmap *= rng_main.normal(1.0, grain, heatmap.shape)
        heatmap = np.clip(heatmap, 0, 1)
        if invert_colors: heatmap = 1.0 - heatmap
        
        final_img = np.flipud(heatmap.T)
        frames_list.append((resize(final_img, (1080, int(1080 * iw/ih))) * 255).astype(np.uint8))
        bar.progress((i+1)/total_frames)

    b = io.BytesIO()
    if len(frames_list) > 1: imageio.mimsave(b, frames_list, format='GIF', fps=25, loop=0); fmt = "gif"
    else: imageio.imwrite(b, frames_list[0], format='PNG'); fmt = "png"
    return b.getvalue(), fmt

# MAIN INTERFACE
if generate_btn:
    try:
        data, fmt = run_render()
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": time.strftime("%H:%M:%S")})
        st.image(data)
    except Exception as e: st.error(f"Render Error: {e}")

# GALLERY
if st.session_state.history:
    st.divider()
    st.header("🎞️ Session Gallery")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            st.download_button(f"Download {item['fmt'].upper()}", item['data'], f"sand_{idx}.{item['fmt']}", key=f"dl_{idx}")
