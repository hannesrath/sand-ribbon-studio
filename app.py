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

# FORCE LIGHT THEME & UI STYLING
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f5f5f5 !important; }
    .stApp { background-color: #ffffff !important; color: #000000 !important; }
    .stMarkdown, .stText, h1, h2, h3, h4, p, label { color: #000000 !important; }
    div.stButton > button {
        background-color: #000000 !important; border-radius: 4px; border: none; padding: 0.5rem 1rem; color: #ffffff !important;
    }
    div.stButton > button p { color: #ffffff !important; }
    .gallery-item { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# INITIALIZE SESSION STATE
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
        
    with st.expander("Look Development"):
        invert_colors = st.checkbox("Invert Colors", value=False)
        exposure = st.slider("Exposure", 1.0, 5.0, 2.5)
        gamma = st.slider("Gamma", 0.3, 1.0, 0.60)
        grain = st.slider("Grain", 0.0, 1.0, 0.40)
        blur = st.slider("Blur", 0.0, 3.0, 0.8)
        
    col_gen, col_res = st.columns(2)
    with col_gen:
        generate_btn = st.button("Generate", type="primary", use_container_width=True)
    with col_res:
        st.button("Reset", on_click=reset_app, use_container_width=True)

# --- CORE MATH ---
def sample_image_density(pil_img, num_particles, seed):
    img_arr = np.array(pil_img).astype(float)
    img_arr = np.power(img_arr, 3.0) 
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
    eff_seed = np.random.randint(0, 99999)
    rng_main = np.random.RandomState(eff_seed)

    if is_morph_mode:
        w_s, h_s = st.session_state.img_start.size
        img_end_res = st.session_state.img_end.resize((w_s, h_s))
        ix1, iy1, iw, ih = sample_image_density(st.session_state.img_start, p_count, eff_seed)
        ix2, iy2, _, _ = sample_image_density(img_end_res, p_count, eff_seed+1)
        total_frames = 125 
        bounds_x, bounds_y = [0, iw], [0, ih]
    elif is_still_image_mode:
        ix1, iy1, iw, ih = sample_image_density(st.session_state.img_start, p_count, eff_seed)
        total_frames, bounds_x, bounds_y = 1, [0, iw], [0, ih]
    else:
        # Default ribbon geometry fallback
        iw, ih, total_frames = 1920, 1080, 1
        ix1, iy1, bounds_x, bounds_y = np.zeros(p_count), np.zeros(p_count), [0, 1], [0, 1]

    for i in range(total_frames):
        if is_morph_mode:
            if i < 25: t, noise = 0.0, 0.0
            elif i > 100: t, noise = 1.0, 0.0
            else:
                prog = (i - 25) / 75
                t = (1 - np.cos(prog * np.pi)) / 2
                noise = np.sin(prog * np.pi) * 6.0
            xr, yr = ix1*(1-t) + ix2*t + rng_main.normal(0, noise, p_count), iy1*(1-t) + iy2*t + rng_main.normal(0, noise, p_count)
        else:
            xr, yr = ix1, iy1

        heatmap, _, _ = np.histogram2d(xr, yr, bins=[int(iw*res_scale), int(ih*res_scale)], range=[bounds_x, bounds_y])
        if blur > 0: heatmap = gaussian_filter(heatmap, sigma=blur)
        heatmap = np.power(np.log1p(heatmap / (np.max(heatmap) + 1e-8) * exposure * 5), gamma)
        if grain > 0: heatmap *= np.random.normal(1.0, grain, heatmap.shape)
        heatmap = np.clip(heatmap, 0, 1)
        if invert_colors: heatmap = 1.0 - heatmap
        
        final_img = np.flipud(heatmap.T)
        frames_list.append((resize(final_img, (1080, int(1080 * iw/ih))) * 255).astype(np.uint8))
        bar.progress((i+1)/total_frames)

    b = io.BytesIO()
    if len(frames_list) > 1:
        imageio.mimsave(b, frames_list, format='GIF', fps=25, loop=0)
        fmt = "gif"
    else:
        imageio.imwrite(b, frames_list[0], format='PNG')
        fmt = "png"
    return b.getvalue(), fmt

# MAIN INTERFACE
if generate_btn:
    try:
        data, fmt = run_render()
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.history.insert(0, {"data": data, "fmt": fmt, "time": timestamp})
        st.success(f"Generated at {timestamp}")
        st.image(data)
    except Exception as e:
        st.error(f"Error: {e}")

# GALLERY SECTION
if st.session_state.history:
    st.divider()
    st.header("🎞️ Session Gallery")
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.markdown(f"**Rendered at {item['time']}**")
            st.image(item['data'], use_container_width=True)
            st.download_button(f"Download {item['fmt'].upper()}", item['data'], f"sand_{idx}.{item['fmt']}", key=f"dl_{idx}")