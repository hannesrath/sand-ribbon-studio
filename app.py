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

# UI STYLING (DARK MODE)
st.markdown("""
    <style>
    .stApp { background-color: #0e1117 !important; color: #e0e0e0 !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    .metadata-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 12px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.8rem; color: #bdc1c6; margin-top: 8px; margin-bottom: 12px;
    }
    .hero-container {
        border-radius: 15px; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem; margin-bottom: 2rem; background-color: #000000;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    h1, h2, h3 { color: #ffffff !important; }
    div.stButton > button { background-color: #238636 !important; color: white !important; border-radius: 6px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITIES ---
def hex_to_rgb(hex_color):
    """Converts a Hex color string to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def random_hex():
    """Generates a random hex color code."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# --- SESSION STATE ---
if 'history' not in st.session_state: st.session_state.history = []
if 'img_start' not in st.session_state: st.session_state.img_start = None
if 'img_end' not in st.session_state: st.session_state.img_end = None

keys_defaults = {
    'render_mode_radio': "Still Ribbon", 'seed_val_input': 0, 'exposure_slider': 2.8,
    'gamma_slider': 0.65, 'grain_slider': 0.35, 'blur_slider': 0.6,
    'invert_colors_check': False, 'complexity_slider': 3, 'quality_preset_slider': "Normal",
    'bg_color_picker': '#000000', 'sand_color_picker': '#D4AF37'
}
for key, val in keys_defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# --- CALLBACKS ---
def callback_randomize():
    st.session_state['seed_val_input'] = random.randint(1, 999999)
    st.session_state['exposure_slider'] = round(random.uniform(2.0, 4.5), 1)
    st.session_state['gamma_slider'] = round(random.uniform(0.5, 0.8), 2)
    st.session_state['grain_slider'] = round(random.uniform(0.2, 0.5), 2)
    st.session_state['blur_slider'] = round(random.uniform(0.4, 1.2), 1)
    st.session_state['bg_color_picker'] = random_hex()
    st.session_state['sand_color_picker'] = random_hex()
    st.toast("New competitors entered the arena! 🎲")

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
    if "Bg" in meta: st.session_state['bg_color_picker'] = meta["Bg"]
    if "Sand" in meta: st.session_state['sand_color_picker'] = meta["Sand"]

def reset_app():
    st.session_state.img_start = None
    st.session_state.img_end = None
    st.session_state.history = []
    st.rerun()

# --- MAIN PAGE ---
st.title("⏳ Sands of Time Generator")

with st.expander("📖 First Time Here? Read the Quick Start Guide!", expanded=False):
    st.markdown("""
    Welcome to the **Sands of Time Generator**! Here is how to create your first simulation:
    
    1. **Choose an Algorithm:** Head to the Sidebar. *Ribbon modes* generate abstract, mathematical flowing waves. *Image modes* let you upload your own pictures and turn them into gathering sand.
    2. **Set Particle Density:** Higher density ("Ultra") looks incredibly detailed but takes slightly longer to render. 
    3. **Pick Your Colors:** Scroll down to *Visual Styling* to set your Background and Sand colors. Feeling lucky? Click **🎲 Surprise Me!** for a random color and physics combination.
    4. **Fine-Tune Optics:** Adjust *Exposure* (glow brightness), *Blur* (softness), and *Grain* (texture) to give it a cinematic feel. 
    5. **Execute & Save:** Hit **EXECUTE RENDER**. Once finished, your simulation will appear in the Gallery below where you can download it, delete it, or use the 🔄 button to restore those exact settings later!
    """)

preview_placeholder = st.empty()

# SIDEBAR
with st.sidebar:
    st.header("Studio Controls")
    render_mode = st.radio("Core Algorithm", ["Still Ribbon", "Animation Loop (Ribbon)", "Image to Sand (Still)", "Image Morph (Animation)"], key="render_mode_radio", help="Select how the sands are formed. Ribbons are math-generated. Images use your own uploads.")
    
    is_ribbon_mode = "Ribbon" in render_mode
    is_morph_mode = "Morph" in render_mode
    is_still_image_mode = render_mode == "Image to Sand (Still)"
    
    if is_ribbon_mode:
        aspect_ratio = st.selectbox("Aspect Ratio", ["16:9", "9:16", "1:1"], index=0, help="The frame shape of the output. 16:9 is landscape, 9:16 is for mobile/stories, 1:1 is square.")
        complexity = st.slider("Complexity", 2, 8, key="complexity_slider", help="Higher values create more intense mathematical 'competitions' with more waves.")
    elif is_still_image_mode:
        up = st.file_uploader("Source Image", type=['png', 'jpg', 'jpeg'], help="Upload an image. The sand particles will gather in the bright areas of your photo.")
        if up: st.session_state.img_start = Image.open(up).convert("L")
    elif is_morph_mode:
        up1 = st.file_uploader("Start Target", type=['png', 'jpg'], key="up1", help="The initial shape the sand forms before it blows away.")
        if up1: st.session_state.img_start = Image.open(up1).convert("L")
        up2 = st.file_uploader("End Target", type=['png', 'jpg'], key="up2", help="The final shape the sand will form at the end of the animation.")
        if up2: st.session_state.img_end = Image.open(up2).convert("L")

    quality_preset = st.select_slider("Particle Density", options=["Draft", "Normal", "Ultra"], key="quality_preset_slider", help="Controls how many individual grains of sand are simulated. Draft: 200k. Normal: 800k. Ultra: 1.5M.")
    p_count = 200000 if quality_preset == "Draft" else 800000 if quality_preset == "Normal" else 1500000
    res_scale = 1.0 if quality_preset == "Draft" else 1.5 if quality_preset == "Normal" else 2.0
        
    with st.expander("Visual Styling", expanded=True):
        st.button("🎲 Surprise Me!", on_click=callback_randomize, use_container_width=True, help="Randomizes the colors, seed, and optic sliders below to give you a unique look instantly.")
        st.divider()
        
        # COLOR CONTROLS
        c1, c2 = st.columns(2)
        with c1:
            bg_color_input = st.color_picker("Background", key="bg_color_picker", help="The color of the canvas/void behind the sand particles.")
        with c2:
            sand_color_input = st.color_picker("Sand", key="sand_color_picker", help="The color of the actual sand grains/particles.")
        
        seed_input = st.number_input("Seed", min_value=0, step=1, key="seed_val_input", help="The exact DNA of the sand distribution. Reusing a seed gives you the exact same shape.")
        invert_colors = st.checkbox("Swap Colors (Invert)", key="invert_colors_check", help="Inverts the rendering math. Sometimes creates interesting negative-space effects.")
        exposure = st.slider("Exposure", 1.0, 5.0, step=0.1, key="exposure_slider", help="Glow intensity. Higher values make the overlapping sand burn brighter.")
        gamma = st.slider("Gamma", 0.3, 1.0, step=0.05, key="gamma_slider", help="Mid-tone contrast. Lower values make the faint sand trails more visible.")
        grain = st.slider("Grain", 0.0, 1.0, step=0.05, key="grain_slider", help="Adds an organic, film-like noise texture over the final image.")
        blur = st.slider("Blur", 0.0, 3.0, step=0.1, key="blur_slider", help="Softness. Adds a Gaussian blur to make the sand look more like smoke or glowing gas.")
        
    st.divider()
    execute_render = st.button("EXECUTE RENDER", type="primary", use_container_width=True, help="Starts the simulation. High density animations may take a minute!")
    st.button("Clear History", on_click=reset_app, use_container_width=True, help="Wipes your entire gallery clean. Cannot be undone!")

# --- RENDER ENGINE ---
if execute_render:
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        bar = st.progress(0, text="Calculating Mathematical Competition...")
        
        frames_list = []
        final_seed = seed_input if seed_input > 0 else np.random.randint(0, 999999)
        rng_main = np.random.RandomState(final_seed)

        # Pre-calculate RGB colors for the renderer
        bg_rgb = np.array(hex_to_rgb(bg_color_input), dtype=float)
        fg_rgb = np.array(hex_to_rgb(sand_color_input), dtype=float)

        if is_ribbon_mode:
            # RESTORED COMPETITION MATH
            def generate_dna(c, s):
                np.random.seed(s)
                p = []
                sc = np.random.uniform(1.8, 2.5)
                for i in range(1, c + 1):
                    p.append({
                        'freq': i, 
                        'amp_a': np.random.uniform(-1, 1, 3) * sc / (i**0.8), 
                        'amp_b': np.random.uniform(-1, 1, 3) * sc / (i**0.8), 
                        'phases': np.random.uniform(0, 2*np.pi, 3)
                    })
                return p, np.radians(np.random.uniform(20, 80)), np.radians(np.random.uniform(0, 360))
            
            width, height = (1920, 1080) if "16:9" in aspect_ratio else (1080, 1920) if "9:16" in aspect_ratio else (1080, 1080)
            dna, tx, ty = generate_dna(complexity, final_seed)
            t_vals = rng_main.rand(p_count) * 2 * np.pi
            total_frames = 100 if "Animation" in render_mode else 1
            bounds_x, bounds_y = [-5, 5], [-5, 5]
            iw, ih = width, height
            sx, sy, sz = [rng_main.normal(0, 0.15, p_count) for _ in range(3)]
            thickness = (np.sin(t_vals * 2.0 + rng_main.rand()*6) + 1.2) * 0.5
        elif is_morph_mode or is_still_image_mode:
            def sample_img(pil_img, n, s):
                arr = np.power(np.array(pil_img).astype(float) / 255.0, 2.2)
                total = np.sum(arr)
                cdf = np.cumsum(arr.flatten()) / (total + 1e-10)
                indices = np.searchsorted(cdf, np.random.RandomState(s).rand(n))
                y, x = np.unravel_index(indices, arr.shape)
                return x.astype(float), (arr.shape[0] - y.astype(float)), pil_img.width, pil_img.height
            ix1, iy1, iw, ih = sample_img(st.session_state.img_start, p_count, final_seed)
            if is_morph_mode:
                ix2, iy2, _, _ = sample_img(st.session_state.img_end.resize(st.session_state.img_start.size), p_count, final_seed + 1)
                total_frames, bounds_x, bounds_y = 125, [0, iw], [0, ih]
            else: total_frames, bounds_x, bounds_y = 1, [0, iw], [0, ih]

        for i in range(total_frames):
            prog = i / total_frames if total_frames > 1 else 0.0
            if is_ribbon_mode:
                # COMPETING WAVEFORMS
                x, y, z = np.zeros_like(t_vals), np.zeros_like(t_vals), np.zeros_like(t_vals)
                ca, sa = np.cos(prog*2*np.pi), np.sin(prog*2*np.pi)
                for l in dna:
                    ax, ay, az = [l['amp_a'][j] * ca + l['amp_b'][j] * sa for j in range(3)]
                    x += ax * np.cos(l['freq'] * t_vals + l['phases'][0])
                    y += ay * np.sin(l['freq'] * t_vals + l['phases'][1])
                    z += az * np.cos(l['freq'] * t_vals + l['phases'][2])
                
                # Apply depth and thickness
                x, y, z = x + sx*thickness, y + sy*thickness, z + sz*thickness
                yr_r, zr_r = y*np.cos(tx)-z*np.sin(tx), y*np.sin(tx)+z*np.cos(tx)
                xr, yr = x*np.cos(ty)+zr_r*np.sin(ty), yr_r
                w_final, grain_seed = np.exp(-(zr_r - zr_r.min()) / (zr_r.max() - zr_r.min() + 1e-6) * 1.5), final_seed + i
            elif is_morph_mode:
                if i < 25: grain_seed, tm, n = final_seed, 0.0, 0.0
                elif i > 100: grain_seed, tm, n = final_seed + 999, 1.0, 0.0
                else: grain_seed, tp = final_seed + i, (i - 25) / 75; tm, n = (1 - np.cos(tp * np.pi)) / 2, np.sin(tp * np.pi) * 4.0
                xr, yr, w_final = ix1*(1-tm) + ix2*tm + rng_main.normal(0, n, p_count), iy1*(1-tm) + iy2*tm + rng_main.normal(0, n, p_count), None
            else: xr, yr, grain_seed, w_final = ix1, iy1, final_seed, None

            # RENDER Heatmap (CINEMATIC GLOW)
            h_map, _, _ = np.histogram2d(xr, yr, bins=[int(iw*res_scale/2), int(ih*res_scale/2)], range=[bounds_x, bounds_y], weights=w_final)
            if blur > 0: h_map = gaussian_filter(h_map, sigma=blur)
            h_map = h_map / (np.max(h_map) + 1e-10)
            h_map = np.log1p(h_map * exposure * 10) / np.log1p(exposure * 10)
            h_map = np.power(h_map, gamma)
            if grain > 0: h_map *= np.random.RandomState(grain_seed).normal(1.0, grain, h_map.shape)
            h_map = np.clip(h_map, 0, 1)
            if invert_colors: h_map = 1.0 - h_map
            
            # COLOR INTERPOLATION
            # Resize the grayscale density map to final resolution
            h_map_resized = resize(np.flipud(h_map.T), (1080, int(1080 * iw/ih)))
            # Expand dimensions to blend with RGB
            h_map_3d = h_map_resized[..., np.newaxis]
            # Blend background and foreground colors linearly based on density
            colored_frame = bg_rgb * (1.0 - h_map_3d) + fg_rgb * h_map_3d
            
            frames_list.append(colored_frame.astype(np.uint8))
            bar.progress((i+1)/total_frames)

        b = io.BytesIO()
        if len(frames_list) > 1: imageio.mimsave(b, frames_list, format='GIF', fps=25, loop=0); fmt = "gif"
        else: imageio.imwrite(b, frames_list[0], format='PNG'); fmt = "png"
        
        meta = {"Mode": render_mode, "Seed": final_seed, "Exp": exposure, "Gamma": gamma, "Grain": grain, "Blur": blur, "Dens": quality_preset, "Inv": invert_colors, "Bg": bg_color_input, "Sand": sand_color_input}
        if is_ribbon_mode: meta["Complexity"] = complexity
        st.session_state.history.insert(0, {"data": b.getvalue(), "fmt": fmt, "time": time.strftime("%H:%M:%S"), "meta": meta})
        st.rerun()

# HERO DISPLAY
elif st.session_state.history:
    latest = st.session_state.history[0]
    with preview_placeholder.container():
        st.markdown('<div class="hero-container">', unsafe_allow_html=True)
        st.image(latest['data'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# GALLERY & EXPORT
if st.session_state.history:
    st.divider()
    g_col1, g_col2 = st.columns([3, 1])
    with g_col1: st.subheader("Your Gallery")
    with g_col2:
        z_data = io.BytesIO()
        with zipfile.ZipFile(z_data, "w") as zf:
            for i, item in enumerate(st.session_state.history):
                zf.writestr(f"sand_{i}.{item['fmt']}", item['data'])
        st.download_button("📦 DOWNLOAD ALL (ZIP)", data=z_data.getvalue(), file_name="sands_of_time.zip", use_container_width=True, help="Package all your generated images into a single ZIP file.")

    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history):
        with cols[idx % 3]:
            st.image(item['data'], use_container_width=True)
            m = item['meta']
            
            # Updated Metadata Card to include color readouts
            st.markdown(f"""<div class="metadata-card"><b>{m['Mode']}</b> • {item['time']}<br>
            Seed: {m['Seed']} | Exp: {m['Exp']} | Gamma: {m['Gamma']}<br>
            Grain: {m['Grain']} | Blur: {m['Blur']} | Density: {m['Dens']}<br>
            BG: <span style="color:{m.get('Bg', '#000000')}">{m.get('Bg', '#000000')}</span> | 
            Sand: <span style="color:{m.get('Sand', '#D4AF37')}">{m.get('Sand', '#D4AF37')}</span>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            with c1: st.download_button("💾", item['data'], f"sand_{idx}.{item['fmt']}", key=f"dl_{idx}", help="Save this specific image or animation to your computer.")
            with c2: st.button("🔄", key=f"res_{idx}", on_click=callback_restore, args=(m,), help="Restore all sidebar settings to exactly match this render.")
            with c3: st.button("🗑️", key=f"del_{idx}", on_click=lambda i=idx: (st.session_state.history.pop(i), st.rerun()), help="Permanently delete this item from the gallery.")
