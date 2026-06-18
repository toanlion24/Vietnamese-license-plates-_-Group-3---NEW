"""
VN License Plate Recognition - Enterprise Dashboard
==================================================
Professional AI Dashboard for Vietnamese License Plate Recognition.
Enterprise-grade UI with real-time metrics and analytics.

Usage:
    streamlit run src/app/demo.py
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Project imports
from src.detector.yolov8_detector import YoloV8PlateDetector
from src.ocr.qwen_adapter import Qwen2VLPlateOcr
from src.ocr.easyocr_adapter import EasyOcrAdapter
from src.postprocess.plate_rules import normalize_plate_text, advanced_repair_ocr_text
from src.preprocess.ops import crop_plate
from src.utils.types import FrameData

# Constants
PROJECT_ROOT = Path("d:/ComputerVisionNew")
DEFAULT_YOLO_MODEL = PROJECT_ROOT / "runs/detect/yolo_cropped_v2/weights/best.pt"
DEFAULT_LORA_PATH = PROJECT_ROOT / "experiments/qwen2vl_crops_lora"
DEFAULT_BASE_MODEL = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"


# ============================================================================
# THEME CONFIGURATION
# ============================================================================

THEME = {
    "dark": {
        "primary": "#00D4AA",
        "secondary": "#6366F1",
        "background": "#0F172A",
        "surface": "#1E293B",
        "card": "#334155",
        "text": "#F8FAFC",
        "text_secondary": "#94A3B8",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "border": "#475569",
    },
    "light": {
        "primary": "#059669",
        "secondary": "#6366F1",
        "background": "#F8FAFC",
        "surface": "#FFFFFF",
        "card": "#F1F5F9",
        "text": "#0F172A",
        "text_secondary": "#64748B",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "border": "#E2E8F0",
    }
}


def get_theme() -> dict:
    """Get current theme colors."""
    mode = st.session_state.get("theme_mode", "dark")
    return THEME[mode]


def apply_custom_css():
    """Apply enterprise-grade custom CSS."""
    theme = get_theme()
    
    st.html(f"""
    <style>
    /* Global Styles */
    .stApp {{
        background-color: {theme["background"]};
        color: {theme["text"]};
    }}
    
    /* Main Content Container */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1400px;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {theme["surface"]};
        border-right: 1px solid {theme["border"]};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {theme["text"]};
        font-weight: 600;
    }}
    
    /* Metric Containers */
    [data-testid="stMetric"] {{
        background-color: {theme["surface"]};
        border: 1px solid {theme["border"]};
        border-radius: 8px;
        padding: 1rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {theme["text_secondary"]};
        font-size: 0.8rem;
    }}
    [data-testid="stMetricValue"] {{
        color: {theme["text"]};
        font-size: 1.5rem;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid {theme["border"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 0;
        padding: 12px 24px;
        color: {theme["text_secondary"]};
        font-weight: 500;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {theme["text"]};
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {theme["primary"]};
        border-bottom: 2px solid {theme["primary"]};
    }}
    
    /* Buttons */
    .stButton > button {{
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }}
    [data-testid="stMainBlockContainer"] button:hover {{
        transform: translateY(-1px);
    }}
    
    /* Data Table */
    .dataframe {{
        border: none !important;
        background-color: {theme["surface"]};
    }}
    .dataframe th {{
        background-color: {theme["card"]} !important;
        color: {theme["text"]} !important;
        font-weight: 600;
    }}
    .dataframe td {{
        color: {theme["text"]};
    }}
    
    /* Divider */
    hr {{
        border-color: {theme["border"]};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {theme["surface"]};
        border-radius: 6px;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed {theme["border"]};
        border-radius: 8px;
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {theme["background"]};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {theme["border"]};
        border-radius: 3px;
    }}
    
    /* Image container */
    [data-testid="stImage"] {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* Hide default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Status indicator */
    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }}
    .status-online {{background-color: {theme["success"]};}}
    .status-offline {{background-color: {theme["error"]};}}
    .status-loading {{background-color: {theme["warning"]}; animation: pulse 2s infinite;}}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    </style>
    """)


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "detector": None,
        "ocr": None,
        "ocr_engine": "easyocr",
        "results_history": [],
        "stats": {
            "total_processed": 0,
            "total_detected": 0,
            "avg_latency": 0,
            "start_time": None
        },
        "theme_mode": "dark",
        "auto_loaded": False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_detector(yolo_model_path: str, conf_threshold: float = 0.15):
    """Load and cache YOLO detector."""
    return YoloV8PlateDetector(
        model_path=yolo_model_path,
        conf_threshold=conf_threshold,
        iou=0.45,
        imgsz=640,
    )


@st.cache_resource
def load_easyocr():
    """Load EasyOCR on GPU if available."""
    return EasyOcrAdapter(lang_list=["en", "vi"], use_gpu=_has_cuda())


@st.cache_resource
def load_qwen_ocr(use_lora: bool = False, lora_path: str = None):
    """Load Qwen2-VL OCR."""
    has_cuda = _has_cuda()
    use_lora_on_device = use_lora and has_cuda
    model_name = lora_path if use_lora_on_device else DEFAULT_BASE_MODEL
    return Qwen2VLPlateOcr(
        model_name=model_name,
        device="cuda" if has_cuda else "cpu",
        use_lora_adapter=use_lora_on_device,
        use_bnb_quant=use_lora_on_device,
    )


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_gpu_info() -> dict:
    """Get GPU information."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "name": torch.cuda.get_device_name(0),
                "memory": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "available": True
            }
    except Exception:
        pass
    return {"name": None, "memory": 0, "available": False}


# ============================================================================
# INFERENCE
# ============================================================================

def draw_bounding_box(image, bbox, text, score):
    """Draw bounding box with text on image."""
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    font_scale = 0.8
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(image, (x1, y1 - text_h - 15), (x1 + text_w + 10, y1), color, -1)
    cv2.putText(image, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (0, 0, 0), 2)
    return image


def process_image(
    img: Image.Image,
    detector: YoloV8PlateDetector,
    ocr,
    run_ocr: bool = True
) -> tuple[np.ndarray, dict]:
    """Process image with detector and OCR."""
    img_array = np.array(img)
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    
    # Detection
    start_det = time.time()
    frame_data = FrameData(image_id=image_id, frame=img_array, source="upload")
    detections = detector.predict(frame_data)
    det_time = (time.time() - start_det) * 1000
    
    # Prepare output
    annotated = img_array.copy()
    result = {
        "image_id": image_id,
        "plate_text": "",
        "confidence": 0.0,
        "det_time_ms": det_time,
        "ocr_time_ms": 0,
        "total_time_ms": det_time,
        "has_detection": len(detections) > 0,
        "annotated": annotated,
    }
    
    if detections:
        best_det = max(detections, key=lambda d: d.score)
        result["confidence"] = best_det.score
        
        if run_ocr:
            start_ocr = time.time()
            plate_crop = crop_plate(frame_data, best_det, margin_ratio=0.05)
            ocr_out = ocr.recognize(plate_crop, plate_crop.crop)
            result["ocr_time_ms"] = (time.time() - start_ocr) * 1000
            result["plate_text"] = advanced_repair_ocr_text(
                ocr_out.text_norm or normalize_plate_text(ocr_out.text_raw)
            )
            result["total_time_ms"] = det_time + result["ocr_time_ms"]
        
        # Draw box
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        annotated_bgr = draw_bounding_box(
            annotated_bgr,
            best_det.bbox_xyxy,
            result["plate_text"] or "UNKNOWN",
            best_det.score
        )
        result["annotated"] = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
    return result["annotated"], result


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    """Render enterprise sidebar."""
    theme = get_theme()
    gpu_info = get_gpu_info()
    
    with st.sidebar:
        # Header
        st.html(f'''
        <div style="padding: 1rem 0 1.5rem 0;">
            <div style="font-size: 1.2rem; font-weight: 700; color: {theme["text"]}; margin-bottom: 4px;">
                VN-LPR Dashboard
            </div>
            <div style="font-size: 0.75rem; color: {theme["text_secondary"]};">
                License Plate Recognition System
            </div>
        </div>
        ''')
        
        st.divider()
        
        # Theme Toggle
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Dark" if st.session_state.theme_mode == "light" else "Light", 
                        use_container_width=True, key="theme_toggle"):
                st.session_state.theme_mode = "dark" if st.session_state.theme_mode == "light" else "light"
                st.rerun()
        with col2:
            st.caption("")
        
        st.divider()
        
        # OCR Engine Selection
        st.subheader("OCR Engine")
        ocr_engine = st.radio(
            "Select Engine",
            options=["easyocr", "qwen2vl"],
            format_func=lambda x: "EasyOCR (Fast)" if x == "easyocr" else "Qwen2-VL (Accurate)",
            label_visibility="collapsed",
            horizontal=True,
        )
        
        if ocr_engine != st.session_state.ocr_engine:
            st.session_state.ocr_engine = ocr_engine
            st.session_state.auto_loaded = False
            st.session_state.ocr = None
            st.cache_resource.clear()
        
        st.caption(f"Est. ~{300 if ocr_engine == 'easyocr' else 6000}ms per inference")
        
        st.divider()
        
        # Detection Settings
        st.subheader("Detection Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.05, 0.9, 0.15, 0.05)
        
        st.divider()
        
        # System Status
        st.subheader("System Status")
        
        # GPU Status
        if gpu_info["available"]:
            st.html(f'<span class="status-dot status-online"></span>GPU: {gpu_info["name"]} ({gpu_info["memory"]:.0f}GB)')
        else:
            st.html('<span class="status-dot status-offline"></span>GPU: Not available')
        
        # Model Status
        det_status = "Ready" if st.session_state.detector else "Loading"
        det_class = "online" if st.session_state.detector else "loading"
        st.html(f'<span class="status-dot status-{det_class}"></span>Detector: {det_status}')
        
        ocr_status = "Ready" if st.session_state.ocr else "Loading"
        ocr_class = "online" if st.session_state.ocr else "loading"
        st.html(f'<span class="status-dot status-{ocr_class}"></span>OCR: {ocr_status}')
        
        st.divider()
        
        # Load Models Button
        if st.button("Load Models", use_container_width=True, type="primary"):
            with st.spinner("Loading models..."):
                try:
                    st.session_state.detector = load_detector(str(DEFAULT_YOLO_MODEL), conf_threshold)
                    
                    if ocr_engine == "easyocr":
                        st.session_state.ocr = load_easyocr()
                    else:
                        st.session_state.ocr = load_qwen_ocr(use_lora=False)
                    
                    st.session_state.auto_loaded = True
                    st.success("Models loaded successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Advanced Settings
        with st.expander("Advanced Settings"):
            st.text_input("YOLO Model", value=str(DEFAULT_YOLO_MODEL), disabled=True)
            st.text_input("OCR Model", value=DEFAULT_BASE_MODEL[:40] + "...", disabled=True)
        
        # Reset Stats
        st.divider()
        if st.button("Reset Statistics", use_container_width=True):
            st.session_state.stats = {
                "total_processed": 0,
                "total_detected": 0,
                "avg_latency": 0,
                "start_time": datetime.now()
            }
            st.session_state.results_history = []
            st.rerun()
        
        return conf_threshold


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def render_dashboard(conf_threshold: float):
    """Render main dashboard."""
    theme = get_theme()
    
    # Page header
    col_title, col_time = st.columns([3, 1])
    with col_title:
        st.title("License Plate Recognition")
        st.caption("Real-time Vietnamese LPR System")
    with col_time:
        st.markdown(f"<div style='text-align: right; padding-top: 20px; color: {theme['text_secondary']}; font-size: 0.85rem;'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Metric cards row
    stats = st.session_state.stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", str(stats["total_processed"]))
    with col2:
        st.metric("Detections", str(stats["total_detected"]))
    with col3:
        st.metric("Avg Latency", f"{stats['avg_latency']:.0f} ms")
    with col4:
        uptime = ""
        if stats["start_time"]:
            elapsed = (datetime.now() - stats["start_time"]).total_seconds()
            uptime = f"{int(elapsed)}s"
        st.metric("Uptime", uptime)
    
    st.divider()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Inference", "Analytics", "History"])
    
    with tab1:
        render_inference_tab()
    
    with tab2:
        render_analytics_tab()
    
    with tab3:
        render_history_tab()


def render_inference_tab():
    """Render inference tab."""
    col_upload, col_result = st.columns([1, 1], gap="large")
    
    with col_upload:
        st.subheader("Input")
        
        input_mode = st.radio("Input Mode", ["Image File", "URL", "Sample"], 
                             horizontal=True, label_visibility="collapsed")
        
        uploaded_file = None
        img = None
        
        if input_mode == "Image File":
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp"])
            if uploaded_file:
                img = Image.open(uploaded_file)
        elif input_mode == "URL":
            url = st.text_input("Image URL")
            if url:
                try:
                    img = Image.open(BytesIO(__import__("requests").get(url).content))
                except:
                    st.error("Failed to load image")
        else:
            sample_dir = PROJECT_ROOT / "data/samples"
            if sample_dir.exists():
                samples = list(sample_dir.glob("*.jpg"))[:5]
                if samples:
                    selected = st.selectbox("Select sample", samples)
                    if selected:
                        img = Image.open(selected)
        
        if img:
            st.image(img, use_container_width=True)
    
    with col_result:
        st.subheader("Result")
        
        if st.session_state.detector is None or st.session_state.ocr is None:
            st.info("Load models from sidebar to start inference")
        elif img is None:
            st.info("Upload or select an image to process")
        else:
            with st.spinner("Processing..."):
                annotated, result = process_image(
                    img,
                    st.session_state.detector,
                    st.session_state.ocr,
                    run_ocr=True
                )
            
            st.image(annotated, use_container_width=True)
            
            # Result info
            if result["has_detection"]:
                st.success(f"Detected: {result['plate_text'] or 'N/A'}")
                st.caption(f"Confidence: {result['confidence']:.1%}")
            else:
                st.warning("No plate detected")
            
            # Timing breakdown
            with st.expander("Performance Details"):
                col_det, col_ocr, col_total = st.columns(3)
                with col_det:
                    st.metric("Detection", f"{result['det_time_ms']:.0f}ms")
                with col_ocr:
                    st.metric("OCR", f"{result['ocr_time_ms']:.0f}ms")
                with col_total:
                    st.metric("Total", f"{result['total_time_ms']:.0f}ms")
            
            update_stats(result)


def render_analytics_tab():
    """Render analytics tab."""
    if len(st.session_state.results_history) < 2:
        st.info("Process more images to see analytics")
        return
    
    # Timing distribution chart
    st.subheader("Processing Time Distribution")
    timings = [r["total_time_ms"] for r in st.session_state.results_history]
    
    chart_data = {
        "values": timings,
    }
    
    st.bar_chart(timings)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min Latency", f"{min(timings):.0f}ms")
    with col2:
        st.metric("Max Latency", f"{max(timings):.0f}ms")
    with col3:
        st.metric("Avg Latency", f"{sum(timings)/len(timings):.0f}ms")
    with col4:
        st.metric("Total Runs", str(len(timings)))


def render_history_tab():
    """Render history tab with data table."""
    if not st.session_state.results_history:
        st.info("No results yet. Process some images to build history.")
        return
    
    # Convert to DataFrame
    import pandas as pd
    
    history = st.session_state.results_history
    df = pd.DataFrame(history)
    
    # Prepare display
    display_df = pd.DataFrame({
        "Plate": [r.get("plate_text", "") for r in history],
        "Confidence": [f"{r.get('confidence', 0):.1%}" for r in history],
        "Latency (ms)": [f"{r.get('total_time_ms', 0):.0f}" for r in history],
        "Detected": ["Yes" if r.get("has_detection") else "No" for r in history],
        "Time": [r.get("timestamp", "")[-8:] if r.get("timestamp") else "" for r in history],
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button
    if len(df) > 0:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "lpr_results.csv",
            "text/csv",
            use_container_width=True
        )


def update_stats(result: dict):
    """Update running statistics."""
    stats = st.session_state.stats
    
    if stats["start_time"] is None:
        stats["start_time"] = datetime.now()
    
    stats["total_processed"] += 1
    
    if result["has_detection"]:
        stats["total_detected"] += 1
    
    n = stats["total_processed"]
    old_avg = stats["avg_latency"]
    new_latency = result["total_time_ms"]
    stats["avg_latency"] = ((n - 1) * old_avg + new_latency) / n
    
    st.session_state.results_history.append({
        "plate_text": result["plate_text"],
        "confidence": result["confidence"],
        "total_time_ms": result["total_time_ms"],
        "has_detection": result["has_detection"],
        "timestamp": datetime.now().isoformat(),
    })
    
    if len(st.session_state.results_history) > 100:
        st.session_state.results_history = st.session_state.results_history[-100:]


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    st.set_page_config(
        page_title="VN-LPR Dashboard",
        page_icon="LPR",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    init_session_state()
    apply_custom_css()
    
    conf_threshold = render_sidebar()
    render_dashboard(conf_threshold)


if __name__ == "__main__":
    main()
