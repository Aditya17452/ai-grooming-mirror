import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import tempfile
import urllib.request
import os
import json
from datetime import datetime
import cv2
from dotenv import load_dotenv

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Grooming Mirror",
    page_icon="🪞",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Import Google GenAI
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Google GenAI package not available. Please install: pip install google-genai")

# Import MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    st.error("MediaPipe package not available. Please install: pip install mediapipe")

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0A0A0F;
        color: #FFFFFF;
    }
    .score-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .score-number {
        font-size: 4rem;
        font-weight: bold;
        font-family: 'DM Mono', monospace;
    }
    .verdict-excellent { color: #4CCCA8; }
    .verdict-good { color: #4CCCA8; }
    .verdict-fair { color: #C9A84C; }
    .verdict-poor { color: #FF6B6B; }
    .category-bar {
        background: #333;
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
    }
    .category-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    .tab-label {
        font-weight: bold;
        font-family: 'DM Mono', monospace;
    }
    .coaching-box {
        background: #1a1a2e;
        border-left: 4px solid #C9A84C;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'api_key' not in st.session_state:
    # Pre-fill from GEMINI_API_KEY in .env if set; can still be entered in the sidebar
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

def get_score_color(score):
    """Get color based on score"""
    if score >= 85:
        return "#4CCCA8", "EXCELLENT"
    elif score >= 70:
        return "#4CCCA8", "GOOD"
    elif score >= 55:
        return "#C9A84C", "FAIR"
    elif score >= 40:
        return "#FFB347", "NEEDS WORK"
    else:
        return "#FF6B6B", "POOR"

def download_mediapipe_model():
    """Download MediaPipe face detection model"""
    model_path = os.path.join(tempfile.gettempdir(), "blaze_face_short_range.tflite")
    if not os.path.exists(model_path):
        with st.spinner("Downloading MediaPipe model..."):
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
                    model_path
                )
            except Exception as e:
                st.warning(f"Failed to download MediaPipe model: {e}")
                return None
    return model_path

def detect_face_with_mediapipe(image):
    """Detect face using MediaPipe"""
    if not MEDIAPIPE_AVAILABLE:
        return image, "MediaPipe not available - using original image"
    
    try:
        model_path = download_mediapipe_model()
        if not model_path:
            return image, "Failed to download model - using original image"
        
        # Convert PIL to numpy array
        image_np = np.array(image)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        
        # Create face detector
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.4
        )
        
        with mp_vision.FaceDetector.create_from_options(options) as detector:
            detections = detector.detect(mp_image)
            
            if detections.detections:
                # Draw bounding box
                annotated_image = image_np.copy()
                for detection in detections.detections:
                    bbox = detection.bounding_box
                    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                    
                    # Draw green rectangle
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                return Image.fromarray(annotated_image), f"Found {len(detections.detections)} face(s)"
            else:
                return image, "No face detected"
                
    except Exception as e:
        return image, f"MediaPipe error: {str(e)} - using original image"

def prepare_image_for_gemini(image):
    """Prepare image for Gemini API - resize and convert to base64"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to max 1024px on longest side
    max_size = 1024
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_b64

def analyze_grooming_with_gemini(image, api_key):
    """Analyze grooming using Google Gemini API"""
    if not GEMINI_AVAILABLE:
        return {"error": "Google GenAI not available"}
    
    try:
        # Prepare image
        img_b64 = prepare_image_for_gemini(image)
        
        # Create client
        client = genai.Client(api_key=api_key)
        
        # Grooming analysis prompt
        prompt = """You are an expert grooming analyst. Analyze this person's FULL BODY appearance.

Examine EVERYTHING visible:
- HAIR: style, cleanliness, tidiness, condition
- FACE: skin condition, grooming, freshness
- FACIAL HAIR (if any): neat or unkempt
- EYES: alert or tired
- NECK & EARS: cleanliness
- HANDS & NAILS (if visible): condition
- CLOTHING: cleanliness, wrinkles, stains, fit
- OVERALL HYGIENE IMPRESSION

Return ONLY valid JSON, no extra text, no markdown fences:
{
  "overall_score": <integer 0-100>,
  "verdict": "<one sentence summary>",
  "verdict_level": "<excellent|good|fair|poor>",
  "categories": {
    "hair": {
      "score": <0-100>,
      "observation": "<specific observation>",
      "feedback": "<improvement or positive note>"
    },
    "face_skin": {
      "score": <0-100>,
      "observation": "<specific observation>",
      "feedback": "<suggestion>"
    },
    "facial_hair": {
      "score": <0-100>,
      "observation": "<clean shaven / stubble / beard and condition>",
      "feedback": "<suggestion>"
    },
    "clothing_condition": {
      "score": <0-100>,
      "observation": "<describe cleanliness, wrinkles, fit>",
      "feedback": "<suggestion>"
    },
    "hands_nails": {
      "score": <0-100>,
      "observation": "<if visible describe, else say not visible>",
      "feedback": "<suggestion if visible>"
    },
    "overall_hygiene_impression": {
      "score": <0-100>,
      "observation": "<holistic impression>",
      "feedback": "<overall coaching>"
    }
  },
  "top_improvements": [
    "<most impactful specific change>",
    "<second improvement>",
    "<third improvement>"
  ],
  "strengths": [
    "<what is already good>"
  ],
  "coaching_message": "<2-3 sentences warm honest coaching advice>"
}"""
        
        # Make API call
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=img_b64
                            )
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1500,
            )
        )
        
        result_text = response.text
        
        # Parse JSON response
        try:
            # Remove markdown fences if present
            if result_text.startswith('```json'):
                result_text = result_text[7:]
            if result_text.endswith('```'):
                result_text = result_text[:-3]
            
            analysis = json.loads(result_text)
            return analysis
            
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON response: {e}",
                "raw_response": result_text
            }
            
    except Exception as e:
        if "429" in str(e):
            return {"error": "Rate limit exceeded. Please wait 65 seconds and try again."}
        elif "API key" in str(e) or "invalid" in str(e).lower():
            return {"error": "Invalid API key. Please check your Google AI Studio API key."}
        else:
            return {"error": f"Gemini API error: {str(e)}"}

def display_analysis_results(analysis):
    """Display the grooming analysis results"""
    if "error" in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        if "raw_response" in analysis:
            st.text_area("Raw API Response", analysis["raw_response"], height=200)
        return
    
    # Overall score display
    score = analysis["overall_score"]
    color, verdict_text = get_score_color(score)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <div class="score-number" style="color: {color}">{score}</div>
            <div style="color: {color}; font-weight: bold; text-transform: uppercase;">{verdict_text}</div>
            <div style="color: #ccc; font-size: 0.9rem; margin-top: 10px;">{analysis['verdict']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Category Breakdown")
        
        # Create progress bars for each category
        for category_key, category_data in analysis["categories"].items():
            score_val = category_data["score"]
            color_val, _ = get_score_color(score_val)
            
            st.markdown(f"**{category_key.replace('_', ' ').title()}**")
            st.markdown(f"<div class='category-bar'><div class='category-fill' style='width: {score_val}%; background-color: {color_val}'></div></div>", unsafe_allow_html=True)
            st.markdown(f"<small>{category_data['observation']} - {category_data['feedback']}</small>", unsafe_allow_html=True)
            st.markdown("---")
    
    # Improvements and strengths
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Top Improvements")
        for i, improvement in enumerate(analysis["top_improvements"][:3], 1):
            st.markdown(f"{i}. {improvement}")
    
    with col2:
        st.markdown("### ✨ Strengths")
        for strength in analysis["strengths"]:
            st.markdown(f"• {strength}")
    
    # Coaching message
    st.markdown("### 💬 Coaching Message")
    st.markdown(f'<div class="coaching-box">{analysis["coaching_message"]}</div>', unsafe_allow_html=True)

def main():
    st.title("🪞 AI Grooming Mirror")
    st.markdown("Upload a photo or use your webcam to get a complete grooming analysis!")
    
    # API Key input
    st.sidebar.title("Settings")
    api_key_input = st.sidebar.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
    
    if api_key_input:
        st.session_state.api_key = api_key_input
    
    if not st.session_state.api_key:
        st.sidebar.warning("Please enter your Gemini API key to use the analysis feature.")
        st.sidebar.markdown("Get a free API key at [Google AI Studio](https://aistudio.google.com)")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📷 Webcam", "🖼️ Upload Photo", "📊 History"])
    
    with tab1:
        st.markdown("### Take a photo with your webcam")
        
        # Webcam input
        webcam_image = st.camera_input("Take a photo")
        
        if webcam_image:
            # Process image
            image = Image.open(webcam_image)
            
            # MediaPipe face detection
            with st.spinner("Analyzing face with MediaPipe..."):
                annotated_image, mp_message = detect_face_with_mediapipe(image)
            
            st.success(mp_message)
            st.image(annotated_image, caption="MediaPipe Analysis", width=700)
            
            # Gemini analysis
            if st.session_state.api_key:
                if st.button("Analyze Grooming with AI"):
                    with st.spinner("Analyzing grooming with Gemini AI..."):
                        analysis = analyze_grooming_with_gemini(image, st.session_state.api_key)
                        
                        if "error" not in analysis:
                            # Save to history
                            st.session_state.history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "score": analysis["overall_score"],
                                "verdict": analysis["verdict"],
                                "analysis": analysis,
                                "image": image
                            })
                        
                        st.session_state.current_analysis = analysis
                        display_analysis_results(analysis)
            else:
                st.warning("Please enter your Gemini API key in the sidebar to analyze grooming.")
    
    with tab2:
        st.markdown("### Upload a photo from your device")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process image
            image = Image.open(uploaded_file)
            
            # Display original
            st.image(image, caption="Uploaded Image", width=700)
            
            # MediaPipe face detection
            with st.spinner("Analyzing face with MediaPipe..."):
                annotated_image, mp_message = detect_face_with_mediapipe(image)
            
            st.success(mp_message)
            st.image(annotated_image, caption="MediaPipe Analysis", width=700)
            
            # Gemini analysis
            if st.session_state.api_key:
                if st.button("Analyze Grooming with AI"):
                    with st.spinner("Analyzing grooming with Gemini AI..."):
                        analysis = analyze_grooming_with_gemini(image, st.session_state.api_key)
                        
                        if "error" not in analysis:
                            # Save to history
                            st.session_state.history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "score": analysis["overall_score"],
                                "verdict": analysis["verdict"],
                                "analysis": analysis,
                                "image": image
                            })
                        
                        st.session_state.current_analysis = analysis
                        display_analysis_results(analysis)
            else:
                st.warning("Please enter your Gemini API key in the sidebar to analyze grooming.")
    
    with tab3:
        st.markdown("### Analysis History")
        
        if st.session_state.history:
            for i, record in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Analysis from {record['timestamp']} - Score: {record['score']}"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        color, verdict_text = get_score_color(record['score'])
                        st.markdown(f"<div style='color: {color}; font-size: 2rem; font-weight: bold;'>{record['score']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color: {color};'>{verdict_text}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**{record['verdict']}**")
                        st.image(record['image'], width=300)
                        
                        if st.button(f"View Details #{len(st.session_state.history) - i}"):
                            display_analysis_results(record['analysis'])
        else:
            st.info("No analysis history yet. Upload a photo or take a webcam photo to get started!")

if __name__ == "__main__":
    main()