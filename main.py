import hashlib
import json
import datetime
import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import librosa
import torchvision.transforms as transforms
import timm
import os
import tempfile
import pickle
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
try:
    import clip
except ImportError:
    import clip as clip_local
    clip = clip_local

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp.isoformat()
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, datetime.datetime.now(), "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

    def save_to_disk(self, filename=None):
        if filename is None:
            filename = BLOCKCHAIN_FILE
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_disk(cls, filename=None):
        if filename is None:
            filename = BLOCKCHAIN_FILE
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return cls()

class Node:
    def __init__(self, host, port, blockchain):
        self.host = host
        self.port = port
        self.blockchain = blockchain
        self.peers = set()

    def start(self):
        # Placeholder for node start logic
        print(f"Node started on {self.host}:{self.port}")

    def broadcast_new_block(self, new_block):
        # Placeholder for broadcasting logic
        print(f"Broadcasting new block: {new_block.hash}")

class ViTDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTDeepfakeDetector, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_features = self.vit.head.in_features
        self.vit.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)

# Initialize ViT model - checkpoint loading is optional
try:
    if os.path.exists(MODEL_PATH):
        vit_checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        vit_model = ViTDeepfakeDetector()
        vit_model.load_state_dict(vit_checkpoint['model'] if 'model' in vit_checkpoint else vit_checkpoint, strict=False)
        vit_model.eval()
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Using pretrained ViT.")
        vit_model = ViTDeepfakeDetector()
except Exception as e:
    print(f"Warning: Could not load model checkpoint: {e}. Using pretrained ViT.")
    vit_model = ViTDeepfakeDetector()

preprocess_vit = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file, sr=None)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.min)
    return mel

def extract_audio_from_video(video_path, audio_output_path):
    try:
        # Use ffmpeg via subprocess if available, else skip audio
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-q:a', '9', '-n', audio_output_path
            ], capture_output=True, timeout=30)
            if os.path.exists(audio_output_path):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        print(f"Warning: Could not extract audio using ffmpeg. Using silence.")
        return False
    except Exception as e:
        print(f"An error occurred while extracting audio: {e}")
        return False

def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_video_with_audio(video_path, audio_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (500, 500))
        frames.append(frame)
        frame_count += 1

    cap.release() 

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 fps if unable to determine
        print(f"Warning: Unable to determine FPS. Using default value of {fps}")

    duration = frame_count / fps if fps > 0 else 0

    if len(frames) == 0:
        raise ValueError("No frames could be extracted from the video.")

    # Try to load audio spectrogram, if audio file exists
    mel = None
    if os.path.exists(audio_path):
        try:
            mel = get_spectrogram(audio_path)
            mel = cv2.resize(mel, (frames[0].shape[1] * len(frames), frames[0].shape[0]))
            mel = np.expand_dims(mel, axis=2)
            mel = np.repeat(mel, 3, axis=2)
        except Exception as e:
            print(f"Error processing audio: {e}")
            mel = None
    
    # If no audio, use black frames as placeholder
    if mel is None:
        mel = np.zeros((frames[0].shape[0], frames[0].shape[1] * len(frames), 3), dtype=np.uint8)

    combined_images = []
    for i in range(0, len(frames), 5):  # Window size = 5
        window_frames = frames[i:i+5]
        combined_frame = np.concatenate(window_frames, axis=1)
        
        start = int((i / len(frames)) * mel.shape[1])
        end = int(((i + 5) / len(frames)) * mel.shape[1])
        mel_section = mel[:, start:end]
        mel_section = cv2.resize(mel_section, (combined_frame.shape[1], combined_frame.shape[0]))

        combined_image = np.concatenate((mel_section, combined_frame), axis=0)
        combined_images.append(combined_image)
    
    return combined_images, frame_count, fps, duration

def send_email(subject, body, to_email):
    # Email configuration - use environment variables from config
    # Skip email if credentials not configured
    if SENDER_EMAIL == 'your_email@gmail.com' or SENDER_PASSWORD == 'your_app_password':
        print("Warning: Email credentials not configured in .env file. Skipping email notification.")
        return

    # Create the email message
    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send the email
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(message)
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def is_video_deepfaked(video_path, blockchain, node):
    try:
        # Extract audio from video
        audio_path = f"{video_path}_audio.wav"
        extract_audio_from_video(video_path, audio_path)
        
        # Preprocess video and audio
        combined_images, frame_count, fps, duration = preprocess_video_with_audio(video_path, audio_path)

        # Perform deepfake detection
        vit_predictions = []
        frame_predictions = []
        
        for i, img in enumerate(combined_images):
            img_tensor = preprocess_vit(img)
            
            with torch.no_grad():
                vit_output = vit_model(img_tensor.unsqueeze(0))
                vit_probabilities = torch.softmax(vit_output, dim=1)
                vit_fake_prob = vit_probabilities[0][1].item() 
                vit_predictions.append(vit_fake_prob)
                frame_predictions.append((i*5, min((i+1)*5, frame_count), vit_fake_prob))

        avg_vit_prediction = sum(vit_predictions) / len(vit_predictions) if vit_predictions else 0

        is_deepfaked = avg_vit_prediction > FAKE_THRESHOLD
        result = "Fake" if is_deepfaked else "Not Fake"
        
        # Add result to blockchain
        data = {
            "video_path": video_path,
            "result": result,
            "confidence": avg_vit_prediction,
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "timestamp": datetime.datetime.now().isoformat()
        }
        new_block = Block(len(blockchain.chain), datetime.datetime.now(), data, blockchain.get_latest_block().hash)
        blockchain.add_block(new_block)
        node.broadcast_new_block(new_block)
        
        # Save blockchain to disk after adding new block
        blockchain.save_to_disk()
        
        # Prepare detailed email content
        email_body = f"""
    Deepfake Detection Report

    Video Information:
    - Path: {video_path}
    - Frame Count: {frame_count}
    - FPS: {fps:.2f}
    - Duration: {duration:.2f} seconds

    Detection Result: {result}
    Overall Confidence: {avg_vit_prediction:.2f}

    Frame-by-Frame Analysis:
    """
        for start_frame, end_frame, prob in frame_predictions:
            email_body += f"Frames {start_frame}-{end_frame}: Fake Probability = {prob:.2f}\n"

        email_body += f"\nNote: Probabilities above 0.5 indicate a higher likelihood of being fake."

        # Send email with detailed report
        subject = f"Deepfake Detection Report: {'FAKE DETECTED' if is_deepfaked else 'No Fake Detected'}"
        email_body = f"""
Deepfake Detection Report

Video Information:
- Frame Count: {frame_count}
- FPS: {fps:.2f}
- Duration: {duration:.2f} seconds

Detection Result: {result}
Overall Confidence: {avg_vit_prediction:.2f}

Note: Probabilities above 0.5 indicate a higher likelihood of being fake."""
        send_email(subject, email_body, RECIPIENT_EMAIL)
        
        return result, avg_vit_prediction, frame_count, fps, duration, vit_predictions, frame_predictions

    except Exception as e:
        error_message = f"An error occurred during video processing: {str(e)}"
        print(error_message)
        return "Error", 0, 0, 0, 0, [], []

def main():
    # Set page config for better aesthetics
    st.set_page_config(
        page_title="🎬 Deepfake Detector",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .title-section {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 1rem;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Title with custom styling
    st.markdown("<div class='title-section'><h1>🎬 AI Deepfake Detection System</h1><p>Powered by Vision Transformer & Blockchain</p></div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        detection_threshold = st.slider("Detection Threshold", 0.0, 1.0, FAKE_THRESHOLD, 0.05)
        show_blockchain = st.checkbox("📊 Show Blockchain Records", False)
    
    # Load or create blockchain
    blockchain = Blockchain.load_from_disk()
    
    # Create a node
    node = Node(NODE_HOST, NODE_PORT, blockchain)
    node_thread = threading.Thread(target=node.start, daemon=True)
    node_thread.start()
    
    # Main upload section
    st.markdown("### 📹 Upload Video File")
    uploaded_file = st.file_uploader("Select a video file", type=["mp4", "mov", "avi"], key="video_uploader")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            video_path = temp_file.name
        
        # Video preview
        st.markdown("### 🎥 Video Preview")
        st.video(uploaded_file, format="video/mp4")
        
        # Processing section
        st.markdown("---")
        st.markdown("### 🔍 Analysis Results")
        
        with st.spinner("🤖 Analyzing video... This may take a moment..."):
            result, confidence, frame_count, fps, duration, vit_predictions, frame_predictions = is_video_deepfaked(video_path, blockchain, node)
        
        if result == "Error":
            st.error("❌ An error occurred during video processing.")
        else:
            # Display main result with color coding
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result == "Fake":
                    st.error("🚨 FAKE DETECTED")
                else:
                    st.success("✅ GENUINE VIDEO")
            
            with col2:
                # Confidence gauge
                confidence_percent = confidence * 100
                st.metric("Confidence Score", f"{confidence_percent:.1f}%", delta=None)
            
            with col3:
                st.metric("Video Duration", f"{duration:.1f}s")
            
            # Video metadata
            st.markdown("---")
            st.markdown("### 📋 Video Information")
            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            
            with meta_col1:
                st.info(f"📌 **Total Frames**\n{frame_count}")
            with meta_col2:
                st.info(f"⏱️ **FPS**\n{fps:.1f}")
            with meta_col3:
                st.info(f"⏳ **Duration**\n{duration:.2f}s")
            with meta_col4:
                st.info(f"🔍 **Analyzed Windows**\n{len(vit_predictions)}")
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📊 Detection Analytics")
            
            tab1, tab2, tab3 = st.tabs(["📈 Confidence Chart", "🎯 Frame Analysis", "⛓️ Blockchain"])
            
            with tab1:
                # Line chart for confidence scores
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=vit_predictions,
                    mode='lines+markers',
                    name='Fake Probability',
                    line=dict(color='#ff6b6b', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.2)'
                ))
                fig.add_hline(y=detection_threshold, line_dash="dash", line_color="red", 
                             annotation_text="Detection Threshold", annotation_position="right")
                fig.update_layout(
                    title="Confidence Score Across Frames",
                    xaxis_title="Frame Window",
                    yaxis_title="Fake Probability",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Frame-by-frame distribution
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=vit_predictions,
                    nbinsx=10,
                    marker_color='#667eea',
                    name='Distribution'
                ))
                fig_dist.update_layout(
                    title="Probability Distribution",
                    xaxis_title="Fake Probability",
                    yaxis_title="Frequency",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Statistics
                st.markdown("#### 📊 Statistics")
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Max Confidence", f"{max(vit_predictions):.3f}")
                with stat_col2:
                    st.metric("Min Confidence", f"{min(vit_predictions):.3f}")
                with stat_col3:
                    st.metric("Average", f"{np.mean(vit_predictions):.3f}")
                with stat_col4:
                    st.metric("Std Dev", f"{np.std(vit_predictions):.3f}")
            
            with tab3:
                # Blockchain records
                st.markdown("#### ⛓️ Latest Blockchain Records")
                
                # Display last 5 blocks
                blocks_to_show = blockchain.chain[-6:-1] if len(blockchain.chain) > 1 else []
                
                if blocks_to_show:
                    for block in reversed(blocks_to_show):
                        if isinstance(block.data, dict) and "result" in block.data:
                            col_time, col_result, col_conf = st.columns([1, 1, 1])
                            with col_time:
                                st.text(f"⏰ {block.data['timestamp']}")
                            with col_result:
                                if block.data['result'] == "Fake":
                                    st.error(f"🚨 {block.data['result']}")
                                else:
                                    st.success(f"✅ {block.data['result']}")
                            with col_conf:
                                st.info(f"📊 {block.data['confidence']:.2%}")
                else:
                    st.info("No detection records yet")
            
            # Download results
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            # Create JSON report
            report = {
                "detection": result,
                "confidence": float(confidence),
                "timestamp": datetime.datetime.now().isoformat(),
                "video_info": {
                    "frames": frame_count,
                    "fps": fps,
                    "duration": duration
                },
                "frame_predictions": [
                    {"window": i, "confidence": float(pred)} 
                    for i, pred in enumerate(vit_predictions)
                ]
            }
            
            col_json, col_csv = st.columns(2)
            with col_json:
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"deepfake_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col_csv:
                csv_data = "Frame_Window,Fake_Probability\n"
                for i, pred in enumerate(vit_predictions):
                    csv_data += f"{i},{pred:.6f}\n"
                st.download_button(
                    label="📥 Download CSV Data",
                    data=csv_data,
                    file_name=f"predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Blockchain viewer in sidebar
    if show_blockchain and len(blockchain.chain) > 1:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### ⛓️ Blockchain Explorer")
            st.markdown(f"**Total Blocks:** {len(blockchain.chain)}")
            
            for i, block in enumerate(reversed(blockchain.chain[-5:])):
                with st.expander(f"Block #{len(blockchain.chain) - 1 - i}"):
                    st.json(block.__dict__)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem 0;'>
    🔐 Secure Deepfake Detection | ⛓️ Blockchain Verified | 🤖 AI Powered
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()