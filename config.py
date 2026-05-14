# Configuration file for Deepfake Detection Project
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Email Configuration
SENDER_EMAIL = os.environ.get('SENDER_EMAIL', 'your_email@gmail.com')
SENDER_PASSWORD = os.environ.get('SENDER_PASSWORD', 'your_app_password')
RECIPIENT_EMAIL = os.environ.get('RECIPIENT_EMAIL', SENDER_EMAIL)
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))

# Model Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', './models/ckpt.pth')
ViT_MODEL_NAME = 'vit_base_patch16_224'

# Data Configuration
VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv', 'flv']
AUDIO_SAMPLE_RATE = int(os.environ.get('AUDIO_SR', 22050))

# Processing Configuration
WINDOW_SIZE = 5  # Number of frames per window
RESIZE_FRAME_SIZE = (500, 500)
RESIZE_INPUT_SIZE = (224, 224)

# Detection Threshold
FAKE_THRESHOLD = 0.5  # Probability threshold above which video is considered fake

# Blockchain Configuration
BLOCKCHAIN_FILE = os.environ.get('BLOCKCHAIN_FILE', './blockchain.pkl')

# Node Configuration
NODE_HOST = os.environ.get('NODE_HOST', 'localhost')
NODE_PORT = int(os.environ.get('NODE_PORT', 5000))

# Logging
DEBUG_MODE = os.environ.get('DEBUG_MODE', 'False').lower() == 'true'
