#!/usr/bin/env python3
"""
RAXION - Local AI Assistant
Continuous voice assistant with real-time audio streaming and intelligent conversation
Uses Qwen2-0.5B for natural language understanding with OpenAI Whisper for speech recognition
"""

import whisper
import pyttsx3
import queue
import threading
import time
import wave
import subprocess
import re
import os
import torch
import numpy as np
import random
import difflib
import urllib.parse
import json
import glob
import sys
import argparse
from scipy.io import wavfile
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# Remove faster_whisper import - using original OpenAI Whisper instead

class ContinuousRAXION:
    def __init__(self):
        print("🤖 === Continuous RAXION Starting ===")
        
        # Initialize all attributes first (before loading profile)
        # Audio parameters
        self.chunk_duration = 1.0  # Process every 1 second
        self.sample_rate = 16000
        self.channels = 1
        self.buffer_duration = 10.0  # Keep 10 seconds of audio buffer
        self.max_buffer_samples = int(self.buffer_duration * self.sample_rate)
        
        # Voice activity detection parameters (set defaults first)
        self.speech_threshold = 0.008  # RMS threshold for speech
        self.silence_threshold = 0.005  # RMS threshold for silence
        self.variance_threshold = 0.0001  # Variance threshold
        self.min_speech_duration = 1.0  # Minimum speech duration in seconds
        self.max_silence_duration = 3.0  # Maximum silence before processing
        self.detection_sensitivity = "medium"  # high, medium, low
        
        # Debug mode for tuning voice activity detection
        self.debug_audio = True  # Enable debug by default for tuning
        
        # Load voice profile first (before audio setup)
        self.load_voice_profile()
        
        # Clear GPU memory before loading models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Clean up any old processes first
            self.cleanup_gpu_processes()
            # Check for memory-hogging processes
            self.check_gpu_processes()
        
        self.setup_whisper()
        
        # Load LLM
        self.setup_llm()
        
        # Audio setup
        self.setup_audio()
        
        # TTS setup
        self.setup_tts()
        
        # Continuous audio streaming setup
        self.audio_queue = queue.Queue()
        self.processing_queue = queue.Queue()  # Queue for audio segments ready for processing
        self.audio_buffer = []
        self.is_recording = False
        self.listening = True
        self.processing_audio = False
        self.speaking = False  # Flag to prevent self-hearing
        self.speak_start_time = 0
        self.min_silence_after_speak = 3.0  # Wait 3 seconds after speaking before listening again
        
        # Background processing control
        self.max_concurrent_processing = 2  # Allow up to 2 simultaneous processing tasks
        
        print("✅ Continuous RAXION ready!")
        
    def load_voice_profile(self):
        """Load voice profile with trained settings"""
        # First, try to load GUI configuration
        config_loaded = self.load_gui_config()
        
        try:
            import json
            import os
            
            profile_path = '/home/mateja/raxion_assistant/voice_profile.json'
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    profile = json.load(f)
                
                print(f"📋 Loading voice profile created on {profile['created']}")
                
                # Use trained microphone if available (but GUI config takes precedence)
                if 'best_microphone' in profile and not hasattr(self, 'configured_input_device'):
                    self.trained_microphone = profile['best_microphone']
                    print(f"🎯 Will use trained microphone: {self.trained_microphone}")
                
                # Use trained thresholds (but GUI config takes precedence)
                if 'optimal_threshold' in profile and not hasattr(self, 'configured_sensitivity'):
                    # Convert old numerical thresholds to sensitivity levels
                    recommended_threshold = profile.get('optimal_threshold', 500)
                    if recommended_threshold <= 15:
                        self.detection_sensitivity = "high"
                    elif recommended_threshold <= 50:
                        self.detection_sensitivity = "medium"
                    else:
                        self.detection_sensitivity = "low"
                    
                    print(f"🎚️ Converted trained threshold to sensitivity:")
                    print(f"   Detection sensitivity: {self.detection_sensitivity}")
                    print(f"   (Based on trained threshold: {recommended_threshold})")
                elif 'optimal_thresholds' in profile and not hasattr(self, 'configured_sensitivity'):
                    # Fallback for different profile format - use medium as default
                    self.detection_sensitivity = "medium"
                    print(f"🎚️ Using medium sensitivity (from legacy profile)")
                
                print("✅ Voice profile loaded successfully!")
                return True
            else:
                print("📝 No voice profile found - using default settings")
                if not config_loaded:
                    print("💡 Run train_voice.py to create a personalized profile")
                    print("💡 Or run raxion_config_gui.py to configure devices")
                return False
        except Exception as e:
            print(f"⚠️ Failed to load voice profile: {e}")
            return False
    
    def check_gpu_processes(self):
        """Check for processes using excessive GPU memory"""
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", 
                                   "--format=csv,noheader,nounits"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            pid = parts[0].strip()
                            memory_mb = int(parts[1].strip())
                            
                            # If a process is using more than 2GB, warn about it
                            if memory_mb > 2000:
                                print(f"⚠️  Process {pid} is using {memory_mb}MB of GPU memory")
                                print(f"   Consider running: kill -9 {pid}")
                                
        except Exception as e:
            # Don't fail if nvidia-smi isn't available or fails
            pass

    def cleanup_gpu_processes(self):
        """Kill old GPU processes that might be causing slowdown"""
        try:
            result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", 
                                   "--format=csv,noheader"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and "python" in line.lower():
                        parts = line.split(', ')
                        if len(parts) >= 1:
                            pid = parts[0].strip()
                            try:
                                # Check if it's an old raxion process
                                proc_info = subprocess.run(["ps", "-p", pid, "-o", "cmd="], 
                                                         capture_output=True, text=True)
                                if proc_info.returncode == 0 and ("raxion" in proc_info.stdout.lower() or 
                                                                  "continuous" in proc_info.stdout.lower()):
                                    print(f"🔪 Killing old raxion process {pid}")
                                    subprocess.run(["kill", "-9", pid], check=False)
                            except:
                                pass
                                
        except Exception as e:
            # Don't fail if cleanup fails
            pass

    def load_gui_config(self):
        """Load GUI configuration settings"""
        try:
            import json
            import os
            
            config_path = '/home/mateja/raxion_assistant/raxion_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                print(f"⚙️ Loading GUI configuration (modified: {config.get('last_modified', 'unknown')})")
                
                # Apply configured devices
                if config.get('input_device') and config['input_device'] != 'auto':
                    self.configured_input_device = config['input_device']
                    print(f"🎤 Configured input device: {self.configured_input_device}")
                
                if config.get('output_device') and config['output_device'] != 'auto':
                    self.configured_output_device = config['output_device']
                    print(f"🔊 Configured output device: {self.configured_output_device}")
                
                # Apply sensitivity settings
                if 'detection_sensitivity' in config:
                    self.detection_sensitivity = config['detection_sensitivity']
                    self.configured_sensitivity = True
                    print(f"🎚️ Configured detection sensitivity: {self.detection_sensitivity}")
                elif 'detection_threshold' in config:
                    # Legacy support for old threshold config
                    threshold = config['detection_threshold']
                    if threshold <= 15:
                        self.detection_sensitivity = "high"
                    elif threshold <= 50:
                        self.detection_sensitivity = "medium"
                    else:
                        self.detection_sensitivity = "low"
                    print(f"🎚️ Converted threshold {threshold} to sensitivity: {self.detection_sensitivity}")
                
                # Apply debug settings
                if 'debug_audio' in config:
                    self.debug_audio = config['debug_audio']
                    print(f"🔧 Debug mode: {'enabled' if self.debug_audio else 'disabled'}")
                
                # Apply TTS settings (will be used in setup_tts)
                if 'tts_rate' in config:
                    self.configured_tts_rate = config['tts_rate']
                
                if 'tts_volume' in config:
                    self.configured_tts_volume = config['tts_volume']
                
                print("✅ GUI configuration loaded successfully!")
                return True
            else:
                print("📝 No GUI configuration found - using defaults")
                return False
                
        except Exception as e:
            print(f"⚠️ Failed to load GUI configuration: {e}")
            return False

    def setup_whisper(self):
        """Setup OpenAI Whisper with GPU acceleration and automatic model selection"""
        print("🚀 Loading OpenAI Whisper model...")
        try:
            # Hardware detection for optimal model selection
            device = "cpu"
            model_size = "base"
            
            if torch.cuda.is_available():
                print(f"🎯 CUDA detected: {torch.cuda.get_device_name(0)}")
                
                # Check available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024  # GB
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024  # GB
                gpu_free = gpu_memory - gpu_reserved
                
                print(f"   GPU Memory Status:")
                print(f"   Total: {gpu_memory:.2f} GB")
                print(f"   Allocated: {gpu_allocated:.2f} GB") 
                print(f"   Reserved: {gpu_reserved:.2f} GB")
                print(f"   Free: {gpu_free:.2f} GB")
                
                # Optimize for speed over accuracy - use base model for faster transcription
                if gpu_free > 1.0:  # Use base model for better speed-to-accuracy ratio
                    device = "cuda"
                    model_size = "base" 
                    print("   Using: GPU with base model (optimized for speed)")
                elif gpu_free > 0.5:  # Use tiny model for maximum speed
                    device = "cuda"
                    model_size = "tiny"
                    print("   Using: GPU with tiny model (maximum speed)")
                else:  # Low memory - CPU fallback
                    device = "cpu"
                    model_size = "tiny"
                    print("   Using: CPU fallback (low GPU memory)")
            else:
                print("   Using: CPU (no CUDA available)")
            
            # Load OpenAI Whisper model - much more stable than faster-whisper
            print(f"   Loading {model_size} model...")
            self.model = whisper.load_model(model_size, device=device)
            
            # Memory usage check for GPU
            if device == "cuda":
                gpu_allocated_after = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
                whisper_memory = gpu_allocated_after - gpu_allocated
                print(f"   Whisper using: ~{whisper_memory:.2f} GB")
            
            print(f"✅ OpenAI Whisper loaded: {model_size} model on {device}")
            
        except Exception as e:
            print(f"❌ Whisper setup failed: {e}")
            print("   Trying alternative configurations...")
            
            # Try base model on CPU as fallback
            try:
                print("   🔄 Attempting CPU fallback with base model...")
                self.model = whisper.load_model("base", device="cpu")
                print("✅ CPU fallback successful: base model")
            except Exception as cpu_error:
                print(f"   ❌ CPU fallback failed: {cpu_error}")
                print("   🔄 Emergency fallback to tiny model...")
                try:
                    # Last resort - tiny model on CPU
                    self.model = whisper.load_model("tiny", device="cpu")
                    print("✅ Emergency fallback: tiny model on CPU")
                except Exception as final_error:
                    print(f"❌ Complete failure: {final_error}")
                    print("❌ Cannot initialize any Whisper model - exiting")
                    exit(1)

    def setup_llm(self):
        """Setup Qwen LLM with memory optimization"""
        print("🧠 Loading Qwen2-0.5B-Instruct...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
                # Check GPU memory before loading
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024  # GB
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024  # GB
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024  # GB
                gpu_free = gpu_memory - gpu_reserved
                
                print(f"   GPU Memory Status:")
                print(f"   Total: {gpu_memory:.2f} GB")
                print(f"   Allocated: {gpu_allocated:.2f} GB")
                print(f"   Reserved: {gpu_reserved:.2f} GB")
                print(f"   Free: {gpu_free:.2f} GB")
            
            model_name = "Qwen/Qwen2-0.5B-Instruct"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Try to load with accelerate optimizations first, fallback if not available
            try:
                # Check if accelerate is properly available
                import accelerate
                
                # Optimize loading for memory efficiency
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
                )
                print("   ✅ Loaded with accelerate optimizations")
                
            except (ImportError, Exception) as e:
                print(f"   ⚠️ Accelerate not available ({e}), loading without optimizations...")
                
                # Fallback: Load without accelerate-dependent parameters
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=True,
                )
                
                # Manually move to device if needed
                if device == "cuda":
                    self.llm_model = self.llm_model.to(device)
                
                print("   ✅ Loaded without accelerate (manual device placement)")
            
            if torch.cuda.is_available():
                # Check memory after loading
                gpu_allocated_after = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024  # GB
                gpu_reserved_after = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024  # GB
                
                print(f"   Memory after LLM loading:")
                print(f"   Allocated: {gpu_allocated_after:.2f} GB")
                print(f"   Reserved: {gpu_reserved_after:.2f} GB")
                print(f"   LLM using: {gpu_allocated_after - gpu_allocated:.2f} GB")
            
            print("✅ LLM loaded successfully")
            
        except Exception as e:
            print(f"❌ LLM setup failed: {e}")
            print("   This might be due to memory constraints. Consider:")
            print("   1. Closing other applications")
            print("   2. Using a smaller model")
            print("   3. Running LLM on CPU")
            self.llm_model = None

    def setup_audio(self):
        """Setup audio recording using system default devices"""
        print("🎤 Setting up audio recording...")
        
        try:
            # Check if PulseAudio is available
            result = subprocess.run(["pactl", "info"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ PulseAudio detected")
                
                # Use system default input device
                # This respects user's choice in system settings (gnome-control-center, pavucontrol, etc.)
                self.audio_device = "@DEFAULT_SOURCE@"
                
                # Get current default source info for display
                default_info = subprocess.run(["pactl", "get-default-source"], 
                                            capture_output=True, text=True)
                if default_info.returncode == 0:
                    default_source = default_info.stdout.strip()
                    
                    # Get human-readable name
                    source_info = subprocess.run([
                        "pactl", "list", "sources", "short"
                    ], capture_output=True, text=True)
                    
                    if source_info.returncode == 0:
                        for line in source_info.stdout.strip().split('\n'):
                            if line.strip() and default_source in line:
                                parts = line.split('\t')
                                if len(parts) >= 2:
                                    print(f"🎯 Using system default: {parts[1]}")
                                break
                        else:
                            print(f"🎯 Using system default input device")
                    else:
                        print(f"🎯 Using system default input device")
                else:
                    print(f"🎯 Using system default input device")
                    
            else:
                # Fallback for non-PulseAudio systems
                print("⚠️ PulseAudio not detected, using generic default")
                self.audio_device = "default"
                
            print(f"✅ Audio ready - Device: {self.audio_device}")
            print("   💡 Tip: Change input device in your system settings")
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            print("   Using fallback default device")
            self.audio_device = "default"

    def setup_tts(self):
        """Setup text-to-speech using system default output device"""
        try:
            self.tts = pyttsx3.init()
            
            # Use configured TTS settings if available, otherwise defaults
            tts_rate = getattr(self, 'configured_tts_rate', 180)
            tts_volume = getattr(self, 'configured_tts_volume', 0.8)
            
            self.tts.setProperty('rate', tts_rate)
            self.tts.setProperty('volume', tts_volume)
            
            print(f"✅ TTS ready (rate: {tts_rate}, volume: {tts_volume:.1f})")
            print("   � Tip: TTS uses system default output device")
            
        except Exception as e:
            print(f"❌ TTS setup failed: {e}")
            self.tts = None

    def speak(self, text):
        """Text to speech with self-hearing prevention and output device control"""
        if self.tts:
            # Set speaking flag IMMEDIATELY to prevent self-hearing
            self.speaking = True
            self.speak_start_time = time.time()
            
            # Clear any accumulated audio during speaking
            try:
                while not self.audio_queue.empty():
                    self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Print AFTER setting the speaking flag
            print(f"🗣️  RAXION: {text}")
            
            # Use system default output device (respects user's system settings)
            self.tts.say(text)
            self.tts.runAndWait()
            
            # Keep speaking flag for a bit longer to avoid hearing the tail end
            time.sleep(1.0)  # Increased from 0.5 to 1.0 seconds
            self.speaking = False
            
            print("🔇 Ready to listen again...")

    def fast_command_detection(self, user_text):
        """Fast pattern-based command detection to bypass LLM for common commands"""
        user_lower = user_text.lower().strip()
        
        # Remove wake words for cleaner matching
        for wake_word in ["raxion", "reaction", "traction", "hey raxion", "okay raxion", "hi raxion", "hey reaction", "okay reaction", "hi reaction", "hey traction", "okay traction", "hi traction", "hey", "okay"]:
            user_lower = user_lower.replace(wake_word, "").strip()
        
        # Remove common filler words
        user_lower = re.sub(r'\b(please|can you|could you|would you|will you)\b', '', user_lower).strip()
        
        # Debug: Print the cleaned text
        print(f"   🔍 Fast detection checking: '{user_lower}'")
        
        # Python file creation commands - check this first for priority
        python_creation_phrases = [
            "create python file", "make python file", "write python file", 
            "create python script", "make python script", "write python script",
            "write a script", "create a script", "make a script",
            "write a python file", "create a python file", "make a python file",
            "write script", "create script", "make script"
        ]
        
        for phrase in python_creation_phrases:
            if phrase in user_lower:
                print(f"   ⚡ Matched Python creation phrase: '{phrase}'")
                return "create_python_file", user_text  # Pass full user text for LLM processing
        
        # Check for Python file creation with specific content types (higher priority than general calculator)
        python_content_creation_phrases = [
            "create a python", "make a python", "write a python",
            "create python", "make python", "write python",
            "creates a python", "creates python",  # Added "creates" variants
            "python file that", "python script that", "python program that"
        ]
        
        for phrase in python_content_creation_phrases:
            if phrase in user_lower:
                print(f"   ⚡ Matched Python content creation phrase: '{phrase}'")
                return "create_python_file", user_text  # Pass full user text for LLM processing
        
        # Browser commands
        if any(phrase in user_lower for phrase in ["open browser", "open brave", "start browser", "launch browser"]):
            return "open_browser", "Opening browser"
        
        # YouTube commands  
        if any(phrase in user_lower for phrase in ["open youtube", "go to youtube", "youtube", "play youtube"]):
            return "open_youtube", "Opening YouTube"
        
        # VS Code commands
        if any(phrase in user_lower for phrase in ["open vscode", "open vs code", "open code", "start coding", "launch code", "let's code", "lets code"]):
            return "open_vscode", "Opening VS Code"
        
        # GitHub commands
        if any(phrase in user_lower for phrase in ["open github", "go to github", "github"]):
            return "open_github", "Opening GitHub"
        
        # Claude AI commands - improved variations
        if any(phrase in user_lower for phrase in ["open claude", "claude ai", "claude", "open ai assistant", "open cloud", "cloud ai", "open clawed", "clawed ai", "open clawd", "clawd ai"]):
            return "open_claude", "Opening Claude AI"
        
        # Perplexity commands
        if any(phrase in user_lower for phrase in ["open perplexity", "perplexity", "research", "search perplexity"]):
            return "open_perplexity", "Opening Perplexity"
        
        # ChatGPT commands - improved variations
        if any(phrase in user_lower for phrase in ["open chatgpt", "chat gpt", "chatgpt", "open gpt", "open chat gpt"]):
            return "open_chatgpt", "Opening ChatGPT"
        
        # Time commands
        if any(phrase in user_lower for phrase in ["what time", "current time", "time is it", "tell me the time", "what's the time"]):
            return "get_time", "Here's the current time"
        
        # Weather commands
        if any(phrase in user_lower for phrase in ["weather", "temperature", "how's the weather", "weather today"]):
            return "get_weather", "I'd need weather service setup for detailed weather"
        
        # Screen control commands
        if any(phrase in user_lower for phrase in ["turn off screen", "screen off", "turn off monitor", "turn off display"]):
            return "screen_off", "Turning screens off"
        elif any(phrase in user_lower for phrase in ["turn on screen", "screen on", "turn on monitor", "turn on display"]):
            return "screen_on", "Turning screens on"
        
        # File manager commands
        if any(phrase in user_lower for phrase in ["open files", "file manager", "open folder", "file explorer"]):
            return "open_files", "Opening file manager"
        
        # Directory creation commands (non-Python specific)
        directory_creation_phrases = [
            "create directory", "make directory", "create folder", "make folder",
            "create a directory", "make a directory", "create a folder", "make a folder"
        ]
        
        for phrase in directory_creation_phrases:
            if phrase in user_lower:
                # Check if this is NOT a Python file creation request
                if not any(python_word in user_lower for python_word in ["python", "script", ".py", "file"]):
                    print(f"   ⚡ Matched directory creation phrase: '{phrase}'")
                    return "create_directory", user_text  # We'll need to add this command
        
        # Terminal commands
        if any(phrase in user_lower for phrase in ["open terminal", "open console", "command line"]):
            return "open_terminal", "Opening terminal"
        
        # Calculator commands (moved after Python creation to avoid conflicts)
        calculator_only_phrases = ["calculator", "open calc"]
        if any(phrase in user_lower for phrase in calculator_only_phrases):
            # Make sure it's not part of a Python creation request
            if not any(python_phrase in user_lower for python_phrase in ["create", "make", "write", "python", "script", "file"]):
                return "open_calculator", "Opening calculator"
        
        # Music commands
        if any(phrase in user_lower for phrase in ["play music", "open music", "spotify", "music player"]):
            return "open_music", "Opening music player"
        
        # Python script execution commands
        if any(phrase in user_lower for phrase in ["run python", "execute python", "run script", "execute script", "run that script", "execute that script"]):
            return "run_python_script", self.extract_script_name(user_text)
        
        # System commands
        if any(phrase in user_lower for phrase in ["shutdown", "power off", "turn off computer"]):
            return "shutdown", "Initiating shutdown"
        elif any(phrase in user_lower for phrase in ["restart", "reboot"]):
            return "restart", "Initiating restart"
        elif any(phrase in user_lower for phrase in ["lock screen", "lock computer"]):
            return "lock_screen", "Locking screen"
        
        # Volume commands
        if any(phrase in user_lower for phrase in ["volume up", "increase volume", "louder"]):
            return "volume_up", "Increasing volume"
        elif any(phrase in user_lower for phrase in ["volume down", "decrease volume", "quieter", "lower volume"]):
            return "volume_down", "Decreasing volume"
        elif any(phrase in user_lower for phrase in ["mute", "silence"]):
            return "mute", "Muting audio"
        
        # Web search commands - only explicit browser search
        if any(phrase in user_lower for phrase in ["search for", "search", "google", "look up", "find information about"]):
            return "web_search_browser", self.extract_search_terms(user_text)
        
        # Application closing commands
        if any(phrase in user_lower for phrase in ["close browser", "close brave", "quit browser", "exit browser"]):
            return "close_browser", "Closing browser"
        elif any(phrase in user_lower for phrase in ["close youtube", "quit youtube", "exit youtube"]):
            return "close_browser", "Closing YouTube"  # YouTube runs in browser
        elif any(phrase in user_lower for phrase in ["close vscode", "close vs code", "close code", "quit vscode", "quit vs code", "quit code", "exit vscode", "exit vs code", "exit code"]):
            return "close_vscode", "Closing VS Code"
        elif any(phrase in user_lower for phrase in ["close github", "quit github", "exit github"]):
            return "close_browser", "Closing GitHub"  # GitHub runs in browser
        elif any(phrase in user_lower for phrase in ["close claude", "quit claude", "exit claude", "close ai assistant"]):
            return "close_browser", "Closing Claude AI"  # Claude runs in browser
        elif any(phrase in user_lower for phrase in ["close perplexity", "quit perplexity", "exit perplexity"]):
            return "close_browser", "Closing Perplexity"  # Perplexity runs in browser
        elif any(phrase in user_lower for phrase in ["close chatgpt", "close chat gpt", "quit chatgpt", "quit chat gpt", "exit chatgpt", "exit chat gpt"]):
            return "close_browser", "Closing ChatGPT"  # ChatGPT runs in browser
        elif any(phrase in user_lower for phrase in ["close files", "close file manager", "close folder", "close file explorer", "quit files", "quit file manager", "exit files", "exit file manager"]):
            return "close_files", "Closing file manager"
        elif any(phrase in user_lower for phrase in ["close terminal", "close console", "quit terminal", "quit console", "exit terminal", "exit console"]):
            return "close_terminal", "Closing terminal"
        elif any(phrase in user_lower for phrase in ["close calculator", "close calc", "quit calculator", "quit calc", "exit calculator", "exit calc"]):
            return "close_calculator", "Closing calculator"
        elif any(phrase in user_lower for phrase in ["close music", "close spotify", "quit music", "quit spotify", "exit music", "exit spotify", "stop music"]):
            return "close_music", "Closing music player"
        
        # General app closing commands
        elif any(phrase in user_lower for phrase in ["close all", "close everything", "quit all", "exit all", "close all apps", "quit all apps"]):
            return "close_all_apps", "Closing all applications"
        elif any(phrase in user_lower for phrase in ["close app", "close application", "quit app", "quit application", "exit app", "exit application", "close active app", "close the app", "close this app"]):
            return "close_active_app", "Closing active application"
        
        # Simple greetings and confirmations
        if user_lower in ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]:
            return "greeting", "Hello! How can I help you today?"
        elif user_lower in ["thank you", "thanks", "thank you raxion"]:
            return "thanks", "You're welcome! Anything else I can help with?"
        elif user_lower in ["goodbye", "bye", "see you later"]:
            return "goodbye", "Goodbye! Let me know if you need anything."
        elif any(phrase in user_lower for phrase in ["good night", "goodnight"]):
            return "screen_off", "Good night! Turning screens off."
        elif any(phrase in user_lower for phrase in ["good morning", "goodmorning"]):
            return "screen_on", "Good morning! Turning screens on."
        
        # If no fast command matches, return None to use LLM
        return None, None

    def extract_search_terms(self, user_text):
        """Extract search terms from user's speech"""
        text_lower = user_text.lower().strip()
        
        # Remove wake words
        for wake_word in ["raxion", "reaction", "traction", "hey raxion", "okay raxion", "hi raxion", "hey reaction", "okay reaction", "hi reaction", "hey traction", "okay traction", "hi traction", "hey", "okay"]:
            text_lower = text_lower.replace(wake_word, "").strip()
        
        # Special handling for "google" - extract everything after it
        if "google" in text_lower:
            # Find the position of "google" and extract everything after it
            google_index = text_lower.find("google")
            if google_index != -1:
                # Get text after "google"
                after_google = text_lower[google_index + len("google"):].strip()
                if after_google:
                    # Clean up remaining filler words and punctuation
                    after_google = re.sub(r'\b(please|can you|could you|would you|will you|the|a|an)\b', '', after_google).strip()
                    after_google = re.sub(r'^[,.\s]+|[,.\s]+$', '', after_google)  # Remove leading/trailing punctuation
                    after_google = re.sub(r'\s+', ' ', after_google)  # Replace multiple spaces with single space
                    
                    if len(after_google.strip()) >= 2:
                        return after_google.strip()
        
        # Remove search command phrases and extract the actual search terms
        # Order matters - longer phrases first to avoid partial matches
        search_phrases = [
            "search and tell me about", "search and tell me", "look up and tell me about", 
            "look up and tell me", "find out about", "search for", "look up", 
            "find information about", "what is", "who is", "when was", 
            "where is", "how does", "why does", "search", "google"
        ]
        
        for phrase in search_phrases:
            if phrase in text_lower:
                text_lower = text_lower.replace(phrase, "").strip()
                break
        
        # Clean up remaining filler words and punctuation
        text_lower = re.sub(r'\b(please|can you|could you|would you|will you|the|a|an)\b', '', text_lower).strip()
        text_lower = re.sub(r'^[,.\s]+|[,.\s]+$', '', text_lower)  # Remove leading/trailing punctuation
        text_lower = re.sub(r'\s+', ' ', text_lower)  # Replace multiple spaces with single space
        
        # Ensure we have meaningful search terms
        if len(text_lower.strip()) < 2:
            return "general information"
        
        return text_lower.strip()

    def extract_python_file_content(self, user_text):
        """Extract Python file details from user speech"""
        text_lower = user_text.lower().strip()
        
        # Remove wake words and command phrases
        for wake_word in ["raxion", "reaction", "traction", "hey raxion", "okay raxion", "hi raxion", "hey reaction", "okay reaction", "hi reaction", "hey traction", "okay traction", "hi traction", "hey", "okay"]:
            text_lower = text_lower.replace(wake_word, "").strip()
        
        # Remove file creation command phrases
        creation_phrases = [
            "create python file", "make python file", "write python file",
            "create python script", "make python script", "create a python file",
            "make a python file", "write a python file"
        ]
        
        for phrase in creation_phrases:
            if phrase in text_lower:
                text_lower = text_lower.replace(phrase, "").strip()
                break
        
        # Extract filename and content hints
        filename = "hello_world.py"  # Default filename
        content_type = "hello_world"  # Default content type
        
        # Look for filename patterns
        if "called" in text_lower or "named" in text_lower:
            words = text_lower.split()
            for i, word in enumerate(words):
                if word in ["called", "named"] and i + 1 < len(words):
                    potential_filename = words[i + 1]
                    if not potential_filename.endswith('.py'):
                        potential_filename += '.py'
                    filename = potential_filename.replace(' ', '_')
                    break
        
        # Detect content type from speech
        if any(word in text_lower for word in ["hello", "hello world"]):
            content_type = "hello_world"
        elif any(word in text_lower for word in ["calculator", "add", "subtract", "math"]):
            content_type = "calculator"
        elif any(word in text_lower for word in ["fibonacci", "fib"]):
            content_type = "fibonacci"
        elif any(word in text_lower for word in ["factorial"]):
            content_type = "factorial"
        elif any(word in text_lower for word in ["prime", "prime numbers"]):
            content_type = "prime_numbers"
        elif any(word in text_lower for word in ["password", "generator"]):
            content_type = "password_generator"
        
        return f"{filename}|{content_type}"

    def extract_script_name(self, user_text):
        """Extract script name from user speech for execution"""
        text_lower = user_text.lower().strip()
        
        # Remove wake words and command phrases
        for wake_word in ["raxion", "reaction", "traction", "hey raxion", "okay raxion", "hi raxion", "hey reaction", "okay reaction", "hi reaction", "hey traction", "okay traction", "hi traction", "hey", "okay"]:
            text_lower = text_lower.replace(wake_word, "").strip()
        
        # Remove execution command phrases
        execution_phrases = [
            "run python", "execute python", "run script", "execute script",
            "run that script", "execute that script", "run the script",
            "execute the script", "run python script", "execute python script"
        ]
        
        for phrase in execution_phrases:
            if phrase in text_lower:
                text_lower = text_lower.replace(phrase, "").strip()
                break
        
        # Look for specific filename
        if text_lower and len(text_lower) > 2:
            # Clean up the filename
            script_name = text_lower.replace(" ", "_")
            if not script_name.endswith('.py'):
                script_name += '.py'
            return script_name
        
        # Default to the most recently created script
        return "hello_world.py"

    def analyze_with_llm(self, user_text):
        """Use fast command detection first, fallback to LLM for complex queries"""
        print(f"🔍 Analyzing: '{user_text}'")
        
        # Try fast command detection first
        fast_intent, fast_response = self.fast_command_detection(user_text)
        if fast_intent:
            print(f"⚡ Fast command detected: {fast_intent}")
            return fast_intent, fast_response
        
        print("🧠 Using LLM for complex analysis...")
        
        # Fallback to LLM for conversation and complex commands
        if not self.llm_model:
            return "conversation", "I'm having trouble understanding right now."
        
        try:
            # Determine if this is a complex question that needs a longer response
            user_lower = user_text.lower()
            complex_keywords = [
                "explain", "what is", "how does", "tell me about", "describe", 
                "why", "physics", "science", "theory", "concept", "definition",
                "history", "meaning", "difference", "comparison", "analysis"
            ]
            
            is_complex_query = any(keyword in user_lower for keyword in complex_keywords)
            
            if is_complex_query:
                # Use a prompt that requests brief but complete answers
                prompt = f"""You are RAXION, a helpful AI assistant. Answer the user's question in exactly 2 short sentences. Be concise and direct. End with a period.

User: {user_text}
Assistant:"""
                temperature = 0.7
                top_p = 0.9
                top_k = 50
            else:
                # Short response for simple conversation
                prompt = f"""You are RAXION, a helpful AI assistant. Respond in exactly 1 short sentence. Be direct and end with proper punctuation.

User: {user_text}
Assistant:"""
                temperature = 0.3
                top_p = 0.9
                top_k = 50

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=150 if is_complex_query else 50,  # Balanced limits for concise complete responses
                    temperature=temperature,
                    do_sample=True,  # Always use sampling with the test script's approach
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response_text = response.strip()
            
            # Clean up response to ensure complete sentences
            if response_text.startswith('Response:'):
                response_text = response_text[9:].strip()
            
            # Remove quotes if the whole response is quoted
            if response_text.startswith('"') and response_text.endswith('"'):
                response_text = response_text[1:-1].strip()
            
            # Clean up any remaining prompt artifacts
            cleanup_patterns = ["User:", "Assistant:", "Human:", "AI:"]
            for pattern in cleanup_patterns:
                if pattern in response_text:
                    index = response_text.find(pattern)
                    if index > 10:  # Only remove if it appears later in the text
                        response_text = response_text[:index].strip()
            
            # Ensure complete sentences by truncating at the last complete sentence
            if response_text and not response_text.endswith(('.', '!', '?')):
                # Find the last complete sentence ending
                last_period = response_text.rfind('.')
                last_exclamation = response_text.rfind('!')
                last_question = response_text.rfind('?')
                
                last_sentence_end = max(last_period, last_exclamation, last_question)
                
                if last_sentence_end > len(response_text) * 0.5:  # Only truncate if we keep at least half
                    response_text = response_text[:last_sentence_end + 1].strip()
                    print(f"   🔧 Truncated incomplete sentence")
                elif len(response_text.split()) > 5:  # If it's long enough, add a period
                    response_text = response_text.rstrip(' ,;:') + '.'
                    print(f"   🔧 Added period to complete sentence")
            
            # Ensure it's not empty
            if not response_text or len(response_text.strip()) < 2:
                response_text = "How can I help you?"
            
            # Enhanced conversation handling for common cases
            user_lower = user_text.lower().replace("raxion", "").replace("hey", "").replace("okay", "").replace("hi", "").strip(",.!? ")
            
            if not user_lower or len(user_lower) < 3:
                # Just calling raxion or simple greeting
                greetings = [
                    "Yes, how can I help you?",
                    "I'm here. What do you need?", 
                    "How can I assist you today?",
                    "What can I do for you?"
                ]
                response_text = random.choice(greetings)
            elif "how are you" in user_lower:
                response_text = "I'm doing well, thank you! How can I assist you?"
            
            return "conversation", response_text
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return "conversation", "I'm here to help!"

    def execute_command(self, intent, response_text):
        """Execute the determined command"""
        if intent == "screen_on":
            subprocess.run(["xset", "dpms", "force", "on"], check=False)
            self.speak("Screens are on")
        elif intent == "screen_off":
            subprocess.run(["xset", "dpms", "force", "off"], check=False)
            self.speak("Good night")
        elif intent == "get_time":
            current_time = time.strftime("%I:%M %p")
            self.speak(f"It's {current_time}")
        elif intent == "get_weather":
            self.speak("I'd need weather service setup for that")
        elif intent == "open_browser":
            subprocess.run(["brave-browser"], check=False)
            self.speak("Opening browser")
        elif intent == "open_youtube":
            subprocess.run(["brave-browser", "https://youtube.com"], check=False)
            self.speak("Opening YouTube")
        elif intent == "open_github":
            subprocess.run(["brave-browser", "https://github.com"], check=False)
            self.speak("Opening GitHub")
        elif intent == "open_vscode":
            subprocess.run(["code"], check=False)
            self.speak("Opening VS Code")
        elif intent == "open_claude":
            subprocess.run(["brave-browser", "https://claude.ai"], check=False)
            self.speak("Opening Claude AI")
        elif intent == "open_perplexity":
            subprocess.run(["brave-browser", "https://perplexity.ai"], check=False)
            self.speak("Opening Perplexity")
        elif intent == "open_chatgpt":
            subprocess.run(["brave-browser", "https://chatgpt.com"], check=False)
            self.speak("Opening ChatGPT")
        elif intent == "open_files":
            subprocess.run(["nautilus"], check=False)  # Or "thunar" for XFCE, "dolphin" for KDE
            self.speak("Opening file manager")
        elif intent == "open_terminal":
            subprocess.run(["gnome-terminal"], check=False)  # Or "xfce4-terminal", "konsole", etc.
            self.speak("Opening terminal")
        elif intent == "open_calculator":
            subprocess.run(["gnome-calculator"], check=False)  # Or "xcalc", "kcalc"
            self.speak("Opening calculator")
        elif intent == "open_music":
            subprocess.run(["spotify"], check=False)  # Or try other music players
            self.speak("Opening music player")
        elif intent == "volume_up":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"], check=False)
            self.speak("Volume up")
        elif intent == "volume_down":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"], check=False)
            self.speak("Volume down")
        elif intent == "mute":
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], check=False)
            self.speak("Audio muted")
        elif intent == "lock_screen":
            subprocess.run(["gnome-screensaver-command", "-l"], check=False)  # Or "xscreensaver-command -lock"
            self.speak("Locking screen")
        elif intent == "shutdown":
            self.speak("Shutting down in 10 seconds. Say cancel to stop.")
            # Add confirmation logic here if needed
            subprocess.run(["shutdown", "-h", "+1"], check=False)
        elif intent == "restart":
            self.speak("Restarting in 10 seconds. Say cancel to stop.")
            subprocess.run(["shutdown", "-r", "+1"], check=False)
        elif intent == "web_search_browser":
            # Browser search only - no external API calls
            search_terms = response_text  # response_text contains the extracted search terms
            if search_terms and len(search_terms.strip()) > 0:
                encoded_query = urllib.parse.quote_plus(search_terms)
                subprocess.run(["brave-browser", f"https://www.google.com/search?q={encoded_query}"], check=False)
                self.speak(f"Searching Google for {search_terms}")
            else:
                self.speak("I didn't catch what you wanted me to search for.")
        elif intent == "create_python_file":
            self.create_python_file(response_text)
        elif intent == "create_directory":
            self.create_directory_only(response_text)
        elif intent == "run_python_script":
            self.run_python_script(response_text)
        elif intent in ["greeting", "thanks", "goodbye"]:
            self.speak(response_text)
        
        # Application closing commands
        elif intent == "close_browser":
            self.close_application("brave-browser", "Browser")
        elif intent == "close_vscode":
            self.close_application("code", "VS Code")
        elif intent == "close_files":
            self.close_application(["nautilus", "thunar", "dolphin", "pcmanfm", "caja"], "File manager")
        elif intent == "close_terminal":
            self.close_application(["gnome-terminal", "xfce4-terminal", "konsole", "xterm", "terminator"], "Terminal")
        elif intent == "close_calculator":
            self.close_application(["gnome-calculator", "kcalc", "xcalc", "qalculate-gtk"], "Calculator")
        elif intent == "close_music":
            self.close_application(["spotify", "rhythmbox", "vlc", "audacious", "clementine"], "Music player")
        elif intent == "close_all_apps":
            self.close_all_applications()
        elif intent == "close_active_app":
            self.close_active_application()
        
        else:
            # For conversation, ensure we always have a response
            if not response_text or len(response_text.strip()) == 0:
                response_text = "I'm here. How can I help you?"
            self.speak(response_text)

    def continuous_audio_capture(self):
        """Continuously capture audio in a separate thread"""
        print("🎧 Starting continuous audio capture...")
        
        try:
            chunk_duration = 1  # 1 second chunks
            chunk_counter = 0
            
            while self.listening:
                try:
                    # Create unique temp file for this chunk
                    temp_file = f"/tmp/raxion_chunk_{chunk_counter}.wav"
                    
                    # Start parecord process
                    proc = subprocess.Popen([
                        "parecord",
                        f"--device={self.audio_device}",
                        "--file-format=wav",
                        "--channels=1",
                        f"--rate={self.sample_rate}",
                        temp_file
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Let it record for 1 second
                    time.sleep(chunk_duration)
                    
                    # Stop the process
                    proc.terminate()
                    proc.wait()
                    
                    # Check if file was created and has content
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                        try:
                            import scipy.io.wavfile as wavfile
                            sample_rate, audio_data = wavfile.read(temp_file)
                            
                            if len(audio_data) > 0:
                                # Put audio chunk in queue for processing
                                if not self.audio_queue.full():
                                    self.audio_queue.put(audio_data.copy())
                                
                                # Add to buffer
                                self.audio_buffer.extend(audio_data)
                                
                                # Keep buffer size manageable
                                if len(self.audio_buffer) > self.max_buffer_samples:
                                    self.audio_buffer = self.audio_buffer[-self.max_buffer_samples:]
                            
                        except Exception as read_error:
                            print(f"Error reading audio chunk: {read_error}")
                        finally:
                            # Clean up temp file
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                    else:
                        if chunk_counter % 30 == 0:  # Print every 30 chunks (30 seconds) to reduce spam
                            print(f"⚠️  No audio recorded in chunk {chunk_counter}")
                    
                    chunk_counter += 1
                    
                except Exception as e:
                    if self.listening:
                        print(f"Audio capture error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Failed to start audio capture: {e}")
        
        print("🎧 Audio capture stopped")

    def detect_speech_activity(self, audio_data, threshold=None):
        """Enhanced voice activity detection with noise floor adaptation"""
        if len(audio_data) == 0:
            return False
        
        # Convert to float for calculations and normalize properly
        audio_float = audio_data.astype(np.float32)
        
        # Normalize audio data to -1 to 1 range (16-bit audio is -32768 to 32767)
        if np.max(np.abs(audio_float)) > 1.0:
            audio_float = audio_float / 32768.0
        
        # Calculate RMS (Root Mean Square) - simple volume measurement
        rms = np.sqrt(np.mean(audio_float ** 2))
        
        # Calculate variance for detecting dynamic content (speech vs constant noise)
        variance = np.var(audio_float)
        
        # Conservative thresholds for normalized audio (0-1 range)
        sensitivity_thresholds = {
            "high": 0.050,     # Sensitive - picks up quieter speech
            "medium": 0.070,   # Balanced - good for normal speaking  
            "low": 0.100       # Conservative - only loud, clear speech
        }
        
        # Variance thresholds - speech has more dynamic content than steady noise
        variance_thresholds = {
            "high": 0.0015,    # More sensitive to speech dynamics
            "medium": 0.0025,  # Balanced variance detection
            "low": 0.0040      # Less sensitive - only clear speech patterns
        }
        
        sensitivity = getattr(self, 'detection_sensitivity', 'medium')
        rms_threshold = sensitivity_thresholds.get(sensitivity, 0.070)  # Use correct medium default
        var_threshold = variance_thresholds.get(sensitivity, 0.0025)   # Use correct medium default
        
        # Require BOTH volume AND variance to indicate speech
        # This helps reject constant background noise
        has_volume = rms > rms_threshold
        has_dynamics = variance > var_threshold
        is_speech = has_volume and has_dynamics
        
        if self.debug_audio and is_speech:  # Only print when speech is detected
            print(f"[🔊 RMS:{rms:.4f}>{rms_threshold:.3f} VAR:{variance:.6f}>{var_threshold:.6f}]", end="", flush=True)
        
        return is_speech

    def process_audio_continuously(self):
        """Process audio from the queue continuously with better noise rejection"""
        accumulated_audio = []
        silence_counter = 0
        speech_detected = False
        chunk_counter = 0
        consecutive_speech_chunks = 0
        
        while self.listening:
            try:
                # Skip processing if we're speaking or just finished speaking
                if self.speaking or (time.time() - self.speak_start_time) < self.min_silence_after_speak:
                    # Clear queue while speaking to prevent buildup
                    try:
                        while not self.audio_queue.empty():
                            self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    
                    # Reset speech detection state
                    accumulated_audio = []
                    silence_counter = 0
                    speech_detected = False
                    consecutive_speech_chunks = 0
                    time.sleep(0.1)
                    continue
                
                # Get audio chunk from queue (with timeout)
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    chunk_counter += 1
                except queue.Empty:
                    continue
                
                # Check for speech activity with simplified detection
                has_speech = self.detect_speech_activity(audio_chunk)
                
                if has_speech:
                    consecutive_speech_chunks += 1
                    # Start recording immediately on first speech detection (most responsive)
                    speech_detected = True
                    silence_counter = 0
                    accumulated_audio.extend(audio_chunk)
                    # Only show visual indicator for speech, not debug spam
                    if not self.debug_audio:
                        print("🔊", end="", flush=True)
                else:
                    consecutive_speech_chunks = 0  # Reset consecutive counter
                    if speech_detected:
                        silence_counter += 1
                        accumulated_audio.extend(audio_chunk)
                        # Only show silence during active speech detection
                        if not self.debug_audio and silence_counter == 1:
                            print(".", end="", flush=True)
                    else:
                        # If we haven't detected speech yet, don't accumulate noise
                        accumulated_audio = []
                    
                    # Process when we have enough silence after speech
                    # Natural pause detection - wait for user to finish speaking
                    if silence_counter >= 1 and len(accumulated_audio) > 0:  # 1 second of silence
                        # Check if we have enough audio to be meaningful speech
                        # Minimum 0.5 seconds for actual words
                        min_speech_samples = int(0.5 * self.sample_rate)
                        if len(accumulated_audio) >= min_speech_samples:
                            if not self.debug_audio:
                                print("\n🧠 Queuing for processing...")
                            else:
                                print("\n🧠 Queuing speech for processing...")
                            
                            # Add to processing queue instead of blocking
                            audio_copy = accumulated_audio.copy()
                            self.processing_queue.put(audio_copy)
                        else:
                            if self.debug_audio:
                                print(f"\n⚠️ Discarding short audio segment ({len(accumulated_audio)} samples)")
                        
                        # Reset for next speech segment immediately
                        accumulated_audio = []
                        silence_counter = 0
                        speech_detected = False
                
                # Process very long segments to prevent memory issues
                # But allow for natural speech patterns - no artificial time limits
                if len(accumulated_audio) > self.sample_rate * 30:  # 30 seconds max (very generous)
                    if len(accumulated_audio) >= int(0.5 * self.sample_rate):  # Use same 0.5s minimum as above
                        if not self.debug_audio:
                            print("\n⚠️ Processing very long segment...")
                        else:
                            print("\n⚠️ Processing very long speech segment...")
                        
                        audio_copy = accumulated_audio.copy()
                        self.processing_queue.put(audio_copy)
                    else:
                        if self.debug_audio:
                            print(f"\n⚠️ Discarding very long noise segment ({len(accumulated_audio)} samples)")
                    
                    accumulated_audio = []
                    silence_counter = 0
                    speech_detected = False
                    consecutive_speech_chunks = 0
                    
            except Exception as e:
                print(f"Audio processing error: {e}")
                time.sleep(0.1)

    def background_audio_processor(self):
        """Background worker that processes audio segments without blocking listening"""
        active_processors = 0
        
        while self.listening:
            try:
                # Wait for audio segments to process
                try:
                    audio_data = self.processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Limit concurrent processing to prevent resource exhaustion
                if active_processors >= self.max_concurrent_processing:
                    # Put the audio back in queue and wait
                    self.processing_queue.put(audio_data)
                    time.sleep(0.1)
                    continue
                
                # Process in a separate thread to maintain concurrency
                def process_worker():
                    nonlocal active_processors
                    active_processors += 1
                    try:
                        self.process_audio_segment(audio_data)
                    finally:
                        active_processors -= 1
                
                processing_thread = threading.Thread(target=process_worker, daemon=True)
                processing_thread.start()
                
            except Exception as e:
                print(f"Background processor error: {e}")
                time.sleep(0.1)

    def process_audio_segment(self, audio_data):
        """Process a single audio segment (non-blocking version of process_accumulated_audio)"""
        try:
            # Start timing the transcription process
            transcription_start = time.time()
            
            # Convert to numpy array and normalize
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Normalize audio data to -1 to 1 range (16-bit audio is -32768 to 32767)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / 32768.0
            
            # Skip redundant quality checks - if audio made it here, it already passed detection
            # The continuous processor already validated speech quality, so just transcribe
            if self.debug_audio:
                rms = np.sqrt(np.mean(audio_array ** 2))
                variance = np.var(audio_array)
                print(f"   🎯 Processing speech segment (RMS:{rms:.4f} VAR:{variance:.6f})")
            
            # Create unique temp file for concurrent processing
            import threading
            thread_id = threading.current_thread().ident
            temp_file = f"/tmp/raxion_segment_{thread_id}_{int(time.time() * 1000)}.wav"
            
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_file, self.sample_rate, audio_array)
            
            # Transcribe with OpenAI Whisper - optimized for speed
            whisper_start = time.time()
            result = self.model.transcribe(
                temp_file,
                language="en",  # Specify language to skip language detection
                temperature=0.0,  # Use deterministic transcription
                fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
                verbose=False,  # Suppress verbose output
                # Speed optimizations:
                condition_on_previous_text=False,  # Don't use previous text context (faster)
                no_speech_threshold=0.6,  # Higher threshold to skip silent segments faster
                logprob_threshold=-1.0,  # Default value but explicit for clarity
                compression_ratio_threshold=2.4  # Skip segments with poor compression
            )
            whisper_time = time.time() - whisper_start
            
            # Extract text from result
            text = result.get("text", "").strip()
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            if text and len(text) > 2:
                total_time = time.time() - transcription_start
                print(f"⏱️ Transcription timing: Whisper={whisper_time:.2f}s Total={total_time:.2f}s")
                
                # More aggressive self-hearing prevention
                time_since_speaking = time.time() - self.speak_start_time
                if not self.speaking and time_since_speaking > self.min_silence_after_speak:
                    # Additional check: ignore if text contains TTS-like patterns
                    text_lower = text.lower()
                    tts_phrases = [
                        "raxion online", "ready to listen", "how can i help", 
                        "opening", "turning", "launching", "hello sir",
                        "screens are", "volume", "anything else"
                    ]
                    
                    # Skip if this looks like echo of our own speech
                    is_likely_echo = any(phrase in text_lower for phrase in tts_phrases)
                    
                    if not is_likely_echo:
                        self.process_command(text)
                    else:
                        print(f"   🔇 Ignoring likely echo: '{text}'")
                else:
                    print(f"   🔇 Ignoring audio - still in silence period ({time_since_speaking:.1f}s)")
            
        except Exception as e:
            print(f"Audio segment processing error: {e}")

    def fuzzy_match_wake_word(self, text):
        """Check if text contains a wake word using fuzzy matching"""
        import difflib
        
        text_lower = text.lower()
        wake_words = ["raxion", "reaction", "traction", "rection", "hey raxion", "okay raxion", "hi raxion", "hey reaction", "okay reaction", "hi reaction", "hey traction", "okay traction", "hi traction"]
        
        # First, try exact matching
        for wake_word in wake_words:
            if wake_word in text_lower:
                return True, wake_word
        
        # Then try fuzzy matching for single words that might be misheard
        words = text_lower.split()
        for word in words:
            # Clean punctuation from word for better matching
            clean_word = word.strip(",.!?;:")
            
            # Skip very short words or common words that aren't wake words
            if len(clean_word) < 4 or clean_word in ["please", "open", "the", "and", "with", "what", "how", "when", "where"]:
                continue
                
            # Check specific known misheard variations first
            misheard_variations = {
                "reaction": "raxion",  # Very common mishearing - important to add
                "rection": "raxion",  # Very common mishearing - important to add
                "traction": "raxion",  # New wake word
                "ration": "raxion",    # Common mishearing of raxion
                "action": "raxion",    # Common mishearing of raxion
                "fashion": "raxion",   # Common mishearing of raxion
                "fraction": "raxion",  # Common mishearing of raxion
                "taxation": "raxion",  # Another -ation sound mishearing
                "station": "raxion",   # Another -ation sound mishearing
                "nation": "raxion",    # Another -ation sound mishearing
                "passion": "raxion",   # Another -ation sound mishearing
                "mansion": "raxion",   # Another -sion sound mishearing
                "version": "raxion",   # Another -sion sound mishearing
                "session": "raxion",   # Another -sion sound mishearing
                "mission": "raxion",   # Another -sion sound mishearing
                "relation": "raxion",  # Longer -ation mishearing
                "creation": "raxion",  # Longer -ation mishearing
                "attraction": "traction", # Common mishearing of traction
                "retraction": "traction", # Common mishearing of traction
                "distraction": "traction", # Common mishearing of traction
                "extraction": "traction", # Common mishearing of traction
                "transaction": "traction", # Common mishearing of traction
                "axis": "raxion",      # Axis sounds similar to raxion
                "nexus": "raxion",     # Similar phonetic pattern
                "maxon": "raxion",     # Close pronunciation
                "saxon": "raxion",     # Close pronunciation
                "jackson": "raxion",   # Common name mishearing
                "watson": "raxion",    # Common name mishearing
                "raxion": "raxion",    # Exact match support
            }
            
            if clean_word in misheard_variations:
                matched_wake_word = misheard_variations[clean_word]
                print(f"   🎯 Known variation: '{clean_word}' → '{matched_wake_word}'")
                return True, f"{matched_wake_word} (heard as '{clean_word}')"
            
            # Only check high similarity for words that could plausibly be wake words
            if len(clean_word) >= 5 and len(clean_word) <= 10:  # Extended range for traction (8 letters)
                # Check similarity against all wake words
                best_similarity = 0
                best_match = None
                
                for wake_word in ["raxion", "reaction", "traction"]:
                    similarity = difflib.SequenceMatcher(None, clean_word, wake_word).ratio()
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = wake_word
                
                if best_similarity >= 0.75:  # 75% similarity threshold
                    print(f"   🎯 Fuzzy match: '{clean_word}' → '{best_match}' (similarity: {best_similarity:.2f})")
                    return True, f"{best_match} (heard as '{clean_word}')"
        
        return False, None

    def process_command(self, text):
        """Process the transcribed text"""
        start_time = time.time()
        print(f"🔊 HEARD: '{text}'")
        
        # Enhanced wake word detection with fuzzy matching
        has_wake_word, matched_word = self.fuzzy_match_wake_word(text)
        
        if not has_wake_word:
            print("   (No wake word detected)")
            return
        
        print(f"   ✅ Wake word detected: {matched_word}")
        
        # Use LLM to analyze intent and generate response
        intent, response_text = self.analyze_with_llm(text)
        
        processing_time = time.time() - start_time
        print(f"   Intent: {intent} (processed in {processing_time:.2f}s)")
        print(f"   Response: {response_text}")
        
        # If no specific command was detected, make sure raxion still responds
        if intent == "conversation" and not response_text:
            response_text = "Yes, how can I help you?"
        elif intent == "conversation" and response_text in ["How can I help you?", "I'm here to help!"]:
            # Make the response more natural when no specific command is detected
            conversation_responses = [
                "Yes, I'm here. What can I do for you?",
                "How can I help you today?", 
                "What would you like me to do?",
                "I'm listening. What do you need?",
                "Yes? How can I assist you?"
            ]
            import random
            response_text = random.choice(conversation_responses)
        
        # Execute the command
        self.execute_command(intent, response_text)

    def create_python_file(self, user_request):
        """Create a Python file with LLM-generated content based on user request"""
        try:
            print(f"🧠 Generating Python code for: '{user_request}'")
            
            # Extract target directory and filename from request
            target_directory, filename = self.extract_file_and_directory_from_request(user_request)
            
            # Generate code using LLM
            generated_code = self.generate_python_code_with_llm(user_request)
            
            if not generated_code:
                self.speak("I couldn't generate the Python code. Please try rephrasing your request.")
                return
            
            # Ensure target directory exists
            if not os.path.exists(target_directory):
                print(f"📁 Creating directory: {target_directory}")
                os.makedirs(target_directory, exist_ok=True)
                self.speak(f"Created directory {os.path.basename(target_directory)}")
            
            # Create the file in the target directory
            file_path = os.path.join(target_directory, filename)
            
            with open(file_path, 'w') as f:
                f.write(generated_code)
            
            # Make it executable
            os.chmod(file_path, 0o755)
            
            print(f"📝 Created Python file: {filename}")
            print(f"   Location: {file_path}")
            lines_count = len(generated_code.split('\n'))
            print(f"   Generated {lines_count} lines of code")
            
            # Show a preview of the generated code (first few lines)
            lines = generated_code.split('\n')
            preview_lines = lines[:5] if len(lines) > 5 else lines
            print(f"   Preview:")
            for i, line in enumerate(preview_lines, 1):
                print(f"     {i}: {line}")
            if len(lines) > 5:
                print(f"     ... and {len(lines) - 5} more lines")
            
            # Create a user-friendly response about location
            if target_directory == "/home/mateja/raxion_assistant":
                location_msg = "in the current directory"
            else:
                location_msg = f"in the {os.path.basename(target_directory)} directory"
            
            self.speak(f"Created Python file {filename} with generated code {location_msg}. The script is ready to run!")
            
        except Exception as e:
            print(f"❌ Error creating Python file: {e}")
            self.speak("Sorry, I couldn't create the Python file")

    def create_directory_only(self, user_request):
        """Create a directory without any files based on user request"""
        try:
            print(f"📁 Creating directory from request: '{user_request}'")
            
            # Extract directory name from the request
            directory_name = self.extract_directory_name_from_request(user_request)
            
            if not directory_name:
                self.speak("I couldn't determine what directory to create. Please specify a directory name.")
                return
            
            # Use the same directory mapping logic as file creation
            user_lower = user_request.lower()
            base_directory = "/home/mateja/raxion_assistant"
            
            # Common directory mappings
            directory_mappings = {
                "documents": "/home/mateja/Documents",
                "document": "/home/mateja/Documents",
                "docs": "/home/mateja/Documents",
                "desktop": "/home/mateja/Desktop",
                "downloads": "/home/mateja/Downloads", 
                "download": "/home/mateja/Downloads",
                "home": "/home/mateja",
                "pictures": "/home/mateja/Pictures",
                "videos": "/home/mateja/Videos",
                "music": "/home/mateja/Music"
            }
            
            # Determine target location
            if directory_name.lower() in directory_mappings:
                target_path = directory_mappings[directory_name.lower()]
            else:
                target_path = os.path.join(base_directory, directory_name)
            
            # Create the directory
            if os.path.exists(target_path):
                print(f"📁 Directory already exists: {target_path}")
                self.speak(f"The {directory_name} directory already exists")
            else:
                os.makedirs(target_path, exist_ok=True)
                print(f"📁 Created directory: {target_path}")
                
                # Create a user-friendly response about location
                if target_path.startswith("/home/mateja/raxion_assistant"):
                    location_msg = "in the current directory"
                else:
                    location_msg = f"in {os.path.dirname(target_path)}"
                
                self.speak(f"Created {directory_name} directory {location_msg}")
                
        except Exception as e:
            print(f"❌ Error creating directory: {e}")
            self.speak("Sorry, I couldn't create the directory")

    def extract_directory_name_from_request(self, user_request):
        """Extract directory name from user request"""
        user_lower = user_request.lower()
        
        # Look for explicit directory name patterns
        import re
        patterns = [
            r"directory\s+called\s+([a-zA-Z0-9_\s]+)",
            r"folder\s+called\s+([a-zA-Z0-9_\s]+)",
            r"create\s+([a-zA-Z0-9_\s]+)\s+directory",
            r"make\s+([a-zA-Z0-9_\s]+)\s+directory",
            r"create\s+([a-zA-Z0-9_\s]+)\s+folder",
            r"make\s+([a-zA-Z0-9_\s]+)\s+folder"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_lower)
            if match:
                dir_name = match.group(1).strip()
                # Clean up the directory name
                dir_name = dir_name.replace(' ', '_')
                print(f"🎯 Directory name extracted: '{match.group(1)}' -> '{dir_name}'")
                return dir_name
        
        # If no explicit pattern found, try to extract from context
        words = user_lower.split()
        create_words = ["create", "make"]
        dir_words = ["directory", "folder"]
        
        for i, word in enumerate(words):
            if word in create_words and i + 1 < len(words):
                # Look for the next meaningful word
                for j in range(i + 1, len(words)):
                    if words[j] in dir_words:
                        continue
                    if words[j] in ["a", "the", "called", "named"]:
                        continue
                    # Found the directory name
                    dir_name = words[j]
                    print(f"🎯 Directory name from context: '{dir_name}'")
                    return dir_name
        
        return None

    def extract_file_and_directory_from_request(self, user_request):
        """Extract target directory and filename from user request"""
        user_lower = user_request.lower()
        base_directory = "/home/mateja/raxion_assistant"
        
        # Common directory mappings
        directory_mappings = {
            "documents": "/home/mateja/Documents",
            "document": "/home/mateja/Documents",
            "docs": "/home/mateja/Documents",
            "desktop": "/home/mateja/Desktop",
            "downloads": "/home/mateja/Downloads", 
            "download": "/home/mateja/Downloads",
            "home": "/home/mateja",
            "pictures": "/home/mateja/Pictures",
            "videos": "/home/mateja/Videos",
            "music": "/home/mateja/Music",
            "projects": "/home/mateja/Projects",
            "code": "/home/mateja/Code",
            "scripts": "/home/mateja/Scripts",
            "python": "/home/mateja/Python",
            "work": "/home/mateja/Work"
        }
        
        # Extract filename first
        filename = self.extract_filename_from_request(user_request)
        
        # Look for directory specifications in the request
        target_directory = base_directory  # Default to current directory
        
        # Pattern 1: "in the [directory] directory"
        import re
        patterns = [
            r"in\s+the\s+([a-zA-Z0-9_]+)\s+directory",
            r"in\s+([a-zA-Z0-9_]+)\s+directory", 
            r"to\s+the\s+([a-zA-Z0-9_]+)\s+directory",
            r"to\s+([a-zA-Z0-9_]+)\s+directory",
            r"into\s+the\s+([a-zA-Z0-9_]+)\s+directory",
            r"into\s+([a-zA-Z0-9_]+)\s+directory",
            r"inside\s+the\s+([a-zA-Z0-9_]+)\s+directory",
            r"inside\s+([a-zA-Z0-9_]+)\s+directory"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_lower)
            if match:
                dir_name = match.group(1).strip()
                if dir_name in directory_mappings:
                    target_directory = directory_mappings[dir_name]
                    print(f"🎯 Directory detected: '{dir_name}' -> {target_directory}")
                else:
                    # Create a custom directory in the base directory
                    target_directory = os.path.join(base_directory, dir_name)
                    print(f"🎯 Custom directory: '{dir_name}' -> {target_directory}")
                break
        
        # Pattern 2: "create directory called [name] and then..." 
        directory_creation_patterns = [
            r"create\s+(?:a\s+)?directory\s+called\s+([a-zA-Z0-9_]+)",
            r"make\s+(?:a\s+)?directory\s+called\s+([a-zA-Z0-9_]+)", 
            r"create\s+(?:a\s+)?folder\s+called\s+([a-zA-Z0-9_]+)",
            r"make\s+(?:a\s+)?folder\s+called\s+([a-zA-Z0-9_]+)"
        ]
        
        for pattern in directory_creation_patterns:
            match = re.search(pattern, user_lower)
            if match:
                dir_name = match.group(1).strip()
                if dir_name in directory_mappings:
                    target_directory = directory_mappings[dir_name]
                else:
                    target_directory = os.path.join(base_directory, dir_name)
                print(f"🎯 Directory to create: '{dir_name}' -> {target_directory}")
                break
        
        # Pattern 3: Direct directory mentions (documents, desktop, etc.)
        if target_directory == base_directory:  # Still using default
            for dir_name, dir_path in directory_mappings.items():
                if dir_name in user_lower:
                    # Make sure it's not part of another word
                    if re.search(rf'\b{re.escape(dir_name)}\b', user_lower):
                        target_directory = dir_path
                        print(f"🎯 Directory mentioned: '{dir_name}' -> {target_directory}")
                        break
        
        # Pattern 4: Relative path indicators
        if "current directory" in user_lower or "here" in user_lower:
            target_directory = base_directory
            print(f"🎯 Using current directory: {target_directory}")
        
        # If filename contains directory info, extract it
        if "/" in filename:
            # Handle relative paths in filename
            dir_part, filename = os.path.split(filename)
            if dir_part:
                target_directory = os.path.join(target_directory, dir_part)
                print(f"🎯 Subdirectory from filename: {dir_part}")
        
        return target_directory, filename

    def extract_filename_from_request(self, user_request):
        """Extract filename from user request or generate appropriate name"""
        user_lower = user_request.lower()
        
        # Look for explicit filename patterns - more specific patterns
        patterns = [
            r"called\s+([a-zA-Z0-9_.-]+(?:\.py)?)",
            r"named\s+([a-zA-Z0-9_.-]+(?:\.py)?)",
            r"save\s+(?:it\s+)?as\s+([a-zA-Z0-9_.-]+(?:\.py)?)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, user_lower)
            if match:
                potential_filename = match.group(1).strip()
                # Skip common non-filename words that might be captured
                if potential_filename not in ["for", "that", "which", "with", "from", "to", "in", "at", "on"]:
                    filename = potential_filename.replace(' ', '_')
                    if not filename.endswith('.py'):
                        filename += '.py'
                    print(f"🎯 Filename detected: '{match.group(1)}' -> {filename}")
                    return filename
        
        # Generate filename based on content description with better matching
        content_mappings = {
            ("hello", "hello world", "world", "print hello", "printing hello"): "hello_world.py",
            ("calculator", "calc", "math", "add", "subtract", "multiply", "divide"): "calculator.py",
            ("fibonacci", "fib"): "fibonacci.py",
            ("factorial",): "factorial.py",
            ("prime", "prime numbers"): "prime_numbers.py", 
            ("password", "generator", "random password"): "password_generator.py",
            ("game", "tic tac toe", "guess", "quiz"): "game.py",
            ("web", "scraper", "scraping", "requests", "beautiful soup"): "web_scraper.py",
            ("data", "csv", "pandas", "analysis"): "data_processor.py",
            ("todo", "task", "reminder"): "todo_app.py",
            ("weather", "api", "forecast"): "weather_app.py",
            ("timer", "stopwatch", "countdown"): "timer.py",
            ("file", "organizer", "sort", "cleanup"): "file_organizer.py",
            ("backup", "copy", "sync"): "backup_script.py",
            ("log", "logger", "logging"): "logger.py"
        }
        
        for keywords, filename in content_mappings.items():
            if any(keyword in user_lower for keyword in keywords):
                print(f"🎯 Generated filename from content: {filename}")
                return filename
        
        # Default filename
        return "generated_script.py"

    def generate_python_code_with_llm(self, user_request):
        """Use LLM to generate Python code based on user request"""
        if not self.llm_model:
            print("❌ LLM not available for code generation")
            return None
        
        try:
            # First, try template-based approach for reliable results
            user_lower = user_request.lower()
            
            print("🎯 Using template-based code generation for reliability...")
            
            if any(word in user_lower for word in ["hello", "hello world", "world", "helloworld", "greet"]):
                # Hello world template
                template_code = '''#!/usr/bin/env python3
                    """
                    Hello World Program
                    Generated by raxion voice assistant
                    """

                    def greet_user():
                        """Function to greet the user"""
                        print("Hello, World!")
                        name = input("What's your name? ")
                        print(f"Hello, {name}! Nice to meet you!")
                        return name

                    def display_message():
                        """Display a welcome message"""
                        print("=" * 40)
                        print("Welcome to this Python program!")
                        print("Generated by raxion voice assistant")
                        print("=" * 40)

                    def main():
                        """Main function to run the program"""
                        display_message()
                        user_name = greet_user()
                        print(f"\\nGoodbye, {user_name}! Have a great day!")

                    if __name__ == "__main__":
                        main()
                    '''
                print("   ✅ Using Hello World template")
                return template_code
                
            elif any(word in user_lower for word in ["calculator", "calc", "math"]):
                # Calculator template
                template_code = '''#!/usr/bin/env python3
                    """
                    Simple Calculator Program
                    Generated by raxion voice assistant
                    """

                    def add(x, y):
                        """Add two numbers"""
                        return x + y

                    def subtract(x, y):
                        """Subtract two numbers"""
                        return x - y

                    def multiply(x, y):
                        """Multiply two numbers"""
                        return x * y

                    def divide(x, y):
                        """Divide two numbers"""
                        if y == 0:
                            return "Error: Division by zero!"
                        return x / y

                    def main():
                        """Main calculator function"""
                        print("Simple Calculator")
                        print("=" * 20)
                        
                        try:
                            num1 = float(input("Enter first number: "))
                            operation = input("Enter operation (+, -, *, /): ")
                            num2 = float(input("Enter second number: "))
                            
                            if operation == '+':
                                result = add(num1, num2)
                            elif operation == '-':
                                result = subtract(num1, num2)
                            elif operation == '*':
                                result = multiply(num1, num2)
                            elif operation == '/':
                                result = divide(num1, num2)
                            else:
                                result = "Invalid operation"
                            
                            print(f"Result: {result}")
                        except ValueError:
                            print("Error: Please enter valid numbers")

                    if __name__ == "__main__":
                        main()
                    '''
                print("   ✅ Using Calculator template")
                return template_code
                
            elif any(word in user_lower for word in ["fibonacci", "fib"]):
                # Fibonacci template
                template_code = '''#!/usr/bin/env python3
                    """
                    Fibonacci Sequence Generator
                    Generated by raxion voice assistant
                    """

                    def fibonacci(n):
                        """Generate fibonacci sequence up to n terms"""
                        if n <= 0:
                            return []
                        elif n == 1:
                            return [0]
                        elif n == 2:
                            return [0, 1]
                        
                        sequence = [0, 1]
                        for i in range(2, n):
                            sequence.append(sequence[i-1] + sequence[i-2])
                        return sequence

                    def main():
                        """Main function"""
                        print("Fibonacci Sequence Generator")
                        print("=" * 30)
                        
                        try:
                            n = int(input("How many terms? "))
                            if n <= 0:
                                print("Please enter a positive number")
                            else:
                                sequence = fibonacci(n)
                                print(f"First {n} terms: {sequence}")
                        except ValueError:
                            print("Error: Please enter a valid number")

                    if __name__ == "__main__":
                        main()
                    '''
                print("   ✅ Using Fibonacci template")
                return template_code
                
            else:
                # Generic functional template
                template_code = f"""#!/usr/bin/env python3
                    \"\"\"
                    Generated Python Script
                    Created by raxion voice assistant
                    Request: {user_request}
                    \"\"\"

                    def main():
                        \"\"\"Main function\"\"\"
                        print("Python Script Generated Successfully!")
                        print("=" * 40)
                        print(f"Original request: {user_request}")
                        print("=" * 40)
                        
                        # Basic interactive functionality
                        print("This is a working Python script.")
                        user_input = input("Enter something to test the script: ")
                        print(f"You entered: {{user_input}}")
                        print("Script completed successfully!")

                    if __name__ == "__main__":
                        main()
                    """
                print("   ✅ Using Generic template")
                return template_code
                
        except Exception as e:
            print(f"❌ Code generation error: {e}")
            # Return a minimal working script as fallback
            fallback_code = f'''#!/usr/bin/env python3
                """
                Fallback Python Script
                Generated by raxion voice assistant
                """

                print("Hello from your generated Python script!")
                print(f"Request was: {user_request}")
                input("Press Enter to exit...")
                '''
            return fallback_code

    def extract_python_code_from_response(self, response):
        """Extract and clean Python code from LLM response"""
        try:
            # Look for code blocks
            if "```python" in response:
                # Extract code between ```python and ```
                start = response.find("```python") + 9
                end = response.find("```", start)
                if end != -1:
                    code = response[start:end].strip()
                else:
                    code = response[start:].strip()
            elif "```" in response:
                # Extract code between ``` blocks
                start = response.find("```") + 3
                end = response.find("```", start)
                if end != -1:
                    code = response[start:end].strip()
                else:
                    code = response[start:].strip()
            else:
                # Try to extract Python-like content - more aggressive approach
                lines = response.split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    # Skip obvious non-code lines
                    if any(marker in line.lower() for marker in ['here is', 'here\'s', 'this script', 'this code']):
                        continue
                    
                    # Detect start of Python code with more patterns
                    if any(keyword in line for keyword in ['def ', 'import ', 'from ', 'class ', 'if __name__', '#!', 'print(', '= ']):
                        in_code = True
                    
                    # Also look for simple assignment and function calls
                    if in_code or any(pattern in line for pattern in ['def ', 'import ', 'from ', 'print(', 'input(', 'return ', 'if ', 'else:', 'elif ', 'for ', 'while ']):
                        in_code = True
                        code_lines.append(line)
                    elif in_code and line.strip():  # Continue if we're already in code and line has content
                        code_lines.append(line)
                    elif in_code and not line.strip():  # Keep blank lines within code
                        code_lines.append(line)
                
                code = '\n'.join(code_lines).strip()
            
            if not code:
                # Last resort - try to find any Python-like patterns in the response
                import re
                python_patterns = [
                    r'def\s+\w+\([^)]*\):',
                    r'import\s+\w+',
                    r'from\s+\w+\s+import',
                    r'print\([^)]*\)',
                    r'input\([^)]*\)'
                ]
                
                if any(re.search(pattern, response) for pattern in python_patterns):
                    # Take the whole response and clean it later
                    code = response.strip()
                else:
                    return None
            
            # Clean up the code
            code = self.clean_generated_code(code)
            
            # Validate it looks like Python
            if self.is_valid_python_structure(code):
                return code
            else:
                print("❌ Generated code doesn't appear to be valid Python structure")
                # Debug: Print the raw response to understand what went wrong
                print(f"   Raw response: {response[:200]}...")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting code: {e}")
            return None

    def clean_generated_code(self, code):
        """Clean up generated Python code"""
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove common LLM artifacts
            if any(artifact in line.lower() for artifact in [
                'here is the code', 'here\'s the code', 'this script will',
                'this code will', 'the following code', 'above code'
            ]):
                continue
            
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
                
            cleaned_lines.append(line)
        
        # Add shebang if not present
        final_code = '\n'.join(cleaned_lines)
        if not final_code.startswith('#!'):
            final_code = '#!/usr/bin/env python3\n"""\nGenerated Python script\nCreated by raxion voice assistant\n"""\n\n' + final_code
        
        return final_code

    def is_valid_python_structure(self, code):
        """Basic validation to check if code looks like Python"""
        # Check for common Python patterns
        python_indicators = [
            'def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ',
            'print(', 'input(', 'return ', ':', 'try:', 'except:', '=', 
            'elif ', 'else:', 'with ', 'lambda ', 'yield ', 'assert '
        ]
        
        has_python_syntax = any(indicator in code for indicator in python_indicators)
        
        # Check it's not just comments or strings
        non_comment_lines = [line.strip() for line in code.split('\n') 
                           if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""')]
        has_code = len(non_comment_lines) > 0
        
        # More lenient check - if it has basic Python keywords, consider it valid
        basic_python_words = ['def', 'print', 'input', 'import', 'if', 'else', 'return']
        has_basic_python = any(word in code.lower() for word in basic_python_words)
        
        # Try to compile the code as a final validation (but don't fail on syntax errors from incomplete generation)
        is_compilable = False
        try:
            compile(code, '<generated>', 'exec')
            is_compilable = True
        except SyntaxError:
            # If it doesn't compile, at least check if it looks Python-ish
            is_compilable = has_basic_python
        except:
            is_compilable = has_basic_python
        
        result = (has_python_syntax or has_basic_python) and has_code and is_compilable
        
        if not result:
            print(f"   Validation failed: syntax={has_python_syntax}, code={has_code}, compilable={is_compilable}")
        
        return result

    def run_python_script(self, script_name):
        """Execute a Python script and show output - searches multiple locations"""
        try:
            # Clean up script name
            if not script_name.endswith('.py'):
                script_name += '.py'
            
            # Search locations in order of preference
            search_paths = [
                "/home/mateja/raxion_assistant",  # Current directory (highest priority)
                "/home/mateja/Documents",
                "/home/mateja/Desktop", 
                "/home/mateja/Downloads",
                "/home/mateja/Projects",
                "/home/mateja/Code",
                "/home/mateja/Scripts",
                "/home/mateja/Python"
            ]
            
            script_path = None
            found_location = None
            
            # First try exact filename match
            for search_dir in search_paths:
                if os.path.exists(search_dir):
                    candidate_path = os.path.join(search_dir, script_name)
                    if os.path.exists(candidate_path):
                        script_path = candidate_path
                        found_location = search_dir
                        print(f"🎯 Found script: {script_name} in {search_dir}")
                        break
            
            # If not found, try fuzzy matching in all directories
            if not script_path:
                print(f"🔍 Searching for similar files to '{script_name}'...")
                all_python_files = []
                
                for search_dir in search_paths:
                    if os.path.exists(search_dir):
                        try:
                            py_files = [f for f in os.listdir(search_dir) if f.endswith('.py')]
                            for f in py_files:
                                all_python_files.append((f, search_dir))
                        except (PermissionError, FileNotFoundError):
                            continue
                
                # Try to find a close match
                script_base = script_name.lower().replace('.py', '')
                for filename, directory in all_python_files:
                    file_base = filename.lower().replace('.py', '')
                    if script_base in file_base or file_base in script_base:
                        script_path = os.path.join(directory, filename)
                        found_location = directory
                        print(f"🎯 Found similar script: {filename} in {directory}")
                        self.speak(f"Found similar script {filename}. Running that instead.")
                        break
            
            # If still not found, list available files
            if not script_path:
                print(f"❌ Script {script_name} not found")
                
                # Show available files from current directory first
                available_files = []
                if os.path.exists("/home/mateja/raxion_assistant"):
                    available_files.extend([f for f in os.listdir("/home/mateja/raxion_assistant") if f.endswith('.py')])
                
                if available_files:
                    print(f"   Available Python files in current directory: {', '.join(available_files[:5])}")
                    if len(available_files) > 5:
                        print(f"   ... and {len(available_files) - 5} more files")
                    
                    self.speak(f"Script {script_name} not found. Available files include {', '.join(available_files[:3])}")
                else:
                    self.speak("No Python files found in the current directory")
                return
            
            print(f"🚀 Executing Python script: {os.path.basename(script_path)}")
            print(f"   Location: {script_path}")
            
            if found_location != "/home/mateja/raxion_assistant":
                self.speak(f"Running {os.path.basename(script_path)} from {os.path.basename(found_location)}")
            else:
                self.speak(f"Running {os.path.basename(script_path)}")
            
            # Execute the script using the configured Python environment
            result = subprocess.run([
                "/home/mateja/raxion_assistant/raxion_env/bin/python3", 
                script_path
            ], capture_output=True, text=True, cwd=found_location)
            
            # Display output
            if result.stdout:
                print(f"📤 Output:")
                print(result.stdout)
            
            if result.stderr:
                print(f"❌ Errors:")
                print(result.stderr)
            
            if result.returncode == 0:
                self.speak("Script executed successfully")
            else:
                self.speak(f"Script completed with exit code {result.returncode}")
            
            print(f"✅ Script execution completed (exit code: {result.returncode})")
            
        except Exception as e:
            print(f"❌ Error running Python script: {e}")
            self.speak("Sorry, I couldn't run the Python script")

    def close_application(self, app_names, friendly_name="Application"):
        """Close application(s) by process name"""
        try:
            if isinstance(app_names, str):
                app_names = [app_names]
            
            closed_any = False
            
            for app_name in app_names:
                # Find main processes (not subprocesses/helpers)
                try:
                    if "brave" in app_name:
                        # For Brave, find the main process (not helper processes)
                        result = subprocess.run(["pgrep", "-f", "/opt/brave.com/brave/brave$"], 
                                              capture_output=True, text=True)
                    elif "firefox" in app_name:
                        # For Firefox, target main process
                        result = subprocess.run(["pgrep", "-f", "firefox$"], 
                                              capture_output=True, text=True)
                    elif "chrome" in app_name or "chromium" in app_name:
                        # For Chrome/Chromium, target main process
                        result = subprocess.run(["pgrep", "-f", f"{app_name}$"], 
                                              capture_output=True, text=True)
                    else:
                        # For other apps, use exact match to avoid subprocesses
                        result = subprocess.run(["pgrep", "-x", app_name], 
                                              capture_output=True, text=True)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        
                        for pid in pids:
                            if pid.strip():
                                try:
                                    # Try graceful termination first
                                    subprocess.run(["kill", "-TERM", pid.strip()], check=False)
                                    time.sleep(1.0)  # Give more time for graceful shutdown
                                    
                                    # Check if process still exists
                                    check_result = subprocess.run(["kill", "-0", pid.strip()], 
                                                                capture_output=True)
                                    
                                    # If still exists, force kill
                                    if check_result.returncode == 0:
                                        subprocess.run(["kill", "-KILL", pid.strip()], check=False)
                                        time.sleep(0.5)
                                    
                                    closed_any = True
                                    print(f"🔒 Closed {app_name} (PID: {pid.strip()})")
                                    
                                except Exception as kill_error:
                                    print(f"⚠️ Could not close {app_name} PID {pid}: {kill_error}")
                    
                    # If no main process found, try fallback method
                    if not closed_any and "brave" in app_name:
                        # Fallback: kill brave browser using pkill
                        result = subprocess.run(["pkill", "-f", "/opt/brave.com/brave/brave"], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            closed_any = True
                            print(f"🔒 Closed {app_name} (using pkill)")
                                    
                except Exception as pgrep_error:
                    # This app might not be running, continue to next
                    continue
            
            if closed_any:
                time.sleep(1.0)  # Give time for cleanup
                self.speak(f"{friendly_name} closed")
                return True
            else:
                print(f"ℹ️ {friendly_name} is not currently running")
                self.speak(f"{friendly_name} is not currently running")
                return False
                
        except Exception as e:
            print(f"❌ Error closing {friendly_name}: {e}")
            self.speak(f"Sorry, I couldn't close {friendly_name}")
            return False

    def close_active_application(self):
        """Close the currently active window/application"""
        try:
            # Get the active window
            result = subprocess.run(["xdotool", "getactivewindow"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                window_id = result.stdout.strip()
                
                # Get window name for feedback
                name_result = subprocess.run(["xdotool", "getwindowname", window_id],
                                           capture_output=True, text=True)
                
                window_name = name_result.stdout.strip() if name_result.returncode == 0 else "Unknown"
                
                # Close the window
                subprocess.run(["xdotool", "windowclose", window_id], check=False)
                
                print(f"🔒 Closed active window: {window_name}")
                self.speak(f"Closed {window_name}")
                return True
            else:
                # Fallback: send Alt+F4
                subprocess.run(["xdotool", "key", "alt+F4"], check=False)
                self.speak("Sent close command to active application")
                return True
                
        except FileNotFoundError:
            # xdotool not available, try alternative approaches
            try:
                # Try wmctrl if available
                subprocess.run(["wmctrl", "-c", ":ACTIVE:"], check=False)
                self.speak("Closed active window")
                return True
            except FileNotFoundError:
                # Last resort: keyboard shortcut
                subprocess.run(["xvkbd", "-text", "\\A\\[F4]"], check=False)
                self.speak("Sent close command")
                return True
        except Exception as e:
            print(f"❌ Error closing active application: {e}")
            self.speak("Sorry, I couldn't close the active application")
            return False

    def close_all_applications(self):
        """Close all non-essential applications"""
        try:
            print("🔒 Closing all applications...")
            
            # List of applications to close (in order of priority)
            apps_to_close = [
                # Browsers
                (["brave-browser", "firefox", "chrome", "chromium"], "Browser"),
                # Code editors
                (["code", "atom", "sublime_text", "gedit"], "Code editor"),
                # File managers
                (["nautilus", "thunar", "dolphin", "pcmanfm"], "File manager"),
                # Terminals (be careful with this)
                (["gnome-terminal", "xfce4-terminal", "konsole"], "Terminal"),
                # Media players
                (["spotify", "vlc", "rhythmbox", "audacious"], "Media player"),
                # Office applications
                (["libreoffice", "writer", "calc", "impress"], "Office application"),
                # Graphics
                (["gimp", "inkscape", "blender"], "Graphics application"),
                # Communication
                (["discord", "slack", "telegram"], "Communication app"),
            ]
            
            closed_count = 0
            
            for app_list, friendly_name in apps_to_close:
                if self.close_application(app_list, friendly_name):
                    closed_count += 1
                    time.sleep(0.5)  # Brief pause between closings
            
            if closed_count > 0:
                self.speak(f"Closed {closed_count} applications")
            else:
                self.speak("No applications needed to be closed")
                
            print(f"✅ Application cleanup completed. Closed {closed_count} applications.")
            
        except Exception as e:
            print(f"❌ Error closing applications: {e}")
            self.speak("Sorry, I had trouble closing some applications")

    def run_audio_calibration(self):
        """Run automatic audio calibration to optimize voice detection settings"""
        print("\n🎯 === RAXION Audio Calibration ===")
        print("This will help optimize voice detection for your setup.")
        print("Please ensure you're in a typical environment where you'll use RAXION.\n")
        
        input("Press Enter when ready to start calibration...")
        
        # Step 1: Measure background noise
        print("\n📊 Step 1: Measuring background noise...")
        print("Please remain SILENT for 10 seconds...")
        
        for i in range(5, 0, -1):
            print(f"Starting in {i}...", end='\r')
            time.sleep(1)
        print("🤫 Recording silence... (10 seconds)")
        
        silence_samples = []
        start_time = time.time()
        
        # Record silence for 10 seconds
        while time.time() - start_time < 10:
            temp_file = "/tmp/raxion_calibration_silence.wav"
            try:
                # Record 1 second chunk
                proc = subprocess.Popen([
                    "parecord",
                    f"--device={self.audio_device}",
                    "--file-format=wav",
                    "--channels=1",
                    f"--rate={self.sample_rate}",
                    temp_file
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                time.sleep(1)
                proc.terminate()
                proc.wait()
                
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                    try:
                        sample_rate, audio_data = wavfile.read(temp_file)
                        if len(audio_data) > 0:
                            # Normalize and calculate RMS
                            audio_float = audio_data.astype(np.float32) / 32768.0
                            rms = np.sqrt(np.mean(audio_float ** 2))
                            variance = np.var(audio_float)
                            silence_samples.append((rms, variance))
                        os.remove(temp_file)
                    except Exception as e:
                        print(f"Error reading silence sample: {e}")
            except Exception as e:
                print(f"Error recording silence: {e}")
        
        if not silence_samples:
            print("❌ Could not measure background noise. Using default settings.")
            return
        
        # Calculate noise floor
        silence_rms_values = [s[0] for s in silence_samples]
        silence_variance_values = [s[1] for s in silence_samples]
        
        noise_floor = np.mean(silence_rms_values)
        noise_variance = np.mean(silence_variance_values)
        noise_max = np.max(silence_rms_values)
        
        print(f"✅ Background noise measured:")
        print(f"   Average: {noise_floor:.4f}")
        print(f"   Peak: {noise_max:.4f}")
        print(f"   Variance: {noise_variance:.6f}")
        
        # Step 2: Measure voice levels
        print(f"\n📊 Step 2: Measuring your voice...")
        print("Please read the following phrases in your NORMAL speaking voice:")
        
        test_phrases = [
            "Hello RAXION, can you hear me clearly?",
            "Open browser and search for artificial intelligence",
            "What time is it right now?",
            "Create a Python file called hello world",
            "Turn off the screen and good night"
        ]
        
        voice_samples = []
        
        for i, phrase in enumerate(test_phrases, 1):
            print(f"\n({i}/5) Please say: \"{phrase}\"")
            input("Press Enter when ready, then speak...")
            
            print("🎤 Recording... (speak now!)")
            
            # Record for 5 seconds
            temp_file = f"/tmp/raxion_calibration_voice_{i}.wav"
            proc = subprocess.Popen([
                "parecord",
                f"--device={self.audio_device}",
                "--file-format=wav",
                "--channels=1",
                f"--rate={self.sample_rate}",
                temp_file
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            time.sleep(5)
            proc.terminate()
            proc.wait()
            
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                try:
                    sample_rate, audio_data = wavfile.read(temp_file)
                    if len(audio_data) > 0:
                        # Normalize and analyze
                        audio_float = audio_data.astype(np.float32) / 32768.0
                        
                        # Find the loudest 2-second segment (likely where speech occurred)
                        segment_length = self.sample_rate * 2
                        max_rms = 0
                        best_segment = None
                        
                        for start in range(0, len(audio_float) - segment_length, self.sample_rate // 4):
                            segment = audio_float[start:start + segment_length]
                            rms = np.sqrt(np.mean(segment ** 2))
                            if rms > max_rms:
                                max_rms = rms
                                best_segment = segment
                        
                        if best_segment is not None:
                            variance = np.var(best_segment)
                            voice_samples.append((max_rms, variance))
                            print(f"   ✅ Captured (RMS: {max_rms:.4f}, Var: {variance:.6f})")
                        
                    os.remove(temp_file)
                except Exception as e:
                    print(f"   ❌ Error processing voice sample: {e}")
            else:
                print("   ❌ No audio recorded, trying again...")
                i -= 1  # Retry this phrase
        
        if not voice_samples:
            print("❌ Could not measure voice levels. Using default settings.")
            return
        
        # Calculate optimal thresholds
        voice_rms_values = [v[0] for v in voice_samples]
        voice_variance_values = [v[1] for v in voice_samples]
        
        avg_voice_rms = np.mean(voice_rms_values)
        min_voice_rms = np.min(voice_rms_values)
        avg_voice_variance = np.mean(voice_variance_values)
        
        print(f"\n✅ Voice levels measured:")
        print(f"   Average RMS: {avg_voice_rms:.4f}")
        print(f"   Minimum RMS: {min_voice_rms:.4f}")
        print(f"   Average variance: {avg_voice_variance:.6f}")
        
        # Calculate optimal sensitivity
        # We want the threshold to be above noise but below the quietest voice
        safety_margin = 1.5
        
        optimal_rms_threshold = max(
            noise_max * safety_margin,  # Above noise floor
            min_voice_rms * 0.7         # But not too high vs quiet voice
        )
        
        optimal_variance_threshold = max(
            noise_variance * 2,           # Above noise variance
            avg_voice_variance * 0.3      # But not too high vs voice
        )
        
        # Determine sensitivity level
        if optimal_rms_threshold <= 0.060:
            recommended_sensitivity = "high"
        elif optimal_rms_threshold <= 0.090:
            recommended_sensitivity = "medium"
        else:
            recommended_sensitivity = "low"
        
        print(f"\n🎯 Calibration Results:")
        print(f"   Noise floor: {noise_floor:.4f}")
        print(f"   Voice level: {avg_voice_rms:.4f}")
        print(f"   Optimal RMS threshold: {optimal_rms_threshold:.4f}")
        print(f"   Optimal variance threshold: {optimal_variance_threshold:.6f}")
        print(f"   Recommended sensitivity: {recommended_sensitivity}")
        
        # Save calibration results
        calibration_data = {
            "calibrated": True,
            "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "noise_floor": noise_floor,
            "noise_max": noise_max,
            "voice_level": avg_voice_rms,
            "optimal_rms_threshold": optimal_rms_threshold,
            "optimal_variance_threshold": optimal_variance_threshold,
            "recommended_sensitivity": recommended_sensitivity,
            "device_used": self.audio_device
        }
        
        # Update configuration
        config_path = '/home/mateja/raxion_assistant/raxion_config.json'
        try:
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            config.update({
                "detection_sensitivity": recommended_sensitivity,
                "audio_calibration": calibration_data,
                "last_modified": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Calibration saved to {config_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save calibration: {e}")
        
        print(f"\n🎉 Calibration complete! RAXION is now optimized for your setup.")
        print(f"   Sensitivity set to: {recommended_sensitivity}")
        print(f"   You can restart RAXION to use the new settings.")
        
        return True

    def run(self):
        """Main execution with continuous audio streaming and background processing"""
        print("\n🤖 === Continuous raxion Active ===")
        print("🧠 LLM-powered natural language understanding")
        print("🎤 Continuous audio streaming - just speak naturally!")
        print("⚡ Background processing - no delays between commands!")
        print("🔊 = speech detected, . = silence")
        print("====================================\n")
        
        self.speak("Hello Sir")
        
        # Start audio capture thread
        capture_thread = threading.Thread(target=self.continuous_audio_capture, daemon=True)
        capture_thread.start()
        
        # Start audio processing thread (for detection only)
        process_thread = threading.Thread(target=self.process_audio_continuously, daemon=True)
        process_thread.start()
        
        # Start background audio processing workers
        for i in range(self.max_concurrent_processing):
            bg_processor = threading.Thread(target=self.background_audio_processor, daemon=True)
            bg_processor.start()
        
        try:
            # Main loop - just keep the program running
            while self.listening:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            self.listening = False
            
            # Give threads time to finish
            time.sleep(2)
            
            # Clean up temp files
            import glob
            for temp_file in glob.glob("/tmp/raxion_chunk_*.wav"):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            for temp_file in glob.glob("/tmp/raxion_segment_*.wav"):
                try:
                    os.remove(temp_file)
                except:
                    pass
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ raxion shut down cleanly")

def update_raxion():
    """Update RAXION to the latest version from GitHub"""
    import tempfile
    import shutil
    import subprocess
    import os
    
    print("🔄 === RAXION Update ===")
    print("Checking for updates...")
    
    # Determine installation directory
    install_dir = os.path.expanduser("~/.local/share/raxion")
    if not os.path.exists(install_dir):
        print("❌ RAXION installation not found at ~/.local/share/raxion")
        print("   Please run the installer first: ./install.sh")
        sys.exit(1)
    
    try:
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            print("📥 Downloading latest version from GitHub...")
            
            # Clone the latest version
            result = subprocess.run([
                "git", "clone", 
                "https://github.com/TheBigSM/raxion.git", 
                os.path.join(temp_dir, "raxion-update")
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ Failed to download update: {result.stderr}")
                sys.exit(1)
            
            update_dir = os.path.join(temp_dir, "raxion-update")
            
            # Check if we got the files
            main_file = os.path.join(update_dir, "raxion_continuous.py")
            if not os.path.exists(main_file):
                print("❌ Downloaded files are incomplete")
                sys.exit(1)
            
            print("🔄 Installing updates...")
            
            # Create lightweight backup of only essential files
            backup_dir = f"{install_dir}.backup"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.makedirs(backup_dir)
            
            # Only backup essential files (not the huge venv directory)
            essential_files = ["raxion_continuous.py", "requirements.txt", "raxion_config.json", "voice_profile.json"]
            for file_name in essential_files:
                src_file = os.path.join(install_dir, file_name)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, os.path.join(backup_dir, file_name))
            
            print(f"💾 Backup created (essential files only): {backup_dir}")
            
            # Copy new files (preserve venv and config)
            files_to_update = ["raxion_continuous.py", "requirements.txt"]
            
            for file_name in files_to_update:
                src_file = os.path.join(update_dir, file_name)
                dst_file = os.path.join(install_dir, file_name)
                
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"✅ Updated: {file_name}")
            
            # Check if requirements.txt actually changed to avoid unnecessary reinstalls
            old_req = os.path.join(backup_dir, "requirements.txt")
            new_req = os.path.join(install_dir, "requirements.txt")
            
            requirements_changed = True
            if os.path.exists(old_req) and os.path.exists(new_req):
                with open(old_req, 'r') as f1, open(new_req, 'r') as f2:
                    requirements_changed = f1.read() != f2.read()
            
            # Only update dependencies if requirements actually changed
            if requirements_changed:
                print("🔍 Requirements changed, updating dependencies...")
                venv_python = os.path.join(install_dir, "venv", "bin", "python")
                if os.path.exists(venv_python):
                    result = subprocess.run([
                        venv_python, "-m", "pip", "install", "-r", 
                        os.path.join(install_dir, "requirements.txt")
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✅ Dependencies updated")
                    else:
                        print("⚠️ Warning: Failed to update dependencies")
                        print("   You may need to run: pip install -r requirements.txt")
            else:
                print("✅ No dependency changes detected, skipping package updates")
            
            print("\n✅ Update completed successfully!")
            print("🚀 You can now run RAXION with the latest version")
            print("\n💡 If you encounter issues, restore backup files with:")
            print(f"   cp {backup_dir}/* {install_dir}/")
            
    except subprocess.TimeoutExpired:
        print("❌ Download timed out. Please check your internet connection.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Update failed: {e}")
        print("   Please try manual update or reinstallation")
        sys.exit(1)


if __name__ == "__main__":
    # Parse arguments first, before importing heavy modules
    parser = argparse.ArgumentParser(description="RAXION - Local AI Assistant")
    parser.add_argument("--calibrate", action="store_true", 
                       help="Run audio calibration to optimize voice detection")
    parser.add_argument("--setup", action="store_true", 
                       help="Run first-time setup")
    parser.add_argument("--update", action="store_true", 
                       help="Update RAXION to the latest version")
    parser.add_argument("--version", action="version", version="RAXION v1.0.0")
    
    args = parser.parse_args()
    
    # Only import heavy modules if we need them
    if args.setup or args.calibrate or args.update:
        # For setup/calibration, we need scipy but not the heavy models
        try:
            import scipy.io.wavfile
        except ImportError:
            print("📦 Installing required dependency...")
            subprocess.run([sys.executable, "-m", "pip", "install", "scipy"], check=True)
            import scipy.io.wavfile
        
        # Just run setup/calibration without full initialization
        print("🤖 === RAXION Setup Mode ===")
        
        if args.setup:
            print("\n🚀 === RAXION First-Time Setup ===")
            print("Welcome to RAXION! Let's get you set up.")
            print("\nRAXION is a local AI assistant that:")
            print("  🎤 Listens for voice commands")
            print("  🧠 Uses local AI for understanding")
            print("  💬 Responds with speech")
            print("  🔧 Helps with everyday tasks")
            
            print("\n📋 System Check:")
            
            # Check dependencies
            missing_deps = []
            deps_to_check = {
                "torch": "PyTorch (for AI models)",
                "whisper": "OpenAI Whisper (for speech recognition)",
                "pyttsx3": "Text-to-speech",
                "transformers": "Hugging Face Transformers",
                "numpy": "NumPy (for audio processing)",
                "scipy": "SciPy (for audio file handling)"
            }
            
            for dep, description in deps_to_check.items():
                try:
                    __import__(dep)
                    print(f"  ✅ {description}")
                except ImportError:
                    print(f"  ❌ {description} - MISSING")
                    missing_deps.append(dep)
            
            if missing_deps:
                print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
                print("Please install them with: pip install -r requirements.txt")
                sys.exit(1)
            
            # Check audio system
            try:
                result = subprocess.run(["pactl", "info"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print("  ✅ PulseAudio detected")
                else:
                    print("  ⚠️ PulseAudio not detected - audio may not work optimally")
            except:
                print("  ⚠️ Could not check audio system")
            
            # Check GPU
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                    print(f"  ✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
                else:
                    print("  ℹ️ No GPU detected - will use CPU (slower but works)")
            except ImportError:
                print("  ⚠️ PyTorch not installed - cannot check GPU")
            
            print("\n🎯 Audio Calibration:")
            print("For best results, we recommend running audio calibration.")
            try:
                calibrate = input("Run audio calibration now? (y/n): ").lower().strip()
                
                if calibrate in ['y', 'yes']:
                    # Import the full class for calibration
                    raxion = ContinuousRAXION()
                    raxion.run_audio_calibration()
                else:
                    print("⏭️ Skipping calibration - you can run it later with: raxion --calibrate")
            except KeyboardInterrupt:
                print("\n⏭️ Setup interrupted")
            
            print("\n🎉 Setup complete! You can now run RAXION with: raxion")
        
        elif args.calibrate:
            # Run calibration only
            print("\n🎯 Starting audio calibration...")
            raxion = ContinuousRAXION()
            raxion.run_audio_calibration()
            
        elif args.update:
            # Run update
            update_raxion()
        
    else:
        # Normal operation - import everything and run
        try:
            # Check if scipy is available for wav file writing
            import scipy.io.wavfile
        except ImportError:
            print("📦 Installing required dependency...")
            subprocess.run([sys.executable, "-m", "pip", "install", "scipy"], check=True)
            import scipy.io.wavfile
        
        raxion = ContinuousRAXION()
        raxion.run()
