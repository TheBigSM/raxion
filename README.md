# 🤖 RAXION AI Assistant

**A powerful, privacy-focused local AI assistant for Linux**

RAXION is an open-source voice-controlled AI assistant that runs entirely on your local machine. No cloud services, no data sharing - just intelligent assistance that respects your privacy.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Platform](https://img.shields.io/badge/platform-Linux-orange.svg)

## ✨ Key Features

🎤 **Natural Voice Commands** - Just say "Hey RAXION" and speak naturally  
🧠 **Local AI Processing** - Powered by Qwen2-0.5B for fast, private responses  
🗣️ **Advanced Speech Recognition** - OpenAI Whisper for accurate transcription  
⚡ **Real-time Processing** - Continuous listening with background processing  
🔧 **Smart Automation** - Control your desktop, create files, search the web  
🎯 **Auto Calibration** - Optimizes audio settings for your environment  
🛡️ **Privacy First** - Everything runs locally, your data never leaves your machine  
🚀 **Easy Installation** - One-command setup with automatic dependency management  

## 🚀 Quick Start

### One-Command Installation

```bash
git clone https://github.com/TheBigSM/raxion.git
cd raxion
./install.sh
```

### First-Time Setup

```bash
raxion --setup
```

This will:
- ✅ Check system dependencies
- ✅ Verify audio system compatibility
- ✅ Detect GPU capabilities
- ✅ Run audio calibration (optional)
- ✅ Download AI models

### Start RAXION

```bash
raxion
```

That's it! RAXION will start listening for voice commands.

## 📋 System Requirements

### Minimum Requirements
- **OS:** Linux (Ubuntu 20.04+, Fedora 35+, Arch Linux, openSUSE Leap 15.4+)
- **CPU:** 64-bit processor (x86_64)
- **RAM:** 4 GB (8 GB recommended)
- **VRAM:** 2 GB (for GPU acceleration)
- **Storage:** 5 GB free space
- **Audio:** Working microphone and speakers/headphones
- **Python:** 3.8 or newer

### Recommended for Best Performance
- **GPU:** NVIDIA GPU with CUDA support (GTX 1060 6GB or better)
- **VRAM:** 6 GB (tested on GTX 1060 6GB with excellent performance)
- **RAM:** 8+ GB
- **CPU:** Multi-core processor (Intel i5+ or AMD Ryzen 5+)

### GPU Memory Requirements
- **Minimum VRAM:** 2 GB (basic functionality)
- **Recommended VRAM:** 6 GB (optimal performance)
- **Peak Usage:** 1.4 GB (during full conversations)
- **Model Downloads:**
  - Whisper Base: ~290 MB download, ~280 MB VRAM
  - Qwen2-0.5B: ~1.1 GB download, ~1.1 GB VRAM

### Performance Metrics (Tested on GTX 1060 6GB)
- **VRAM Efficiency:** 23.8% peak usage (1.4 GB / 6 GB)
- **Response Time:** 1.5 seconds average (speech-to-response)
- **Speech Processing:** 0.12 seconds (real-time)
- **Text Generation:** 1.4 seconds (conversational responses)
- **Available Headroom:** 76.2% GPU memory free for other tasks

### Real-World Performance Testing
Comprehensive testing on GTX 1060 6GB shows excellent efficiency:

| Test Scenario | VRAM Usage | Processing Time | Status |
|---------------|-----------|-----------------|---------|
| Speech Recognition (1-10s) | 280 MB | 0.12s | ✅ Excellent |
| Short LLM Response (20 tokens) | 1.12 GB | 0.6s | ⚡ Fast |
| Long LLM Response (200 tokens) | 1.12 GB | 5.7s | ✅ Good |
| **Full Conversation Cycle** | **1.41 GB** | **1.5s** | **🏆 Outstanding** |

**Result:** System runs at only 24% GPU capacity, leaving massive headroom for stability and future enhancements.

### Tested Linux Distributions
- ✅ **Ubuntu** 20.04 LTS, 22.04 LTS, 23.04+
- ✅ **Fedora** 35, 36, 37, 38, 39
- ✅ **Arch Linux** (latest)
- ✅ **Manjaro** 22.0+
- ✅ **Pop!_OS** 22.04+
- ✅ **openSUSE Leap** 15.4+
- ✅ **Debian** 11, 12
- ✅ **Elementary OS** 6.1+

## 🗣️ Voice Commands

### Wake Words
- **"Hey RAXION"** or **"RAXION"** or **"OK RAXION"**

### Example Commands

**🌐 Web & Applications**
- "Open browser" → Opens your default web browser
- "Search for artificial intelligence" → Searches Google
- "Open VS Code" → Launches Visual Studio Code
- "Open YouTube" → Opens YouTube in browser
- "Open GitHub" → Opens GitHub

**📁 File Operations**
- "Create a Python file" → Creates a new Python script
- "Make a calculator script" → Creates a calculator program
- "Create a folder called projects" → Makes a new directory

**🖥️ System Control**
- "What time is it?" → Tells current time
- "Turn off screen" → Turns off display
- "Volume up" → Increases system volume
- "Lock screen" → Locks the desktop
- "Open file manager" → Opens file explorer

**💬 Conversation**
- Ask questions about any topic
- Request explanations
- Have natural conversations
- Get help with tasks

### More Commands
- **Applications:** calculator, terminal, music player, file manager
- **System:** restart, shutdown, volume control, screen control
- **Development:** Python script creation and execution
- **Web search:** Google search with voice queries

## ⚙️ Configuration

RAXION automatically creates configuration files in:
- **Installation:** `~/.local/share/raxion/`
- **Configuration:** `~/.local/share/raxion/raxion_config.json`
- **Launcher:** `~/.local/bin/raxion`

### Audio Calibration

For optimal performance, run audio calibration after installation:

```bash
raxion --calibrate
```

This process:
1. **Measures background noise** (10-second silence recording)
2. **Analyzes your voice** (5 test phrases)
3. **Calculates optimal settings** automatically
4. **Saves configuration** for immediate use

### Manual Configuration

Edit `~/.local/share/raxion/raxion_config.json`:

```json
{
  "detection_sensitivity": "medium",  // "high", "medium", "low"
  "debug_audio": false,              // Enable audio debugging
  "wake_words": ["raxion", "hey raxion"],
  "tts_rate": 180,                   // Speech rate (words per minute)
  "tts_volume": 0.8,                 // Speech volume (0.0-1.0)
  "input_device": "auto",            // Audio input device
  "output_device": "auto"            // Audio output device
}
```

## 🔧 Command Reference

| Command | Description |
|---------|-------------|
| `raxion` | Start RAXION assistant |
| `raxion --setup` | First-time setup wizard |
| `raxion --calibrate` | Audio calibration only |
| `raxion --help` | Show help information |
| `raxion --version` | Show version |

## 🏗️ Architecture

RAXION consists of several key components:

### Core Components
1. **🎤 Audio Processing** - Continuous capture and voice activity detection
2. **🗣️ Speech Recognition** - OpenAI Whisper for transcription
3. **🧠 Language Understanding** - Qwen2-0.5B for intent recognition
4. **⚡ Command Execution** - System integration and automation
5. **⚙️ Configuration** - Settings management and calibration

### AI Models Used
- **Speech Recognition:** OpenAI Whisper (base model)
- **Language Understanding:** Qwen2-0.5B-Instruct
- **Text-to-Speech:** System TTS (pyttsx3)

### Performance
- **VRAM Usage:** 1.4 GB peak (23.8% of GTX 1060 6GB)
- **Model Memory:**
  - Whisper Base: 280 MB VRAM
  - Qwen2-0.5B: 1.1 GB VRAM
- **Response Time:** 1.5 seconds average (real-time conversations)
- **Speech Processing:** 0.12 seconds (near-instantaneous)
- **GPU Efficiency:** 76.2% memory headroom available
- **Tested Hardware:** GTX 1060 6GB (excellent performance)

## 🛠️ Development

### Project Structure
```
raxion/
├── raxion_continuous.py    # Main application
├── install.sh             # Linux installer
├── uninstall.sh           # Clean removal
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── CONTRIBUTING.md        # Developer guide
└── .gitignore             # Git ignore rules
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/TheBigSM/raxion.git
cd raxion

# Manual installation
python -m venv raxion_env
source raxion_env/bin/activate
pip install -r requirements.txt

# Run setup
python raxion_continuous.py --setup

# Start RAXION
python raxion_continuous.py
```

### Dependencies

**Core AI Libraries:**
- PyTorch (1.13.1+cu117) - Deep learning framework (GTX 1060 optimized)
- Transformers (4.39.0) - Hugging Face model library
- Tokenizers (0.15.2) - Fast tokenization
- Accelerate (0.20.3) - Model acceleration with fallback support
- OpenAI Whisper (latest) - Speech recognition
- pyttsx3 - Text-to-speech

**Audio Processing:**
- sounddevice - Audio I/O
- soundfile - Audio file handling
- scipy - Scientific computing
- numpy (1.26.4) - Numerical arrays (PyTorch compatible)

**System Integration:**
- PulseAudio (Linux audio system)
- subprocess - System commands
- threading - Concurrent processing
- psutil - System monitoring

### AI Models Used
- **Speech Recognition:** OpenAI Whisper Base (~290 MB download)
- **Language Model:** Qwen2-0.5B-Instruct (~1.1 GB download)
- **Text-to-Speech:** System TTS (pyttsx3)

### Model Performance & Memory
| Model | Download Size | VRAM Usage | Load Time | Performance |
|-------|---------------|-----------|-----------|-------------|
| Whisper Base | 290 MB | 280 MB | 1.0s | Real-time speech recognition |
| Qwen2-0.5B | 1.1 GB | 1.1 GB | 1.8s | Fast conversational AI |
| **Combined** | **1.4 GB** | **1.4 GB** | **2.8s** | **Full system ready** |

## 🔍 Troubleshooting

### Audio Issues

**No microphone detected:**
```bash
# Check PulseAudio
pulseaudio --check

# List audio devices
pactl list sources short
pactl list sinks short

# Test microphone
arecord -d 5 test.wav && aplay test.wav
```

**Poor voice recognition:**
```bash
# Run calibration
raxion --calibrate

# Check sensitivity settings
# Edit ~/.local/share/raxion/raxion_config.json
```

**No sound output:**
- Check system audio settings
- Verify default output device
- Test with: `speaker-test -t wav -c 2`

### Performance Issues

### Performance Issues

**Slow responses:**
```bash
# Check GPU availability and memory
nvidia-smi

# Monitor VRAM usage (should be ~1.4 GB peak)
watch -n 1 nvidia-smi

# Check system resources
htop

# CPU fallback is automatic if GPU unavailable
```

**High VRAM usage (>2 GB):**
- Check for other GPU processes: `nvidia-smi`
- Restart RAXION to clear GPU cache
- Ensure GTX 1060 compatible PyTorch version (1.13.1+cu117)

**Memory optimization:**
- Current system uses only 1.4 GB peak VRAM (excellent efficiency)
- GTX 1060 6GB provides 76.2% memory headroom
- No optimization needed with current configuration

**Memory issues:**
```bash
# Check available memory
free -h

# Clear model cache
rm -rf ~/.cache/huggingface/

# Restart RAXION
raxion
```

### Installation Issues

**Missing dependencies:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip pulseaudio-utils

# Fedora
sudo dnf install python3 python3-pip pulseaudio-utils

# Arch Linux
sudo pacman -S python python-pip pulseaudio
```

**Permission errors:**
```bash
# Fix permissions
chmod +x install.sh
chmod +x ~/.local/bin/raxion

# Add to PATH (if needed)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Common Fixes

```bash
# Restart audio system
pulseaudio -k && pulseaudio --start

# Reset configuration
rm ~/.local/share/raxion/raxion_config.json
raxion --setup

# Reinstall (clean)
./uninstall.sh
./install.sh
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing procedures
- Pull request process

### Quick Contribution Setup
```bash
git clone https://github.com/TheBigSM/raxion.git
cd raxion
python -m venv dev-env
source dev-env/bin/activate
pip install -r requirements.txt
# Make your changes
# Test thoroughly
# Submit pull request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

RAXION builds upon excellent open-source projects:

- **[OpenAI Whisper](https://github.com/openai/whisper)** - Robust speech recognition
- **[Qwen2](https://huggingface.co/Qwen)** - Efficient language model by Alibaba Cloud
- **[PyTorch](https://pytorch.org/)** - Deep learning framework
- **[Hugging Face](https://huggingface.co/)** - Model hosting and transformers library
- **[pyttsx3](https://pyttsx3.readthedocs.io/)** - Cross-platform text-to-speech

## 📞 Support & Community

- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/TheBigSM/raxion/issues)
-  **Documentation:** [README.md](https://github.com/TheBigSM/raxion/blob/main/README.md) & [CONTRIBUTING.md](https://github.com/TheBigSM/raxion/blob/main/CONTRIBUTING.md)
- 🌐 **Contact:** [thebigsm.github.io](https://thebigsm.github.io/)

## ⭐ Star History

If you find RAXION useful, please give it a star! It helps others discover the project.

---

<div align="center">

**🤖 RAXION: Your Private, Powerful AI Assistant**

*Made with ❤️ for the open source community*

[![GitHub stars](https://img.shields.io/github/stars/TheBigSM/raxion?style=social)](https://github.com/TheBigSM/raxion/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/TheBigSM/raxion?style=social)](https://github.com/TheBigSM/raxion/network/members)

</div>
