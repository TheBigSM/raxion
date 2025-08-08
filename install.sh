#!/bin/bash

# RAXION Linux Installer
# Automated installation script for RAXION AI Assistant

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for better UX
ROBOT="🤖"
CHECK="✅"
CROSS="❌"
WRENCH="🔧"
PACKAGE="📦"
ROCKET="🚀"

print_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════╗"
    echo "║            RAXION AI Assistant                ║"
    echo "║         Linux Installation Script             ║"
    echo "║                                               ║"
    echo "║  🎤 Voice-controlled local AI assistant       ║"
    echo "║  🧠 Powered by Qwen2-0.5B & OpenAI Whisper   ║"
    echo "║  🔧 Easy setup with automatic calibration     ║"
    echo "╚═══════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "Please do not run this script as root/sudo"
        echo "RAXION should be installed in user space for security and proper audio access."
        exit 1
    fi
}

# Detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    else
        error "Cannot detect Linux distribution"
        exit 1
    fi
    
    log "Detected: $PRETTY_NAME"
}

# Install system dependencies
install_system_deps() {
    log "${PACKAGE} Installing system dependencies..."
    
    case $DISTRO in
        ubuntu|debian|pop|elementary)
            log "Installing packages for Debian/Ubuntu-based system..."
            sudo apt update
            sudo apt install -y \
                python3 \
                python3-pip \
                python3-venv \
                pulseaudio \
                pulseaudio-utils \
                alsa-utils \
                sox \
                git \
                curl \
                wget \
                build-essential \
                python3-dev \
                libssl-dev \
                libffi-dev \
                libasound2-dev \
                portaudio19-dev \
                espeak \
                espeak-data \
                libespeak-dev \
                ffmpeg
            ;;
        
        fedora|centos|rhel|rocky|almalinux)
            log "Installing packages for Red Hat-based system..."
            sudo dnf install -y \
                python3 \
                python3-pip \
                python3-virtualenv \
                pulseaudio \
                pulseaudio-utils \
                alsa-utils \
                sox \
                git \
                curl \
                wget \
                gcc \
                gcc-c++ \
                python3-devel \
                openssl-devel \
                libffi-devel \
                alsa-lib-devel \
                portaudio-devel \
                espeak \
                espeak-devel \
                ffmpeg
            ;;
        
        arch|manjaro|endeavouros)
            log "Installing packages for Arch-based system..."
            sudo pacman -Sy --needed --noconfirm \
                python \
                python-pip \
                python-virtualenv \
                pulseaudio \
                pulseaudio-alsa \
                alsa-utils \
                sox \
                git \
                curl \
                wget \
                base-devel \
                openssl \
                libffi \
                alsa-lib \
                portaudio \
                espeak \
                ffmpeg
            ;;
        
        opensuse*|sles)
            log "Installing packages for openSUSE..."
            sudo zypper install -y \
                python3 \
                python3-pip \
                python3-virtualenv \
                pulseaudio \
                pulseaudio-utils \
                alsa-utils \
                sox \
                git \
                curl \
                wget \
                gcc \
                gcc-c++ \
                python3-devel \
                libopenssl-devel \
                libffi-devel \
                alsa-devel \
                portaudio-devel \
                espeak \
                ffmpeg
            ;;
        
        *)
            warn "Unsupported distribution: $DISTRO"
            warn "Please install the following packages manually:"
            echo "  - Python 3.8+ with pip and venv"
            echo "  - PulseAudio and ALSA utilities"
            echo "  - Development tools (gcc, python3-dev, etc.)"
            echo "  - Audio libraries (portaudio, espeak, ffmpeg)"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
    
    success "System dependencies installed"
}

# Check Python version
check_python() {
    log "${PYTHON} Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        error "Python 3 not found. Please install Python 3.8 or newer."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_VERSION_NUM=$(python3 -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)")
    
    if [ "$PYTHON_VERSION_NUM" -lt 38 ]; then
        error "Python $PYTHON_VERSION found, but Python 3.8+ required"
        exit 1
    fi
    
    success "Python $PYTHON_VERSION found"
}

# Check audio system
check_audio() {
    log "${AUDIO} Checking audio system..."
    
    if ! command -v pactl &> /dev/null; then
        warn "PulseAudio not found - audio may not work properly"
        warn "Please ensure your audio system is working"
    else
        if pactl info &> /dev/null; then
            success "PulseAudio is running"
        else
            warn "PulseAudio is installed but not running"
            warn "You may need to start it manually"
        fi
    fi
    
    # Check for audio devices
    if command -v pactl &> /dev/null; then
        AUDIO_SOURCES=$(pactl list sources short 2>/dev/null | wc -l)
        AUDIO_SINKS=$(pactl list sinks short 2>/dev/null | wc -l)
        
        if [ "$AUDIO_SOURCES" -gt 0 ] && [ "$AUDIO_SINKS" -gt 0 ]; then
            success "Audio input and output devices detected"
        else
            warn "Limited audio devices detected - check your audio setup"
        fi
    fi
}

# Create installation directory
create_install_dir() {
    INSTALL_DIR="$HOME/.local/share/raxion"
    BIN_DIR="$HOME/.local/bin"
    
    log "${WRENCH} Creating installation directory..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    
    success "Created directories: $INSTALL_DIR, $BIN_DIR"
}

# Install RAXION
install_raxion() {
    log "${PACKAGE} Installing RAXION..."
    
    cd "$INSTALL_DIR"
    
    # Copy source files
    cp "$SOURCE_DIR"/*.py ./
    cp "$SOURCE_DIR"/requirements.txt ./
    
    # Create virtual environment
    log "Creating Python virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment and install dependencies
    log "Installing Python dependencies..."
    source venv/bin/activate
    
    # Upgrade pip first
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected - installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log "Installing PyTorch for CPU..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    pip install -r requirements.txt
    
    # Download models on first install
    log "Pre-downloading AI models (this may take a few minutes)..."
    python -c "
import whisper
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Whisper model...')
whisper.load_model('base')
print('Downloading Qwen model...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B-Instruct', trust_remote_code=True)
print('Models downloaded successfully!')
"
    
    deactivate
    
    success "RAXION installed successfully"
}

# Create launcher script
create_launcher() {
    log "${ROCKET} Creating launcher script..."
    
    cat > "$BIN_DIR/raxion" << 'EOF'
#!/bin/bash

# RAXION Launcher Script
RAXION_DIR="$HOME/.local/share/raxion"

if [ ! -d "$RAXION_DIR" ]; then
    echo "❌ RAXION not found. Please reinstall."
    exit 1
fi

cd "$RAXION_DIR"
source venv/bin/activate
python raxion_continuous.py "$@"
EOF
    
    chmod +x "$BIN_DIR/raxion"
    
    success "Launcher created: $BIN_DIR/raxion"
}

# Create desktop entry
create_desktop_entry() {
    log "Creating desktop entry..."
    
    DESKTOP_DIR="$HOME/.local/share/applications"
    mkdir -p "$DESKTOP_DIR"
    
    cat > "$DESKTOP_DIR/raxion.desktop" << EOF
[Desktop Entry]
Name=RAXION AI Assistant
Comment=Local AI-powered voice assistant
Exec=$HOME/.local/bin/raxion
Icon=audio-input-microphone
Terminal=true
Type=Application
Categories=Office;Audio;Utility;
Keywords=ai;assistant;voice;speech;
StartupNotify=false
EOF
    
    success "Desktop entry created"
}

# Update shell profile
update_shell_profile() {
    log "Updating shell profile..."
    
    # Add ~/.local/bin to PATH if not already present
    SHELL_RC=""
    if [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -f "$HOME/.profile" ]; then
        SHELL_RC="$HOME/.profile"
    fi
    
    if [ -n "$SHELL_RC" ] && [ -f "$SHELL_RC" ]; then
        if ! grep -q '$HOME/.local/bin' "$SHELL_RC"; then
            echo '' >> "$SHELL_RC"
            echo '# Added by RAXION installer' >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
            success "Updated $SHELL_RC to include ~/.local/bin in PATH"
        fi
    fi
}

# Main installation process
main() {
    print_banner
    
    # Get the directory where this script is located
    SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    log "Starting RAXION installation..."
    log "Source directory: $SOURCE_DIR"
    
    check_root
    detect_distro
    check_python
    
    # Ask for confirmation
    echo
    echo -e "${CYAN}This will install RAXION AI Assistant on your system.${NC}"
    echo "Installation directory: $HOME/.local/share/raxion"
    echo "Executable: $HOME/.local/bin/raxion"
    echo
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Installation cancelled"
        exit 0
    fi
    
    install_system_deps
    check_audio
    create_install_dir
    install_raxion
    create_launcher
    create_desktop_entry
    update_shell_profile
    
    echo
    success "${ROBOT} RAXION installation completed!"
    echo
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Restart your terminal or run: source ~/.bashrc"
    echo "2. Run setup: raxion --setup"
    echo "3. Start RAXION: raxion"
    echo
    echo -e "${CYAN}Useful commands:${NC}"
    echo "  raxion --setup     - First-time setup"
    echo "  raxion --calibrate - Audio calibration"
    echo "  raxion --help      - Show help"
    echo "  raxion             - Start RAXION"
    echo
    echo -e "${YELLOW}Note:${NC} If you encounter audio issues, try running 'raxion --calibrate'"
    echo
}

# Run main function
main "$@"
