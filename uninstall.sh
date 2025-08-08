#!/bin/bash

# RAXION Uninstaller
# Removes RAXION AI Assistant from the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════╗"
    echo "║            RAXION AI Assistant                ║"
    echo "║              Uninstaller                      ║"
    echo "╚═══════════════════════════════════════════════╝"
    echo -e "${NC}"
}

main() {
    print_banner
    
    INSTALL_DIR="$HOME/.local/share/raxion"
    BIN_DIR="$HOME/.local/bin"
    DESKTOP_DIR="$HOME/.local/share/applications"
    
    echo "This will remove RAXION AI Assistant from your system."
    echo
    echo "Files to be removed:"
    echo "  - $INSTALL_DIR"
    echo "  - $BIN_DIR/raxion"
    echo "  - $DESKTOP_DIR/raxion.desktop"
    echo
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Uninstallation cancelled"
        exit 0
    fi
    
    # Remove installation directory
    if [ -d "$INSTALL_DIR" ]; then
        log "Removing installation directory..."
        rm -rf "$INSTALL_DIR"
        log "✅ Removed $INSTALL_DIR"
    else
        warn "Installation directory not found: $INSTALL_DIR"
    fi
    
    # Remove launcher
    if [ -f "$BIN_DIR/raxion" ]; then
        log "Removing launcher..."
        rm -f "$BIN_DIR/raxion"
        log "✅ Removed $BIN_DIR/raxion"
    else
        warn "Launcher not found: $BIN_DIR/raxion"
    fi
    
    # Remove desktop entry
    if [ -f "$DESKTOP_DIR/raxion.desktop" ]; then
        log "Removing desktop entry..."
        rm -f "$DESKTOP_DIR/raxion.desktop"
        log "✅ Removed $DESKTOP_DIR/raxion.desktop"
    else
        warn "Desktop entry not found: $DESKTOP_DIR/raxion.desktop"
    fi
    
    # Note about system packages
    echo
    warn "System packages installed by RAXION are NOT removed."
    echo "If you want to remove them, you can run:"
    echo "  Ubuntu/Debian: sudo apt autoremove pulseaudio-utils sox espeak ffmpeg"
    echo "  Fedora:        sudo dnf remove pulseaudio-utils sox espeak ffmpeg"
    echo "  Arch:          sudo pacman -R pulseaudio sox espeak ffmpeg"
    
    echo
    log "🗑️ RAXION has been successfully uninstalled!"
    echo "Thank you for trying RAXION AI Assistant."
}

main "$@"
