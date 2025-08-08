---
name: Bug Report
about: Create a report to help us improve RAXION
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## 🐛 Bug Description
A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## ✅ Expected Behavior
A clear and concise description of what you expected to happen.

## ❌ Actual Behavior
A clear and concise description of what actually happened.

## 🖥️ System Information
- **OS:** [e.g. Ubuntu 22.04 LTS]
- **Python Version:** [e.g. 3.10.12]
- **GPU:** [e.g. GTX 1060 6GB]
- **RAXION Version:** [e.g. latest/commit hash]

## 📋 Audio System
- **PulseAudio Version:** [Run `pulseaudio --version`]
- **Audio Devices:** [Run `pactl list sources short`]
- **Microphone Test:** [Does `arecord -d 5 test.wav && aplay test.wav` work?]

## 📊 GPU Information (if relevant)
```bash
# Paste output of:
nvidia-smi
```

## 📝 Error Logs
```
Paste any error messages or logs here
```

## 🔧 Configuration
```json
// Paste your raxion_config.json (remove any sensitive info)
```

## 📱 Additional Context
Add any other context about the problem here, such as:
- When did this start happening?
- Does it happen consistently?
- Any recent system changes?
- Screenshots if applicable
