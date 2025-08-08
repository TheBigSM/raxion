# Contributing to RAXION AI Assistant

Thank you for your interest in contributing to RAXION! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Provide detailed information about your system and the issue
- Include logs and error messages when possible
- Check if the issue already exists before creating a new one

### Suggesting Features
- Open a GitHub issue with the "feature request" label
- Describe the feature and its benefits
- Discuss implementation approaches if you have ideas

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Ensure code passes all tests and linting
6. Submit a pull request

## 🏗️ Development Setup

### Prerequisites
- Python 3.8+
- Linux system (Ubuntu 20.04+ recommended)
- Git
- PulseAudio development libraries

### Setup Development Environment
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/raxion.git
cd raxion

# Create virtual environment
python -m venv dev-env
source dev-env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=raxion

# Run specific test file
pytest tests/test_audio.py

# Run with verbose output
pytest -v
```

### Code Style
We use `black` for code formatting and `flake8` for linting:

```bash
# Format code
black raxion_continuous.py

# Check linting
flake8 raxion_continuous.py

# Run all checks
make check  # If Makefile exists
```

## 📝 Code Guidelines

### Python Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write descriptive docstrings for functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 50 characters
- Add detailed description if needed

Example:
```
Add audio calibration feature

- Implement automatic noise floor detection
- Add voice level measurement
- Save calibration results to config
- Include interactive prompts for user guidance
```

### Code Organization
- Keep the main application logic in `raxion_continuous.py`
- Separate configuration handling
- Use appropriate error handling and logging
- Comment complex algorithms and audio processing logic

## 🧪 Testing

### Test Coverage
- Aim for good test coverage of core functionality
- Mock external dependencies (audio devices, models)
- Test both success and error cases
- Include integration tests for key workflows

### Test Categories
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Audio Tests**: Test audio processing (with mocked audio)
- **System Tests**: End-to-end functionality

### Running Specific Tests
```bash
# Audio processing tests
pytest tests/test_audio.py

# Voice recognition tests
pytest tests/test_voice.py

# Configuration tests
pytest tests/test_config.py
```

## 📚 Architecture Overview

### Core Components
1. **Audio Processing**: Continuous audio capture and voice activity detection
2. **Speech Recognition**: OpenAI Whisper integration
3. **Language Understanding**: Qwen2-0.5B model for intent recognition
4. **Command Execution**: System integration and automation
5. **Configuration**: Settings management and audio calibration

### Key Classes and Functions
- `ContinuousRAXION`: Main application class
- `run_audio_calibration()`: Audio setup and optimization
- `detect_speech_activity()`: Voice activity detection
- `analyze_with_llm()`: Intent recognition and response generation
- `execute_command()`: Command execution and system integration

## 🔧 Adding New Features

### New Voice Commands
1. Add command pattern to `fast_command_detection()`
2. Implement handler in `execute_command()`
3. Add tests for the new command
4. Update documentation

### Audio Improvements
1. Understand the current audio pipeline
2. Test changes with different microphones and environments
3. Consider cross-platform compatibility
4. Add appropriate error handling

### Model Updates
1. Ensure compatibility with existing configurations
2. Test performance impact
3. Update requirements if needed
4. Document any breaking changes

## 🐛 Debugging

### Audio Issues
- Use `--debug` flag for verbose audio output
- Check PulseAudio logs: `journalctl --user -u pulseaudio`
- Test microphone: `arecord -d 5 test.wav && aplay test.wav`
- Verify audio devices: `pactl list sources short`

### Model Issues
- Check GPU memory: `nvidia-smi`
- Monitor CPU usage during processing
- Test with CPU-only mode for debugging
- Check model download and cache

### Common Debug Commands
```bash
# Run with debug output
python raxion_continuous.py --debug

# Test audio calibration
python raxion_continuous.py --calibrate

# Check dependencies
python -c "import torch, whisper, transformers; print('OK')"

# Test PulseAudio
pactl info
```

## 📋 Pull Request Process

1. **Before Submitting**:
   - Ensure all tests pass
   - Run code formatting and linting
   - Update documentation if needed
   - Test on a clean system if possible

2. **PR Description**:
   - Clearly describe what the PR does
   - Reference any related issues
   - Include screenshots for UI changes
   - List any breaking changes

3. **Review Process**:
   - Maintainers will review your PR
   - Address feedback and requested changes
   - Keep discussions focused and respectful

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### High Priority
- Cross-platform audio support (Windows, macOS)
- Performance optimizations
- Better error handling and recovery
- Improved documentation and examples

### Medium Priority
- Additional voice commands
- Better wake word detection
- UI improvements for configuration
- Integration with more applications

### Low Priority
- Additional language models
- Advanced audio processing features
- Plugins and extensions system

## 📞 Getting Help

- **Questions**: Open a GitHub discussion
- **Chat**: Join our community Discord (if available)
- **Documentation**: Check the README and wiki
- **Issues**: Search existing issues first

## 🏆 Recognition

Contributors will be:
- Listed in the project's contributors section
- Mentioned in release notes for significant contributions
- Given appropriate attribution in code comments

Thank you for helping make RAXION better! 🚀
