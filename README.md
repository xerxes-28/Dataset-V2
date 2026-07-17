AEGIS v2 Dataset
Training data for AEGIS v2, a QLoRA + DPO fine-tune of DeepSeek-R1-Distill-Qwen-32B specialized for embedded systems, drone firmware, and robotics code generation.
Unlike AEGIS v1's ~37K hand-curated Alpaca-format samples, this dataset targets 340K–420K+ examples with a key upgrade: compiler-in-the-loop verification. Every code sample is checked against a real toolchain (arduino-cli, arm-none-eabi-gcc, xtensa-esp32-elf-gcc) before it's used, so the model learns from code that actually builds — not just code that looks plausible.

Contents:
SFT data — ChatML-format instruction/response pairs (with <think> reasoning traces) covering embedded (ESP32/STM32/AVR), drone (ArduPilot/PX4), and robotics (ROS2) domains
DPO preference pairs — chosen/rejected completions labeled automatically by compile pass/fail, not human judgment
