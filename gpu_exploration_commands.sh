#!/bin/bash

# Apple Silicon GPU Exploration Commands
# Collection of useful terminal commands to understand your GPU

echo "🍎 Apple Silicon GPU Exploration Commands"
echo "========================================"

echo -e "\n📋 1. System Hardware Overview:"
echo "system_profiler SPHardwareDataType"
system_profiler SPHardwareDataType

echo -e "\n🖥️  2. Graphics and Display Information:"
echo "system_profiler SPDisplaysDataType"
system_profiler SPDisplaysDataType

echo -e "\n💾 3. Memory Information:"
echo "sysctl -a | grep mem"
echo "Total Memory: $(echo "scale=2; $(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc) GB"
echo "Physical Memory: $(sysctl -n hw.physmem | numfmt --to=iec)"
echo "Usable Memory: $(sysctl -n hw.usermem | numfmt --to=iec)"

echo -e "\n🔧 4. CPU Information:"
echo "sysctl -n machdep.cpu.brand_string"
sysctl -n machdep.cpu.brand_string
echo "CPU Cores: $(sysctl -n hw.ncpu)"
echo "Physical CPU Cores: $(sysctl -n hw.physicalcpu)"
echo "Logical CPU Cores: $(sysctl -n hw.logicalcpu)"

echo -e "\n🌡️  5. Thermal State:"
echo "powermetrics --samplers smc -n 1 -i 1000 | grep -A 10 'SMC sensors'"

echo -e "\n⚡ 6. Power and Performance:"
echo "pmset -g batt"
pmset -g batt

echo -e "\n🔍 7. Metal Support Detection:"
echo "system_profiler SPDisplaysDataType | grep -A 5 'Metal Support'"
system_profiler SPDisplaysDataType | grep -A 5 "Metal Support"

echo -e "\n🏗️  8. Architecture Information:"
echo "uname -m"
uname -m
echo "Architecture: $(arch)"

echo -e "\n📊 9. Process Information:"
echo "ps aux | grep -E '(WindowServer|kernel_task)' | head -5"
ps aux | grep -E "(WindowServer|kernel_task)" | head -5

echo -e "\n🧮 10. Quick GPU Test Commands:"
echo "# Test PyTorch MPS availability:"
echo "python3 -c \"import torch; print('MPS Available:', torch.backends.mps.is_available())\""

echo -e "\n# Test Metal GPU info:"
echo "python3 -c \""
echo "import Metal"
echo "device = Metal.MTLCreateSystemDefaultDevice()"
echo "if device:"
echo "    print('Metal Device:', device.name())"
echo "    print('Max Threads Per Group:', device.maxThreadsPerThreadgroup())"
echo "else:"
echo "    print('No Metal device found')"
echo "\""

echo -e "\n💡 Additional Useful Commands:"
echo "# Monitor GPU usage in real-time:"
echo "sudo powermetrics --samplers gpu_power -n 0"
echo ""
echo "# Check system load:"
echo "uptime"
echo ""
echo "# Memory pressure:"
echo "memory_pressure"
echo ""
echo "# Activity Monitor from command line:"
echo "top -l 1 -s 0 | grep 'CPU usage'"

echo -e "\n🔗 Quick Access Commands:"
cat << 'EOF'
# Save these as aliases in your ~/.zshrc:
alias gpu-info="python3 apple_silicon_gpu_info.py"
alias gpu-specs="system_profiler SPDisplaysDataType SPHardwareDataType"
alias gpu-memory="echo 'GPU Memory: Shared with system ($(echo "scale=2; $(sysctl -n hw.memsize) / 1024 / 1024 / 1024" | bc) GB total)'"
alias gpu-test="python3 src/gpu_utils.py"

# To add these aliases permanently:
# echo 'alias gpu-info="python3 apple_silicon_gpu_info.py"' >> ~/.zshrc
# source ~/.zshrc
EOF
