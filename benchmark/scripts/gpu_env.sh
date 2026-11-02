# Pin all GPU work to the discrete NVIDIA card.
# The AMD integrated GPU is display-only and must never be selected for compute.

# --- CUDA ---------------------------------------------------------------
# CUDA only enumerates NVIDIA devices, but pin explicitly so a second card
# (or a future MIG split) cannot change which device is used.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# --- WebGPU (Dawn / wgpu-native) ----------------------------------------
# Force the Vulkan backend: the GL backend would land on the AMD iGPU.
# WGPU_BACKEND is a wgpu-native variable, not a Dawn one, and is NOT read by this
# project. TORTOISE_WEBGPU_ADAPTER_TYPE, however, IS now honoured by SelectAdapter
# (it had no reader until 2026-08-20). Keeping it here pins this host to the
# discrete NVIDIA card explicitly rather than by default. See CLAUDE.md 3.1 for the
# macOS/Metal values.
export WGPU_BACKEND=vulkan
export TORTOISE_WEBGPU_ADAPTER_TYPE=discrete

# This one IS read (SelectAdapter, webgpu_context.cxx) and overrides the vendor check.
export TORTOISE_WEBGPU_VENDOR_ID=0x10DE          # NVIDIA
# Vulkan loader: restrict ICDs to the NVIDIA one where the file exists.
# Measured on this host: without this, Vulkan exposes three devices - the AMD
# Raphael iGPU (0x1002, integrated), the RTX 4070 Ti SUPER (0x10DE, discrete)
# and llvmpipe (software, type=cpu). With it, only the NVIDIA device is visible.
_nv_icd=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ -f "$_nv_icd" ]; then
    export VK_ICD_FILENAMES=$_nv_icd     # older loaders
    export VK_DRIVER_FILES=$_nv_icd      # loader >= 1.3.234
fi
# Belt and braces for the NVIDIA loader on hybrid-graphics systems.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
