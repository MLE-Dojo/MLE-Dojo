# 1. Select NVIDIA CUDA Development Base Image
# Using 'devel' tag for broader compatibility, including compiling CUDA extensions.
# Using Ubuntu 22.04 LTS for stability.
# ARG allows easy modification of these versions at build time if needed.
ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${OS_VERSION}

# 2. Set Environment Variables
# - Avoid interactive prompts during package installation.
# - Ensure Python output is sent straight to terminal (no buffering).
# - Ensure container runtime has access to all GPUs and necessary capabilities.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 3. Install System Dependencies, Python 3.11, pip, and basic tools
# - Update package lists.
# - Install 'software-properties-common' to manage repositories (for PPA).
# - Add the 'deadsnakes' PPA for newer Python versions.
# - Install Python 3.11, development headers (-dev), venv, and pip.
# - Install 'build-essential' (for compiling C/C++ code) and 'git' (for `pip install -e`).
# - Install common utilities like 'curl', 'wget', 'ca-certificates'.
# - Set Python 3.11 as the default 'python' and 'python3'.
# - Clean up apt cache afterwards to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    && apt-get remove -y python3-pip && apt-get autoremove -y && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Verify Python and Pip versions (Optional - uncomment during debugging)
# RUN python --version
# RUN pip --version

# 4. Set Working Directory
WORKDIR /home

# 5. Copy Project Code
# Copy the current directory contents into the container at /home
COPY . /home/

# 6. Install Python Dependencies
# - Upgrade pip, setuptools, and wheel to the latest versions first.
# - Install project dependencies using 'pip install -e .'
#   (Editable mode often used for development).
# - Use --no-cache-dir to reduce layer size.
# - Use 'python -m pip' to ensure the correct pip associated with python3.11 is used.
RUN python -m pip install --no-cache-dir --upgrade pip wheel setuptools
RUN python -m pip install --no-cache-dir -e .

# Note: If your Python packages have additional system library dependencies
# (e.g., libgl1-mesa-glx, libsm6, libxext6 for OpenCV),
# you MUST install them using 'apt-get install' in step 3 above.

# 7. Set Entrypoint
# Specifies the command to run when the container starts.
ENTRYPOINT ["/home/entrypoint.sh"]
