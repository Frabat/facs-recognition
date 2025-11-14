FROM --platform=linux/amd64 ubuntu:16.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- Base dependencies (system toolchain + OpenCV 2.4) ----
RUN apt-get update && apt-get install -y \
    git cmake build-essential pkg-config \
    ffmpeg \
    python3 python3-dev python3-pip \
    libopencv-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libtbb-dev \
    libeigen3-dev \
    libx11-dev \
    libgtk2.0-dev \
    libpng-dev \
    libjpeg-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
    mosquitto-clients \
    wget \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# ---- Install modern CMake and override system cmake ----
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0-linux-x86_64.sh \
    && chmod +x cmake-3.22.0-linux-x86_64.sh \
    && ./cmake-3.22.0-linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.22.0-linux-x86_64.sh \
    && ln -sf /usr/local/bin/cmake /usr/bin/cmake

# ---- Install newer GCC/G++ (8.x) as required by OpenFace ----
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get update \
    && apt-get install -y g++-8 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install numpy before dlib ----
RUN pip3 install numpy==1.17

# ---- Install dlib 19.13 ----
WORKDIR /opt
RUN wget http://dlib.net/files/dlib-19.13.tar.bz2 \
    && tar xjf dlib-19.13.tar.bz2 \
    && cd dlib-19.13 \
    && python3 setup.py install --yes BUILD_SHARED_LIBS=ON

# ---- Minimal dlibConfig.cmake so CMake can find dlib ----
RUN mkdir -p /usr/local/lib/cmake/dlib \
    && printf "set(dlib_FOUND TRUE)\nset(dlib_INCLUDE_DIRS /usr/local/include)\nset(dlib_LIBRARIES /usr/local/lib/libdlib.so)\n" \
        > /usr/local/lib/cmake/dlib/dlibConfig.cmake

# ---- Clone OpenFace ----
WORKDIR /opt
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git

# ---- Relax OpenCV requirement: allow system OpenCV (2.4/3.x) instead of forcing 4.0 ----
RUN sed -i 's/OpenCV 4.0/OpenCV/g' /opt/OpenFace/CMakeLists.txt

# ---- Relax dlib requirement: drop explicit 19.13 version check ----
RUN sed -i 's/find_package(dlib 19.13 REQUIRED)/find_package(dlib REQUIRED)/' /opt/OpenFace/CMakeLists.txt

# ---- Build OpenFace with GCC 8 (C++17-capable) ----
WORKDIR /opt/OpenFace/build
RUN CC=gcc-8 CXX=g++-8 cmake .. && \
    CC=gcc-8 CXX=g++-8 make -j"$(nproc)"

# ---- Runtime ----
COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

CMD ["/opt/start.sh"]
