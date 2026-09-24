#!/bin/bash
# Configure the macOS build trees. Records every dependency path in one place so
# the M0 provenance artefact can quote it. Usage: mac_configure.sh <webgpu|metal>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LIB=${TORTOISE_LIBS:-$HOME/src/tortoise_libraries}
SDK=$(xcrun --show-sdk-path)
BOOST=$(brew --prefix boost)
FFTW=$(brew --prefix fftw)
# Boost.System is header-only since 1.69 and Homebrew 1.90 ships no stub library, so
# the Apple branch of CMakeLists.txt requests only iostreams+filesystem.
COMMON=(-DUSE_VTK=0 -DCMAKE_BUILD_TYPE=Release
  -DITK_DIR="$LIB/InsightToolkit-6.0b02_build"
  -DEigen3_DIR="$LIB/local/share/eigen3/cmake"
  -DBoost_INCLUDE_DIR="$BOOST/include"
  -DBoost_IOSTREAMS_LIBRARY_RELEASE="$BOOST/lib/libboost_iostreams.a"
  -DBoost_FILESYSTEM_LIBRARY_RELEASE="$BOOST/lib/libboost_filesystem.a"
  -DFFTW_ROOT="$FFTW"
  -DZLIB_LIBRARY="$SDK/usr/lib/libz.tbd"
  -DZLIB_INCLUDE_DIR="$SDK/usr/include")
case "${1:-webgpu}" in
  webgpu) cmake -S "$REPO/TORTOISEV4" -B "$REPO/build_webgpu" -DUSECUDA=0 -DUSEWEBGPU=1 \
            -DDAWN_DIR="$LIB/dawn" "${COMMON[@]}" ;;
  metal)  cmake -S "$REPO/TORTOISEV4" -B "$REPO/build_metal"  -DUSECUDA=0 -DUSEMETAL=1 \
            "${COMMON[@]}" ;;
  *) echo "usage: mac_configure.sh <webgpu|metal>" >&2; exit 2 ;;
esac
