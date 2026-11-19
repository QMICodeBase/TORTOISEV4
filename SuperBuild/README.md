# Native dependency superbuild

The superbuild downloads checksum-pinned Boost 1.86.0, Eigen 3.4.0,
FFTW 3.3.10, zlib 1.3.1, and ITK 6.0b02, builds them in a private prefix,
and builds and installs TORTOISE. AFNI remains a runtime dependency:
`3dSkullStrip` must be on `PATH` for automatic brain masks.

## Apple Silicon with Homebrew GCC 15

From the repository root (CMake, Ninja, Xcode command-line tools, and GCC 15
must already be installed):

```sh
cmake -S SuperBuild -B build/macos-gcc15 -G Ninja \
  -DCMAKE_C_COMPILER="$(command -v gcc-15)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++-15)" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORTOISE_BUILD_JOBS=4
cmake --build build/macos-gcc15 --parallel 2
cmake --install build/macos-gcc15
export PATH="$PWD/build/macos-gcc15/install/bin:$PATH"
```

`--parallel 2` limits simultaneous external projects; `TORTOISE_BUILD_JOBS`
limits jobs *inside each* external project. Start conservatively because ITK
and TORTOISE compile large C++ templates. Re-run the build to resume after an
interruption. TORTOISE sources are checked for changes on every outer build.

Use a fresh build directory for a different compiler, architecture, SDK, or
C++ runtime. GCC 15 and Clang with libomp are candidates; this is not a claim
that every compiler configuration has been validated. For Clang, provide the
OpenMP runtime discovery hints to the inner project if CMake cannot find it.
GCC 14 encountered SDK-header errors in the process-launch adapter locally.

## Layout and options

- `dependencies/`: private headers and libraries; never installed system-wide.
- `downloads/`: reusable archives, verified against the pinned SHA-256 hashes.
- `<dependency>-prefix/`: unpacked sources, build trees, and step logs.
- `tortoise-build/`: normal inner CMake build, including `bin/` and `settings/`.
- `install/bin/`, `install/settings/`: installed TORTOISE runtime layout.
- `build-manifest.txt`: compiler, platform, build type, and dependency versions.

Set `TORTOISE_DOWNLOAD_DIR` to reuse archives across builds (names are
`boost.tar.gz`, `eigen.tar.gz`, `fftw.tar.gz`, `zlib.tar.gz`, `itk.tar.gz`).
Set `CMAKE_INSTALL_PREFIX` to change the final TORTOISE destination, and
`TORTOISE_DEPENDENCY_PREFIX` to change its private dependency prefix.
Do not share a dependency prefix across incompatible compiler configurations.

CPU-only and no VTK are the defaults. `USECUDA=ON` requires an installed CUDA
toolkit and is rejected on macOS. `USE_VTK=ON` requires an existing compatible
VTK installation supplied with `VTK_DIR`; this first superbuild does not build
VTK. Dependency libraries are requested static, but the OS and compiler/OpenMP
runtimes remain dynamically linked. The install tree is a local installation,
not yet a self-contained, signed macOS distribution.

## Using existing dependencies

The ordinary build remains available without the outer project:

```sh
cmake -S TORTOISEV4 -B build/direct -G Ninja \
  -DCMAKE_C_COMPILER="$(command -v gcc-15)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++-15)" \
  -DCMAKE_PREFIX_PATH="$PWD/build/macos-gcc15/dependencies" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/direct/install"
cmake --build build/direct --parallel 4
cmake --install build/direct
```

Provide `ITK_DIR`, `Boost_DIR`, `Eigen3_DIR`, or FFTW discovery cache variables
when using dependencies outside a common prefix. Use compatible C++ runtimes
throughout. This build uses C++17 to match the ITK 6 dependency.

## Validation

The default build runs CTest before installing TORTOISE. Tests exercise the
mask adapter with a fake AFNI command (including failure and cleanup paths)
and check the main CPU command-line usage menus. Run them again with:

```sh
ctest --test-dir build/macos-gcc15/tortoise-build --output-on-failure
```

Use `TORTOISE_BUILD_TESTING=OFF` to skip building the adapter test.
A successful compile is only the first check. Before scientific use, run the
command-line tools and `CreateMask` on representative b=0/T2 images, inspect
mask geometry and coverage, and compare a complete processing run with Linux.
The switch from BET to AFNI changes brain extraction and does not promise
numerically identical results. No representative processing dataset is
bundled in this repository.

### Verified local configuration

The native arm64 Release build with Homebrew GCC 15.3.0 completed and installed
96 executables with CUDA and VTK disabled. All four CTest checks passed, and
the installed `TORTOISEProcess`, `DRBUDDI`, and `CreateMask` usage checks passed.
`otool -L` on the installed main executable lists macOS system libraries plus
GCC 15's `libgomp` and `libstdc++`; it does not depend on the private build-tree
ITK, Boost, FFTW, or zlib libraries. AFNI is invoked at runtime.

This verifies compilation, installation, and adapter mechanics. Actual AFNI
brain extraction and end-to-end processing on scientific datasets have not
been validated by these tests. Intel macOS, Linux, Clang, CUDA, and VTK builds
have not been exercised with this superbuild yet.
