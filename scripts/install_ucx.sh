export ROCM_PATH=/opt/rocm
export UCX_HOME=/usr/local/ucx

cd /usr/local/src
git clone https://github.com/ROCm/ucx.git
cd ucx
git checkout da3fac2a   # match Dockerfile.rocm unless you know a newer compatible commit

./autogen.sh
mkdir -p build && cd build
../configure \
  --prefix="${UCX_HOME}" \
  --enable-shared \
  --disable-static \
  --disable-doxygen-doc \
  --enable-optimizations \
  --enable-devel-headers \
  --with-rocm="${ROCM_PATH}" \
  --with-verbs \
  --with-dm \
  --enable-mt

make -j"$(nproc)"
make install

export PATH="${UCX_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${UCX_HOME}/lib:${LD_LIBRARY_PATH}"
