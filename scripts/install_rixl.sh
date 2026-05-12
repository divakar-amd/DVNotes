export RIXL_HOME=/usr/local/rixl
export UCX_HOME=/usr/local/ucx
export RIXL_REPO=https://github.com/ROCm/RIXL.git
export RIXL_BRANCH=f33a5599   # match Dockerfile.rocm

git clone "${RIXL_REPO}" /opt/rixl
cd /opt/rixl
git checkout "${RIXL_BRANCH}"

meson setup build --prefix="${RIXL_HOME}" \
  -Ducx_path="${UCX_HOME}" \
  -Drocm_path="${ROCM_PATH}"

cd build
ninja
ninja install

mkdir -p /tmp/rixl-wheels
cd /opt/rixl
./contrib/build-wheel.sh \
  --output-dir /tmp/rixl-wheels \
  --rocm-dir "${ROCM_PATH}" \
  --ucx-plugins-dir "${UCX_HOME}/lib/ucx" \
  --nixl-plugins-dir "${RIXL_HOME}/lib/x86_64-linux-gnu/plugins"

python3 -m pip install /tmp/rixl-wheels/*.whl
