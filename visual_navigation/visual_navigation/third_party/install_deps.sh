export CUDA_HOME=/usr/local/cuda-12.6/
export BUILD_WITH_CUDA=True
export AM_I_DOCKER=False
# sudo python -m pip install -e segment_anything
sudo pip install --no-build-isolation -e Grounded-Segment-Anything/GroundingDINO
sudo pip install nvidia_radio/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# sudo pip install --upgrade diffusers[torch]
# sudo pip install -r ./recognize-anything/requirements.txt
# sudo pip install -e ./recognize-anything/
# sudo pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel