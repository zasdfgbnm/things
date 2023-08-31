FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
RUN apt update
RUN apt install -y python3 python3-pip git
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
ENV LD_LIBRARY_PATH=/w/build:/usr/local/lib/python3.8/dist-packages/torch/lib
