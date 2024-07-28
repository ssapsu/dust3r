#! /bin/bash

docker build -t dust3r:webserver -f ./docker/cuda.fastapi.Dockerfile .

docker run -it --name dust3r --gpus '"device=2"'  \
        -e DEVICE=cuda \
        -e MODEL=DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
        -p 20520:7860 \
        -v /home/ssapsu/dust3r:/dust3r \
        --cap-add=IPC_LOCK --cap-add=SYS_RESOURCE   dust3r:webserver
