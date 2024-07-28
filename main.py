#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# FastAPI webserver for 3D reconstruction
# --------------------------------------------------------
import os
import shutil
import copy
import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta

from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl
pl.ion()

app = FastAPI()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

model = None
device = None
image_size = None
output_directory = "/dust3r/output"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Scheduler for deleting files after a certain period
scheduler = BackgroundScheduler()

def delete_old_files(path: str, age_seconds: int):
    now = datetime.now()
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if (now - file_creation_time).total_seconds() > age_seconds:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

# Run the cleanup job every hour
scheduler.add_job(delete_old_files, 'interval', hours=1, args=[output_directory, 24 * 60 * 60])
scheduler.start()

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False, transparent_cams=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera if transparent_cams is False
    if not transparent_cams:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                          imgs[i], focals[i],
                          imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    print(f'(exporting 3D scene to {outfile} )')
    scene.export(file_obj=outfile)
    return outfile

def get_3D_model_from_scene(outdir, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size)

def get_reconstructed_scene(outdir, model, device, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    imgs = load_images(filelist, size=image_size)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    if not os.path.exists(outfile):
        print(f"Error: Output file {outfile} was not created.")
        return None

    return outfile

@app.on_event("startup")
async def startup_event():
    global model, device, image_size
    # Load your model and set device here
    model_path = "./docker/files/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"  # Change this to your model weights path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    image_size = 512  # You can change the default image size here

@app.post("/reconstruct")
async def reconstruct(files: List[UploadFile] = File(...),
                      image_size: int = Form(...),
                      min_conf_thr: float = Form(...),
                      as_pointcloud: bool = Form(...),
                      mask_sky: bool = Form(...),
                      clean_depth: bool = Form(...),
                      transparent_cams: bool = Form(...),
                      cam_size: float = Form(...),
                      scenegraph_type: str = Form(...),
                      winsize: int = Form(...),
                      refid: int = Form(...),
                      schedule: str = Form(...),
                      niter: int = Form(...)):
    # Create a subdirectory for each request to avoid conflicts
    request_id = os.urandom(8).hex()
    request_output_directory = os.path.join(output_directory, request_id)
    os.makedirs(request_output_directory, exist_ok=True)

    try:
        file_paths = []
        for file in files:
            file_path = os.path.join(request_output_directory, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            file_paths.append(file_path)

        output_file = get_reconstructed_scene(request_output_directory, model, device, image_size, file_paths, schedule, niter,
                                              min_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams,
                                              cam_size, scenegraph_type, winsize, refid)

        if not output_file:
            raise HTTPException(status_code=500, detail="Failed to generate 3D model")

        return FileResponse(output_file, media_type='application/octet-stream', filename='scene.glb')

    finally:
        # Clean up the request-specific directory after 24 hours
        scheduler.add_job(shutil.rmtree, 'date', run_date=datetime.now() + timedelta(hours=24), args=[request_output_directory])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
