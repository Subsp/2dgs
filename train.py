#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import re
import torch
import torch.nn.functional as torch_F
from glob import glob
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def _extract_trailing_int(stem):
    match = re.search(r"(\d+)$", stem)
    if match is None:
        return None
    return int(match.group(1))


def _sort_key_from_stem(stem):
    idx = _extract_trailing_int(stem)
    if idx is None:
        return (1, 0, stem)
    return (0, idx, stem)


def _build_external_prior_index(train_cameras, external_prior_root, prior_subdir="priors", exts=("png", "jpg", "jpeg", "webp")):
    root = os.path.abspath(os.path.expanduser(external_prior_root))
    candidates = []
    if prior_subdir:
        candidates.append(os.path.join(root, prior_subdir))
    candidates.append(root)

    stem_to_path = {}
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        for ext in exts:
            pattern = os.path.join(folder, "**", f"*.{ext}")
            for path in sorted(glob(pattern, recursive=True)):
                stem = os.path.splitext(os.path.basename(path))[0]
                if stem not in stem_to_path:
                    stem_to_path[stem] = path

    index = {}
    used_paths = set()
    exact_count = 0
    numeric_count = 0
    order_count = 0

    for camera in train_cameras:
        path = stem_to_path.get(camera.image_name)
        if path is None or path in used_paths:
            continue
        index[camera.image_name] = path
        used_paths.add(path)
        exact_count += 1

    unmatched = [cam for cam in train_cameras if cam.image_name not in index]
    idx_to_paths = {}
    for stem, path in sorted(stem_to_path.items(), key=lambda x: _sort_key_from_stem(x[0])):
        idx = _extract_trailing_int(stem)
        if idx is None:
            continue
        idx_to_paths.setdefault(idx, []).append(path)

    for camera in unmatched:
        idx = _extract_trailing_int(camera.image_name)
        if idx is None:
            continue
        candidates = [p for p in idx_to_paths.get(idx, []) if p not in used_paths]
        if not candidates:
            continue
        picked = candidates[0]
        index[camera.image_name] = picked
        used_paths.add(picked)
        numeric_count += 1

    unmatched = [cam for cam in train_cameras if cam.image_name not in index]
    if unmatched:
        available_paths = []
        for stem, path in sorted(stem_to_path.items(), key=lambda x: _sort_key_from_stem(x[0])):
            if path in used_paths:
                continue
            available_paths.append(path)

        unmatched_sorted = sorted(unmatched, key=lambda cam: _sort_key_from_stem(cam.image_name))
        pair_count = min(len(unmatched_sorted), len(available_paths))
        for i in range(pair_count):
            camera = unmatched_sorted[i]
            picked = available_paths[i]
            index[camera.image_name] = picked
            used_paths.add(picked)
            order_count += 1

    missing = len(train_cameras) - len(index)
    print(
        "[2DGS-PRIOR] external index size: "
        f"{len(index)} matched / {len(train_cameras)} train views "
        f"(missing={missing}, root={root})"
    )
    if len(index) > 0:
        print(
            "[2DGS-PRIOR] external match breakdown: "
            f"exact={exact_count}, numeric={numeric_count}, order={order_count}"
        )
    return index


class ImageTensorBank:
    def __init__(self, index, mode="rgb"):
        self.index = index
        self.mode = mode
        self.cache = {}

    def _load_tensor(self, image_name):
        path = self.index.get(image_name)
        if path is None or not os.path.exists(path):
            return None
        from PIL import Image

        if self.mode == "mask":
            arr = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0).contiguous()
        else:
            arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def get(self, image_name, width, height, device):
        if image_name not in self.cache:
            tensor = self._load_tensor(image_name)
            if tensor is None:
                return None
            self.cache[image_name] = tensor

        tensor = self.cache[image_name]
        if tensor.shape[-2:] != (height, width):
            interp_mode = "bilinear" if tensor.shape[0] > 1 else "nearest"
            tensor = torch_F.interpolate(
                tensor.unsqueeze(0),
                size=(height, width),
                mode=interp_mode,
                align_corners=False if interp_mode == "bilinear" else None,
            )[0]
        return tensor.to(device=device)


def _masked_l1(pred, target, mask):
    diff = (pred - target).abs()
    if mask is None:
        return diff.mean()
    weighted = diff * mask
    denom = torch.clamp(mask.sum() * diff.shape[0], min=1.0)
    return weighted.sum() / denom


def _laplacian_highfreq_rgb(image):
    kernel = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(image.shape[0], 1, 1, 1)
    out = torch_F.conv2d(image.unsqueeze(0), kernel, padding=1, groups=image.shape[0])[0]
    return out


def _load_prior_supervision(prior_bank, mask_bank, camera, device):
    if prior_bank is None or camera is None:
        return None
    prior_image = prior_bank.get(
        camera.image_name,
        width=int(camera.image_width),
        height=int(camera.image_height),
        device=device,
    )
    if prior_image is None:
        return None
    mask = None
    if mask_bank is not None:
        mask = mask_bank.get(
            camera.image_name,
            width=int(camera.image_width),
            height=int(camera.image_height),
            device=device,
        )
        if mask is not None:
            mask = mask.clamp(0.0, 1.0)
    return {"prior_image": prior_image, "mask": mask}


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, prior_args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_l1_for_log = 0.0
    ema_prior_hf_for_log = 0.0

    prior_bank = None
    mask_bank = None
    if getattr(prior_args, "external_prior_root", ""):
        prior_exts = tuple(
            tok.strip().lower()
            for tok in str(getattr(prior_args, "external_prior_exts", "png,jpg,jpeg,webp")).split(",")
            if tok.strip()
        )
        train_cameras = scene.getTrainCameras()
        prior_index = _build_external_prior_index(
            train_cameras=train_cameras,
            external_prior_root=prior_args.external_prior_root,
            prior_subdir=getattr(prior_args, "external_prior_subdir", "priors"),
            exts=prior_exts,
        )
        prior_bank = ImageTensorBank(prior_index, mode="rgb")
        if getattr(prior_args, "external_prior_mask_subdir", ""):
            mask_index = _build_external_prior_index(
                train_cameras=train_cameras,
                external_prior_root=prior_args.external_prior_root,
                prior_subdir=prior_args.external_prior_mask_subdir,
                exts=prior_exts,
            )
            mask_bank = ImageTensorBank(mask_index, mode="mask")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        prior_l1_loss = torch.zeros((), device=image.device)
        prior_hf_loss = torch.zeros((), device=image.device)

        prior_pack = _load_prior_supervision(prior_bank, mask_bank, viewpoint_cam, image.device)
        if prior_pack is not None:
            prior_image = prior_pack["prior_image"]
            prior_mask = prior_pack["mask"]
            if prior_mask is not None and float(getattr(prior_args, "prior_mask_floor", 0.0)) > 0.0:
                prior_mask = (prior_mask >= float(prior_args.prior_mask_floor)).float()
            if float(getattr(prior_args, "prior_l1_weight", 0.0)) > 0.0:
                prior_l1_loss = _masked_l1(image, prior_image, prior_mask)
            if float(getattr(prior_args, "prior_hf_weight", 0.0)) > 0.0:
                image_hf = _laplacian_highfreq_rgb(image)
                prior_hf = _laplacian_highfreq_rgb(prior_image)
                hf_mask = None if prior_mask is None else prior_mask.expand_as(image_hf[:1])
                prior_hf_loss = _masked_l1(image_hf, prior_hf, hf_mask)
            loss = (
                loss
                + float(getattr(prior_args, "prior_l1_weight", 0.0)) * prior_l1_loss
                + float(getattr(prior_args, "prior_hf_weight", 0.0)) * prior_hf_loss
            )
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_prior_l1_for_log = 0.4 * prior_l1_loss.item() + 0.6 * ema_prior_l1_for_log
            ema_prior_hf_for_log = 0.4 * prior_hf_loss.item() + 0.6 * ema_prior_hf_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "prior_l1": f"{ema_prior_l1_for_log:.{5}f}",
                    "prior_hf": f"{ema_prior_hf_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/prior_l1_loss', ema_prior_l1_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/prior_hf_loss', ema_prior_hf_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--external_prior_root", type=str, default="")
    parser.add_argument("--external_prior_subdir", type=str, default="priors")
    parser.add_argument("--external_prior_mask_subdir", type=str, default="")
    parser.add_argument("--external_prior_exts", type=str, default="png,jpg,jpeg,webp")
    parser.add_argument("--prior_l1_weight", type=float, default=0.0)
    parser.add_argument("--prior_hf_weight", type=float, default=0.0)
    parser.add_argument("--prior_mask_floor", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args,
    )

    # All done
    print("\nTraining complete.")
