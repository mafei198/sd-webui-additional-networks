import json
import os
import time
from typing import List
import hashlib
import filelock
import torch
import numpy as np
from pydantic import BaseModel, Field

import modules.scripts as scripts
from modules import shared, script_callbacks, hashes
import gradio as gr

import modules.ui
from modules.api import api

from scripts import lora_compvis, model_util, metadata_editor, xyz_grid_support
from scripts.model_util import lora_models, MAX_MODEL_COUNT

memo_symbol = '\U0001F4DD'  # 📝
addnet_paste_params = {"txt2img": [], "img2img": []}


class Script(scripts.Script):
    latest_params = [(None, None, None, None)] * MAX_MODEL_COUNT
    changed = False
    latest_networks = []
    latest_model_hash = ""

    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Additional networks for generating"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def set_infotext_fields(self, p, params):
        for i, t in enumerate(params):
            module, model, weight_unet, weight_tenc = t
            if model is None or model == "None" or len(model) == 0 or (weight_unet == 0 and weight_tenc == 0):
                continue
            p.extra_generation_params.update({
                "AddNet Enabled": True,
                f"AddNet Module {i + 1}": module,
                f"AddNet Model {i + 1}": model,
                f"AddNet Weight A {i + 1}": weight_unet,
                f"AddNet Weight B {i + 1}": weight_tenc,
            })

    def set_infotext_fields(self, p, params):
        for i, t in enumerate(params):
            module, model, weight_unet, weight_tenc = t
            if model is None or model == "None" or len(model) == 0 or (weight_unet == 0 and weight_tenc == 0):
                continue
            p.extra_generation_params.update({
                "AddNet Enabled": True,
                f"AddNet Module {i + 1}": module,
                f"AddNet Model {i + 1}": model,
                f"AddNet Weight A {i + 1}": weight_unet,
                f"AddNet Weight B {i + 1}": weight_tenc,
            })

    def restore_networks(self, sd_model):
        unet = sd_model.model.diffusion_model
        text_encoder = sd_model.cond_stage_model

        if len(Script.latest_networks) > 0:
            print("restoring last networks")
            for network, _ in Script.latest_networks[::-1]:
                network.restore(text_encoder, unet)
            Script.latest_networks.clear()

    def process_batch(self, p, *args, **kwargs):
        unet = p.sd_model.model.diffusion_model
        text_encoder = p.sd_model.cond_stage_model

        if Script.latest_params is None:
            self.restore_networks(p.sd_model)
            return

        Script.latest_model_hash = p.sd_model.sd_model_hash
        if Script.changed:
            self.restore_networks(p.sd_model)
            for module, model, weight_unet, weight_tenc in Script.latest_params:
                if model is None or model == "None" or len(model) == 0:
                    continue
                if weight_unet == 0 and weight_tenc == 0:
                    print(f"ignore because weight is 0: {model}")
                    continue

                model_path = lora_models.get(model, None)
                if model_path is None:
                    raise RuntimeError(f"model not found: {model}")

                if model_path.startswith("\"") and model_path.endswith("\""):  # trim '"' at start/end
                    model_path = model_path[1:-1]
                if not os.path.exists(model_path):
                    print(f"file not found: {model_path}")
                    continue

                print(f"{module} weight_unet: {weight_unet}, weight_tenc: {weight_tenc}, model: {model}")
                if module == "LoRA":
                    if os.path.splitext(model_path)[1] == '.safetensors':
                        from safetensors.torch import load_file
                        du_state_dict = load_file(model_path)
                    else:
                        du_state_dict = torch.load(model_path, map_location='cpu')

                    network, info = lora_compvis.create_network_and_apply_compvis(du_state_dict, weight_tenc,
                                                                                  weight_unet, text_encoder, unet)
                    # in medvram, device is different for u-net and sd_model, so use sd_model's
                    network.to(p.sd_model.device, dtype=p.sd_model.dtype)

                    print(f"LoRA model {model} loaded: {info}")
                    Script.latest_networks.append((network, model))

        if len(Script.latest_networks) > 0:
            print("setting (or sd model) changed. new networks created.")

        # apply mask: currently only top 3 networks are supported
        # if len(self.latest_networks) > 0:
        #     mask_image = args[-2]
        #     if mask_image is not None:
        #         mask_image = mask_image.astype(np.float32) / 255.0
        #         print(f"use mask image to control LoRA regions.")
        #         for i, (network, model) in enumerate(self.latest_networks[:3]):
        #             if not hasattr(network, "set_mask"):
        #                 continue
        #             mask = mask_image[:, :, i]  # R,G,B
        #             if mask.max() <= 0:
        #                 continue
        #             mask = torch.tensor(mask, dtype=p.sd_model.dtype, device=p.sd_model.device)
        #             network.set_mask(mask, height=p.height, width=p.width)
        #             print(f"apply mask. channel: {i}, model: {model}")
        #     else:
        #         for network, _ in self.latest_networks:
        #             if hasattr(network, "set_mask"):
        #                 network.set_mask(None)

        self.set_infotext_fields(p, Script.latest_params)
        Script.changed = False

        Script.latest_model_hash = ""


def on_script_unloaded():
    if shared.sd_model:
        for s in scripts.scripts_txt2img.alwayson_scripts:
            if isinstance(s, Script):
                s.restore_networks(shared.sd_model)
                break




def on_infotext_pasted(infotext, params):
    if "AddNet Enabled" not in params:
        params["AddNet Enabled"] = "False"

    # TODO changing "AddNet Separate Weights" does not seem to work
    if "AddNet Separate Weights" not in params:
        params["AddNet Separate Weights"] = "False"

    for i in range(MAX_MODEL_COUNT):
        # Convert combined weight into new format
        if f"AddNet Weight {i + 1}" in params:
            params[f"AddNet Weight A {i + 1}"] = params[f"AddNet Weight {i + 1}"]
            params[f"AddNet Weight B {i + 1}"] = params[f"AddNet Weight {i + 1}"]

        if f"AddNet Module {i + 1}" not in params:
            params[f"AddNet Module {i + 1}"] = "LoRA"
        if f"AddNet Model {i + 1}" not in params:
            params[f"AddNet Model {i + 1}"] = "None"
        if f"AddNet Weight A {i + 1}" not in params:
            params[f"AddNet Weight A {i + 1}"] = "0"
        if f"AddNet Weight B {i + 1}" not in params:
            params[f"AddNet Weight B {i + 1}"] = "0"

        params[f"AddNet Weight {i + 1}"] = params[f"AddNet Weight A {i + 1}"]

        if params[f"AddNet Weight A {i + 1}"] != params[f"AddNet Weight B {i + 1}"]:
            params["AddNet Separate Weights"] = "True"

        # Convert potential legacy name/hash to new format
        params[f"AddNet Model {i + 1}"] = str(model_util.find_closest_lora_model_name(params[f"AddNet Model {i + 1}"]))

        xyz_grid_support.update_axis_params(i, params[f"AddNet Module {i + 1}"], params[f"AddNet Model {i + 1}"])


xyz_grid_support.initialize(Script)

script_callbacks.on_script_unloaded(on_script_unloaded)
script_callbacks.on_infotext_pasted(on_infotext_pasted)


class ApiHijack(api.Api):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_api_route("/select_lora", select_lora, methods=["POST"],
                           response_model=SelectLoRAResponse)
        self.add_api_route("/refresh_lora", refresh_lora, methods=["GET"])
        self.add_api_route("/refresh_hash_cache", refresh_hash_cache, methods=["GET"])


class ModelParam(BaseModel):
    model_name: str
    unet_weight: float
    text_encoder_weight: float


class SelectLoRARequest(BaseModel):
    models: List[ModelParam]


class SelectLoRAResponse(BaseModel):
    result: str = Field(default=None, title="result", description="result of loading LoRA")


def select_lora(select_lora_request: SelectLoRARequest):
    params = []
    for model in select_lora_request.models:
        t = ("LoRA", model.model_name, model.unet_weight, model.text_encoder_weight)
        params.append(t)

    if Script.latest_params != params:
        Script.changed = True
        Script.latest_params = params
        print("[select_lora] lora models changed: ", Script.latest_params)

    return SelectLoRAResponse(result="success")


def refresh_lora():
    model_util.update_models()
    return {
        "result": "success"
    }


class RefreshModelHashCacheRequest(BaseModel):
    model_uri: str


# 用于更新ckpt model hash cache
def refresh_hash_cache(req: RefreshModelHashCacheRequest):
    with filelock.FileLock(hashes.cache_filename + ".lock"):
        if not os.path.isfile(hashes.cache_filename):
            hashes.cache_data = {}
        else:
            with open(hashes.cache_filename, "r", encoding="utf8") as file:
                hashes.cache_data = json.load(file)
    filename = "checkpoint/" + os.path.basename(req.model_uri)
    hashes_dict = hashes.cache("hashes")

    hashes_dict[filename] = {
        "mtime": time.gmtime(time.time()),
        "sha256": hashlib.sha256(filename.encode('utf-8'))
    }
    print(filename, "\n", hashes_dict[filename])
    return {
        "result": "success"
    }


api.Api = ApiHijack
