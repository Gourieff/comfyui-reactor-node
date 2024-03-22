import os, glob, sys

import torch
from torchvision.transforms.functional import normalize
import numpy as np
import cv2
from typing import List
from PIL import Image
from scipy import stats
from insightface.app.common import Face

from modules.processing import StableDiffusionProcessingImg2Img
from comfy_extras.chainner_models import model_loading
import comfy.model_management as model_management
import comfy.utils
import folder_paths

import scripts.reactor_version
from scripts.reactor_faceswap import (
    FaceSwapScript,
    get_models,
    get_current_faces_model,
    analyze_faces,
    half_det_size
)
from scripts.reactor_logger import logger
from reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    tensor_to_pil,
    img2tensor,
    tensor2img,
    save_face_model,
    load_face_model,
    download
)
from reactor_log_patch import apply_logging_patch
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from r_basicsr.utils.registry import ARCH_REGISTRY
import scripts.r_archs.codeformer_arch


models_dir = folder_paths.models_dir
REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")

if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = ([dir_facerestore_models], folder_paths.supported_pt_extensions)


def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_restorers():
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".pth")]
    if len(models) == 0:
        fr_urls = [
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth"
        ]
        for model_url in fr_urls:
            model_name = os.path.basename(model_url)
            model_path = os.path.join(dir_facerestore_models, model_name)
            download(model_url, model_path, model_name)
        models = glob.glob(models_path)
        models = [x for x in models if x.endswith(".pth")]
    return models

def get_model_names(get_models):
    models = get_models()
    names = ["none"]
    for x in models:
        names.append(os.path.basename(x))
    return names

def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


class reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),               
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        self.face_helper = None

    def restore_face(
            self,
            input_image,
            face_restore_model,
            face_restore_visibility,
            codeformer_weight,
            facedetection
        ):

        result = input_image

        if face_restore_model != "none" and not model_management.processing_interrupted():

            logger.status(f"Restoring with {face_restore_model}")

            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)

            device = model_management.get_torch_device()
            
            if "codeformer" in face_restore_model.lower():
                
                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                checkpoint = torch.load(model_path)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                facerestore_model = codeformer_net.eval()
            
            else:

                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                facerestore_model = model_loading.load_state_dict(sd).eval()

            facerestore_model.to(device)
            
            if self.face_helper is None:
                self.face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)

            image_np = 255. * result.cpu().numpy()

            total_images = image_np.shape[0]
            out_images = np.ndarray(shape=image_np.shape)

            for i in range(total_images):
                cur_image_np = image_np[i,:, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return result

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()

                restored_face = None
                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:
                        with torch.no_grad():
                            output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except Exception as error:
                        print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                    
                    if face_restore_visibility < 1:
                        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility
                    
                    restored_face = restored_face.astype('uint8')
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

                self.face_helper.clean_all()

                out_images[i] = restored_img

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

        return result
    
    def execute(self, enabled, input_image, swap_model, detect_gender_source, detect_gender_input, source_faces_index, input_faces_index, console_log_level, face_restore_model, face_restore_visibility, codeformer_weight, facedetection, source_image=None, face_model=None):
        apply_logging_patch(console_log_level)

        if not enabled:
            return (input_image,face_model)
        elif source_image is None and face_model is None:
            logger.error("Please provide 'source_image' or `face_model`")
            return (input_image,face_model)

        if face_model == "none":
            face_model = None
        
        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)
        if source_image is not None:
            source = tensor_to_pil(source_image)
        else:
            source = None
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(
            p=p,
            img=source,
            enable=True,
            source_faces_index=source_faces_index,
            faces_index=input_faces_index,
            model=swap_model,
            swap_in_source=True,
            swap_in_generated=True,
            gender_source=detect_gender_source,
            gender_target=detect_gender_input,
            face_model=face_model,
        )
        result = batched_pil_to_tensor(p.init_images)

        if face_model is None:
            current_face_model = get_current_faces_model()
            face_model_to_provide = current_face_model[0] if (current_face_model is not None and len(current_face_model) > 0) else face_model
        else:
            face_model_to_provide = face_model
        
        result = reactor.restore_face(self,result,face_restore_model,face_restore_visibility,codeformer_weight,facedetection)

        return (result,face_model_to_provide)


class LoadFaceModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model": (get_model_names(get_facemodels),),
            }
        }
    
    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "🌌 ReActor"

    def load_model(self, face_model):
        self.face_model = face_model
        self.face_models_path = FACE_MODELS_PATH
        if self.face_model != "none":
            face_model_path = os.path.join(self.face_models_path, self.face_model)
            out = load_face_model(face_model_path)
        else:
            out = None
        return (out, )

class BuildFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "compute_method": (["Mean", "Median", "Mode"], {"default": "Mean"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "blend_faces"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def build_face_model(self, image: Image.Image, det_size=(640, 640)):
        if image is None:
            error_msg = "Please load an Image"
            logger.error(error_msg)
            return error_msg
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_model = analyze_faces(image, det_size)

        if len(face_model) == 0:
            det_size_half = half_det_size(det_size)
            face_model = analyze_faces(image, det_size_half)
        
        if face_model is not None and len(face_model) > 0:
            return face_model[0]
        else:
            no_face_msg = "No face found, please try another image"
            logger.error(no_face_msg)
            return no_face_msg
    
    def blend_faces(self, images, save_mode, face_model_name, compute_method):
        if save_mode and images is not None:

            faces = []
            embeddings = []
            images_list: List[Image.Image] = batch_tensor_to_pil(images)

            apply_logging_patch(1)

            n = len(images_list)
            import logging

            logging.StreamHandler.terminator = " "
            for i,image in enumerate(images_list):
                logger.status(f"Building Face Model {i+1} of {n}...")
                face = self.build_face_model(image)
                print(f"{int(((i+1)/n)*100)}%")
                if isinstance(face, str):
                    # logger.error(f"No faces found in {images_names[i]}, skipping")
                    continue
                faces.append(face)
                embeddings.append(face.embedding)
            logging.StreamHandler.terminator = "\n"
            if len(faces) > 0:
                compute_method_name = "Mean" if compute_method == 0 else "Median" if compute_method == 1 else "Mode"
                logger.status(f"Blending with Compute Method {compute_method_name}...")
                blended_embedding = np.mean(embeddings, axis=0) if compute_method == "Mean" else np.median(embeddings, axis=0) if compute_method == "Median" else stats.mode(embeddings, axis=0)[0].astype(np.float32)
                blended_face = Face(
                    bbox=faces[0].bbox,
                    kps=faces[0].kps,
                    det_score=faces[0].det_score,
                    landmark_3d_68=faces[0].landmark_3d_68,
                    pose=faces[0].pose,
                    landmark_2d_106=faces[0].landmark_2d_106,
                    embedding=blended_embedding,
                    gender=faces[0].gender,
                    age=faces[0].age
                )
                if blended_face is not None:
                    face_model_path = os.path.join(FACE_MODELS_PATH, face_model_name + ".safetensors")
                    save_face_model(blended_face,face_model_path)
                    logger.status("--Done!--")
                    # done_msg = f"Face model has been saved to '{face_model_path}'"
                    # logger.status(done_msg)
                    return face_model_name
                else:
                    no_face_msg = "Something went wrong, please try another set of images"
                    logger.error(no_face_msg)
                    return face_model_name
            logger.status("--Done!--")
        if images is None:
            logger.error("Please provide `images`")
        return face_model_name


class SaveFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "select_face_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def save_model(self, save_mode, face_model_name, select_face_index, image=None, face_model=None, det_size=(640, 640)):
        if save_mode and image is not None:
            source = tensor_to_pil(image)
            source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            apply_logging_patch(1)
            logger.status("Building Face Model...")
            face_model_raw = analyze_faces(source, det_size)
            if len(face_model_raw) == 0:
                det_size_half = half_det_size(det_size)
                face_model_raw = analyze_faces(source, det_size_half)
            try:
                face_model = face_model_raw[select_face_index]
            except:
                logger.error("No face(s) found")
                return face_model_name
            logger.status("--Done!--")
        if save_mode and (face_model != "none" or face_model is not None):
            face_model_path = os.path.join(self.output_dir, face_model_name + ".safetensors")
            save_face_model(face_model,face_model_path)
        if image is None and face_model is None:
            logger.error("Please provide `face_model` or `image`")
        return face_model_name


class RestoreFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),               
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "model": (get_model_names(get_restorers),),
                "visibility": ("FLOAT", {"default": 1, "min": 0.0, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        self.face_helper = None

    def execute(self, image, model, visibility, codeformer_weight, facedetection):
        result = reactor.restore_face(self,image,model,visibility,codeformer_weight,facedetection)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ReActorFaceSwap": reactor,
    "ReActorLoadFaceModel": LoadFaceModel,
    "ReActorSaveFaceModel": SaveFaceModel,
    "ReActorRestoreFace": RestoreFace,
    "ReActorBuildFaceModel": BuildFaceModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReActorFaceSwap": "ReActor - Fast Face Swap",
    "ReActorLoadFaceModel": "Load Face Model",
    "ReActorSaveFaceModel": "Save Face Model",
    "ReActorRestoreFace": "Restore Face",
    "ReActorBuildFaceModel": "Build Blended Face Model",
}
