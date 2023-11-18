import os, glob, sys
import shutil
from modules.processing import StableDiffusionProcessingImg2Img
from scripts.reactor_faceswap import FaceSwapScript, get_models, get_current_faces_model
from scripts.reactor_logger import logger
from reactor_utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil, img2tensor, tensor2img, move_path, save_face_model, load_face_model
from reactor_log_patch import apply_logging_patch

import model_management
import torch
import comfy.utils
import numpy as np
import cv2
# import math
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
# from facelib.detection.retinaface import retinaface
from torchvision.transforms.functional import normalize
from comfy_extras.chainner_models import model_loading
import folder_paths

models_dir = folder_paths.models_dir
REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")
if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_restorers():
    # basedir = os.path.abspath(os.path.dirname(__file__))
    # global MODELS_DIR
    # models_path = os.path.join(basedir, "models/facerestore_models/*")
    models_path = os.path.join(models_dir, "facerestore_models/*")
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

models_dir_old = os.path.join(os.path.dirname(__file__), "models")
old_dir_facerestore_models = os.path.join(models_dir_old, "facerestore_models")

dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = ([dir_facerestore_models], folder_paths.supported_pt_extensions)

if os.path.exists(old_dir_facerestore_models):
    move_path(old_dir_facerestore_models,dir_facerestore_models)
if os.path.exists(dir_facerestore_models) and os.path.exists(old_dir_facerestore_models):
    shutil.rmtree(old_dir_facerestore_models)

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
                # "coderformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.1}), # list(np.arange(0,1,0.1)
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "source_faces_index": ("STRING", {"default": "0"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
                "enable_threads":("BOOLEAN",{"default": True} ),
                "n_thread":("INT", {"default": 5,"min": 1,"step":1}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    FUNCTION = "execute"
    CATEGORY = "ReActor"

    def __init__(self):
        self.face_helper = None

    def execute(self, enabled, input_image, swap_model, detect_gender_source, detect_gender_input, source_faces_index, input_faces_index, console_log_level, face_restore_model, facedetection,enable_threads,n_thread, source_image=None, face_model=None):
        apply_logging_patch(console_log_level)

        if not enabled:
            return (input_image,face_model)
        elif source_image is None and face_model is None:
            logger.error("Please provide 'source_image' or `face_model`")
            # print("ReActor Node: Please provide 'source_image' or `face_model`")
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

        if n_thread==1:
            enable_threads=False

        if enable_threads:
            script.process_parallel(
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
                n_thread=n_thread,
            )
        else:            
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
            face_model_to_provide = get_current_faces_model()[0]
        else:
            face_model_to_provide = face_model

        # face restoration

        if face_restore_model != "none":

            logger.status(f"Restoring with {face_restore_model}")
            # print(f"Restoring with {face_restore_model}")

            # model_path = os.path.join(os.path.dirname(__file__), "models", "facerestore_models", face_restore_model)
            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            facerestore_model = model_loading.load_state_dict(sd).eval()

            device = model_management.get_torch_device()
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
                    return (result,face_model_to_provide)

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
                            output = facerestore_model(cropped_face_t)[0]
                            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except Exception as error:
                        print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

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

            return (restored_img_tensor,face_model_to_provide)
        
        else:
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
    CATEGORY = "ReActor"

    def load_model(self, face_model):
        self.face_model = face_model
        self.face_models_path = FACE_MODELS_PATH
        if self.face_model != "none":
            face_model_path = os.path.join(self.face_models_path, self.face_model)
            out = load_face_model(face_model_path)
        else:
            out = None
        return (out, )

class SaveFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model": ("FACE_MODEL",),
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"

    OUTPUT_NODE = True

    CATEGORY = "ReActor"

    def save_model(self, face_model, save_mode, face_model_name):
        if face_model != "none" and save_mode:
            face_model_path = os.path.join(self.output_dir, face_model_name + ".safetensors")
            save_face_model(face_model,face_model_path)
        return face_model_name


NODE_CLASS_MAPPINGS = {
    "ReActorFaceSwap": reactor,
    "ReActorLoadFaceModel": LoadFaceModel,
    "ReActorSaveFaceModel": SaveFaceModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReActorFaceSwap": "ReActor - Fast Face Swap",
    "ReActorLoadFaceModel": "Load Face Model",
    "ReActorSaveFaceModel": "Save Face Model",
}
