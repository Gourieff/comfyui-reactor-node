import os, glob
from modules.processing import StableDiffusionProcessingImg2Img
from scripts.reactor_faceswap import FaceSwapScript, get_models
from reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    tensor_to_pil,
    img2tensor,
    tensor2img,
)
from reactor_log_patch import apply_logging_patch

import model_management
import torch
import comfy.utils
import numpy as np
import cv2

# import math
from facelib.utils.face_restoration_helper import FaceRestoreHelper

# from facelib.detection.retinaface import retinaface
from torchvision.transforms.functional import normalize
from comfy_extras.chainner_models import model_loading
import folder_paths


def get_restorers():
    models_path = os.path.join(folder_paths.models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".pth")]
    return models


def restorer_names():
    models = get_restorers()
    names = ["none"]
    for x in models:
        names.append(os.path.basename(x))
    return names


def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


dir_facerestore_models = os.path.join(folder_paths.models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)


class reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (
                    [
                        "retinaface_resnet50",
                        "retinaface_mobile0.25",
                        "YOLOv5l",
                        "YOLOv5n",
                    ],
                ),
                "face_restore_model": (restorer_names(),),
                "detect_gender_source": (["no", "female", "male"], {"default": "no"}),
                "detect_gender_input": (["no", "female", "male"], {"default": "no"}),
                "source_faces_index": ("STRING", {"default": "0"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    def __init__(self):
        self.face_helper = None

    def execute(
        self,
        source_image,
        input_image,
        swap_model,
        detect_gender_source,
        detect_gender_input,
        source_faces_index,
        input_faces_index,
        console_log_level,
        face_restore_model,
        facedetection,
    ):
        apply_logging_patch(console_log_level)

        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)
        source = tensor_to_pil(source_image)
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
        )
        result = batched_pil_to_tensor(p.init_images)

        # face restoration

        if face_restore_model != "none":
            model_path = os.path.join(
                folder_paths.models_dir,
                "facerestore_models",
                face_restore_model,
            )
            sd = comfy.utils.load_torch_file(model_path, safe_load=True)
            facerestore_model = model_loading.load_state_dict(sd).eval()

            device = model_management.get_torch_device()
            facerestore_model.to(device)
            if self.face_helper is None:
                self.face_helper = FaceRestoreHelper(
                    1,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model=facedetection,
                    save_ext="png",
                    use_parse=True,
                    device=device,
                )

            image_np = 255.0 * result.cpu().numpy()

            total_images = image_np.shape[0]
            out_images = np.ndarray(shape=image_np.shape)

            for i in range(total_images):
                cur_image_np = image_np[i, :, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return (result,)

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(
                    only_center_face=False, resize=640, eye_dist_threshold=5
                )
                self.face_helper.align_warp_face()

                restored_face = None
                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                    cropped_face_t = img2tensor(
                        cropped_face / 255.0, bgr2rgb=True, float32=True
                    )
                    normalize(
                        cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True
                    )
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:
                        with torch.no_grad():
                            output = facerestore_model(cropped_face_t)[0]
                            restored_face = tensor2img(
                                output, rgb2bgr=True, min_max=(-1, 1)
                            )
                        del output
                        torch.cuda.empty_cache()
                    except Exception as error:
                        print(
                            f"\tFailed inference for CodeFormer: {error}",
                            file=sys.stderr,
                        )
                        restored_face = tensor2img(
                            cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                        )

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(
                        restored_img,
                        (0, 0),
                        fx=original_resolution[1] / restored_img.shape[1],
                        fy=original_resolution[0] / restored_img.shape[0],
                        interpolation=cv2.INTER_LINEAR,
                    )

                self.face_helper.clean_all()

                out_images[i] = restored_img

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            return (restored_img_tensor,)

        else:
            return (result,)


NODE_CLASS_MAPPINGS = {
    "ReActorFaceSwap": reactor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReActorFaceSwap": "ReActor - Fast Face Swap",
}
