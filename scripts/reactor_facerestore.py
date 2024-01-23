import torch
from torchvision.transforms.functional import normalize
import numpy as np
import cv2

from comfy_extras.chainner_models import model_loading
import model_management
import comfy.utils
import folder_paths

from scripts.reactor_logger import logger
from reactor_utils import img2tensor, tensor2img
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY
import scripts.r_archs.codeformer_arch


def restore_face(
        face_helper,
        image,
        model,
        visibility,
        codeformer_weight,
        facedetection,
        face_model_to_provide=None
    ):

    result = image

    if model != "none":

        logger.status(f"Restoring with {model}")

        model_path = folder_paths.get_full_path("facerestore_models", model)

        device = model_management.get_torch_device()
        
        if "codeformer" in model.lower():
            
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
        
        if face_helper is None:
            face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)

        image_np = 255. * result.cpu().numpy()

        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=image_np.shape)

        for i in range(total_images):
            cur_image_np = image_np[i,:, :, ::-1]

            original_resolution = cur_image_np.shape[0:2]

            if facerestore_model is None or face_helper is None:
                if face_model_to_provide is not None:
                    return (result,face_model_to_provide)
                else:
                    return (result,)

            face_helper.clean_all()
            face_helper.read_image(cur_image_np)
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            restored_face = None
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in model.lower() else facerestore_model(cropped_face_t)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                
                if visibility < 1:
                    restored_face = cropped_face * (1 - visibility) + restored_face * visibility
                
                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face)

            face_helper.get_inverse_affine(None)

            restored_img = face_helper.paste_faces_to_input_image()
            restored_img = restored_img[:, :, ::-1]

            if original_resolution != restored_img.shape[0:2]:
                restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

            face_helper.clean_all()

            out_images[i] = restored_img

        restored_img_np = np.array(out_images).astype(np.float32) / 255.0
        restored_img_tensor = torch.from_numpy(restored_img_np)

        result = restored_img_tensor

    if face_model_to_provide is not None:
        return (result,face_model_to_provide)
    else:
        return (result,)
