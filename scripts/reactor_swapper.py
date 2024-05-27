import copy
import os
import shutil
from dataclasses import dataclass
from typing import List, Union
import torch
from torchvision.transforms.functional import normalize

import cv2
import numpy as np
from PIL import Image

import insightface
from insightface.app.common import Face
try:
    import torch.cuda as cuda
except:
    cuda = None

from scripts.reactor_logger import logger
from reactor_utils import (
    move_path,
    get_image_md5hash,
    img2tensor,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face
)
import folder_paths
import comfy.model_management as model_management
from modules.shared import state

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

if cuda is not None:
    if cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

def get_restored_face(cropped_face,
                      face_restore_model,
                      face_restore_visibility,
                      codeformer_weight,):
    logger.status(f"Restoring with {face_restore_model}")

    faceSize = 512
    if "1024" in face_restore_model.lower():
        faceSize = 1024
    elif "2048" in face_restore_model.lower():
        faceSize = 2048

    scale = faceSize / cropped_face.shape[0]
    cropped_face = cv2.resize(cropped_face, (faceSize, faceSize),
                              interpolation=cv2.INTER_CUBIC)

    # For upscaling the base 128px face, I found bicubic interpolation to be the best compromise targeting antialiasing
    # and detail preservation. Nearest is predictably unusable, Linear produces too much aliasing, and Lanczos produces
    # too many hallucinations and artifacts/fringing.

    model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)
    device = model_management.get_torch_device()

    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

    try:

        with torch.no_grad():

            if ".onnx" in face_restore_model:  # ONNX models

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

                for ort_session_input in ort_session.get_inputs():
                    if ort_session_input.name == "input":
                        cropped_face_prep = prepare_cropped_face(cropped_face)
                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                    if ort_session_input.name == "weight":
                        weight = np.array([1], dtype=np.double)
                        ort_session_inputs[ort_session_input.name] = weight

                output = ort_session.run(None, ort_session_inputs)[0][0]
                restored_face = normalize_cropped_face(output)

            else:  # PTH models

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

                output = facerestore_model(cropped_face_t, w=codeformer_weight)[
                    0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        del output
        torch.cuda.empty_cache()

    except Exception as error:

        print(f"\tFailed inference: {error}", file=sys.stderr)
        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

    if face_restore_visibility < 1:
        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

    restored_face = restored_face.astype("uint8")
    return restored_face, scale

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # if order == "large-small":
    #     return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        order: str,
):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    # If index is outside of bounds, return None, avoid exception
    if face_index >= len(gender):
        logger.status("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(gender))
        return None, 0
    face_gender = gender[face_index]
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.status("OK - Detected Gender matches Condition")
        try:
            faces_sorted = sort_by_order(face, order)
            return faces_sorted[face_index], 0
            # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.status("WRONG - Detected Gender doesn't match Condition")
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 1
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 1

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel(det_size)
    faces = face_analyser.get(img_data)

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0


# The following code is almost entirely copied from INSwapper; the only change here is that we want to use Lanczos
# interpolation for the warpAffine call. Now that the face has been restored, Lanczos represents a good compromise
# whether the restored face needs to be upscaled or downscaled.
def in_swap(img, bgr_fake, M):
    target_img = img
    IM = cv2.invertAffineTransform(M)
    img_white = np.full((bgr_fake.shape[0], bgr_fake.shape[1]), 255, dtype=np.float32)

    # Note the use of Lanczos here; this is functionally the only change from the source code
    bgr_fake = cv2.warpAffine(bgr_fake, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0,
                              flags=cv2.INTER_LANCZOS4)

    img_white = cv2.warpAffine(img_white, IM, (target_img.shape[1], target_img.shape[0]), borderValue=0.0)
    img_white[img_white > 20] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))
    k = max(mask_size // 10, 10)
    # k = max(mask_size//20, 6)
    # k = 6
    kernel = np.ones((k, k), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    k = max(mask_size // 20, 5)
    # k = 3
    # k = 3
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
    k = 5
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask /= 255
    # img_mask = fake_diff
    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(np.float32)
    fake_merged = fake_merged.astype(np.uint8)
    return fake_merged


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    restore_immediately: bool = True,
    face_restore_model = None,
    face_restore_visibility = 1,
    codeformer_weight = 0.5,
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                result = target_img
                model_path = model_path = os.path.join(insightface_path, model)
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        if target_face is not None and wrong_gender == 0:
                            logger.status(f"Swapping...")
                            if restore_immediately:
                                logger.status(f"Immediate restore")
                                bgr_fake, M = face_swapper.get(result, target_face, source_face, paste_back=False)
                                bgr_fake, scale = get_restored_face(bgr_fake, face_restore_model, face_restore_visibility,
                                                             codeformer_weight)
                                M *= scale
                                result = in_swap(target_img, bgr_fake, M)
                            else:
                                logger.status(f"Swapping as-is")
                                result = face_swapper.get(result, target_face, source_face)
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            # Keep searching for other faces if wrong gender is detected, enhancement
                            #if source_face_idx == len(source_faces_index):
                            #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                            #    return result_image
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        # Keep searching for other faces if wrong gender is detected, enhancement
                        #if source_face_idx == len(source_faces_index):
                        #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        #    return result_image
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_image

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    result_images = target_imgs

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in target_imgs]

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_faces = []
            for i, target_img in enumerate(target_imgs):
                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    break
                
                target_image_md5hash = get_image_md5hash(target_img)
                if len(TARGET_IMAGE_LIST_HASH) == 0:
                    TARGET_IMAGE_LIST_HASH = [target_image_md5hash]
                    target_image_same = False
                elif len(TARGET_IMAGE_LIST_HASH) == i:
                    TARGET_IMAGE_LIST_HASH.append(target_image_md5hash)
                    target_image_same = False
                else:
                    target_image_same = True if TARGET_IMAGE_LIST_HASH[i] == target_image_md5hash else False
                    if not target_image_same:
                        TARGET_IMAGE_LIST_HASH[i] = target_image_md5hash
                
                logger.info("(Image %s) Target Image MD5 Hash = %s", i, TARGET_IMAGE_LIST_HASH[i])
                logger.info("(Image %s) Target Image the Same? %s", i, target_image_same)

                if len(TARGET_FACES_LIST) == 0:
                    logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST = [target_face]
                elif len(TARGET_FACES_LIST) == i and not target_image_same:
                    logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST.append(target_face)
                elif len(TARGET_FACES_LIST) != i and not target_image_same:
                    logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img)
                    TARGET_FACES_LIST[i] = target_face
                elif target_image_same:
                    logger.status("(Image %s) Using Hashed Target Face(s) Model...", i)
                    target_face = TARGET_FACES_LIST[i]
                

                # logger.status(f"Analyzing Target Image {i}...")
                # target_face = analyze_faces(target_img)
                if target_face is not None:
                    target_faces.append(target_face)

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                results = target_imgs
                model_path = model_path = os.path.join(insightface_path, model)
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        # Reading results to make current face swap on a previous face result
                        for i, (target_img, target_face) in enumerate(zip(results, target_faces)):
                            target_face_single, wrong_gender = get_face_single(target_img, target_face, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                            if target_face_single is not None and wrong_gender == 0:
                                logger.status(f"Swapping {i}...")
                                result = face_swapper.get(target_img, target_face_single, source_face)
                                results[i] = result
                            elif wrong_gender == 1:
                                wrong_gender = 0
                                logger.status("Wrong target gender detected")
                                continue
                            else:
                                logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_images
