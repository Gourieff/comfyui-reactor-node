import copy
import os
from dataclasses import dataclass
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

import insightface

from scripts.reactor_logger import logger

import warnings

np.warnings = warnings
np.warnings.filterwarnings("ignore")
import folder_paths

providers = ["CPUExecutionProvider"]

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
swapper_path = os.path.join(models_path, "insightface")


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODEL = None


def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    return ANALYSIS_MODEL


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def get_face_gender(face, face_index, gender_condition, operated: str):
    gender = [x.sex for x in face]
    gender.reverse()
    face_gender = gender[face_index]
    logger.info("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (
        gender_condition == 2 and face_gender == "M"
    ):
        logger.info("OK - Detected Gender matches Condition")
        try:
            return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.info("WRONG - Detected Gender doesn't match Condition")
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 1


def reget_face_single(img_data, det_size, face_index):
    det_size_half = (det_size[0] // 2, det_size[1] // 2)
    return get_face_single(img_data, face_index=face_index, det_size=det_size_half)


def get_face_single(
    img_data: np.ndarray,
    face_index=0,
    det_size=(640, 640),
    gender_source=0,
    gender_target=0,
):
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            return reget_face_single(img_data, det_size, face_index)
        return get_face_gender(face, face_index, gender_source, "Source")

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            return reget_face_single(img_data, det_size, face_index)
        return get_face_gender(face, face_index, gender_target, "Target")

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        return reget_face_single(img_data, det_size, face_index)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
):
    result_image = target_img

    if model is not None:
        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io

            if (
                "base64," in source_img
            ):  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split("base64,")[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)

            source_img = Image.open(io.BytesIO(img_bytes))

        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        source_face, wrong_gender = get_face_single(
            source_img, face_index=source_faces_index[0], gender_source=gender_source
        )

        if (
            len(source_faces_index) != 0
            and len(source_faces_index) != 1
            and len(source_faces_index) != len(faces_index)
        ):
            logger.info(
                f"Source Faces must have no entries (default=0), one entry, or same number of entries as target faces."
            )
        elif source_face is not None:
            result = target_img
            model_path = model_path = os.path.join(swapper_path, model)
            face_swapper = getFaceSwapModel(model_path)

            source_face_idx = 0

            for face_num in faces_index:
                if len(source_faces_index) > 1 and source_face_idx > 0:
                    source_face, wrong_gender = get_face_single(
                        source_img,
                        face_index=source_faces_index[source_face_idx],
                        gender_source=gender_source,
                    )
                source_face_idx += 1

                if source_face is not None and wrong_gender == 0:
                    target_face, wrong_gender = get_face_single(
                        target_img, face_index=face_num, gender_target=gender_target
                    )
                    if target_face is not None and wrong_gender == 0:
                        result = face_swapper.get(result, target_face, source_face)
                    elif wrong_gender == 1:
                        wrong_gender = 0
                        if source_face_idx == len(source_faces_index):
                            result_image = Image.fromarray(
                                cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                            )
                            return result_image
                    else:
                        logger.info(f"No target face found for {face_num}")
                elif wrong_gender == 1:
                    wrong_gender = 0
                    if source_face_idx == len(source_faces_index):
                        result_image = Image.fromarray(
                            cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        )
                        return result_image
                else:
                    logger.info(
                        f"No source face found for face number {source_face_idx}."
                    )

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        else:
            logger.info("No source face(s) found")
    return result_image
