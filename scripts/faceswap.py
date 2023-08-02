import modules.scripts as scripts
from modules.upscaler import Upscaler, UpscalerData
from modules import scripts, shared, images, scripts_postprocessing
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from PIL import Image
import glob
from modules.face_restoration import FaceRestoration

from scripts.logger import logger
from scripts.swapper import UpscaleOptions, swap_face
from scripts.version import version_flag, app_title
import os

def get_models():
    models_path = os.path.join(scripts.basedir(), "models/roop/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models


class FaceSwapScript(scripts.Script):
    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None

    @property
    def upscale_options(self) -> UpscaleOptions:
        return UpscaleOptions(
            do_restore_first = self.restore_first,
            scale=self.upscaler_scale,
            upscaler=self.upscaler,
            face_restorer=self.face_restorer,
            upscale_visibility=self.upscaler_visibility,
            restorer_visibility=self.face_restorer_visibility,
        )

    def process(
        self,
        p: StableDiffusionProcessing,
        img,
        enable,
        source_faces_index,
        faces_index,
        model,
        face_restorer_name,
        face_restorer_visibility,
        restore_first,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        swap_in_source,
        swap_in_generated,
    ):
        self.source = img
        self.face_restorer_name = face_restorer_name
        self.upscaler_scale = upscaler_scale
        self.upscaler_visibility = upscaler_visibility
        self.face_restorer_visibility = face_restorer_visibility
        self.enable = enable
        self.restore_first = restore_first
        self.upscaler_name = upscaler_name       
        self.swap_in_generated = swap_in_generated
        self.model = model
        self.source_faces_index = [
            int(x) for x in source_faces_index.strip(",").split(",") if x.isnumeric()
        ]
        self.faces_index = [
            int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
        ]
        if len(self.source_faces_index) == 0:
            self.source_faces_index = [0]
        if len(self.faces_index) == 0:
            self.faces_index = [0]
        if self.enable:
            if self.source is not None:
                if isinstance(p, StableDiffusionProcessingImg2Img) and swap_in_source:
                    logger.info(f"Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)

                    for i in range(len(p.init_images)):
                        logger.info(f"Swap in %s", i)
                        result = swap_face(
                            self.source,
                            p.init_images[i],
                            source_faces_index=self.source_faces_index,
                            faces_index=self.faces_index,
                            model=self.model,
                            upscale_options=self.upscale_options,
                        )
                        p.init_images[i] = result
            else:
                logger.error(f"Please provide a source face")

    def postprocess_batch(self, p, *args, **kwargs):
        if self.enable:
            images = kwargs["images"]

    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.enable and self.swap_in_generated:
            if self.source is not None:
                logger.info(f"Working: source face index %s, target face index %s", self.source_faces_index, self.faces_index)
                image: Image.Image = script_pp.image
                result = swap_face(
                    self.source,
                    image,
                    source_faces_index=self.source_faces_index,
                    faces_index=self.faces_index,
                    model=self.model,
                    upscale_options=self.upscale_options,
                )
                try:
                    pp = scripts_postprocessing.PostprocessedImage(result)
                    pp.info = {}
                    p.extra_generation_params.update(pp.info)
                    script_pp.image = pp.image
                except:
                    logger.error(f"Cannot create a result image")
