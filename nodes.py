import os
from modules.processing import StableDiffusionProcessingImg2Img
from scripts.faceswap import FaceSwapScript, get_models
from utils import batch_tensor_to_pil, batched_pil_to_tensor, tensor_to_pil
from console_log_patch import apply_logging_patch


def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


class reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "reference_faces_index": ("STRING", {"default": "0"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/postprocessing"

    def execute(self, input_image, reference_image, swap_model, reference_faces_index, input_faces_index, console_log_level):
        apply_logging_patch(console_log_level)

        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)
        source = tensor_to_pil(reference_image)
        p = StableDiffusionProcessingImg2Img(pil_images)
        script.process(
            p=p, img=source, enable=True, source_faces_index=reference_faces_index, faces_index=input_faces_index, model=swap_model, face_restorer_name='None', face_restorer_visibility=1, restore_first=True,upscaler_name=None, upscaler_scale=1, upscaler_visibility=1, swap_in_source=True, swap_in_generated=True
        )
        result = batched_pil_to_tensor(p.init_images)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "ReActorFaceSwap": reactor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ReActorFaceSwap": "ReActor - Fast Face Swap",
}