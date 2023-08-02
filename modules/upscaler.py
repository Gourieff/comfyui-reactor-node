class Upscaler:

    def upscale(self, img, scale, selected_model: str = None):
        pass


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()
