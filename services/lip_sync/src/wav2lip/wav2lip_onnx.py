import numpy as np
import onnxruntime as ort

class Wav2LipOnnx:

    def __init__(self, model_path: str):
        self.model = ort.InferenceSession(model_path)
        self._warmup()

    def _warmup(self):
        mock_mels = np.random.rand(8, 1, 80, 16).astype("float32")
        mock_imgs = np.random.rand(8, 6, 96, 96).astype("float32")

        _ = self.model.run(None, {
            "mel_spectrogram": mock_mels,
            "video_frames": mock_imgs})

    def infer(self, mels: np.ndarray, imgs: np.ndarray):
        """ inference """
        if len(mels.shape) == 3:
            mels = np.expand_dims(mels, axis=0)
        if len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, axis=0)
        outputs = self.model.run(None, {
            "mel_spectrogram": mels,
            "video_frames": imgs})

        return outputs[0]
