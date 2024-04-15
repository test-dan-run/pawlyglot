import yaml
import subprocess
from typing import List, Tuple
import logging
logging.basicConfig(level=logging.INFO)

import cv2
import torch
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

from wav2lip.audio import melspectrogram
from wav2lip.wav2lip_onnx import Wav2LipOnnx

class LipSyncer:
    def __init__(self, config_path: str):

        with open(config_path, "r") as fr:
            cfg = yaml.safe_load(fr)

        self.device = cfg["device"]
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.image_resize_factor = cfg["image_resize_factor"]

        self.face_detector = MTCNN(
            keep_all=cfg["face"]["keep_all"],
            selection_method=cfg["face"]["selection_method"],
            device=self.device)
        self.face_pads = cfg["face"]["pads"]
        self.face_resize = (cfg["face"]["face_resize"], cfg["face"]["face_resize"])
        self.face_batch_size = cfg["face"]["batch_size"]
        
        self.wav2lip = Wav2LipOnnx(cfg["wav2lip"]["model_path"])
        self.sample_rate = cfg["wav2lip"]["sample_rate"]
        self.mel_step_size = cfg["wav2lip"]["mel_step_size"]
        self.wav2lip_batch_size = cfg["wav2lip"]["batch_size"]

        self.video_frames = {}

    def load_video(self, video_filepath: str) -> Tuple[List[np.ndarray], int]:

        video_stream = cv2.VideoCapture(video_filepath)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        video_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break

            if self.image_resize_factor > 1:
                frame = cv2.resize(frame,
                                   (frame.shape[1]//self.image_resize_factor, 
                                    frame.shape[0]//self.image_resize_factor))

            video_frames.append(frame)

        return video_frames, fps
    
    def store_video_frames(
            self, video_frames: List[np.ndarray], fps: int, idx: str = None) -> str:
        
        if idx is None:
            idx = str(len(self.video_frames) + 1)

        self.video_frames[idx] = {
            "frames": video_frames,
            "fps": fps,
            "height": video_frames[0].shape[0],
            "width": video_frames[0].shape[1]
        }

        return idx

    def generate_all_face_bboxes(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, List[List[int]]]:
        frames = [np.asarray(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames]
        bboxes = []

        with torch.no_grad():
            for i in tqdm(range(0, len(frames), self.face_batch_size)):
                boxes_batch, _ = self.face_detector.detect(frames[i:i+self.face_batch_size])
                bboxes.extend([boxes[0] for boxes in boxes_batch])

        def add_pads(rect, frame):
            rect = np.clip(rect, 0, None)
            x1, y1, x2, y2 = map(int, rect)

            y1 = max(0, y1 - self.face_pads["y1"])
            y2 = min(frame.shape[0], y2 + self.face_pads["y2"])
            x1 = max(0, x1 - self.face_pads["x1"])
            x2 = min(frame.shape[1], x2 + self.face_pads["x2"])
            return [y1, y2, x1, x2]

        coords = [add_pads(rect, frame) for rect, frame in zip(bboxes, frames)]
        if self.face_resize:
            faces = [
                cv2.resize(
                    frame[y1:y2, x1:x2],
                    self.face_resize) for frame, (y1, y2, x1, x2) in zip(frames, coords)]
        else:
            faces = [frame[y1:y2, x1:x2] for frame, (y1, y2, x1, x2) in zip(frames, coords)]

        return np.asarray(faces), coords

    def store_all_bboxes(self, idx: str, face_frames: np.ndarray, face_coords: List[List[int]]):

        self.video_frames[idx]["face_frames"] = face_frames
        self.video_frames[idx]["face_coords"] = face_coords

    def load_audio(self, audio_filepath: str) -> np.ndarray:
        wav, _ = librosa.load(audio_filepath, sr=self.sample_rate, mono=True)
        return wav
    
    def generate_all_mel_chunks(self, wav: np.ndarray, fps: int) -> List[np.ndarray]:

        mels = melspectrogram(wav)
        mel_chunks = []
        mel_idx_multiplier = 80./fps
        idx = 0
        while True:
            start_idx = int(idx * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mels[0]):
                mel_chunks.append(mels[:, len(mels[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mels[:, start_idx : start_idx + self.mel_step_size])
            idx += 1

        return mel_chunks

    def generate_all_outputs(
            self, 
            video_frames: List[np.ndarray],
            face_frames: List[np.ndarray],
            face_coords: List[np.ndarray],
            mel_chunks: List[np.ndarray]):

        num_chunks = len(mel_chunks)
        output_frames = []

        batch_size = self.wav2lip_batch_size
        for i in tqdm(range(0, num_chunks, batch_size)):
            frame_batch = np.asarray(video_frames[i:i+batch_size])
            face_batch = face_frames[i:i+batch_size]
            coords_batch = face_coords[i:i+batch_size]
            mel_batch = np.asarray(mel_chunks[i:i+batch_size])

            face_batch = face_batch[:,:,:,::-1]

            face_masked_batch = face_batch.copy()
            face_masked_batch[:, self.face_resize[0]//2:] = 0

            face_batch = np.concatenate((face_masked_batch, face_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [
                len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1
                ])

            face_batch = np.transpose(face_batch, (0,3,1,2)).astype("float32")
            mel_batch = np.transpose(mel_batch, (0,3,1,2)).astype("float32")     


            preds = self.wav2lip.infer(mels=mel_batch, imgs=face_batch)
            preds = preds.transpose(0,2,3,1) * 255.

            for p, f, c in zip(preds, frame_batch, coords_batch):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                output_frames.append(f)

        return output_frames
    
    def write_out(
            self,
            output_frames: List[np.ndarray],
            audio_path: np.ndarray,
            fps: int, width: int, height: int,
            output_path: str) -> None:

        video_out = cv2.VideoWriter(
            "/lipsync/temp/temp.avi",
            cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        for frame in output_frames:
            video_out.write(frame)

        video_out.release()

        command = f'ffmpeg -y -i {audio_path} -i /lipsync/temp/temp.avi -strict -2 -q:v 1 {output_path}'
        subprocess.call(command, shell=True)


if __name__ == "__main__":

    VIDEO_FILEPATH = ""
    AUDIO_FILEPATH = ""
    OUT_FILEPATH = ""

    lipsyncer = LipSyncer("config.yaml")
    import time

    vid_start = time.perf_counter()
    video_frames, fps = lipsyncer.load_video(VIDEO_FILEPATH)
    face_frames, face_coords = lipsyncer.generate_all_face_bboxes(video_frames)
    vid_end = time.perf_counter()

    aud_start = time.perf_counter()
    wav = lipsyncer.load_audio(AUDIO_FILEPATH)
    mel_chunks = lipsyncer.generate_all_mel_chunks(wav, fps)
    output_frames = lipsyncer.generate_all_outputs(video_frames, face_frames, face_coords, mel_chunks)
    aud_end = time.perf_counter()

    lipsyncer.write_out(
        output_frames,
        AUDIO_FILEPATH,
        fps,
        output_frames[0].shape[1],
        output_frames[0].shape[0],
        OUT_FILEPATH
    )


    vid_proc_time = round(vid_end-vid_start, 3)
    aud_proc_time = round(aud_end-aud_start, 3)
    proc_time = round(vid_proc_time+aud_proc_time, 3)
    audio_duration = librosa.get_duration(filename=AUDIO_FILEPATH)
    logging.info("Audio Length   : %s", audio_duration)
    logging.info("Face Detector Proc Time: %s", vid_proc_time)
    logging.info("Face Detector RTF      : %s", round(vid_proc_time/audio_duration, 3))    
    logging.info("Wav2Lip Proc Time: %s", aud_proc_time)
    logging.info("Wav2Lip RTF      : %s", round(aud_proc_time/audio_duration, 3))

    logging.info("Total Proc Time: %s", proc_time)
    logging.info("Total RTF      : %s", round(proc_time/audio_duration, 3))
