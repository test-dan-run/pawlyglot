import os
import yaml
import subprocess
import tempfile
from typing import List, Tuple
import logging
from uuid import uuid4
from datetime import datetime
import asyncio
from concurrent import futures
import grpc
logging.basicConfig(level=logging.INFO)

import cv2
import torch
import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

import ls_pb2
import ls_pb2_grpc

from wav2lip.audio import melspectrogram
from wav2lip.wav2lip_onnx import Wav2LipOnnx

CHUNK_SIZE = 1024 * 1024

os.makedirs("/temp", exist_ok=True)

def save_chunks_to_file(chunks, filename):
    """writer incoming buffer chunks into file"""

    with open(filename, "wb") as f:
        for chunk in chunks:
            f.write(chunk.audio_buffer)

def get_encoded_chunks(encoded: str):

    idx = 0

    while True:
        piece = encoded[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
        if len(piece) == 0:
            return
        idx += 1
        yield ls_pb2.LipSyncResponse(buffer=piece)

def get_file_chunks(filepath: str):
    """ Splits audio file into chunks """

    with open(filepath, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield ls_pb2.LipSyncResponse(image_buffer=piece)

class LipSyncerServer(ls_pb2_grpc.LipSyncerServicer):
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

        return faces, coords

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

            face_batch = np.asarray(face_batch)[:,:,:,::-1]

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

        command = ['ffmpeg', '-y', '-i', audio_path , '-i', '/lipsync/temp/temp.avi', '-strict', '-2', '-c:v', 'h264', '-q:v', '1', output_path]
        subprocess.run(command)

    def _generate_id(self):
        # e.g. '201902-0309-0347-3de72c98-4004-45ab-980f-658ab800ec5d'
        return datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

    def videotransfer(self, request_iterator, context):
        metadata = context.invocation_metadata()

        filename = None
        for key, value in metadata:
            if key != "filename":
                continue
            filename = value
        assert isinstance(filename, str), "filename metadata not found in request."

        output_path = f"/temp/{filename}"
        embed_id = self._generate_id()

        total_buffer = bytearray()
        for request in request_iterator:
            total_buffer.extend(request.image_buffer)

        with open(output_path, 'wb') as f:
            f.write(total_buffer)

        video_frames, fps = self.load_video(output_path)
        self.store_video_frames(video_frames, fps, embed_id)

        return ls_pb2.VideoTransferResponse(embed_id=embed_id)

    def facedetect(self, request: ls_pb2.FaceDetectRequest, context):

        embed_id = request.embed_id
        face_frames, face_bboxes = self.generate_all_face_bboxes(self.video_frames[embed_id]["frames"])
        self.store_all_bboxes(embed_id, face_frames, face_bboxes)

        return ls_pb2.FaceDetectResponse(reply=1)


    def lipsync(self, request_iterator, context):

        metadata = context.invocation_metadata()

        embed_id = None
        for key, value in metadata:
            if key != "embed_id":
                continue
            embed_id = value
        assert isinstance(embed_id, str), "embed_id metadata not found in request."

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav")
        save_chunks_to_file(request_iterator, tmp_wav.name)
        logging.info("File loaded and saved to: %s", tmp_wav.name)

        target_vid = self.video_frames[embed_id]

        wav = self.load_audio(tmp_wav.name)
        mel_chunks = self.generate_all_mel_chunks(wav, target_vid["fps"])

        if len(mel_chunks) > len(target_vid["frames"]):
            num_to_append = len(target_vid["frames"]) - len(mel_chunks)

            last_half_frames = target_vid["frames"][-round(num_to_append/2):]
            last_half_frames_flip = last_half_frames.copy().reverse()

            target_vid["frames"] = target_vid["frames"] + last_half_frames_flip + last_half_frames

            last_half_face_frames = target_vid["face_frames"][-round(num_to_append/2):]
            last_half_face_frames_flip = last_half_face_frames.copy().reverse()

            target_vid["face_frames"] = target_vid["face_frames"] + last_half_face_frames_flip + last_half_face_frames

            last_half_face_coords = target_vid["face_coords"][-round(num_to_append/2):]
            last_half_face_coords_flip = last_half_face_coords.copy().reverse()

            target_vid["face_coords"] = target_vid["face_coords"] + last_half_face_coords_flip + last_half_face_coords

        target_vid["frames"] = target_vid["frames"][:len(mel_chunks)]
        target_vid["face_frames"] = target_vid["face_frames"][:len(mel_chunks)]
        target_vid["face_coords"] = target_vid["face_coords"][:len(mel_chunks)]

        output_frames = self.generate_all_outputs(
            target_vid["frames"],
            target_vid["face_frames"],
            target_vid["face_coords"],
            mel_chunks
        )

        self.write_out(
                output_frames,
                tmp_wav.name,
                target_vid["fps"],
                output_frames[0].shape[1],
                output_frames[0].shape[0],
                embed_id+".mp4"
            )

        chunks_generator = get_file_chunks(embed_id+".mp4")
        return chunks_generator

async def serve():
    """ Serves TTS Model """

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    ls_pb2_grpc.add_LipSyncerServicer_to_server(LipSyncerServer("/lipsync/src/config.yaml"), server)
    server.add_insecure_port("[::]:50056")
    server.start()

    print("Server started, listening on port 50056")
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
