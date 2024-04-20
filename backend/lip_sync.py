import base64
import cv2
import os
from typing import TypedDict
import grpc
import numpy as np
import ls_pb2, ls_pb2_grpc
from uuid import uuid4
from datetime import datetime

# __all__ = ["embed_call", "synthesize_call", "synthesize_async_call", "delete_call"]

CHUNK_SIZE = 1024 * 1024

def get_file_chunks(filepath: str):
    """ Splits audio file into chunks """

    with open(filepath, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield ls_pb2.VideoTransferRequest(image_buffer=piece)

def get_audio_file_chunks(filepath: str):
    """ Splits audio file into chunks """

    with open(filepath, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield ls_pb2.LipSyncRequest(audio_buffer=piece)

def generate_id():
    # e.g. '201902-0309-0347-3de72c98-4004-45ab-980f-658ab800ec5d'
    return datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

def video_transfer_call(
        video_filepath: str,
        host: str = "localhost",
        port: int = 50056
    ) -> str:

    # video_stream = cv2.VideoCapture(video_filepath)
    # embed_id = generate_id()
    # fps = video_stream.get(cv2.CAP_PROP_FPS)

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = ls_pb2_grpc.LipSyncerStub(channel)
        response = stub.videotransfer(
            get_file_chunks(video_filepath),
            metadata=(
                ("filename", os.path.basename(video_filepath)),
            ))

    return response.embed_id

def face_detect_call(
        embed_id: str,
        host: str = "localhost",
        port: int = 50056
    ):

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = ls_pb2_grpc.LipSyncerStub(channel)
        response = stub.facedetect(ls_pb2.FaceDetectRequest(embed_id=embed_id))

    if response.reply != 1:
        print(response)
        return 0

    return 1

def lip_sync_call(
        audio_filepath: str,
        embed_id: str,
        host: str = "localhost",
        port: int = 50056
    ):

    filepath = f"out_{embed_id}.mp4"

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = ls_pb2_grpc.LipSyncerStub(channel)
        chunks_generator = get_audio_file_chunks(audio_filepath)
        for response in stub.lipsync(
            chunks_generator,
            metadata=(("embed_id", embed_id),),
        ):
            with open(filepath, mode="ab") as f:
                f.write(response.image_buffer)

    return filepath