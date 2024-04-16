import base64
import cv2
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
            yield ls_pb2.DetectRequest(buffer=piece)

def generate_id():
    # e.g. '201902-0309-0347-3de72c98-4004-45ab-980f-658ab800ec5d'
    return datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

def get_encoded_chunks(frame: np.ndarray):
    ret, buffer = cv2.imencode(".jpg", frame)
    if ret != 1:
        return
    
    yield ls_pb2.VideoTransferRequest(image_buffer=buffer.tobytes())

def video_transfer_call(
        video_filepath: str,
        host: str = "localhost",
        port: int = 50056
    ) -> str:

    video_stream = cv2.VideoCapture(video_filepath)
    embed_id = generate_id()
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = ls_pb2_grpc.LipSyncerStub(channel)
    
        while True:
            try:
                ret, frame = video_stream.read()
                if not ret:
                    video_stream.release()
                    break
                response = stub.videotransfer(
                    get_encoded_chunks(frame),
                    metadata=(
                        ("fps", str(fps)),
                        ("embed_id", embed_id),
                    ))
                if response.reply != 1:
                    print(response)
            except grpc.RpcError as e:
                print(e.details())

    return embed_id

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

# TODO
def lip_sync_call(
        audio_filepath: str,
        embed_id: str,
        host: str = "localhost",
        port: int = 50056
    ):

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
        chunks_generator = get_file_chunks(audio_filepath)
        response = stub.detect(
            chunks_generator,
            metadata=(("filename", os.path.basename(audio_filepath)),),
        )
    
    return response.timestamps

def synthesize_call(
            text: str,
            embed_id: str,
            language: str = "zh-cn",
            sample_rate: int = 16000,
            host: str = "localhost",
            port: int = 50055
        ) -> VCOutput:
    
    b64encoded = ""
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        for response in stub.synthesize(
            vc_pb2.SynthRequest(
                text=text, 
                language=language, 
                embed_id=embed_id, 
                sample_rate=sample_rate)):
            b64encoded += response.buffer
        
    b64decoded = base64.b64decode(b64encoded)
    arr = np.frombuffer(b64decoded, dtype=np.float32)

    return {"array": arr}

async def synthesize_async_call(
            number: int,
            text: str,
            embed_id: str,
            language: str = "zh-cn",
            sample_rate: int = 16000,
            host: str = "localhost",
            port: int = 50055
        ) -> VCOutput:
    
    b64encoded = ""
    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        for response in stub.synthesize(
            vc_pb2.SynthRequest(
                text=text, 
                language=language, 
                embed_id=embed_id, 
                sample_rate=sample_rate)):
            b64encoded += response.buffer
        
    b64decoded = base64.b64decode(b64encoded)
    arr = np.frombuffer(b64decoded, dtype=np.float32)

    return {"number": number, "array": arr}

def delete_call(
        embed_id: str,
        host: str = "localhost",
        port: int = 50055
        ) -> str:

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        stub.delete(vc_pb2.DeleteRequest(embed_id=embed_id))
