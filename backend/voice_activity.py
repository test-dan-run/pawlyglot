import os
from typing import List
import grpc
import vad_pb2, vad_pb2_grpc

__all__ = ["vad_call"]

CHUNK_SIZE = 1024 * 1024

def get_file_chunks(filepath: str):
    """ Splits audio file into chunks """

    with open(filepath, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield vad_pb2.DetectRequest(buffer=piece)

def vad_call(
        audio_filepath: str, 
        host: str = "localhost", 
        port: int = 50052
    ) -> List[vad_pb2.Timestamp]:

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
        chunks_generator = get_file_chunks(audio_filepath)
        response = stub.detect(
            chunks_generator,
            metadata=(("filename", os.path.basename(audio_filepath)),),
        )
    
    return response.timestamps
