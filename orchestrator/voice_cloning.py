import base64
from typing import TypedDict
import grpc
import numpy as np
import vc_pb2, vc_pb2_grpc

__all__ = ["embed_call", "synthesis_call", "delete_call"]

CHUNK_SIZE = 1024 * 1024

class VCOutput(TypedDict):
    number: int
    array: np.ndarray

def get_encoded_chunks(encoded: str):
    idx = 0
    while True:
        piece = encoded[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
        if len(piece) == 0:
            return
        idx += 1
        yield vc_pb2.EmbedRequest(buffer=piece)

def embed_call(
        audio_array: np.ndarray, 
        sample_rate: int,
        host: str = "localhost",
        port: int = 50055
    ) -> str:

    b64encoded = base64.b64encode(audio_array)
    chunks_generator = get_encoded_chunks(b64encoded)

    with grpc.insecure_channel(f"{host}:{port}") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        response = stub.embed(
            chunks_generator,
            metadata=(("sample_rate", str(sample_rate)),
        ))
    
    return response.embed_id

def synthesize_call(
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
