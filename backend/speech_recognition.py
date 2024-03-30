import base64
from typing import TypedDict
import grpc
import numpy as np
import asr_pb2, asr_pb2_grpc

__all__ = ["asr_call"]

CHUNK_SIZE = 1024 * 1024

class ASROutput(TypedDict):
    number: int
    text: str

def get_encoded_chunks(encoded: str):
    idx = 0
    while True:
        piece = encoded[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
        if len(piece) == 0:
            return
        idx += 1
        yield asr_pb2.RecognizeRequest(buffer=piece)

async def asr_call(
        number: int, 
        array: np.ndarray,
        host: str = "localhost",
        port: int = 50053
    ) -> ASROutput:

    b64encoded = base64.b64encode(array)
    chunks_generator = get_encoded_chunks(b64encoded)

    async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
        stub = asr_pb2_grpc.SpeechRecognizerStub(channel)

        response = await stub.recognize(chunks_generator)

    return {"number": number, "text": response.transcription}
