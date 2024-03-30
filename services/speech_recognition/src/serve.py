""" ASR server scripts """

import base64
import logging
import asyncio
from concurrent import futures

import grpc
import numpy as np
from faster_whisper import WhisperModel

import asr_pb2
import asr_pb2_grpc

MODEL_DIR = "/asr/models/small"
DEVICE = "cuda"  # cuda or cpu
COMPUTE_TYPE = "float32"

class ASRServer(asr_pb2_grpc.SpeechRecognizerServicer):
    """ASR Model Server instance"""

    def __init__(self):
        self.model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)

    def recognize(self, request_iterator: asr_pb2.RecognizeRequest, context):
        """Takes in audio file and generates transcription, returns string"""

        b64encoded = ""
        for chunk in request_iterator:
            b64encoded += chunk.buffer

        b64decoded = base64.b64decode(b64encoded)
        audio_array = np.frombuffer(b64decoded, dtype=np.float32)

        transcript = ""
        segments, _ = self.model.transcribe(audio_array, beam_size=5, language="en")
        for segment in segments:
            transcript += segment.text

        transcript = transcript.strip()

        return asr_pb2.RecognizeResponse(transcription=transcript)


async def serve():
    """ Serves ASR Model Asynchronously """

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=2))
    asr_pb2_grpc.add_SpeechRecognizerServicer_to_server(ASRServer(), server)
    server.add_insecure_port("[::]:50053")
    await server.start()

    print("Server started, listening on port 50053")
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
