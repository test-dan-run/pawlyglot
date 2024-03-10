""" Script to test VAD model serving instance 

# generate protobufs
python3 -m grpc_tools.protoc -I ../proto \
    --python_out=. \
    --pyi_out=. \
    --grpc_python_out=. \
    ../proto/vad.proto

"""

import os
import base64
import logging

import grpc
import numpy as np

import vad_pb2
import vad_pb2_grpc

CHUNK_SIZE = 1024 * 1024
SAMPLE_AUDIO_PATH = "examples/test_audio.wav"


def get_file_chunks(filepath: str):
    """ Splits audio file into chunks """

    with open(filepath, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield vad_pb2.DetectRequest(buffer=piece)


def run():
    """ Sends audio file to model serving instance """

    with grpc.insecure_channel("localhost:50052") as channel:
        stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
        chunks_generator = get_file_chunks(SAMPLE_AUDIO_PATH)
        response = stub.detect(
            chunks_generator,
            metadata=(("filename", os.path.basename(SAMPLE_AUDIO_PATH)),),
        )

        # decode bytes to array, reshape is needed as frombuffer outputs 1d-array
        boundaries_buffer = base64.b64decode(response.b64array)
        boundaries = np.frombuffer(boundaries_buffer, np.float32).reshape((-1, 2))
        print(boundaries)


if __name__ == "__main__":
    logging.basicConfig()
    run()
