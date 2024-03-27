""" Script to test VAD model serving instance 

# generate protobufs
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./tests \
    --pyi_out=./tests \
    --grpc_python_out=./tests \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/mt.proto

"""

import os
import base64
import logging
import librosa
import numpy as np
from typing import Dict, Any

import grpc
import asr_pb2, asr_pb2_grpc
import vad_pb2, vad_pb2_grpc
import mt_pb2, mt_pb2_grpc

CHUNK_SIZE = 1024 * 1024
SAMPLE_AUDIO_PATH = "../examples/test_audio.wav"
SAMPLE_RATE = 16000
""" Script to test VAD model serving instance 

# generate protobufs
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./tests \
    --pyi_out=./tests \
    --grpc_python_out=./tests \
    ./proto/base.proto \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/mt.proto

"""

import os
import base64
import asyncio
import logging
import librosa

import grpc
import asr_pb2, asr_pb2_grpc
import vad_pb2, vad_pb2_grpc
import mt_pb2, mt_pb2_grpc

CHUNK_SIZE = 1024 * 1024
SAMPLE_AUDIO_PATH = "../examples/test_audio.wav"
SAMPLE_RATE = 16000

async def transcribe(
        number: int, 
        array: np.ndarray, 
        ts: vad_pb2.Timestamp
    ) -> Dict[str, Any]:

    start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end) 
    segment = array[start:end]

    b64encoded = base64.b64encode(segment)
    chunks_generator = get_encoded_chunks(b64encoded)

    async with grpc.aio.insecure_channel("localhost:50053") as channel:
        stub = asr_pb2_grpc.SpeechRecognizerStub(channel)

        response = await stub.recognize(chunks_generator)
        # response = await asyncio.wait_for(stub.recognize(chunks_generator), 10)

    return {"number": number, "text": response.transcription}

async def translate(number: int, text: str) -> Dict[str, Any]:

    async with grpc.aio.insecure_channel("localhost:50054") as channel:
        stub = mt_pb2_grpc.TranslatorStub(channel)

        response = await stub.translate(mt_pb2.TranslateRequest(text=text))
        # response = await asyncio.wait_for(stub.translate(mt_pb2.TranslateRequest(text=text)), 10)

    return {"number": number, "text": response.translation}

def get_file_chunks(filepath: str):
    """ Splits audio file into chunks """

    with open(filepath, "rb") as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield vad_pb2.DetectRequest(buffer=piece)

def get_encoded_chunks(encoded: str):

    idx = 0

    while True:
        piece = encoded[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
        if len(piece) == 0:
            return
        idx += 1
        yield asr_pb2.RecognizeRequest(buffer=piece)

async def run():
    """ Sends audio file to model serving instance """

    transcriptions = {}
    transcription_tasks = []
    def process_transcription_response(response: str):
        logging.info(f"Received response for {response}")
        transcriptions[response["number"]] = response["text"]

    translations = {}
    translation_tasks = []
    def process_translation_response(response: str):
        logging.info(f"Received response for {response}")
        translations[response["number"]] = response["text"]

    # VAD
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
        chunks_generator = get_file_chunks(SAMPLE_AUDIO_PATH)
        response = stub.detect(
            chunks_generator,
            metadata=(("filename", os.path.basename(SAMPLE_AUDIO_PATH)),),
        )

    # decode bytes to array, reshape is needed as frombuffer outputs 1d-array
    timestamps = response.timestamps
    audio_arr, _ = librosa.load(SAMPLE_AUDIO_PATH, sr=SAMPLE_RATE, mono=True)

    # ASR
    for i in range(len(timestamps)):
        task = asyncio.create_task(transcribe(i, audio_arr, timestamps[i]))
        task.add_done_callback(lambda t: process_transcription_response(t.result()))

        transcription_tasks.append(task)

    await asyncio.gather(*transcription_tasks, return_exceptions=True)

    # Translate
    for i in range(len(timestamps)):
        task = asyncio.create_task(translate(i, transcriptions[i]))
        task.add_done_callback(lambda t: process_translation_response(t.result()))

        translation_tasks.append(task)

    await asyncio.gather(*translation_tasks, return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
