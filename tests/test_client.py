""" Script to test VAD model serving instance 

# generate protobufs
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./tests \
    --pyi_out=./tests \
    --grpc_python_out=./tests \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/vc.proto \
    ./proto/mt.proto

"""

import os
import json
import base64
import logging
import asyncio
import librosa
import numpy as np
import soundfile as sf
from typing import Dict, List, Any, Union

import grpc
import asr_pb2, asr_pb2_grpc
import vad_pb2, vad_pb2_grpc
import mt_pb2, mt_pb2_grpc
import vc_pb2, vc_pb2_grpc

CHUNK_SIZE = 1024 * 1024
SAMPLE_AUDIO_PATH = "../examples/test_audio.wav"
SAMPLE_RATE = 16000

async def transcribe(
        number: int, 
        array: np.ndarray, 
    ) -> Dict[str, Any]:

    b64encoded = base64.b64encode(array)
    chunks_generator = get_encoded_chunks(b64encoded)

    async with grpc.aio.insecure_channel("localhost:50053") as channel:
        stub = asr_pb2_grpc.SpeechRecognizerStub(channel)

        response = await stub.recognize(chunks_generator)
        # response = await asyncio.wait_for(stub.recognize(chunks_generator), 10)

    return {"number": number, "text": response.transcription}

async def translate(number: int, text: str) -> Dict[str, Union[int, str]]:

    async with grpc.aio.insecure_channel("localhost:50054") as channel:
        stub = mt_pb2_grpc.TranslatorStub(channel)

        response = await stub.translate(mt_pb2.TranslateRequest(text=text))
        # response = await asyncio.wait_for(stub.translate(mt_pb2.TranslateRequest(text=text)), 10)

    return {"number": number, "text": response.translation}

def embed(audio_array: np.ndarray) -> str:

    b64encoded = base64.b64encode(audio_array)
    chunks_generator = get_encoded_chunks(b64encoded)

    with grpc.insecure_channel("localhost:50055") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        response = stub.embed(
            chunks_generator,
            metadata=(("sample_rate", str(SAMPLE_RATE)),
        ))
    
    return response.embed_id

def synthesize(
            number: int, 
            text: str, 
            embed_id: str, 
            language: str = "zh-cn",
            target_sr: int = 16000,
        ) -> Dict[str, Union[int, np.ndarray]]:
    
    b64encoded = ""
    with grpc.insecure_channel("localhost:50055") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        for response in stub.synthesize(vc_pb2.SynthRequest(text=text, language=language, embed_id=embed_id, sample_rate=target_sr)):
            b64encoded += response.buffer
        
    b64decoded = base64.b64decode(b64encoded)
    arr = np.frombuffer(b64decoded, dtype=np.float32)

    return {"number": number, "array": arr}

def delete_embeddings(embed_id: str) -> str:

    with grpc.insecure_channel("localhost:50055") as channel:
        stub = vc_pb2_grpc.VoiceClonerStub(channel)
        response = stub.delete(vc_pb2.DeleteRequest(embed_id=embed_id))

async def pipeline(number: int, array: np.ndarray, timestamp: vad_pb2.Timestamp, embed_id: str, target_sr: int) -> Dict[str, Any]:

    asr_response = await transcribe(number, array)
    mt_response = await translate(number, asr_response["text"])
    vc_response = synthesize(number, mt_response["text"], embed_id, target_sr=target_sr)

    return {
        "number": number,
        "start_time": timestamp.start,
        "end_time": timestamp.end,
        "transcription": asr_response["text"],
        "translation": mt_response["text"],
        "audio": vc_response["array"],
    }

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

    results = {}
    tasks = []

    def process_pipeline_response(response: Dict[str, Any]):
        number = response.pop("number")

        logging.info(f"Received response {number}, content: {response}")
        results[number] = response

    # VAD
    with grpc.insecure_channel("localhost:50052") as channel:
        stub = vad_pb2_grpc.VoiceActivityDetectorStub(channel)
        chunks_generator = get_file_chunks(SAMPLE_AUDIO_PATH)
        response = stub.detect(
            chunks_generator,
            metadata=(("filename", os.path.basename(SAMPLE_AUDIO_PATH)),),
        )

    # Embed
    audio_arr, _ = librosa.load(SAMPLE_AUDIO_PATH, sr=SAMPLE_RATE, mono=True)
    audio_segments = [audio_arr[int(SAMPLE_RATE*ts.start):int(SAMPLE_RATE*ts.end)] for ts in response.timestamps]
    concat_speech = np.concatenate(audio_segments)
    
    embed_id = embed(concat_speech)
    logging.info(f"Voice embedded, id: {embed_id}")

    for idx, ts in enumerate(response.timestamps):
        start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end) 
        segment = audio_arr[start:end]

        task = asyncio.create_task(pipeline(idx, segment, ts, embed_id, SAMPLE_RATE))
        task.add_done_callback(lambda t: process_pipeline_response(t.result()))

        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)
    _ = delete_embeddings(embed_id)

    audio_arrays = [results[i].pop("audio") for i in range(len(results))]
    output_array = np.concatenate(audio_arrays)
    sf.write("output_cloned.wav", output_array, SAMPLE_RATE)

    with open("outputs.json", mode="w") as fw:
        # resort
        results = {k: v for k, v in sorted(results.items(), key=lambda item: item[0])}
        json.dump(results, fw, indent=2)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
