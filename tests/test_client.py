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
import logging
import librosa

import grpc
import asr_pb2, asr_pb2_grpc
import vad_pb2, vad_pb2_grpc
import mt_pb2, mt_pb2_grpc

CHUNK_SIZE = 1024 * 1024
SAMPLE_AUDIO_PATH = "../examples/test_audio.wav"
SAMPLE_RATE = 16000

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

def run():
    """ Sends audio file to model serving instance """

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
    transcriptions = []
    with grpc.insecure_channel("localhost:50053") as channel:
        stub = asr_pb2_grpc.SpeechRecognizerStub(channel)

        for ts in timestamps:
            start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end) 
            segment = audio_arr[start:end]

            b64encoded = base64.b64encode(segment)
            chunks_generator = get_encoded_chunks(b64encoded)
            response = stub.recognize(chunks_generator)
            transcriptions.append(response.transcription)

    # Translate
    translations = []
    with grpc.insecure_channel("localhost:50054") as channel:
        stub = mt_pb2_grpc.TranslatorStub(channel)

        for ts in transcriptions:
            response = stub.translate(mt_pb2.TranslateRequest(text=ts))
            translations.append(response.translation)
            print(response.translation)



if __name__ == "__main__":
    logging.basicConfig()
    run()
