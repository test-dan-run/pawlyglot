""" Orchestrates all of the services together

# generate protobufs
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./backend \
    --pyi_out=./backend \
    --grpc_python_out=./backend \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/vc.proto \
    ./proto/mt.proto

"""

import os
import logging
import librosa
import numpy as np
import soundfile as sf
from typing import TypedDict

import machine_translation as mt
import speech_recognition as asr
import voice_activity as vad
import voice_cloning as vc

logging.basicConfig(level=logging.INFO)

TMP_DIR = "/tmp/pawlyglot/input_audio"
os.makedirs(TMP_DIR, exist_ok=True)

SAMPLE_RATE: int = 16000
LANGUAGE: str = "zh-cn"

HOST: str = "localhost"
VAD_PORT: int = 50052
ASR_PORT: int = 50053
MT_PORT: int = 50054
VC_PORT: int = 50055

class PawlyglotOutput(TypedDict):
    number: int
    start_time: float
    end_time: float
    transcription: str
    translation: str
    audio: np.ndarray

class MinPawlyglotOutput(TypedDict):
    start_time: float
    end_time: float
    transcription: str
    translation: str

# strings up asr, mt, and voice cloning calls
def combined_call(
        array: np.ndarray,
        target_language: str,
        start_time: float,
        end_time: float,
        embed_id: str,
        sample_rate: int
    ) -> PawlyglotOutput:

    asr_response = asr.asr_call(
        array, HOST, ASR_PORT
    )
    mt_response = mt.mt_call(
        asr_response["text"], HOST, MT_PORT
    )
    vc_response = vc.synthesize_call(
        mt_response["text"], embed_id, target_language, sample_rate, HOST, VC_PORT
    )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "transcription": asr_response["text"],
        "translation": mt_response["text"],
        "audio": vc_response["array"],
    }

def run_pipeline(audio_filepath: str):

    # load and convert audio into standardized sample rate
    audio_arr, _ = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
    tmp_audio_filepath = os.path.join(TMP_DIR, os.path.basename(audio_filepath))
    sf.write(tmp_audio_filepath, audio_arr, SAMPLE_RATE)

    # send audio file to vad service
    # TODO: have vad service take in numpy array instead
    # this is to prevent double loading of the same audio
    speech_timestamps = vad.vad_call(tmp_audio_filepath, HOST, VAD_PORT)
    silence_timestamps = [
        (
            speech_timestamps[i].end, 
            speech_timestamps[i+1].start,
        ) for i in range(len(speech_timestamps)-1)
    ]

    # send numpy array to vc embedding service
    # vc service will store the embeddings after processing
    audio_segments = [
        audio_arr[int(SAMPLE_RATE*ts.start):int(SAMPLE_RATE*ts.end)] for ts in speech_timestamps
        ]
    concat_audio = np.concatenate(audio_segments)
    silence_segments = [
        audio_arr[int(SAMPLE_RATE*ts[0]):int(SAMPLE_RATE*ts[1])] for ts in silence_timestamps
        ]

    embed_id = vc.embed_call(concat_audio, SAMPLE_RATE, HOST, VC_PORT)
    logging.info(f"Audio successfully embedded. ID: {embed_id}")

    text_concat = ""

    # set up calls for each audio segment, and execute the calls
    for idx, ts in enumerate(speech_timestamps):
        if idx == len(speech_timestamps):
            break
        start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end)
        segment = audio_arr[start:end]
        result = combined_call(segment, LANGUAGE, ts.start, ts.end, embed_id, SAMPLE_RATE)

        if idx != len(speech_timestamps)-1:
            output_array = np.concatenate([result["audio"], silence_segments[idx]])
        else:
            output_array = result["audio"]

        output_array = output_array / np.abs(output_array).max()
        output_array = (output_array * 32767).astype(np.int16)

        text_concat += f"""[{round(ts.start,2)}:{round(ts.end,2)}]\nTRANSCRIPT: {result["transcription"]}\nTRANSLATE : {result["translation"]}\n-----------\n"""

        yield [(SAMPLE_RATE, output_array), text_concat]

    # delete the embeddings to release vram
    vc.delete_call(embed_id)
    logging.info(f"Audio embedding successfully deleted. ID: {embed_id}")

if __name__ == "__main__":
    pass
