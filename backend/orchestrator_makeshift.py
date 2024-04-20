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
import subprocess
import numpy as np
import soundfile as sf
from uuid import uuid4
from typing import TypedDict
import concurrent.futures

import machine_translation as mt
import speech_recognition as asr
import voice_activity as vad
import voice_cloning as vc
import lip_sync as ls

logging.basicConfig(level=logging.INFO)

TMP_DIR = "/tmp/pawlyglot/input_audio"
os.makedirs(TMP_DIR, exist_ok=True)

SAMPLE_RATE: int = 16000
LANGUAGE: str = "zh-cn"

VAD_HOST: str = "vad"
ASR_HOST: str = "asr"
MT_HOST: str = "mt"
VC_HOST: str = "vc"
LS_HOST: str = "ls"

VAD_PORT: int = 50052
ASR_PORT: int = 50053
MT_PORT: int = 50054
VC_PORT: int = 50055
LS_PORT: int = 50056

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
def combined_audio_ml_call(
        array: np.ndarray,
        target_language: str,
        start_time: float,
        end_time: float,
        embed_id: str,
        sample_rate: int
    ) -> PawlyglotOutput:

    asr_response = asr.asr_call(
        array, ASR_HOST, ASR_PORT
    )
    mt_response = mt.mt_call(
        asr_response["text"], MT_HOST, MT_PORT
    )
    vc_response = vc.synthesize_call(
        mt_response["text"], embed_id, target_language, sample_rate, VC_HOST, VC_PORT
    )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "transcription": asr_response["text"],
        "translation": mt_response["text"],
        "audio": vc_response["array"],
    }

def run_audio_pipeline(video_filepath: str) -> str:

    audio_filepath = "temp.wav"
    subprocess.run(["ffmpeg", "-y", "-i", video_filepath, "-ar", "16000", "-ac", "1", audio_filepath])

    # load and convert audio into standardized sample rate
    audio_arr, _ = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
    tmp_audio_filepath = os.path.join(TMP_DIR, "standardized_" + os.path.basename(audio_filepath))
    sf.write(tmp_audio_filepath, audio_arr, SAMPLE_RATE)

    # send audio file to vad service
    # TODO: have vad service take in numpy array instead
    # this is to prevent double loading of the same audio
    speech_timestamps = vad.vad_call(tmp_audio_filepath, VAD_HOST, VAD_PORT)
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

    embed_id = vc.embed_call(concat_audio, SAMPLE_RATE, VC_HOST, VC_PORT)
    logging.info(f"Audio successfully embedded. ID: {embed_id}")

    output_arrays = []

    # set up calls for each audio segment, and execute the calls
    for idx, ts in enumerate(speech_timestamps):
        if idx == len(speech_timestamps):
            break
        start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end)
        segment = audio_arr[start:end]
        result = combined_audio_ml_call(segment, LANGUAGE, ts.start, ts.end, embed_id, SAMPLE_RATE)

        if idx != len(speech_timestamps)-1:
            output_array = np.concatenate([result["audio"], silence_segments[idx]])
        else:
            output_array = result["audio"]

        output_array = output_array / np.abs(output_array).max()
        output_array = (output_array * 32767).astype(np.int16)
        output_arrays.append(output_array)

    output_array = np.concatenate(output_arrays)

    output_path = f"{embed_id}.wav"
    sf.write(output_path, output_array, samplerate=SAMPLE_RATE)

    # delete the embeddings to release vram
    vc.delete_call(embed_id, VC_HOST, VC_PORT)
    logging.info(f"Audio embedding successfully deleted. ID: {embed_id}")

    return output_path

def run_face_detect_pipeline(video_filepath: str) -> str:
    logging.info("[VIDEO] Running face detector")
    vid_embed_id = ls.video_transfer_call(video_filepath, LS_HOST, LS_PORT)
    logging.info("[VIDEO] Generated embed_id: {vid_embed_id}")
    vid_detect_reply = ls.face_detect_call(vid_embed_id, LS_HOST, LS_PORT)
    logging.info("[VIDEO] Generated face detection boundaries")
    if vid_detect_reply != 1:
        return "gg"
    return vid_embed_id

def run_complete_pipeline(video_filepath: str):

    # vid_embed_id = run_face_detect_pipeline(video_filepath)
    # tts_audio_path = run_audio_pipeline(video_filepath)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_face_detect_pipeline, video_filepath),
            executor.submit(run_audio_pipeline, video_filepath),
            ]

        concurrent.futures.wait(futures)

        vid_embed_id = futures[0].result()
        tts_audio_path = futures[1].result()
    logging.info("[ALL] Running Lip Sync")
    lip_sync_path = ls.lip_sync_call(tts_audio_path, vid_embed_id, LS_HOST, LS_PORT)
    return lip_sync_path


def extract_from_youtube_url(youtube_url: str) -> str:
    logging.info(f"Youtube URL downloaded: {youtube_url}")
    output_path = os.path.join(TMP_DIR, str(uuid4())+".mp4")
    subprocess.run(["yt-dlp", youtube_url, "--format", "mp4", "-o", output_path])
    return output_path

if __name__ == "__main__":
    pass
