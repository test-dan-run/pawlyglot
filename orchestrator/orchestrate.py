""" Orchestrates all of the services together

# generate protobufs
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./orchestrator \
    --pyi_out=./orchestrator \
    --grpc_python_out=./orchestrator \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/vc.proto \
    ./proto/mt.proto

"""

import time
import json
import asyncio
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

# strings up asr, mt, and voice cloning calls
async def distributed_call(
        number: int,
        array: np.ndarray,
        target_language: str,
        start_time: float,
        end_time: float,
        embed_id: str,
        sample_rate: int
    ) -> PawlyglotOutput:

    asr_response = await asr.asr_call(
        number, array, HOST, ASR_PORT
    )
    mt_response = await mt.mt_call(
        number, asr_response["text"], HOST, MT_PORT
    )
    vc_response = vc.synthesize_call(
        number, mt_response["text"], embed_id, target_language, sample_rate, HOST, VC_PORT
    )

    return {
        "number": number,
        "start_time": start_time,
        "end_time": end_time,
        "transcription": asr_response["text"],
        "translation": mt_response["text"],
        "audio": vc_response["array"],
    }

async def run_pipeline(
        input_audio_filepath: str,
        output_audio_filepath: str, 
        output_json_filepath: str
    ) -> None:

    # send audio file to vad service
    # TODO: have vad service take in numpy array instead
    # this is to prevent double loading of the same audio
    timestamps = vad.vad_call(input_audio_filepath, HOST, VAD_PORT)

    # send numpy array to vc embedding service
    # vc service will store the embeddings after processing
    audio_arr, _ = librosa.load(input_audio_filepath, sr=SAMPLE_RATE, mono=True)
    audio_segments = [audio_arr[int(SAMPLE_RATE*ts.start):int(SAMPLE_RATE*ts.end)] for ts in timestamps]
    concat_audio = np.concatenate(audio_segments)

    embed_id = vc.embed_call(concat_audio, SAMPLE_RATE, HOST, VC_PORT)
    logging.info(f"Audio successfully embedded. ID: {embed_id}")
    
    # callback to gather results of each asynchronous task
    tasks = []
    results = {}
    def process_distributed_results(output: PawlyglotOutput) -> None:
        number = output.pop("number")
        logging.info(f"Received response {number}, content: {output}")
        results[number] = output
    
    # set up calls for each audio segment, and execute the calls
    for idx, ts in enumerate(timestamps):
        start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end) 
        segment = audio_arr[start:end]

        task = asyncio.create_task(
            distributed_call(idx, segment, LANGUAGE, ts.start, ts.end, embed_id, SAMPLE_RATE)
        )
        task.add_done_callback(lambda t: process_distributed_results(t.result()))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)

    # delete the embeddings to release vram
    _ = vc.delete_call(embed_id)

    # generate synthesised audio and textual outputs
    audio_arrays = [results[i].pop("audio") for i in range(len(results))]
    output_array = np.concatenate(audio_arrays)
    sf.write(output_audio_filepath, output_array, SAMPLE_RATE)

    with open(output_json_filepath, mode="w") as fw:
        results = {
            k: v for k, v in sorted(
                results.items(), key=lambda item: item[0]
            )}
        json.dump(results, fw, indent=2)

if __name__ == "__main__":
    
    input_audio_filepath = "../examples/test_audio.wav"
    output_audio_filepath = "../examples/output.wav"
    output_json_filepath = "../examples/output.json"

    total_duration = librosa.get_duration(filename=input_audio_filepath)

    perf_start = time.perf_counter()
    asyncio.run(run_pipeline(
        input_audio_filepath,
        output_audio_filepath,
        output_json_filepath,
    ))
    perf_end = time.perf_counter()
    process_time = perf_end-perf_start
    logging.info(f"Audio Duration  : {round(total_duration, 3)}")
    logging.info(f"Processing Time : {round(process_time, 3)}")
    logging.info(f"Real-time Factor: {round(process_time/total_duration, 3)}")
