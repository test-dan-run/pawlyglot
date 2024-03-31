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

import time
import json
import asyncio
import logging
import librosa
import numpy as np
import soundfile as sf
from typing import TypedDict, Tuple, Dict

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

class MinPawlyglotOutput(TypedDict):
    start_time: float
    end_time: float
    transcription: str
    translation: str

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

    asr_response = await asr.asr_async_call(
        number, array, HOST, ASR_PORT
    )
    mt_response = await mt.mt_async_call(
        number, asr_response["text"], HOST, MT_PORT
    )
    vc_response = vc.synthesize_call(
        mt_response["text"], embed_id, target_language, sample_rate, HOST, VC_PORT
    )

    return {
        "number": number,
        "start_time": start_time,
        "end_time": end_time,
        "transcription": asr_response["text"],
        "translation": mt_response["text"],
        "audio": vc_response["array"],
    }

async def run_pipeline(audio_filepath: str) -> Tuple[np.ndarray, Dict[int, MinPawlyglotOutput]]:

    # send audio file to vad service
    # TODO: have vad service take in numpy array instead
    # this is to prevent double loading of the same audio
    speech_timestamps = vad.vad_call(audio_filepath, HOST, VAD_PORT)
    silence_timestamps = [
        (
            speech_timestamps[i].end, 
            speech_timestamps[i+1].start,
        ) for i in range(len(speech_timestamps)-1)
    ]

    # send numpy array to vc embedding service
    # vc service will store the embeddings after processing
    audio_arr, _ = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
    audio_segments = [audio_arr[int(SAMPLE_RATE*ts.start):int(SAMPLE_RATE*ts.end)] for ts in speech_timestamps]
    concat_audio = np.concatenate(audio_segments)
    silence_segments = [audio_arr[int(SAMPLE_RATE*ts[0]):int(SAMPLE_RATE*ts[1])] for ts in silence_timestamps]

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
    for idx, ts in enumerate(speech_timestamps):
        start, end = int(SAMPLE_RATE * ts.start), int(SAMPLE_RATE * ts.end) 
        segment = audio_arr[start:end]

        task = asyncio.create_task(
            distributed_call(idx, segment, LANGUAGE, ts.start, ts.end, embed_id, SAMPLE_RATE)
        )
        task.add_done_callback(lambda t: process_distributed_results(t.result()))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)

    # delete the embeddings to release vram
    vc.delete_call(embed_id)

    # re-sort responses in chronological order
    results = {
        k: v for k, v in sorted(
            results.items(), key=lambda item: item[0]
        )}

    # generate synthesised audio
    # concatenate synthesised audio with original silence segments
    audio_arrays = []
    for i in range(len(results)):
        audio_arrays.append(results[i].pop("audio"))
        if i == len(results)-1: break
        audio_arrays.append(silence_segments[i])
    output_array = np.concatenate(audio_arrays)

    return (output_array, results)

async def run_pipeline_with_write(
        input_audio_filepath: str,
        output_audio_filepath: str, 
        output_json_filepath: str,
    ) -> None:

    output_array, results = run_pipeline(input_audio_filepath)
    sf.write(output_audio_filepath, output_array, SAMPLE_RATE)

    # generate text outputs in a JSON file
    with open(output_json_filepath, mode="w", encoding="utf-8") as fw:
        json.dump(results, fw, indent=2)

if __name__ == "__main__":
    
    INPUT_AUDIO_FILEPATH = "../examples/tom_scott_dubbing_16k.wav"
    OUTPUT_AUDIO_FILEPATH = "../examples/output.wav"
    OUTPUT_JSON_FILEPATH = "../examples/output.json"

    total_duration = librosa.get_duration(filename=INPUT_AUDIO_FILEPATH)

    perf_start = time.perf_counter()
    asyncio.run(run_pipeline_with_write(
        INPUT_AUDIO_FILEPATH,
        OUTPUT_AUDIO_FILEPATH,
        OUTPUT_JSON_FILEPATH,
    ))
    perf_end = time.perf_counter()
    process_time = perf_end-perf_start
    logging.info(f"Audio Duration  : {round(total_duration, 3)}")
    logging.info(f"Processing Time : {round(process_time, 3)}")
    logging.info(f"Real-time Factor: {round(process_time/total_duration, 3)}")
