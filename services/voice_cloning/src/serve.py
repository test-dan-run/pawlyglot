import base64
import logging
import asyncio
import soundfile as sf
from uuid import uuid4
from datetime import datetime
from concurrent import futures

import torch
import torchaudio
import numpy as np
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig

import grpc
import vc_pb2
import vc_pb2_grpc

MAX_REF_LENGTH = 30
GPT_COND_LEN = 6
GPT_COND_CHUNK_LEN = 6
LATENT_SAMPLE_RATE = 22050
SYNTH_SAMPLE_RATE = 24000
TEMPERATURE = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 1024 * 1024

def get_encoded_chunks(encoded: str):

    idx = 0

    while True:
        piece = encoded[idx*CHUNK_SIZE:(idx+1)*CHUNK_SIZE]
        if len(piece) == 0:
            return
        idx += 1
        yield vc_pb2.SynthResponse(buffer=piece)


class TTSServer(vc_pb2_grpc.VoiceClonerServicer):

    def __init__(self):
        config = XttsConfig()
        config.load_json("./models/xtts/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="./models/xtts", use_deepspeed=False)
        self.model.to(DEVICE)

        self.latents = {}
        self.embeddings = {}
        self.count = 0

    def _generate_id(self):
        # e.g. '201902-0309-0347-3de72c98-4004-45ab-980f-658ab800ec5d'
        return datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

    def embed(self, request_iterator: vc_pb2.EmbedRequest, context):

        metadata = context.invocation_metadata()
        input_sample_rate = None
        for key, value in metadata:
            if key != "sample_rate":
                continue
            input_sample_rate = int(value)
        assert isinstance(input_sample_rate, int), "sample_rate metadata not found in request."

        embed_id = self._generate_id()

        b64encoded = ""
        for chunk in request_iterator:
            b64encoded += chunk.buffer

        b64decoded = base64.b64decode(b64encoded)
        audio_array = np.frombuffer(b64decoded, dtype=np.float32)
        audio_array = np.array(audio_array)
        # print(audio_array)

        audio_tensor = torch.from_numpy(audio_array).to(DEVICE)
        audio_tensor = audio_tensor.unsqueeze(0)
        # sf.write("/vc/src/test_check_tensor.wav", audio_tensor.to_numpy(), input_sample_rate)
        
        # resample
        if input_sample_rate != LATENT_SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, 
                input_sample_rate, 
                LATENT_SAMPLE_RATE
            )

        speaker_embedding = self.model.get_speaker_embedding(audio_tensor, LATENT_SAMPLE_RATE)       
        gpt_cond_latents = self.model.get_gpt_cond_latents(
            audio_tensor, LATENT_SAMPLE_RATE, length=GPT_COND_LEN, chunk_length=GPT_COND_CHUNK_LEN
        ) # [1, 2024, T]            

        self.latents[embed_id] = gpt_cond_latents
        self.embeddings[embed_id] = speaker_embedding

        return vc_pb2.EmbedResponse(embed_id=embed_id)

    def synthesize(self, request: vc_pb2.SynthRequest, context):

        out = self.model.inference(
            request.text,
            request.language,
            self.latents[request.embed_id],
            self.embeddings[request.embed_id],
            temperature=TEMPERATURE
        )

        resampled_wav = torchaudio.functional.resample(
            torch.tensor(out["wav"]), SYNTH_SAMPLE_RATE, request.sample_rate)
        resampled_wav = resampled_wav.numpy()
        wav_length = round(resampled_wav.shape[0] / request.sample_rate, 3)
        logging.info(f"voice generated in {request.language}, length: {wav_length}")

        encoded_array = base64.b64encode(resampled_wav)
        chunks_generator = get_encoded_chunks(encoded_array)
        return chunks_generator

    def delete(self, request: vc_pb2.DeleteRequest, context):

        del self.latents[request.embed_id]
        del self.embeddings[request.embed_id]

        return vc_pb2.DeleteResponse(status="success")

async def serve():
    """ Serves TTS Model """

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=1))
    vc_pb2_grpc.add_VoiceClonerServicer_to_server(TTSServer(), server)
    server.add_insecure_port("[::]:50055")
    await server.start()

    print("Server started, listening on port 50055")
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
