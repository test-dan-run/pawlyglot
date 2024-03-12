""" ASR server scripts """

import base64
import logging
from concurrent import futures

import grpc
import torch
import numpy as np
from nemo.collections.asr.models import EncDecCTCModelBPE

import asr_pb2
import asr_pb2_grpc

if torch.cuda.is_available():
    DEVICE = [0]  # use 0th CUDA device
    ACCELERATOR = 'gpu'
else:
    DEVICE = 1
    ACCELERATOR = 'cpu'

MAP_LOCATION = torch.device(f'cuda:{DEVICE[0]}' if ACCELERATOR == 'gpu' else 'cpu')

class ASRServer(asr_pb2_grpc.SpeechRecognizerServicer):
    """ASR Model Server instance"""

    def __init__(self):
        self.model = EncDecCTCModelBPE.restore_from(restore_path="/asr/models/parakeet-ctc-0.6b.nemo")

    def recognize(self, request_iterator: asr_pb2.RecognizeRequest, context):
        """Takes in audio file and generates transcription, returns string"""

        b64encoded = ""
        for chunk in request_iterator:
            b64encoded += chunk.buffer

        b64decoded = base64.b64decode(b64encoded)
        audio_array = np.frombuffer(b64decoded, dtype=np.float32)

        audio_tensor = torch.from_numpy(audio_array)
        audio_length_tensor = torch.tensor([audio_tensor.size(dim=0)])

        audio_tensor = audio_tensor.unsqueeze(0)

        with torch.no_grad():
            logits, logits_len, _ = self.model.forward(
                                input_signal=audio_tensor.to(MAP_LOCATION), 
                                input_signal_length=audio_length_tensor.to(MAP_LOCATION),
                            )
            
            hypotheses, _ = self.model.decoding.ctc_decoder_predictions_tensor(
                                logits, decoder_lengths=logits_len,
                            )

        return asr_pb2.RecognizeResponse(transcription=hypotheses[0])


def serve():
    """ Serves VAD Model """

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    asr_pb2_grpc.add_SpeechRecognizerServicer_to_server(ASRServer(), server)
    server.add_insecure_port("[::]:50053")
    server.start()

    print("Server started, listening on port 50053")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
