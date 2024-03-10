import os
import base64
import logging
from concurrent import futures

import grpc
import vad_pb2, vad_pb2_grpc
from speechbrain.inference.VAD import VAD

def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)

class VADServer(vad_pb2_grpc.VoiceActivityDetectorServicer):

    def __init__(self):

        os.makedirs("tmp", exist_ok=True)
        self.vad = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")

    def Detect(self, request_iterator: vad_pb2.DetectRequest, context):

        metadata = context.invocation_metadata()

        filename = None
        for key, values in metadata:
            if key != "filename": continue
            filename = values
        
        if filename is None:
            raise Exception("Filename metadata not found in request.")

        logging.info(f" File loaded: {filename}")

        save_chunks_to_file(request_iterator, filename)

        boundaries = self.vad.get_speech_segments(filename).numpy()
        logging.info(f" Boundaries detected: {boundaries}")

        response = base64.b64encode(boundaries)
        return vad_pb2.DetectResponse(b64array=response)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    vad_pb2_grpc.add_VoiceActivityDetectorServicer_to_server(VADServer(), server)
    server.add_insecure_port("[::]:50052")
    server.start()

    print(f"Server started, listening on port 50052")
    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
