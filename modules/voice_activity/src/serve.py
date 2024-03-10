""" VAD server scripts """

import os
import base64
import logging
from concurrent import futures

import grpc
import vad_pb2
import vad_pb2_grpc
from speechbrain.inference.VAD import VAD


def save_chunks_to_file(chunks, filename):
    """writer incoming buffer chunks into file"""

    with open(filename, "wb") as f:
        for chunk in chunks:
            f.write(chunk.buffer)


class VADServer(vad_pb2_grpc.VoiceActivityDetectorServicer):
    """VAD Model Server instance"""

    def __init__(self):
        self.tmp_dir = "./tmp"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty",
        )

    def detect(self, request_iterator: vad_pb2.DetectRequest, context):
        """Takes in audio file and generates speech boundaries in numpy array"""

        metadata = context.invocation_metadata()

        filename = None
        for key, value in metadata:
            if key != "filename":
                continue
            filename = value

        assert isinstance(filename, str), "Filename metadata not found in request."

        output_path = os.path.join(self.tmp_dir, filename)
        save_chunks_to_file(request_iterator, output_path)
        logging.info("File loaded and saved to: %s", filename)

        boundaries = self.vad.get_speech_segments(output_path).numpy()
        logging.info(" Boundaries detected: %s", boundaries)

        response = base64.b64encode(boundaries)
        return vad_pb2.DetectResponse(b64array=response)


def serve():
    """ Serves VAD Model """

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    vad_pb2_grpc.add_VoiceActivityDetectorServicer_to_server(VADServer(), server)
    server.add_insecure_port("[::]:50052")
    server.start()

    print("Server started, listening on port 50052")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
