""" VAD server scripts """

import os
import logging
import tempfile
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

        tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[-1])
        save_chunks_to_file(request_iterator, tmp.name)
        logging.info("File loaded and saved to: %s", tmp.name)

        boundaries = self.vad.get_speech_segments(tmp.name).tolist()
        logging.info(" Boundaries detected: %s", boundaries)
        timestamps = [vad_pb2.Timestamp(start=start, end=end) for start, end in boundaries]

        return vad_pb2.DetectResponse(timestamps=timestamps)


def serve():
    """Serves VAD Model"""

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    vad_pb2_grpc.add_VoiceActivityDetectorServicer_to_server(VADServer(), server)
    server.add_insecure_port("[::]:50052")
    server.start()

    print("Server started, listening on port 50052")
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
