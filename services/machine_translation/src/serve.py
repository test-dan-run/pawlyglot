""" MT server scripts """

import logging
import asyncio
from concurrent import futures

import grpc
import torch

import mt_pb2
import mt_pb2_grpc

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MTServer(mt_pb2_grpc.TranslatorServicer):
    " MT Model Server instance "

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    def translate(self, request: mt_pb2.TranslateRequest, context):
        """ Takes in a text and translate CN -> EN """

        tokenized_text = self.tokenizer([request.text], return_tensors="pt")
        with torch.no_grad():
            generated_ids = self.model.generate(**tokenized_text)
        translation = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.info(translation)

        return mt_pb2.TranslateResponse(translation=translation)
    
async def serve():
    """ Serves MT Model """

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=2))
    mt_pb2_grpc.add_TranslatorServicer_to_server(MTServer(), server)
    server.add_insecure_port("[::]:50054")
    await server.start()

    print("Server started, listening on port 50054")
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
