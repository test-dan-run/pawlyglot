from typing import TypedDict
import grpc
import mt_pb2, mt_pb2_grpc

__all__ = ["mt_call"]

CHUNK_SIZE = 1024 * 1024

class MTOutput(TypedDict):
    number: int
    text: str

async def mt_call(
        number: int, 
        text: str, 
        host: str = "localhost",
        port: int = 50054
    ) -> MTOutput:

    async with grpc.aio.insecure_channel(f"{host}:{port}") as channel:
        stub = mt_pb2_grpc.TranslatorStub(channel)

        response = await stub.translate(mt_pb2.TranslateRequest(text=text))

    return {"number": number, "text": response.translation}
