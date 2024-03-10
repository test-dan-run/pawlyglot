from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DetectRequest(_message.Message):
    __slots__ = ("buffer",)
    BUFFER_FIELD_NUMBER: _ClassVar[int]
    buffer: bytes
    def __init__(self, buffer: _Optional[bytes] = ...) -> None: ...

class DetectResponse(_message.Message):
    __slots__ = ("b64array",)
    B64ARRAY_FIELD_NUMBER: _ClassVar[int]
    b64array: str
    def __init__(self, b64array: _Optional[str] = ...) -> None: ...
