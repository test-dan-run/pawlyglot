<p align="center">
    <img src="assets/pawlyglot.png" width="150">
</p>

---

**pawlyglot** is a project that aims to develop an end-to-end multilingual TTS, voice-cloning and lip-syncing pipeline, by combining several open-source projects.
Only EN > CN translations will be supported at the moment.

The services will be hosted in docker containers. For experiment's sake, we will be using `gRPC` to communicate between the microservices. For more information about gRPC, you can read the article in the link [here](https://blog.dreamfactory.com/grpc-vs-rest-how-does-grpc-compare-with-traditional-rest-apis/).

Currently, the plan is to use the following projects. 

| No. | Services | Model | Implemented |
| - | - | - | - |
| 1 | Voice Activity Detection | [SpeechBrain CRDNN](https://huggingface.co/speechbrain/vad-crdnn-libriparty) | Yes |
| 2 | Speech Recognition | [Faster-Whisper (Small)](https://github.com/SYSTRAN/faster-whisper) | Yes |
| 3 | Machine Translation | [Helsinki EN-ZH](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) | Yes | 
| 4 | Voice Cloning and TTS | [Coqui XTTS-V2](https://huggingface.co/coqui/XTTS-v2) | No |
| 5 | Lip Sync | [Wav2Lip-GFPGAN](https://github.com/ajay-sainy/Wav2Lip-GFPGAN) | No |

---

## Setup

1. Clone the repository and build base docker image first. The base image is used as the starting point for all the other services.
```sh
git clone https://github.com/test-dan-run/pawlyglot.git
cd pawlyglot
docker build -f build/dockerfile.base .
```

2. Download the pretrained models
    - Actually all of the models are hot-loaded on build/start-up. Only the ASR model ain't. I need to standardise this lol. But whateva for now. 
    - Download the ASR model files stuff via this [link here]()
    - Extract contents into `./modules/speech_recognition/models/small`

3. Build the rest of the services
```sh
cd build
docker-compose build
```

4. For the `test_client.py` in `./tests` to work, please install `grpcio-tools` in your local environment (or create a virtualenv). And run the following commands to generate the auxillary pb files.
```sh
python3 -m pip install grpcio-tools==1.62.1

# be in the main directory
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./tests \
    --pyi_out=./tests \
    --grpc_python_out=./tests \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/mt.proto
```

## Run
1. Start up the services.
```sh
cd build
docker-compose up
```

2. Run test client.

```sh
cd ./tests
# edit SAMPLE_AUDIO_PATH (Line 26) to whatever you want. Audio file is assumed to be sampled at 16KHz.
python3 test_client.py
```