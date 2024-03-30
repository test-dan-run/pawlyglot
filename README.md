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
| 4 | Voice Cloning and TTS | [Coqui XTTS-V2](https://huggingface.co/coqui/XTTS-v2) | Yes |
| 5 | Lip Sync | [Wav2Lip-GFPGAN](https://github.com/ajay-sainy/Wav2Lip-GFPGAN) | No |

---

## Setup

1. Clone the repository and build base docker image first. The base image is used as the starting point for all the other services.
```sh
git clone https://github.com/test-dan-run/pawlyglot.git
cd pawlyglot
docker build -f dockerfile.base -t pawlyglot/base:1.0.0 .
```

2. Download the pretrained models
    - The VAD and Translation models are hot-loaded on build/start-up.
    - Download the zipped file containing the ASR model files [here](https://drive.google.com/file/d/1Y4WkFfLaOoFZ4G78xhWatnMzyyQy5g0J/view?usp=sharing). Extract the contents into `./services/speech_recognition/models/small`
    - Download the zipped file containing the TTS model files [here](https://drive.google.com/file/d/1lLaFCnE3KY8RBIucWaj66h3YPrtMSdig/view?usp=sharing). Extract the content into `./services/voice_cloning/models/xtts`

3. Build the rest of the services

```sh
# for development, mount models and source code
docker-compose -f docker-compose.dev.yaml build

# for staging/deployment
docker-compose build
```

4. For the `test_client.py` in `./tests` to work, please install `grpcio-tools` in your local environment (or create a virtualenv). And run the following commands to generate the auxillary pb files.
```sh
python3 -m pip install grpcio-tools==1.62.1

# be in the main directory
python3 -m grpc_tools.protoc -I ./proto \
    --python_out=./orchestrator \
    --pyi_out=./orchestrator \
    --grpc_python_out=./orchestrator \
    ./proto/asr.proto \
    ./proto/vad.proto \
    ./proto/mt.proto \
    ./proto/vc.proto
```

## Run
1. Start up the services.
```sh
# for development, mount models and source code
docker-compose -f docker-compose.dev.yaml up

# for staging/deployment
docker-compose up
```

2. Test out the orchestrator.

```sh
cd ./tests
# edit input_audio_filepath (Line 151) to whatever you want. Audio file is assumed to be sampled at 16KHz.
python3 orchestrate.py
```