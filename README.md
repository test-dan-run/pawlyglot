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
| 2 | Speech Recognition | [NVIDIA NeMo Parakeet-CTC (600M)](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | Yes |
| 3 | Machine Translation | [Helsinki (600M)](https://huggingface.co/Helsinki-NLP/opus-mt-en-zh) | Yes | 
| 4 | Voice Cloning and TTS | [Coqui XTTS-V2](https://huggingface.co/coqui/XTTS-v2) | No |
| 5 | Lip Sync | [Wav2Lip-GFPGAN](https://github.com/ajay-sainy/Wav2Lip-GFPGAN) | No |

---
