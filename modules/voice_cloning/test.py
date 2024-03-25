import os
from TTS.api import TTS

tts = TTS("xtts_v2.0.2", gpu=True)

filepaths = [os.path.join("outputs/audio24k", fn) for fn in os.listdir("outputs/audio24k") if fn.endswith(".wav")]

SAMPLE_TEXT = """
Hello

"""

# generate speech by cloning a voice using default settings
tts.tts_to_file(text=SAMPLE_TEXT,
                file_path="output.wav",
                speaker_wav=filepaths,
                language="en")