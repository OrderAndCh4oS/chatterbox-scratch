import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Detect device (Mac with M1/M2/M3/M4)
# Note: issues with mps currently
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
map_location = torch.device(device=device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = ("Best sandwich I ever tasted I found in a plastic bag under a train seat I don't know what was in it orangey "
        "yellow sticky stuff it was sort of quite Tangy with a kind of fishy after taste I still think of that "
        "sandwich sometimes at night if I can't sleep yum yum.")

AUDIO_PROMPT_PATH = "voice-one.wav"
wav = model.generate(
    text, 
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.35,
    cfg_weight=0.5
    )
ta.save("output-1.wav", wav, model.sr)
