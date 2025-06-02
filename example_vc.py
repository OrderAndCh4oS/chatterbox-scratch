from chatterbox.vc import ChatterboxVC

device = "cpu"
model = ChatterboxVC.from_pretrained(device)
wav = model.generate("tests/trimmed_8b7f38b1.wav")
import torchaudio as ta
ta.save("testvc.wav", wav, model.sr)
