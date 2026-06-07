# MOSS-TTS Local Transformer v1.5

This folder contains the public remote-code implementation and realtime
streaming examples for `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`.

MOSS-TTS Local Transformer v1.5 uses the `MossTTSLocal` architecture with
time-synchronous RVQ frame generation and `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
for 48 kHz stereo audio decoding.

## Model

| Item | Value |
|---|---|
| Checkpoint | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` |
| Architecture | `MossTTSLocal` |
| Audio tokenizer | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` |
| Audio format | 48 kHz stereo |
| Attention backends | `flash_attention_2`, `sdpa`, `eager` |

## Batch Inference

```python
from pathlib import Path
import importlib.util

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

pretrained_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


def resolve_attn_implementation() -> str:
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = AutoModel.from_pretrained(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    attn_implementation=resolve_attn_implementation(),
    dtype=dtype,
).to(device)
model.eval()

conversation = [
    processor.build_user_message(
        text="Bonjour, je voudrais essayer une voix française naturelle et stable.",
        language="French",
    )
]

batch = processor([conversation], mode="generation")
input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)

with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4096,
        audio_temperature=1.2,
        audio_top_p=1.0,
        audio_top_k=25,
        audio_repetition_penalty=1.0,
    )

message = processor.decode(outputs)[0]
audio = message.audio_codes_list[0]
if audio.ndim == 1:
    audio = audio.unsqueeze(0)

out_path = Path("output.wav")
torchaudio.save(
    str(out_path),
    audio.detach().cpu().to(torch.float32),
    processor.model_config.sampling_rate,
)
```

## Realtime Streaming Decode

Run a streaming smoke test:

```bash
python moss_tts_local_v1.5/streaming.py \
  --model-dir OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
  --codec-dir OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
  --tts-device cuda:0 \
  --codec-device cuda:1 \
  --text "这是一个实时流式推理测试。" \
  --language Chinese
```

Launch the realtime browser app. It uses FastAPI plus Web Audio playback so
decoded PCM chunks can be played while generation is still running:

```bash
MODEL_DIR=OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
CODEC_DIR=OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
TTS_DEVICE=cuda:0 \
CODEC_DEVICE=cuda:1 \
python moss_tts_local_v1.5/streaming_app.py
```

The app defaults to `flash_attention_2`. If FlashAttention 2 is not available
in the current environment, runtime loading falls back to `sdpa` on CUDA.
For Continuation and Continuation + Clone modes, provide the reference audio
transcript in the separate Reference Audio Transcript field.

The streaming demo defaults are:

| Parameter | Default |
|---|---:|
| Audio Temperature | 1.2 |
| Audio Top P | 1.0 |
| Audio Top K | 25 |
| Audio Repetition Penalty | 1.0 |
| Codec Chunk Frames | 0 (auto, UI range 0-32) |
| Initial Playback Delay | 0.08 s |

Generated audio, token dumps, and metadata are written under
`outputs/moss_tts_local_v1_5_streaming/` by default.
