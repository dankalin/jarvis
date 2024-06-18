import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class SpeechRecognizer:
    def __init__(self):
        self.device = 'cuda:0'
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = 'openai/whisper-medium'
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
        self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
        "automatic-speech-recognition",
        model=self.model,
        tokenizer=self.processor.tokenizer,
        feature_extractor=self.processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=self.torch_dtype,
        device=self.device,
        )

    def recognize(self, audio_path):
        result = self.pipe(audio_path)
        return result