import os
import whisperx
import torch
from typing import Dict, Any
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

class WhisperXTranscriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "int8"
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.model_name = "small.en"

    def load_model(self):
        if self.model is None or self.model.model_name != self.model_name:
            print(f"Loading WhisperX model: {self.model_name}")
            self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
            print("WhisperX model loaded successfully.")

    def load_align_model(self, language_code):
        if self.align_model is None or self.align_metadata['language_code'] != language_code:
            print(f"Loading alignment model for language: {language_code}")
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
            print("Alignment model loaded successfully.")

    def load_diarize_model(self):
        if self.diarize_model is None:
            print("Loading diarization model...")
            models_path = os.path.join(os.path.dirname(__file__), "models")
            
            diarization_pipeline = SpeakerDiarization(
                segmentation=os.path.join(models_path, "segmentation-3.0", "pytorch_model.bin"),
                embedding=os.path.join(models_path, "wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin"),
                clustering="AgglomerativeClustering",
            )
            
            # Load the configuration
            with open(os.path.join(models_path, "speaker-diarization-3.1", "config.yaml"), "r") as f:
                config = yaml.safe_load(f)
            
            # Instantiate the pipeline with the configuration
            self.diarize_model = diarization_pipeline.instantiate(config["pipeline"])
            print("Diarization model loaded successfully.")

    def transcribe_audio(self, audio_file_path: str, align_timestamps: bool = False, diarize: bool = False) -> Dict[str, Any]:
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return None

        self.load_model()

        try:
            audio = whisperx.load_audio(audio_file_path)
            result = self.model.transcribe(audio, batch_size=16)

            if align_timestamps:
                self.load_align_model(result['language'])
                result = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, self.device, return_char_alignments=False)

            if diarize:
                self.load_diarize_model()
                diarize_segments = self.diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            return result
        except Exception as e:
            print(f"An error occurred during transcription: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.model = None  # Force reload of the model