import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                             QTextEdit, QLabel, QProgressBar, QCheckBox, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import whisperx
import torch
from typing import Dict, Any
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import yaml

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
            config_path = os.path.join(models_path, "speaker-diarization-3.1", "config.yaml")
            with open(config_path, "r") as f:
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
                diarize_segments = self.diarize_model(audio_file_path)
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

class TranscriptionThread(QThread):
    update_progress = pyqtSignal(int)
    transcription_complete = pyqtSignal(dict)
    transcription_error = pyqtSignal(str)

    def __init__(self, transcriber, audio_file, align_timestamps, diarize):
        super().__init__()
        self.transcriber = transcriber
        self.audio_file = audio_file
        self.align_timestamps = align_timestamps
        self.diarize = diarize

    def run(self):
        try:
            result = self.transcriber.transcribe_audio(self.audio_file, self.align_timestamps, self.diarize)
            if result:
                self.transcription_complete.emit(result)
            else:
                self.transcription_error.emit("Transcription failed. Please check the console for errors.")
        except Exception as e:
            self.transcription_error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.transcriber = WhisperXTranscriber()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('WhisperX Transcriber')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        file_layout = QHBoxLayout()
        self.file_label = QLabel('No file selected')
        file_button = QPushButton('Select Audio File')
        file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_button)

        model_layout = QHBoxLayout()
        model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny.en", "base.en", "small.en", "medium.en", "large-v2"])
        self.model_combo.setCurrentText("small.en")
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

        options_layout = QHBoxLayout()
        self.align_checkbox = QCheckBox('Align Timestamps')
        self.diarize_checkbox = QCheckBox('Speaker Diarization')
        options_layout.addWidget(self.align_checkbox)
        options_layout.addWidget(self.diarize_checkbox)

        button_layout = QHBoxLayout()
        self.transcribe_button = QPushButton('Transcribe')
        self.transcribe_button.clicked.connect(self.start_transcription)
        self.transcribe_button.setEnabled(False)
        self.save_button = QPushButton('Save Output')
        self.save_button.clicked.connect(self.save_output)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.transcribe_button)
        button_layout.addWidget(self.save_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)

        main_layout.addLayout(file_layout)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(options_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.output_text)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def select_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Select Audio File', '', 'Audio Files (*.mp3 *.wav *.m4a)')
        if file_name:
            self.file_label.setText(os.path.basename(file_name))
            self.audio_file = file_name
            self.transcribe_button.setEnabled(True)

    def change_model(self, model_name):
        self.transcriber.model_name = model_name
        self.transcriber.model = None  # Force reload of the model

    def start_transcription(self):
        self.transcribe_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.output_text.clear()
        self.output_text.append("Transcription in progress...")

        align_timestamps = self.align_checkbox.isChecked()
        diarize = self.diarize_checkbox.isChecked()

        self.transcription_thread = TranscriptionThread(self.transcriber, self.audio_file, align_timestamps, diarize)
        self.transcription_thread.update_progress.connect(self.update_progress)
        self.transcription_thread.transcription_complete.connect(self.display_transcription)
        self.transcription_thread.transcription_error.connect(self.display_error)
        self.transcription_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def display_transcription(self, result):
        self.output_text.clear()
        if 'segments' in result:
            for segment in result["segments"]:
                if self.align_checkbox.isChecked():
                    self.output_text.append(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]")
                if self.diarize_checkbox.isChecked() and 'speaker' in segment:
                    self.output_text.append(f"Speaker {segment['speaker']}:")
                self.output_text.append(segment['text'])
                self.output_text.append("")
        else:
            self.output_text.append(result.get('text', 'No transcription available.'))
        
        self.transcribe_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setValue(100)

    def display_error(self, error_message):
        self.output_text.clear()
        self.output_text.append(f"Error: {error_message}")
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def save_output(self):
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save Output', '', 'Text Files (*.txt)')
        if file_name:
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(self.output_text.toPlainText())
                QMessageBox.information(self, 'Save Successful', 'The output has been saved successfully.')
            except Exception as e:
                QMessageBox.critical(self, 'Save Failed', f'An error occurred while saving the file: {str(e)}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())