import torch
import torch.nn as nn
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
import os
import json

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, checkpoint_path="ast-checkpoint/checkpoint-epoch11/checkpoint-174570"):
        super().__init__()
        # Convert relative path to absolute path and normalize it
        abs_checkpoint_path = os.path.abspath(checkpoint_path)
        
        # Check if the path exists
        if not os.path.exists(abs_checkpoint_path):
            raise ValueError(f"Checkpoint path does not exist: {abs_checkpoint_path}")
            
        print(f"Loading model from: {abs_checkpoint_path}")
        
        try:
            # First try loading with local_files_only=False to download any missing files
            self.model = ASTForAudioClassification.from_pretrained(abs_checkpoint_path, local_files_only=False)
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(abs_checkpoint_path, local_files_only=False)
        except Exception as e:
            print(f"Error loading with local_files_only=False: {e}")
            print("Trying with local_files_only=True...")
            # Fall back to local files only
            self.model = ASTForAudioClassification.from_pretrained(abs_checkpoint_path, local_files_only=True)
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(abs_checkpoint_path, local_files_only=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def forward(self, input_values, attention_mask=None, labels=None, return_dict=None):
        # Forward pass through the model
        if attention_mask is not None:
            # If attention_mask is provided, use it
            return self.model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=return_dict
            )
        else:
            # Otherwise, don't pass attention_mask
            return self.model(
                input_values=input_values,
                labels=labels,
                return_dict=return_dict
            )
    
    def predict_audio(self, audio_path, device="cuda"):
        # Load audio file
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.feature_extractor.sampling_rate)

        # Preprocess the audio
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model = self.model.to(device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        # Get predicted class (0 for fake, 1 for real)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # Map class index to label
        label = "fake" if predicted_class == 0 else "real"

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                "fake": probabilities[0][0].item(),
                "real": probabilities[0][1].item()
            }
        }