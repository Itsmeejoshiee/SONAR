import torch
import torch.nn as nn
from transformers import ASTForAudioClassification
from transformers.modeling_outputs import SequenceClassifierOutput

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
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", pooling_mode='mean'):
        super().__init__()
        self.num_labels = 2  # Binary classification (real vs fake)
        self.pooling_mode = pooling_mode
        
        # Load the AST model
        self.ast = ASTForAudioClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Replace the classifier head with our own
        self.ast.classifier = ClassificationHead(self.ast.config)
        
    def forward(self, input_features, attention_mask=None, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.ast.config.use_return_dict
        
        # Forward pass through AST
        outputs = self.ast(
            input_values=input_features,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict
        )
        
        if not return_dict:
            return outputs[0]
            
        return outputs