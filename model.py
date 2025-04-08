import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

class Wav2Vec2ForCTCnCLS(Wav2Vec2PreTrainedModel):

    def __init__(self, config, cls_len=4, alpha=0.01):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.cls_head = nn.Linear(config.hidden_size, cls_len)
        self.attention_proj = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        self.alpha = alpha


    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()


    def _ctc_loss(self, logits, labels, input_values, attention_mask=None):
        loss = None
        if labels is not None:

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = F.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                    )

        return loss


    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss


    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None, # tuple: (ctc_labels, cls_labels), shape=(batch_size, target_length)
        if_ctc=True,
        if_cls=True,
        ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] # this is the last layer's hidden states
        hidden_states = self.dropout(hidden_states)

        logits_ctc = self.lm_head(hidden_states)

        # --- Attention Pooling for Classification ---
        # hidden_states shape: (batch_size, sequence_length, hidden_size)
        
        # Generate attention mask corresponding to hidden_states sequence length
        if attention_mask is None:
            # If no mask provided, assume all frames are valid
            attention_mask_pool = torch.ones(hidden_states.shape[:2], device=hidden_states.device) # (B, T_hidden)
        else:
            # Calculate sequence lengths after feature extraction + encoder
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)) # (B,)
            # Create a mask for the hidden states based on calculated lengths
            attention_mask_pool = torch.zeros(hidden_states.shape[:2], device=hidden_states.device) # (B, T_hidden)
            for i, length in enumerate(output_lengths):
                attention_mask_pool[i, :length] = 1

        # Calculate attention scores
        attention_scores = self.attention_proj(hidden_states) # (batch_size, sequence_length, 1)

        # Apply the mask to the scores before softmax
        # Use a large negative number for masked positions
        extended_attention_mask = (1.0 - attention_mask_pool.unsqueeze(-1).float()) * torch.finfo(attention_scores.dtype).min
        attention_scores = attention_scores + extended_attention_mask

        # Calculate attention weights using softmax
        attention_weights = F.softmax(attention_scores, dim=1) # (batch_size, sequence_length, 1)

        # Calculate the weighted sum (pooled output)
        pooled_output = torch.sum(hidden_states * attention_weights, dim=1) # (batch_size, hidden_size)
        
        # Calculate classification logits using the pooled output
        logits_cls = self.cls_head(pooled_output)
        # --- End of Attention Pooling ---

        loss = None
        if labels is not None:
            if if_ctc:
                loss_ctc = self._ctc_loss(logits_ctc, labels[0], input_values, attention_mask)
            if if_cls:
                loss_cls = self._cls_loss(logits_cls, labels[1])

            loss = loss_cls + self.alpha * loss_ctc

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=(logits_ctc, logits_cls, attention_weights), hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

        
