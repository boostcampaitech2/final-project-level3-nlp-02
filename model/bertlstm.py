
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaLMHead,
    RobertaPooler,
)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class RobertaModel(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        assert isinstance(config, RobertaConfig)
        self.config = config
        self.lstm_layer_size = int(config.num_hidden_layers / 2)

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        self.lstm_head = nn.LSTM(input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=self.lstm_layer_size,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=False,)
        self.lm_head = RobertaLMHead(config)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids=None,             # source input_ids
        attention_mask=None,        # source attention_mask
        token_type_ids=None,        # source token_type_ids
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,     # target input_ids
        decoder_position_ids=None,  # target position_ids
        decoder_token_type_ids=None,# target token_type_ids
        labels=None,                # target labels
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        input_shape = input_ids.size()
        batch_size, seq_length = input_ids.size()
        device = input_ids.device 

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        encoder_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            encoder_embedding,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0] #  sequence_output (batch_size, seq_size, hidden_size)
        if self.pooler is not None :
            sequence_output = self.pooler(sequence_output) if self.pooler is not None else None    
        else :
            sequence_output = sequence_output[:, 0]

        h_0 = sequence_output.unsqueeze(0) # cls token encoded vector (batch_size, hidden_size)
        h_0 = h_0.repeat((self.lstm_layer_size,1,1))
        c_0 = torch.zeros(h_0.shape).to(device)

        batch_size, decoder_seq_length = decoder_input_ids.size()

        if decoder_token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :decoder_seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, decoder_seq_length)
                decoder_token_type_ids = buffered_token_type_ids_expanded
            else:
                decoder_token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        decoder_embedding = self.embeddings(
            input_ids=decoder_input_ids,
            position_ids=decoder_position_ids,
            token_type_ids=decoder_token_type_ids,
            past_key_values_length=0,
        )
        lstm_output, (h_n,c_n) = self.lstm_head(decoder_embedding, (h_0, c_0))
        lm_logits  = self.lm_head(lstm_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=past_key_values,
            decoder_hidden_states=decoder_embedding,
            encoder_hidden_states=sequence_output,
        )
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

