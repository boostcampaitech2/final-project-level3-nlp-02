
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional

from transformers import logging
from transformers.modeling_outputs import (Seq2SeqLMOutput, 
    BaseModelOutput,
    Seq2SeqModelOutput,
)

from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaEmbeddings,
    RobertaModel,
    RobertaLMHead,
    RobertaPooler,
)

logger = logging.get_logger(__name__)

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids

class LstmDecoder(nn.Module) :
    def __init__(self, config:RobertaConfig,  embeddings: Optional[RobertaEmbeddings] = None) :
        super().__init__()
        self.config = config

        self.padding_idx = config.pad_token_id
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = RobertaEmbeddings(config)

        self.layer = nn.LSTM(input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            dropout=config.hidden_dropout_prob,
            batch_first=True,
            bidirectional=False
        )

        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        assert isinstance(value, RobertaEmbeddings)
        self.embeddings = value

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        token_type_ids=None,
        encoder_hidden_states=None,
        inputs_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):

        all_hidden_states = () if output_hidden_states else None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        all_hidden_states = () if output_hidden_states is not None else None

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        h_0 = encoder_hidden_states.unsqueeze(0) # cls token encoded vector (batch_size, hidden_size)
        h_0 = h_0.repeat((self.config.num_hidden_layers,1,1))
        c_0 = torch.zeros(h_0.shape).to(device)

        lstm_output, (h_n,c_n) = self.layer(hidden_states, (h_0, c_0))
        lstm_output = self.layernorm(lstm_output)
    
        if output_hidden_states :
            all_hidden_states = all_hidden_states + (h_n,)

        if not return_dict:
            return tuple(
                v
                for v in [lstm_output, all_hidden_states]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=lstm_output, hidden_states=all_hidden_states
        )


class RobertaModelWithLSTM(RobertaPreTrainedModel):
    def __init__(self, model_name:str, config: RobertaConfig):
        super().__init__(config)
        self.config = config
        
        self.encoder = RobertaModel.from_pretrained(model_name, config=config)
        self.shared = self.encoder.embeddings
        self.decoder = LstmDecoder(config, self.shared)
        self.pooler = RobertaPooler(config)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        assert isinstance(value, RobertaEmbeddings)
        self.shared = value
        self.encoder.embeddings = self.shared
        self.decoder.embeddings = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[2] if output_hidden_states==True else None,
            )

        hidden_outputs = encoder_outputs[0]
        hidden_outputs = self.pooler(hidden_outputs)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_outputs,
            inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
        )


class RobertaLSTMForConditionalGeneration(RobertaPreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"lm_head\.bias", r"lm_head\.weight"]

    def __init__(self, model_name: str, config: RobertaConfig):
        super().__init__(config)
        self.model = RobertaModelWithLSTM(model_name, config)
        self.lm_head = RobertaLMHead(config)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, config: RobertaConfig) -> RobertaEmbeddings:
        new_embeddings = RobertaEmbeddings(config)
        return new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, config: RobertaConfig):
        self.lm_head = RobertaLMHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        attention_mask = None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0]) # decoder output hidden states

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            decoder_hidden_states=outputs.decoder_hidden_states,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        head_mask=None,
        encoder_outputs=None,
        **kwargs
    ):

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "head_mask": head_mask,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

