
""" Custom PyTorch BART model. """
import math
import random
import torch
import torch.utils.checkpoint
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, List

from dataclasses import dataclass
from transformers.activations import ACT2FN

from transformers.utils import logging
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartModel,
    BartDecoder,
    BartDecoderLayer,
    BartPretrainedModel,
    BartForConditionalGeneration,
    BartLearnedPositionalEmbedding,
    shift_tokens_right,
)

from transformers.models.longformer.modeling_longformer import LongformerSelfAttention, LongformerEmbeddings

logger = logging.get_logger(__name__)

class LongformerBartConfig(BartConfig):
    pad_token_id_idx = 4 # pad token id in Bart Tokenizer
    def __init__(self,
        attention_window_size:int = 128,
        attention_dropout:float = 0.1,
        doc_type_size:int = 4,
        architectures:str = 'LongformerBartConditionalGeneration',
        max_position_embeddings:int = 2048,         # from pretrained로 인해서 1026이 입력됨 -> 수정할려면 불러오고 변경을 해야 함
        max_target_positions:int = 1026, # AssertionError: Sequence length should be multiple of 512. Given 1026
        encoder_layers:int = 6,
        decoder_layers:int = 6,
        encoder_attention_heads:int = 16,
        decoder_attention_heads:int = 16,
        **kwargs):
        """
        Longformer Bart Model의 Configuation
        attention_window: Longformer Encoder의 layer별 window_size
        doc_type_size: document 문서 타입 개수(논문, 사설잡지, 뉴스)
        """

        super().__init__(**kwargs) 
        self.attention_window = [attention_window_size]*self.encoder_layers
        self.attention_dropout = attention_dropout
        self.architectures = [architectures]
        self.max_position_embeddings = max_position_embeddings
        self.max_target_positions = max_target_positions
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_attention_heads = decoder_attention_heads
        self.doc_type_size = doc_type_size
        
        # del self.classif_dropout 
        # del self.extra_pos_embeddings
        del self.id2label
        del self.label2id

class LongformerSelfAttentionForBart(nn.Module):
    def __init__(self, config:LongformerBartConfig, layer_id:int):
        super().__init__()
        self.d_model = config.d_model
        config.num_attention_heads = config.encoder_attention_heads
        config.attention_probs_dropout_prob = config.attention_dropout
        self.longformer_self_attn_layer = LongformerSelfAttention(config, layer_id=layer_id)
        self.outputs = nn.Linear(self.d_model, self.d_model)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):

        self_outputs = self.longformer_self_attn_layer( #att_output, attn_probs, global_attn_probs
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        attn_output = (self.outputs(self_outputs[0]),)

        if output_attentions :
            attn_output += (self_outputs[1])
            if is_global_attn :
                attn_output += (self_outputs[2])
            else :
                attn_output += (None,)
        else:
            attn_output += (None, None,)
            
        return attn_output
        
class LongformerBartEncoderLayer(nn.Module):
    def __init__(self, config: LongformerBartConfig, layer_id: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LongformerSelfAttentionForBart(
            config=config,
            layer_id=layer_id
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        is_index_masked: torch.Tensor,  
        is_index_global_attn: torch.Tensor,    
        is_global_attn:torch.Tensor,    
        output_attentions: bool = False,
    ):

        residual = hidden_states
        hidden_states, output_attn , global_attn = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,           
            is_index_global_attn=is_index_global_attn,  
            is_global_attn=is_global_attn,   
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (output_attn, global_attn)
        else :
            outputs += (None, None)

        return outputs


class LongformerBartEncoderWithDocType(BartPretrainedModel):
    def __init__(self,
                config: LongformerBartConfig,
                embed_tokens: Optional[nn.Embedding] = None,
                doc_type_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(self.embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, self.embed_dim, self.padding_idx)

        if doc_type_tokens is not None:
            self.doc_type_tokens = doc_type_tokens
        else:
            self.doc_type_tokens = nn.Embedding(config.doc_type_size, self.embed_dim, 0)
        
        self.embed_positions = BartLearnedPositionalEmbedding(
            self.max_source_positions,
            self.embed_dim,
        )
        self.layers = nn.ModuleList([LongformerBartEncoderLayer(config, i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(self.embed_dim)

        
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()
    
    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_global_token_ids(self, row):
        new_row = np.zeros_like(row)
        bos, eos = np.where(row==1)[0][[0,-1]]
        new_row[bos] = 1
        new_row[eos] = 1
        return new_row 

    def get_is_index_global_attn(self, attention_mask:torch.Tensor):
        np_attention_mask = np.array(attention_mask.detach().cpu())
        is_index_global_attn = np.apply_along_axis(self.get_global_token_ids, axis=1, arr=np_attention_mask)
        is_index_global_attn = torch.tensor(is_index_global_attn)
        return is_index_global_attn

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        doc_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # encoder layer에 정의가 되어서 들어가야 함
        is_index_masked = attention_mask < 0
        device = attention_mask.device
        is_index_global_attn = self.get_is_index_global_attn(attention_mask).to(device)
        is_global_attn = is_index_global_attn.flatten().any().item()
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        doc_type = self.doc_type_tokens(doc_type_ids)

        hidden_states = inputs_embeds + embed_pos + doc_type
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # longformer self attention 내부에서 따로 처리 -> expand mask가 필요 없음
        # # expand attention_mask
        # if attention_mask is not None:
        #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        #     attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype) # encoder padding mask

        encoder_states = () if output_hidden_states else None
        all_attentions_local = () if output_attentions else None
        all_attentions_global = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        is_index_masked,
                        is_index_global_attn,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        is_index_masked=is_index_masked,
                        is_global_attn=is_global_attn,
                        is_index_global_attn=is_index_global_attn,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions_local = all_attentions_local + (layer_outputs[1],)
                all_attentions_global = all_attentions_global + (layer_outputs[2],)
                
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions_local, all_attentions_global] if v is not None)
        return LongformerBartBaseModeloutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions_local=all_attentions_local,
            attentions_global=all_attentions_global
        )

@dataclass
class LongformerBartBaseModeloutput(ModelOutput):
    """
    LongformerBart encoder의 출력 결과를 dictionary 형태로 변환해주는 객체
    """
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions_local: Optional[Tuple[torch.FloatTensor]] = None
    attentions_global: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LongformerBartSeq2SeqModelOutput(ModelOutput):
    """
    LongformerBart Model 출력 결과를 dictionary 형태로 변환해주는 객체
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions_local: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions_global: Optional[Tuple[torch.FloatTensor]] = None


class CustomBartDecoder(BartDecoder):
    def __init__(self,config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_target_positions,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()
        self.gradient_checkpointing = False


class LongformerBartModel(BartModel):
    def __init__(self, config: LongformerBartConfig):
        super().__init__(config)
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.doc_type_shared = nn.Embedding(config.doc_type_size, embed_dim, 0)
        self.encoder = LongformerBartEncoderWithDocType(config, self.shared, self.doc_type_shared)
        self.decoder = CustomBartDecoder(config, self.shared)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        doc_type_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                doc_type_ids=doc_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, LongformerBartBaseModeloutput):
            encoder_outputs = LongformerBartBaseModeloutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions_local=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                attentions_global=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,                    
            attention_mask=decoder_attention_mask,          
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return LongformerBartSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions_local=encoder_outputs.attentions_local,
            encoder_attentions_global=encoder_outputs.attentions_global,
        )

class LongformerBartWithDoctypeForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: LongformerBartConfig):
        super().__init__(config)
        self.model = LongformerBartModel(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        doc_type_ids=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            doc_type_ids=doc_type_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

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
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions_local, # 수정 필요?
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        doc_type_ids=None,
        past=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
    
        # decoder input ids -> decoder attention mask 를 생성해야 한다.
        batch_size, seq_size = decoder_input_ids.shape
        device = decoder_input_ids.device
        decoder_attention_mask = torch.ones((batch_size,seq_size)).to(device) # (pad_token_id)

        if past is not None:
            decoder_input_ids = decoder_input_ids[:,-1:]

        return {
            "input_ids": None,      # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "doc_type_ids":doc_type_ids, 
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask":decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }