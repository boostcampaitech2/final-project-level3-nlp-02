import torch.nn as nn
from args import ModelArguments, DataTrainingArguments
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from .modeling_longformerbart import LongformerBartWithDoctypeForConditionalGeneration, LongformerBartConfig

def make_model_for_changing_postion_embedding(
    config : LongformerBartConfig,
    data_args : DataTrainingArguments,
    model_args:ModelArguments, 
    ) -> None:
    tmp_config = config
    tmp_config.max_target_positions = 1026
    tmp_config.max_position_embeddings = 1026
    model = LongformerBartWithDoctypeForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=tmp_config)
    model_state_dict = model.state_dict()
    origin_enc_pos_embeds = model_state_dict['model.encoder.embed_positions.weight']
    origin_dec_pos_embeds = model_state_dict['model.decoder.embed_positions.weight']
    new_config = model.config
    new_config.max_position_embeddings = data_args.max_source_length
    new_config.max_target_positions = data_args.max_target_length
    new_config.attention_window_size = model_args.attention_window_size
    
    new_model = LongformerBartWithDoctypeForConditionalGeneration(config=new_config)
    model_state_dict['model.encoder.embed_positions.weight'] = transfer_weights(
        origin_pos_embeds = origin_enc_pos_embeds, 
        current_pos_embeds = new_model.model.encoder.embed_positions.weight.detach(),
        max_position= new_config.max_position_embeddings,
    )

    model_state_dict['model.decoder.embed_positions.weight'] = transfer_weights(
        origin_pos_embeds = origin_dec_pos_embeds,
        current_pos_embeds = new_model.model.decoder.embed_positions.weight.detach(),
        max_position = new_config.max_target_positions,
    )

    new_model.model.encoder.embed_positions = BartLearnedPositionalEmbedding(new_config.max_position_embeddings, new_config.d_model)
    new_model.model.decoder.embed_positions = BartLearnedPositionalEmbedding(new_config.max_target_positions, new_config.d_model)
    new_model.load_state_dict(model_state_dict, strict=True)
    new_model_path = f"{model_args.longformerbart_path}_{new_config.max_position_embeddings}_{new_config.max_target_positions}"
    new_model.save_pretrained(new_model_path)
    print(f"save LongformerBart Model {new_model_path}")

def transfer_weights(origin_pos_embeds, current_pos_embeds, max_position):
    if max_position+2 > origin_pos_embeds.shape[0]:
        current_pos_embeds[:origin_pos_embeds.shape[0]] = origin_pos_embeds
    else:
        current_pos_embeds[:max_position+2] = origin_pos_embeds[:max_position+2]
        current_pos_embeds = current_pos_embeds[:max_position+2] 
    current_pos_embeds = nn.Parameter(current_pos_embeds)
    return current_pos_embeds