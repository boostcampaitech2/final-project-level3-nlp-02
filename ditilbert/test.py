
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig

from transformers import DistilBertTokenizerFast

from transformers.configuration_utils import PretrainedConfig

from model_distilbert import DistilBertForConditionalGeneration

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = DistilBertConfig.from_pretrained("monologg/distilkobert")
config.type_vocab_size=1
config.layer_norm_eps= 1e-05
config.hidden_dropout_prob= 0.1
config.attention_probs_dropout_prob= 0.1
config.intermediate_size= 3072
config.hidden_act="gelu"
config.decoder_start_token_id = torch.tensor(2)
config.use_cache = True

tokenizer = DistilBertTokenizerFast.from_pretrained("monologg/distilkobert")
model = DistilBertForConditionalGeneration("monologg/distilkobert", config).to(device)

text = '근대 경제학에서는 행위의 합리성 문제를 비용 대 편익이라는 효율성 측면에서 접근하는데, 이러한 사고방식은 경제학을 넘어 사회과학 전반으로 널리 확산되 었다. 하지만 경제적 논리로는 설명하기 어렵거나 바람직하지 않은 결과를 낳은 현상도 많이 존재한다. 특히 도덕이나 규범은 경제 현상의 작동 방식에도 영향을 미치기 때문에 이를 설명하는 확장된 경제 이론이 필요하다. 도덕경제론은 바로 이러한 문제를 다루는 접근 방식이다. 이에 따르면 경제 활동에서 개인의 효용 극대화와 합리적 선택을 강조하는 경제학의 기본 가정 역시 초시대적으로 통용 되는 가치가 아니라 자본주의 등장 이후에 형성된 역사적 현상에 불과하다. 경제 적 원칙은 당대의 도덕적 규범에 의해 구성되는 상대적인 가치이기 때문에, 경제 활동이나 영역과 관련된 규범적 측면을 고려해 재구성해야 한다. ‘도덕경제 (moral economy)’론은 바로 이처럼 경제 현상에서 규범이나 문화의 역할 문제 를 다루는 접근 방식이다. 이러한 시도는 경제와 관련된 사회 현상을 이해하는 데 에서 경제학적 접근 방식의 편협성과 한계를 해결하는 데에도 풍부한 시사점을 줄수있을것이다. 이논문은기존의도덕경제론에서다룬주요쟁점과개념을 소개하고, 이 논의가 미디어 경제를 이해하는 데 주는 함의, 쟁점 등을 검토한다.'

input = tokenizer(text, return_tensors='pt')
label = tokenizer('천하통일', return_tensors='pt')

output = model(**input.to(device), labels=label['input_ids'].to(device))