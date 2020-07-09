import torch
import timeit
from transformers import *

# torch.backends.cuda.matmul.allow_tf32 = False

text = """
Transformers provides state-of-the-art general-purpose architectures
for Natural Language Understanding (NLU) and Natural Language Generation (NLG)
with over thousands of pretrained models in 100+ languages and deep
interoperability between PyTorch & TensorFlow 2.0.
""".replace('\n', ' ').strip()

#          Model          | Tokenizer          | Pretrained weights shortcut  | batch size
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased',           128),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt',                  64),
          (GPT2Model,       GPT2Tokenizer,       'gpt2',                        64),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103',            4),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased',            32),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024',           256),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased',       128),
          (RobertaModel,    RobertaTokenizer,    'roberta-base',                128),
          (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base',            128),
         ]

for model_class, tokenizer_class, pretrained_weights, batch_size in MODELS:
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights).cuda()

    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)], device='cuda')
    input_ids = input_ids.expand(batch_size, -1).contiguous()

    last_hidden_states = model(input_ids)[0]
    torch.cuda.synchronize()

    t = timeit.default_timer()
    last_hidden_states = model(input_ids)[0]
    torch.cuda.synchronize()
    t = timeit.default_timer() - t
    print(model_class.__name__, 'forward', t)

    loss = last_hidden_states.sum()
    torch.cuda.synchronize()

    t = timeit.default_timer()
    loss.backward()
    torch.cuda.synchronize()
    t = timeit.default_timer() - t
    print(model_class.__name__, 'backward', t)

