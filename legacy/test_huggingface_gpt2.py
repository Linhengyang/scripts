# test.py
import torch
import typing as t
import os
import math
import json

temp = '../cache/temp/'
gpt2_resource_dir = '../../resource/llm/gpt/gpt2'



from src.kits.huggingface.tokenizer_adapt import gpt2Tokenizer
from src.kits.huggingface.state_dict_adapt import gpt2_state_dict_adaptor
from src.projs.gpt2.network import gpt2, gpt2Config


if __name__ == "__main__":
    # 测试 tokenizer
    tokenizer_path = os.path.join(gpt2_resource_dir, 'tokenizer.json')

    tok = gpt2Tokenizer()
    tok.from_doc(tokenizer_path)

    prompt = 'how to say hello in french'

    # # 测试加载 模型(pure-torch)
    # model_path = os.path.join(gpt2_resource_dir, 'pytorch_model.bin')
    # hf_state_dict = torch.load(model_path, map_location='cpu')

    # adapted_state_dict = gpt2_state_dict_adaptor(hf_state_dict)

    # config = gpt2Config(
    #     embd_size = 768,
    #     vocab_size = 50257, 
    #     embd_p_drop = 0.1,
    #     num_head = 12,
    #     use_bias = True,
    #     max_context_size = 1024,
    #     attn_p_drop = 0.1,
    #     resid_p_drop = 0.1,
    #     use_cached_casual_mask = True,
    #     use_rope = False,
    #     num_block = 12
    #     )
    # net = gpt2(config)
    # net.load_state_dict(adapted_state_dict, strict=True)

    # device = torch.device('cuda')

    # tokens = tok.encode(prompt)
    # input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # [1, L_q]

    # net = net.to(device)
    # input_ids = input_ids.to(device)
    # output_ids = net.generate(input_ids, None, 50, top_k=1, eos_id=50257)
    # output = tok.decode(output_ids.squeeze(0).tolist())

    # print(output)


    # 测试加载 模型(transformers)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(gpt2_resource_dir, local_files_only=True, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(gpt2_resource_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output)