from transformers import GPT2LMHeadModel
from transformers import RobertaTokenizerFast
import torch

print("test1")

tokenizer = RobertaTokenizerFast.from_pretrained("javirandor/passgpt-10characters",
                                                 max_len=12, padding="max_length",
                                                 truncation=True, do_lower_case=False,
                                                 strip_accents=False, mask_token="<mask>",
                                                 unk_token="<unk>", pad_token="<pad>",
                                                 truncation_side="right")

print("test2")

model = GPT2LMHeadModel.from_pretrained("javirandor/passgpt-10characters").eval()

print("test3")

NUM_GENERATIONS = 6

print(tokenizer.bos_token_id)
pw_start = tokenizer.encode('pass')
print(pw_start)
# Generate passwords sampling from the beginning of password token
g = model.generate(torch.tensor([[pw_start]]), do_sample=True,
                   num_return_sequences=NUM_GENERATIONS, max_length=12,
                   pad_token_id=tokenizer.pad_token_id, bad_words_ids=[[tokenizer.bos_token_id]])

print("test4")

# Remove start of sentence token
g = g[:, 1:]

print(g)

decoded = tokenizer.batch_decode(g.tolist())

print("test5")

decoded_clean = [i.split("</s>")[0] for i in decoded] # Get content before end of password token

print("test6")

# Print your sampled passwords!
print(decoded_clean)