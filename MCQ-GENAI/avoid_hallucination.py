"""
This program is build with Flan-T5-XL LLM to be able to answer a question in YES/NO using the provided context as in-context learning.

> The program accepts two parameters provided as a command line input.
> The two inputs represent the context and the question.
> The question output is deterministic i.e. its either YES or NO. You are required to use logits to extract the output.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.
> Remember that your output will be tested against test cases, therefore any deviation from the test cases will be considered incorrect during evaluation.
> Note that the assignment and evaluation test cases are carefully sampled from the model itself, eliminating any chance of hallucination.

Syntax: python template.py <CONTEXT> <QUESTION>
"""

import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

#####
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()

def llm_function(model,tokenizer,context,question):
    input_text = (
         f"Context: {context}\n"
         f"Question: {question}\n"
         f"Answer with only YES or NO:\n"
         f"Answer:"
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=model.device)

    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        next_token_logits = outputs.logits[0, -1, :]

    yes_token_id = tokenizer.encode("YES", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("NO", add_special_tokens=False)[0]

    yes_logit = next_token_logits[yes_token_id].item()
    no_logit = next_token_logits[no_token_id].item()

    if yes_logit > no_logit:
        return "YES"
    else:
        return "NO"

if __name__ == '__main__':
    context = sys.argv[1].strip().lower()
    question = sys.argv[2].strip().lower()

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    torch.manual_seed(42)
    out = llm_function(model,tokenizer,context,question)
    print(out.strip())
