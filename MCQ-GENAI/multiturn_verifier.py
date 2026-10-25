"""
This program is build with Flan-T5-XL LLM to be able to answer the final question using the output from the previous questions as in-context learning/few-shot learning. 

Consider three related questions from a search session: Question 1, Question 2, Question 3
1. Answer to Question 1 needs to be generated. 
2. Answer to Question 2 needs to be generated with the answer to Question 1 as one-shot example / context. 
3. Answer to Question 3 needs to be generated with the answer to Question 2 as one-shot example / context.
4. Answer to Question 3 will be either YES or NO and nothing else.


> The program accepts three parameters provided as a command line input. 
> The three inputs represent the questions.
> The output of the first two question is Generation based whereas the last question output is deterministic i.e. its either YES or NO.
> Output should be in upper-case: YES or NO
> There should be no additional output including any warning messages in the terminal.

Syntax: python template.py <string> <string> <string> 

The following example is given for your reference:

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in America?"
Terminal Output: NO

 Terminal Input: python template.py "Who is Rabindranath Tagore?" "Where was he born?" "Is it in India?"
Terminal Output: YES

You are expected to create some examples of your own to test the correctness of your approach.

ALL THE BEST!!
"""

"""
ALERT: * * * No changes are allowed to import statements  * * *
"""
import sys
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

##### You may comment this section to see verbose -- but you must un-comment this before final submission. ######
transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
#################################################################################################################

def llm_function(model,tokenizer,questions):
    '''
    The steps are given for your reference:

    1. Generate answer for the first question.
    2. Generate answer for the second question use the answer for first question as context.
    3. Generate a deterministic output either 'YES' or 'NO' for the third question using the context from second question.  
    5. Clean output and return.
    6. Output is case-sensative: YES or NO
    Note: The model (Flan-T5-XL) and tokenizer is already initialized. Do not modify that section.
    '''
    device = next(model.parameters()).device

    def generate_text(prompt, max_new_tokens=64):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                early_stopping=True,
                do_sample=False,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


    # Step 1: Generate answer to Question 1 (Zero-shot / Direct factual QA)
    prompt_q1 = f"Answer the question:\nQuestion: {questions[0]}\nAnswer:"
    ans_1 = generate_text(prompt_q1, max_new_tokens=64)

    # Step 2: Answer Question 2 using Answer 1 as context (Resolves co-references like "he", "it", etc.)
    prompt_q2 = (
        f"Context: {ans_1}\n"
        f"Question: {questions[1]}\n"
        f"Answer:"
    )
    ans_2 = generate_text(prompt_q2, max_new_tokens=64)

    # Step 3: Answer Question 3 constrained to YES or NO using Answer 2 as context
    prompt_q3 = (
        f"Context: {ans_2}\n"
        f"Question: {questions[2]}\n"
        f"Answer with only YES or NO:\n"
        f"Answer:"
    )
    raw_ans_3 = generate_text(prompt_q3, max_new_tokens=10).strip().upper()

    # Step 4: Strict deterministic post-processing for YES / NO
    # Look for explicit YES or NO tokens
    match = re.search(r"\b(YES|NO)\b", raw_ans_3)
    if match:
        final_output = match.group(1)
    elif "YES" in raw_ans_3:
        final_output = "YES"
    elif "NO" in raw_ans_3:
        final_output = "NO"
    else:
        # Fallback heuristic if the model generates conversational text
        final_output = "NO"

    return final_output


if __name__ == '__main__':

    question_a = sys.argv[1].strip()
    question_b = sys.argv[2].strip()
    question_c = sys.argv[3].strip()

    questions = [question_a, question_b, question_c]
    ##################### Loading Model and Tokenizer ########################
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
    ##########################################################################

    """  Call to function that will perform the computation. """
    torch.manual_seed(42)
    out = llm_function(model,tokenizer,questions)
    print(out.strip())

    """ End to call """
