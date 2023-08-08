from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# T5-3b and T5-11B are supported!
# We need sharded weights otherwise we get CPU OOM errors
model_id=f"OpenAssistant/falcon-7b-sft-top1-696"

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model_8bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True,trust_remote_code=True)

max_new_tokens = 250
print("-----------------------------------------------------------------------")
print("Good Morning Sir,")
print("I am your personal mental health chatbot")
print("You can ask me any question of your related to any kind of problem of yours")
print("Please keep the prompt detailed about what you really want")
print("------------------------------------------------------------------------")
query = input("User: ")
while True:
  print("\n\n")
  input_data = tokenizer(f"User: {query} \nChatbot:", return_tensors="pt")
  input_ids = input_data.input_ids
  attention_mask = input_data.attention_mask

  outputs = model_8bit.generate(input_ids, max_new_tokens=max_new_tokens,attention_mask=attention_mask)
  print(tokenizer.decode(outputs[0], skip_special_tokens=True))
  query = input("User: ")
  if query == "quit":
    print("\n\n")
    print("----------------------------------")
    print("It was nice talking to you")
    print("Have A Nice day ahead!")
    print("----------------------------------")
    break