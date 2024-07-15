import boto3
import requests
from rich.console import Console
from sacrebleu import sentence_bleu
from rouge_score import rouge_scorer
from collections import Counter
import string
import json
import pandas as pd

model_name = "anthropic_Haiku"

def create_bedrock_request(user_question, provided_text):
    aws_access_key_id = 'access_id'
    aws_secret_access_key = 'secret_access_key'
    region_name = 'us-east-1'

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    bedrock_client = session.client('bedrock-runtime')

    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": f"""Human: You will be acting as an AI assistant named Kadou created by the creative inetell. Your goal is to answer the users questions from the provided text (if the text provided). You will be replying to users who are on the creative intell site and who will be confused if you don't respond in the character of Kadou. 

Here are some important rules for the interaction:
- Always stay in character, as Kadou, an AI from creative intell.  
- If you are unsure how to respond, say "Sorry, I didn't understand that. Could you rephrase your question? 
- Do not reveal that you provide the answer from a prepared text. For example, do not use these statements: "based on the provided text" or "from the provided text," etc.
- If text is not provided (or left empty), then just answer the user question based on the chat history.
- do not say you name and role in every prompt. just when the user asked about or some times that is needed.

Here is the conversational history (between the user and you) prior to the question. It could be empty if there is no history:
<history>
User: Hi, I hope you're well. I just want to let you know that I'm excited to start chatting with you!
Kadou: Good to meet you!  I am Kadou, an AI assistant created by creative intell.  What can I help you with today?
</history>

Here is the user's question:
<question>
{user_question}
</question>

<text>{provided_text}</text>

Assistant: [Kadou from creative intell]"""
                }
            ]

        }
    ]
    inference_config = {
        "maxTokens": 2048,
        "stopSequences": ["\n\nHuman:"],
        "temperature": 1,
        "topP": 1
    }
    additional_model_request_fields = {
        "additionalModelRequestFields": {
            "top_k": 250
        }
    }

    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig=inference_config,
        **additional_model_request_fields
    )

    return response

def exact_match(predicted, ground_truth):
    return predicted.strip().lower() == ground_truth.strip().lower()

def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punctuation(lower(s)))

def f1_score(predicted, ground_truth):
    predicted_tokens = normalize_answer(predicted).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(predicted_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_bleu(reference, candidate):
    score = sentence_bleu(candidate, [reference])
    return score.score / 100

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Example usage with dataset
dataset_path = 'D:/python_projects/CI/Education/prompt_engineering_test/data.jsonl'
with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = f


correct_predictions = 0
# total_questions = len(dataset)
total_questions = 2794
total_f1 = 0
total_bleu = 0
total_rouge1 = 0
total_rouge2 = 0
total_rougeL = 0
url = "https://nlu.saman.cloud/nlu/chatbot_parse_non_clause"
question_count = 0

chat_with_LLM_dict = dict()
chat_with_LLM_dict['question'] = []
chat_with_LLM_dict['model_response'] = []
chat_with_LLM_dict['rasa response'] = []

with open(dataset_path, 'r', encoding='utf-8') as dataset:
    
    for line in dataset:
        print(f"--------sample number {question_count}------------")
        entry = json.loads(line.strip())
        user_question = entry["prompt"]
        chat_with_LLM_dict['question'].append(user_question)

        ground_truth = entry["completion"]

        user_message = {
            "text": user_question
        }

        try:
            Rasa_response = requests.post(url, json=user_message)
            
        except requests.exceptions.RequestException as e:
            print(f"Error in HTTP request: {e}")
            Rasa_result = ""

        # Rasa_response = requests.post(url, json=user_message)
        if (Rasa_response.status_code != 404):
             
            if Rasa_response.status_code == 500:
                provided_text = " "
            else:
                provided_text =  Rasa_response.json()['result']

            response = create_bedrock_request(user_question, provided_text)
            chat_with_LLM_dict['rasa response'].append(provided_text)
            # print("Response status code:", Rasa_response.status_code)
            # print("Response JSON:", Rasa_response.json())

           
            predicted_answer = response['output']['message']['content'][0]['text']
            chat_with_LLM_dict['model_response'].append(predicted_answer) 

            if exact_match(predicted_answer, ground_truth):
                correct_predictions += 1

            total_f1 += f1_score(predicted_answer, ground_truth)
            total_bleu += calculate_bleu(ground_truth, predicted_answer)
            rouge_scores = calculate_rouge(ground_truth, predicted_answer)
            total_rouge1 += rouge_scores['rouge1'].fmeasure
            total_rouge2 += rouge_scores['rouge2'].fmeasure
            total_rougeL += rouge_scores['rougeL'].fmeasure
            question_count += 1

            print(f"Question: {user_question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted Answer: {predicted_answer}")
            print(f"Exact Match: {exact_match(predicted_answer, ground_truth)}")
            print(f"F1 Score: {total_f1}")
            print(f"BLEU Score: {total_bleu}")
            print(f"ROUGE-1: {total_rouge1}")
            print(f"ROUGE-2: {total_rouge2}")
            print(f"ROUGE-L: {total_rougeL}")
            print("")

em_score = correct_predictions / total_questions
average_f1_score = total_f1 / total_questions
average_bleu_score = total_bleu / total_questions
average_rouge1_score = total_rouge1 / total_questions
average_rouge2_score = total_rouge2 / total_questions
average_rougeL_score = total_rougeL / total_questions

print(f"Exact Match (EM) Score: {em_score * 100:.2f}%")
print(f"Average F1 Score: {average_f1_score * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu_score * 100:.2f}%")
print(f"Average ROUGE-1 Score: {average_rouge1_score * 100:.2f}%")
print(f"Average ROUGE-2 Score: {average_rouge2_score * 100:.2f}%")
print(f"Average ROUGE-L Score: {average_rougeL_score * 100:.2f}%")


df = pd.DataFrame(chat_with_LLM_dict)
response_file_name = f"D:/python_projects/CI/Education/prompt_engineering_test/{model_name}.csv"
df.to_csv(response_file_name, index=False)