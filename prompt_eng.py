import boto3
import json
import requests
from colorama import Fore, Back, Style
from termcolor import colored
from rich.console import Console

def create_bedrock_request(user_question, provided_text):
    # Initialize a session using Amazon Bedrock
    aws_access_key_id = 'access_id'
    aws_secret_access_key = 'secret_access_key'
    region_name = 'us-east-1'

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )

    # Create the Bedrock client
    bedrock_client = session.client('bedrock-runtime')

    # Define the parameters for the converse request
    # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
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

Assistant: [Kadou from creative intell]<response>"""
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

    # Send the request to the Bedrock API
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig=inference_config,
        **additional_model_request_fields
    )

    # Return the response
    return response

# Example usage

url = "https://nlu.saman.cloud/nlu/chatbot_parse_non_clause"

check = 0

while check == 0:

    user_message_text = input("Enter your message: ")
    print('')
    print('')

    user_message = {
        "text": user_message_text
    }


    Rasa_response = requests.post(url, json=user_message)
    print("Response status code:", Rasa_response.status_code)
    # print("Response JSON:", Rasa_response.json())

    user_question = user_message_text
    if Rasa_response.status_code == 500:
        provided_text = " "
    else:
        provided_text =  Rasa_response.json()['result']

    
    response = create_bedrock_request(user_question, provided_text)

    print('bedrock response: ', response)
    console = Console()
    console.print(response['output']['message']['content'][0]['text'], style="red")

    print('')
    print('')
    continue_chating = input("Do you have more questions? (y/n) ")
    if continue_chating == 'n':
        check = 1
