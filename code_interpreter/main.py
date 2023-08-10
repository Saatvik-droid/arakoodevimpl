import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


def query_openai_chat_completion(messages, functions=None, function_call="auto"):
    if functions is None:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7)
    else:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=messages, temperature=0.7,
                                                  functions=functions, function_call=function_call)
    reply = completion.choices[0].message
    return reply


class Agent:
    def __init__(self, query):
        self.query = query

    @staticmethod
    def exec_python(code):
        return eval(code)

    def run(self):
        example = """
You are a Reasoning + Acting (React) Chain Bot. You have to be interactive so ask the queries one by one from the user to reach to the final answer. Please provide a single Thought and single Action to the user so that the user can search the query of the action and provide you with the observation. 
The tools you have access to are:
1. CodeInterpreter
2. Finish
Do not perform mathematical or code operations yourself, rather use CodeInterpreter. 
If you have the answer use Finish tool to end the thought process.
For example the chain would be like this:

Question: What is value of pi divided by 2
{
    thought: I need to search the value of pi,
    action: {
        tool: Search
        arg: Value of pi
    }
}
observation: Value of pi is approximately 3.141
{
    thought: I need to divide the value of pi by 2.
    action: {
        tool: CodeInterpreter,
        arg: 3.141/2
    }
}
observation: CodeInterpreter returned 15.705
{
    thought: We have obtained the value of pi divided by 2.
    action: {
        tool: Finish
        arg: 15.705
    }
}
        """
        extracted = """
Question:{query}

EXTRACTED = {json_format}
Return EXTRACTED as a valid JSON object.
        """
        schema = """
{
    thought:: string,
    action: {
        tool: string,
        arg: string
    }
}
        """
        print(example)
        while True:
            extracted_f = extracted.format(query=self.query, json_format=schema)
            print(extracted_f)
            messages = [
                {
                    "role": "system",
                    "content": example
                },
                {
                    "role": "user",
                    "content": extracted_f
                }
            ]
            reply = query_openai_chat_completion(messages).content
            reply_json = json.loads(reply)
            print(f"RESPONSE:\n{reply}")
            if reply_json["action"]["tool"] == "CodeInterpreter":
                val = self.exec_python(reply_json["action"]["arg"])
                print(val)
                self.query += f"\n{reply}\nObservation: CodeInterpreter returned {val}"
            if reply_json["action"]["tool"] == "Finish":
                print(f'FINAL ANSWER:\n{reply_json["action"]["arg"]}"')
                break


if __name__ == "__main__":
    a = Agent("What is 10 exponential 2")
    a.run()
