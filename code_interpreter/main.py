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
    @staticmethod
    def exec_python(code):
        print(eval(code))

    def run(self):
        example = """
You are a Reasoning + Acting (React) Chain Bot. You have to be interactive so ask the queries one by one from the user to reach to the final answer. Please provide a single Thought and single Action to the user so that the user can search the query of the action and provide you with the observation. 
The tools you have access to are:
1.Search
2.CodeInterpreter
Do not perform mathematical or code operations yourself, rather use CodeInterpreter. 
When you have found the answer to the original prompt then the final response should be Action: Finish[Answer to the original prompt].

For example the chain would be like this:

Question: What is value of pi divided by 2
Thought 1: I need to search the value of pi.
Action 1: Search[Value of pi]
Observation 1: Value of pi is approximately 3.141
Thought 2: I need to divide the value of pi by 2.
Action 2: CodeInterpreter[3.141/2]
Observation 2: CodeInterpreter returned 15.705
Thought 3: We have obtained the value of pi divided by 2.
Action 3: Finish[15.705]
        """
        print(example)
        messages = [
            {
                "role": "system",
                "content": example
            },
            {
                "role": "user",
                "content": "What is 10 divided by 2"
            }
        ]
        reply = query_openai_chat_completion(messages).content
        print(reply)
        action = reply.split("\n")[1]
        tool = action[action.find(":")+1:].strip()
        print(tool)
        if "CodeInterpreter" in tool:
            self.exec_python(tool[tool.find("[")+1:tool.find("]")])


if __name__ == "__main__":
    a = Agent()
    a.run()
