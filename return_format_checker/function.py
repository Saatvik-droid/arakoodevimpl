from dotenv import load_dotenv
import os
import openai
import json

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


class Function:
    def __init__(self, name, schema, run):
        self.name = name
        self.schema = schema
        self.run = run


class Reply(Function):
    def __init__(self, format):
        name = "reply_user"
        schema = {
            "name": name,
            "description": "reply to user's query",
            "parameters": {
                "type": "object",
                "properties":
                    format
            }
        }
        super().__init__(name, schema, self.print_reply)

    @staticmethod
    def print_reply(args):
        return args


class Agent:
    def __init__(self, format, query):
        self.format = format
        self.query = query

    def run(self):
        tools = [Reply(self.format)]
        functions = [tool.schema for tool in tools]
        sys_prompt = f"""
Only use function_call to reply to use. Do not use content.
            """
        user_prompt = f"{self.query}"
        print(user_prompt)
        messages = [
            {
                "role": "system", "content": sys_prompt
            },
            {
                "role": "user", "content": user_prompt
            },
        ]
        reply = query_openai_chat_completion(messages, functions, {"name": "reply_user"})
        print(reply)
        try:
            if reply["function_call"]:
                for tool in tools:
                    if tool.name == reply["function_call"]["name"]:
                        tool_res = tool.run(json.loads(reply["function_call"]["arguments"]))
                        return tool_res
        except KeyError as e:
            print("KeyError:" + str(e))


if __name__ == "__main__":
    schema = {'dependencies': {'type': 'array', 'items': {'type': 'string'}}}
    query = "In a shallow wide bowl, whisk together the milk, cornstarch, ground flaxseeds, baking powder, " \
            "and vanilla. Add butter to a pan over medium-high heat and melt. Whisk the batter again right before " \
            "dipping bread, as the cornstarch will settle to the bottom of the bowl. List all items used"
    agent = Agent(schema, query)
    print(agent.run())
