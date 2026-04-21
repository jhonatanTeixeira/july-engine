import re
import uuid

from copy import deepcopy
from typing import Any, Dict, List
from llama_cpp.llama_chat_format import Gemma4ChatHandler


class Gemma4Handler(Gemma4ChatHandler):
    def __call__(self, **kwargs):
        response = super().__call__(**kwargs)

        print('stream', kwargs.get('stream'))
        
        if kwargs.get("stream"):
            channel = None
            tools_calls = ""
            called_tools = False

            for chunk in response:
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")

                if content == '<|channel>':
                    channel = "enter"
                    continue

                if channel == 'enter':
                    if content == 'thought':
                        channel = "think"
                        copy = deepcopy(chunk)
                        copy["choices"][0]["delta"]["content"] = "<think>"
                        yield copy
                        continue
                
                if channel == "think" and content != "<channel|>":
                    yield chunk
                    continue
                    
                if content == "<channel|>":
                    if channel == "think":
                        copy = deepcopy(chunk)
                        copy["choices"][0]["delta"]["content"] = "</think>"
                        channel = None

                        yield copy
                        continue
                
                if content == "<|tool_call>":
                    channel = "tool_call"
                    called_tools = True
                    continue

                if content == "<tool_call|>":
                    channel = None

                    for idx, tool_call in enumerate(tools_calls.replace('<|"|>', '"').split('call:')):
                        unique_id = uuid.uuid4().hex
                        
                        if not tool_call:
                            continue

                        matches = re.match(r'(\w+)(\{.*?\})', tool_call)

                        name = matches.group(1)
                        args = matches.group(2)
                        args = re.sub(r'([{,])\s*([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', args)

                        yield self.parse_tool_calls(self.parse_tool_call(unique_id, name=name))
                        yield self.parse_tool_calls(self.parse_tool_call(unique_id, args=args))

                    tools_calls = ""

                    continue

                if channel == "tool_call":
                    tools_calls += content
                    continue
                
                if chunk.get("choices", [{}])[0].get("finish_reason") == "stop" and called_tools:
                    copy = deepcopy(chunk)
                    copy["choices"][0]["delta"]["content"] = ""
                    copy["choices"][0]["finish_reason"] = "tool_calls"
                    
                    yield copy
                    
                    continue

                yield chunk
        else:
            return response


    def parse_tool_calls(self, tool_call):
        return {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [tool_call]
                    }
                }
            ]
        }

    def parse_tool_call(self, idx, name=None, args=None):
        tool_call = {
            "index": idx,
            "id": f"call_{idx}",
        }

        if name:
            tool_call["type"] = 'function'
            tool_call.setdefault("function", {})["name"] = name
        
        if args:
            tool_call.setdefault("function", {})["arguments"] = args

        return tool_call



            
