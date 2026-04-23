import re
import uuid
import logging

from copy import deepcopy
from typing import Any, Dict, List
from llama_cpp.llama_chat_format import Gemma4ChatHandler

logger = logging.getLogger(__name__)


class Gemma4Handler(Gemma4ChatHandler):
    def __call__(self, **kwargs):
        response = super().__call__(**kwargs)

        if kwargs.get("stream"):
            return self._stream_response(response)
        else:
            return self._parse_response(response)

    def _parse_response(self, response):
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "") or ""

        if not isinstance(content, str):
            return response

        # 1. Extract thinking blocks: <|channel>thought\n...<channel|>
        thinking_pattern = re.compile(
            r'<\|channel>thought([\s\S]+)<channel\|>', re.DOTALL
        )
        think_match = thinking_pattern.search(content)
        reasoning = None

        if think_match:
            reasoning = think_match.group(1).strip()
            content = content.replace(think_match.group(0), "")

        # 2. Extract tool calls: <|tool_call>call:name{args}<tool_call|>
        tool_call_pattern = re.compile(
            r'<\|tool_call>(.*?)<tool_call\|>', re.DOTALL
        )
        parsed_tools = []

        for tc_match in tool_call_pattern.finditer(content):
            raw_tc = tc_match.group(1).replace('<|"|>', '"')

            for call_block in raw_tc.split('call:'):
                if not call_block.strip():
                    continue

                fn_match = re.match(r'(\w+)(\{.*\})', call_block, re.DOTALL)
                if not fn_match:
                    continue

                name = fn_match.group(1)
                args = fn_match.group(2)
                
                # Fix missing quotes around keys (relaxed JSON)
                args = re.sub(r'([{,])\s*([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', args)

                unique_id = uuid.uuid4().hex
                parsed_tools.append({
                    "id": f"call_{unique_id}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    }
                })

        # Remove tool call blocks from content
        content = tool_call_pattern.sub('', content)

        # 3. Unescape remaining Gemma4 quote tokens
        content = content.replace('<|"|>', '"')
        content = content.strip() or None

        # 4. Build the final response
        message["content"] = content

        if reasoning:
            message["reasoning_content"] = reasoning

        if parsed_tools:
            message["tool_calls"] = parsed_tools
            response["choices"][0]["finish_reason"] = "tool_calls"

        return response
        
    def _stream_response(self, response):
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

                    matches = re.match(r'(\w+)(\{.*\})', tool_call, re.DOTALL)

                    if matches:
                        name = matches.group(1)
                        args = matches.group(2)
                        
                        # Fix missing quotes around keys (relaxed JSON)
                        args = re.sub(r'([{,])\s*([a-zA-Z0-9_-]+)\s*:', r'\1"\2":', args)
                        
                        yield self.parse_tool_calls(self.parse_tool_call(unique_id, name=name))
                        yield self.parse_tool_calls(self.parse_tool_call(unique_id, args=args))
                    else:
                        logger.warning(f"Gemma4Handler: Failed to match tool call pattern in: {tool_call}")

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



            
