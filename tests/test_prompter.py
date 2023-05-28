import unittest
import os
from textwrap import dedent
from .test_data.test_data_path import test_data_path

from llm_tuner.utils.prompter import Prompter


class TestPrompter(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_dialogue_v1_human_and_ai_chat(self):
        prompter = Prompter(os.path.join(
            test_data_path,
            'prompt_templates',
            'human_and_ai_chat.json'
        ))
        self.assertEqual(
            prompter.get_stop_sequences(),
            ['\n\n### Human:\n']
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'human', 'message': 'Hi there!'}
            ]),
            dedent(r'''
                ### Human:
                Hi there!

                ### AI:

            ''')[1:-1]
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'human', 'message': 'Hi there!'},
                {'from': 'ai', 'message': 'Hello!'},
                {'from': 'human', 'message': 'How are you?'},
            ]),
            dedent(r'''
                ### Human:
                Hi there!

                ### AI:
                Hello!

                ### Human:
                How are you?

                ### AI:

            ''')[1:-1]
        )

    def test_dialogue_v1_vicuna_v1_1(self):
        prompter = Prompter(os.path.join(
            test_data_path,
            'prompt_templates',
            'vicuna_v1_1.json'
        ))
        self.assertEqual(
            prompter.get_stop_sequences(),
            ['</s>USER: ']
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'}
            ]),
            r"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT:"
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'},
                {'from': 'assistant', 'message': 'Hi!'},
                {'from': 'user', 'message': 'How are you?'},
            ]),
            r"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT:"
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'},
                {'from': 'assistant', 'message': 'Hi!'},
                {'from': 'user', 'message': 'How are you?'},
                {'from': 'assistant', 'message': 'Good! How can I help you today?'},
                {'from': 'user', 'message': "I'd like to know more about LLMs. Can you give me some advice?"},
            ]),
            r"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hello! ASSISTANT: Hi!</s>USER: How are you? ASSISTANT: Good! How can I help you today?</s>USER: I'd like to know more about LLMs. Can you give me some advice? ASSISTANT:"
        )

    def test_chatglm(self):
        prompter = Prompter(os.path.join(
            test_data_path,
            'prompt_templates',
            'chatglm.json'
        ))
        self.assertEqual(
            prompter.get_stop_sequences(),
            [']\n问：']
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'}
            ]),
            dedent(r'''
                [Round 0]
                问：Hello!
                答：
            ''')[1:-1]
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'},
                {'from': 'chatglm', 'message': 'Hi!'},
            ]),
            dedent(r'''
                [Round 0]
                问：Hello!
                答：Hi!
            ''')[1:-1]
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'},
                {'from': 'chatglm', 'message': 'Hi!'},
                {'from': 'user', 'message': 'How are you?'},
            ]),
            dedent(r'''
                [Round 0]
                问：Hello!
                答：Hi!
                [Round 1]
                问：How are you?
                答：
            ''')[1:-1]
        )
        self.assertEqual(
            prompter.generate_dialogue_prompt_v1([
                {'from': 'user', 'message': 'Hello!'},
                {'from': 'chatglm', 'message': 'Hi!'},
                {'from': 'user', 'message': 'How are you?'},
                {'from': 'chatglm', 'message': "I'm an AI, so I don't have feelings, but thank you for asking. How can I assist you today?"},
                {'from': 'user', 'message': 'What is the current weather?'},
            ]),
            dedent(r'''
                [Round 0]
                问：Hello!
                答：Hi!
                [Round 1]
                问：How are you?
                答：I'm an AI, so I don't have feelings, but thank you for asking. How can I assist you today?
                [Round 2]
                问：What is the current weather?
                答：
            ''')[1:-1]
        )
        self.assertEqual(
            prompter.get_response(dedent(r'''
                [Round 0]
                问:早安我的朋友
                答: 早安,愿你今天有一个美好的一天!有什么想聊的吗?
            ''', )[1:-1]),
            "早安,愿你今天有一个美好的一天!有什么想聊的吗?"
        )


if __name__ == '__main__':
    unittest.main()
