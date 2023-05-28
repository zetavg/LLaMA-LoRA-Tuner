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


if __name__ == '__main__':
    unittest.main()
