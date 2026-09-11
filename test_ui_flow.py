"""Exercise real Gradio upload/queue/state serialization with a stubbed model.
No external model request, no reading of the user's .env, no persistent index.
"""
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import dotenv
os.environ['DEEPSEEK_API_KEY'] = 'offline-test-not-a-key'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
with patch.object(dotenv, 'load_dotenv', return_value=False):
    import main

from gradio_client import Client, handle_file
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class StubEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, float(len(t) % 11) / 11, .1] for t in texts]

    def embed_query(self, text):
        return [1.0, .2, .1]


class StubModel:
    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        if 'evidence' not in payload:
            return AIMessage(content=payload['question'])
        return AIMessage(content=json.dumps({
            'answer': '测试回答，引用本次原文。[1]', 'evidence_status': 'partial'}, ensure_ascii=False))


class UIFlowTest(unittest.TestCase):
    def test_two_sessions_and_upload_then_ask(self):
        path = Path(__file__).parent / 'data' / 'Figure AI Report 2024.12.30_09.00.34 (1).pdf'
        if not path.is_file():
            self.skipTest('User PDF not available')
        with patch.object(main, 'get_embeddings', return_value=StubEmbeddings()), \
             patch.object(main, 'make_model', return_value=StubModel()):
            clients = []
            try:
                _, url, _ = main.demo.launch(server_name='127.0.0.1', server_port=7865,
                    inbrowser=False, share=False, prevent_thread_lock=True, quiet=True)
                client = Client(url, verbose=False)
                other = Client(url, verbose=False)
                clients.extend([client, other])
                upload = client.predict([handle_file(str(path))], api_name='/process_pdfs')
                self.assertIn('成功索引 1 份', upload[0])
                response = client.predict('为我介绍 Figure AI', '标准回答', api_name='/chat_respond')
                self.assertIn('Figure AI Report', json.dumps(response[1][-1]['content'], ensure_ascii=False))
                self.assertIn('Founded', response[2])
                other_response = other.predict('为我介绍 Figure AI', '标准回答', api_name='/chat_respond')
                self.assertIn('请先处理', json.dumps(other_response[1][-1]['content'], ensure_ascii=False))
                reset = client.predict([], api_name='/process_pdfs')
                self.assertIn('清空', reset[0])
                after = client.predict('为我介绍 Figure AI', '标准回答', api_name='/chat_respond')
                self.assertIn('请先处理', json.dumps(after[1][-1]['content'], ensure_ascii=False))
            finally:
                for client in clients:
                    client.close()
                main.demo.close()


if __name__ == '__main__':
    unittest.main()
