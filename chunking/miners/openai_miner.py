import os
import openai
from dotenv import load_dotenv, find_dotenv
from chunking.protocol import chunkSynapse
import json
from sr25519 import sign
import bittensor as bt
from retry import retry


class OpenAIError(Exception):
    pass


def miner_init(self):
    """
    Initializes the miner. This function is called once when the miner is created.
    """
    _ = load_dotenv(find_dotenv())
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_API_BASE", "")

    # Set openai key and other args
    self.model = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


@retry(
    tries=3,
    delay=1,
    backoff=2,
)
def miner_process(self, synapse: chunkSynapse) -> chunkSynapse:
    system_prompt = f"""
用户会提供一些文本，请对用户提供的文本内容分块
要求：
1. 把文本分成{synapse.chunk_qty}个块，每个块不能超过{synapse.chunk_size}字符，不能有空块
2. 不要任何分析，直接给我分块结果
3. 分析结果请以json字符串方式返回给我，返回数据格式样例：{{"chunks":["hello","world"]}}
"""
    user_prompt = synapse.document
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    response = self.model.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
    )

    bt.logging.debug(f"model response: {response}")

    content = json.loads(response.choices[0].message.content)
    chunks = content.get("chunks", [])

    if not chunks:
        raise ValueError("Response does not contain 'chunks' or 'chunks' is empty")

    synapse.chunks = chunks

    response_data = {
        "document": synapse.document,
        "chunk_size": synapse.chunk_size,
        "chunk_qty": synapse.chunk_qty,
        "chunks": synapse.chunks,
    }
    synapse.miner_signature = sign(
        (self.wallet.get_hotkey().public_key, self.wallet.get_hotkey().private_key),
        str.encode(json.dumps(response_data)),
    ).hex()

    return synapse
