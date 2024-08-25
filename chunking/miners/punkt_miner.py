import nltk
from nltk.data import find
from nltk.tokenize import sent_tokenize
import bittensor as bt
from chunking.protocol import chunkSynapse
from nltk.tokenize import sent_tokenize
import json
from sr25519 import sign


def download_nltk_data(package_name):
    try:
        # Try to find the package
        find(f"tokenizers/{package_name}")
        bt.logging.debug(f"{package_name} already exists. No need to download.")
    except LookupError:
        # If the package doesn't exist, download it
        bt.logging.debug(f"{package_name} not found. Downloading...")
        nltk.download(package_name, quiet=True)
        bt.logging.debug(f"{package_name} download completed.")


def miner_init(self):
    """
    Initialize the miner.
    """
    download_nltk_data("punkt")


def miner_process(self, synapse: chunkSynapse) -> chunkSynapse:
    """
    Process the miner.
    """
    document = sent_tokenize(synapse.document)

    chunks = []
    while len(document) > 0:
        chunks.append(document[0])
        del document[0]
        while len(document) > 0:
            if len(chunks[-1] + " " + document[0]) > synapse.chunk_size:
                break
            chunks[-1] += " " + document.pop(0)

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
