from mistral_common.protocol.instruct.chunk import TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


def main():
    message = UserMessage.model_validate_ignore_extra(
        {"role": "user", "content": "hello", "name": "ignored"}
    )
    assert message == UserMessage(content="hello")

    chunk = TextChunk.model_validate_ignore_extra(
        {"type": "text", "text": "hi", "annotations": []}
    )
    assert chunk == TextChunk(text="hi")

    tokenizer = MistralTokenizer.v1().instruct_tokenizer
    tokenized = tokenizer.encode_instruct(
        InstructRequest(
            messages=[
                UserMessage(content="a"),
                AssistantMessage(content="b"),
            ]
        )
    )
    assert tokenized.tokens == [1, 733, 16289, 28793, 264, 733, 28748, 16289, 28793, 287, 2]
    assert tokenizer.tokenizer.decode(tokenized.tokens) == "[INST] a [/INST] b"


if __name__ == "__main__":
    main()
