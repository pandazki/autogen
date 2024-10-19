from typing import List, Tuple

from autogen_core.base import CancellationToken
from autogen_core.components import default_subscription
from autogen_core.components.models import ChatCompletionClient, UserMessage
from autogen_magentic_one.utils import message_content_to_str

from ..messages import UserContent
from .base_worker import BaseWorker


@default_subscription
class Reasoner(BaseWorker):
    """An agent specialized in mathematical reasoning, logical deduction, and complex problem-solving."""

    DEFAULT_DESCRIPTION = (
        "A highly capable AI assistant specializing in mathematical reasoning, "
        "logical deduction, and complex problem-solving. It excels in scenarios "
        "that require strong reasoning abilities."
    )

    def __init__(
        self,
        model_client: ChatCompletionClient,
        description: str = DEFAULT_DESCRIPTION,
    ) -> None:
        super().__init__(description)
        self._model_client = model_client

    async def _generate_reply(self, cancellation_token: CancellationToken) -> Tuple[bool, UserContent]:
        """Generate a reasoned response to the given context and question."""

        # Prepare the prompt
        prompt = (
            "You are a highly capable AI assistant specializing in mathematical reasoning, "
            "logical deduction, and complex problem-solving. Your task is to analyze the "
            "given context and question, then provide a well-reasoned answer. Remember that "
            "you don't have access to external data or resources, so use only the information "
            "provided in the context and your inherent knowledge.\n\n"
            "If you encounter a precise mathematical calculation problem, you can generate "
            "a code snippet to solve it. This code will be executed by other workers. "
            "Ensure the code is clear, concise, and solves the specific calculation needed.\n\n"
            "Context and Question:\n"
        )

        # Add the chat history to the prompt
        for message in self._chat_history:
            if isinstance(message, UserMessage):
                prompt += f"Human: {message_content_to_str(message.content)}\n"
            else:
                prompt += f"AI: {message_content_to_str(message.content)}\n"

        # Make an inference to the model
        messages: List[UserMessage] = [UserMessage(content=prompt, source="Human")]
        response = await self._model_client.create(messages, cancellation_token=cancellation_token)

        assert isinstance(response.content, str)
        return False, response.content
