from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.chat_models import ChatOpenAI
from typing import List

class CAMELAgent:

    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        self.messages = 0
        
        if self.messages == 0:
            messages = self.update_messages(input_message)

            output_message = self.model(messages)
            self.update_messages(output_message)
            self.messages += 1
        elif self.messages == 1:
            messages = self.update_messages("What is the suggested marketing strategy?")

            output_message = self.model(messages)
            self.update_messages(output_message)
            self.messages += 1
            
        elif self.messages == 2:
            messsages = self.update_messages("Who is the target market?")
            output_message = self.model(messages)
            self.update_messages(output_message)
            self.messages +=1
                
        return output_message
    


    
        
