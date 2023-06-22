#Importing libraries
import os
from dotenv import load_dotenv
import vector_store as stg
import streamlit as st
import time

#WandB
from wandb.integration.langchain import WandbTracer

#Key helper class
from camel import CAMELAgent

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

wandb_config = {"project": "business-agent"}

load_dotenv()


openai_api_key = os.environ["OPENAI_API_KEY"] 

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Business Agent Marketing Development", page_icon=":robot:")
st.header("Business Agent Marketing Development")

st.write("Enter your OpenAI API key below (not stored between sessions):")
openai_api_key = st.text_input("openai_api_key")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_name():
    user_input = st.text_input("Please give your business name", key="user_input")
    return user_input

def get_industry():
    industry = st.text_input("Please describe your industry vertical", key="industry")
    return industry


user_input = get_name()
industry = get_industry()



if user_input and industry:
    with st.spinner('Wait for it...'):
        time.sleep(2)
    st.success('Done!')
    #st.session_state.generated.append()

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")



assistant_role_name = "Market Research Consultant"
user_role_name = "Startup Founder"
business_name = user_input #bus_name
industry = industry #industry
task = f"Develop a business marketing summary for {business_name} (product or service) in industry vertical {industry}. The Founder will outline the idea initially"
word_limit = 50 # word limit for task brainstorming

task_specifier_sys_msg = SystemMessage(content="Discuss the positives and negatives of the idea and pass unique ideas back and forth.")
task_specifier_prompt = (
"""Here is a task that the {assistant_role_name} will help the {user_role_name} to complete: {task}.
Offer a series of questions that explore the product and market within the industry. Be creative and imaginative with your strategies.
Please reply with the specified task in {word_limit} words or less. Enjoy yourself but don't linger too long. When you are finished simply terminate the conversation."""
)
task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-16k"))
task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                            user_role_name=user_role_name,
                                                            task=task, word_limit=word_limit)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")

specified_task = specified_task_msg.content

assistant_inception_prompt = (
"""Never forget you are a {assistant_role_name} and you are working for and consulting for {user_role_name}. Never flip roles! Only provide advice!
You share a common interest in collaborating to successfully complete a task.
You must help the user to complete the task.
Here is the task: {task}. Never forget your task!
The Founder must instruct you. You will respond based on your expertise and my needs to complete the task.

They will only give you one instruction at a time.
You must write a specific solution that appropriately completes the requested instruction.
You must decline my instruction honestly if you cannot perform the instruction due to physical, moral, legal reasons or your capability and explain the reasons.
Do not add anything else other than your solution to my instruction.
You are never supposed to ask me any questions you only answer questions.
You are never supposed to reply with a flake solution. Explain your solutions concisely.
Your solution must be declarative sentences and simple present tense.
Unless I say the task is completed, you should always start with:

Solution Format: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and provide preferable implementations and examples for each question.
Always end <YOUR_SOLUTION> with: 

Next question is: <NEXT_QUESTION>"""
)

user_inception_prompt = (
"""Never forget you are a {user_role_name} and I am a {assistant_role_name}. Never flip roles! You will always seek advice from me. You are professional and direct.
We share a common interest in collaborating to successfully complete a task.
The agent must help you to complete the task.
Here is the task: {task}. Never forget our task!
You must ask direct questions about your business and your needs to complete the task ONLY in the following way:

1. Advise on the question posed:
Instruction: <YOUR_INSTRUCTION>

You will then wait for instructions.

The "Instruction" describes a task or question. The prior conversation provides further context or information for the requested "Instruction". It is a response from the agent.

You must give me one instruction at a time.
I must write a response that appropriately completes the requested instruction.
I must decline your instruction honestly if I cannot perform the instruction due to physical, moral, legal reasons or my capability and explain the reasons.
You should instruct me not ask me questions.
Now you must start to instruct me using the two ways described above.
Do not add anything else other than your instruction and the optional corresponding input!
Keep giving me instructions and necessary inputs until you think the task is completed.
Do not ask the same question twice.
When the task is completed, you must only reply with a single response <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task."""
)

'''Create a helper helper to get system messages for AI assistant and AI user from role names 
and the task
'''


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]
    
    return assistant_sys_msg, user_sys_msg


assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-16k"))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-16k"))

# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats 
assistant_msg = HumanMessage(
    content=(f"{user_sys_msg.content}. "
                "Now start to give me your advice one by one. "
                "Only reply with Instruction and Input."))

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
#step altered with custom messages
user_msg = assistant_agent.step(user_msg)

#print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

chat_turn_limit, n = 15, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content, callbacks=[WandbTracer(wandb_config)])

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content, callbacks=[WandbTracer(wandb_config)])
    
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    
    #specify filename
    filename = os.path.join("chats", f"{business_name}_{industry}_1.txt")
    
    with open(filename, "a") as f: 
        f.write(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
        f.write(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    

    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break
        

stg.embed_upsert(filename, business_name, industry)

    #Generate Summary
print("@#@@##@#@#  Generating summary....#@#@#@#@##")
n = 0
while n < 2:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)

    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    assistant_ai_msg = assistant_agent.step(user_msg)
    if n == 1:
        assistant_msg = HumanMessage(content='''1. Summarize the chat. 2. Summarize the business 3. Write a 2 paragraph summary which outlines
                                    the product, and market it fits into. When the task is completed, you must only reply with a single response.
                                    ''')
    elif n > 1:
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)

        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        assistant_ai_msg = assistant_agent.step(user_msg)
        
    #Save to file
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        sum_file = os.path.join("chats/summary", f"{business_name}_{industry}_summary.txt")
        with open(sum_file, "a") as g:
            g.write(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
            g.write(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
        break


