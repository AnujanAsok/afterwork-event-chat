from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain import LLMChain, OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.utilities import google_serper
from langchain.chains import ConversationChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# import streamlit as st
# from streamlit_chat import message
import gradio as gr
import time
import gradio.themes.monochrome as theme

# if 'buffer_memory' not in st.session_state:
#     st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3)



# toolkit = GmailToolkit() 

from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
# NOTION_TOKEN= os.getenv('NOTION_TOKEN')
# DATABASE_ID = os.getenv('DATABASE_ID')

# loader = NotionDBLoader(integration_token=NOTION_TOKEN, database_id='5658756c19644a8b8688b56166401951', request_timeout_sec=30)
# loader = CSVLoader(file_path='TaskList5658756c19644a8b8688b56166401951.csv')
# docs = loader.load()
# # index = VectorstoreIndexCreator().from_loaders([loader])

# index_creator = VectorstoreIndexCreator()
# doc_search = index_creator.from_loaders([loader])

# chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=doc_search.vectorstore.as_retriever(), input_key="question")


# query = "summarize what you know"
# response = chain({"question": query})
# print(response['result'])
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap = 20,
#     length_function = len
# )

# texts = text_splitter.create_documents([docs])
# print(texts[0])
# print(texts[1])

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# text = text_splitter.split_documents(docs)

# embeddings = OpenAIEmbeddings()

# db = Chroma.from_documents(text, embeddings)

# query = "tell me something about this document"
# doc_search = db.similarity_search(query=query)

# print(doc_search[0].page_content)
# retriever = db.as_retriever()
# qa = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type='stuff', retriever=retriever)

# query = "summarize the document"

# response = qa.run(query)

# print(response)


memory = ConversationBufferMemory(memory_key="chat_history")
llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain_llm = OpenAI(temperature=0, model='gpt-3.5-turbo')
# search = DuckDuckGoSearchRun()
search =  SerpAPIWrapper(serpapi_api_key=SERPAPI_KEY)
tools = [
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about vendors."
    ),
]

email_tools=[
    Tool(
        name = "Current Search",
        func=search.run,
        description="useful for when you need to answer questions about vendors."
    ),
    # Tool(
    #     name = 'Email',
    #     func=toolkit.get_tools(),
    #     description="useful for when you need to send an email"
    # )
]
# agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
agent_chain = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# agent_prompt = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
agent_prompt = ConversationChain(llm=OpenAI(temperature=0), memory=ConversationBufferMemory(), verbose=True) 
# st.session_state.buffer_memory
agent_emailer = initialize_agent(email_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
output_parser = CommaSeparatedListOutputParser()

# 1. event type (corporate dinner, dinner, or networking)
# 2. the services they require (this can be venue, catering, rental, decor)
# 4. the theme of the event (this can be casual, business-casual, formal)
# 5. the budget for the event
# if any criteria is missing or not specified or null, you must prompt the human to enter the missing field in your response.
# Check the details, if there are no null or not specified values: You must respond with Great, all your details are being processed. Give me a minute to fetch your recommendations.


details_template="""You are an event planner assistant

Your goal is to check the Human below, and return relevant results. 

You must not fill in any information on behalf of the human, they must enter the details themselves.

Only when you feel that you have the adequate details required to find the vendors for the human, you must return the phrase: Great, Give me a minute to find you the best vendors for your event. Additionally, I want you to summarize the event details and return that as well.

Human: {human_input}
Assistant:"""

vendor_template="""You are an event vendor search tool

Your goal is to use the event details below to find vendors that can fulfill the requirements. You must return:
 - The vendors name
 - A brief description of the vendor
 - The address of the vendor
 - The website url of the vendor
 - The contact email for the vendor

You should use the search tool to verify that the information is up to date.

Once you have fulfilled the task, return the list of vendors with the required information as key value pairs. You do not need a tool for this step.
event details: {event_details}
event vendor search tool:"""

emailer_template="""You are a vendor emailer assistant

Your goal is to find the accurate contact emails for the vendors that have been input to you below. 

Your tasks are to:
1. Find the contact emails for each of the vendors in the vendor list using the search tool
2. Craft an email with the following format to each vendor and fill in the placeholders with the event details that have been submitted below:
3. Send the emails to the hello@goafterwork.com. Each vendor should have an individual email sent out.

Once you have correctly fulfilled the task, You must return a message that says you have successfully sent the inquiries to each of the vendors in the vendor list.

vendor list: {vendor_list}
event details: {event_details}

"""

prompt_infos = [
    {
        'name': "details",
        'description': "Good for checking if the initial input has the appropriate details",
        'prompt_template': details_template
    },
    {
        'name': "vendor_query",
        'description': "Good for finding the vendors that have been requested",
        'prompt_template': vendor_template
    }
]

llm_router = OpenAI(temperature=0)

destination_chains = {}


prompt = PromptTemplate(
    input_variables = ["human_input"],
    template = details_template
)

vendor_prompt = PromptTemplate(
    input_variables=["event_details"],
    template=vendor_template)

email_prompt = PromptTemplate(
    input_variables=["vendor_list", "event_details"],
    template=emailer_template
)


# # event_details = []

def fetch_results(user_message):
    formatted_prompt = prompt.format(human_input=user_message)
    response = agent_prompt.predict(input=formatted_prompt)
    return response

def fetch_vendors(event_details):
    formatted_vendors_query = vendor_prompt.format(event_details=event_details)
    vendor_response = agent_chain.run(formatted_vendors_query)
    return vendor_response

def checking_output(input):
    response = fetch_results(input)
    print(response)
    print("best vendors for your event")
    if "fetch your recommendations." in response:
        print("The output contains the specified phrase.")
        vendor_response = fetch_vendors(response)
            # {confirmation_row: gr.update(visible=True)}
        format_vendor = agent_prompt.predict(input="Repeat this in the exact format I am writing it here, do not leave anything out: " + vendor_response)
        return format_vendor
    
    elif "find you the best vendors for your event." in response:
         print("The output contains the specified phrase.")
         vendor_response = fetch_vendors(response)
         format_vendor = agent_prompt.predict(input="Repeat this in the exact format I am writing here, do not leave anything out: " + vendor_response)
         return vendor_response

    else:
        print("The output does not contain the specified phrase.")
    
    return response

with gr.Blocks(css="footer {visibility: hidden}, header {visibility: hidden}", title="EventPlannerGPT", theme="soft") as demo:
    eventPlannerGPT = gr.Chatbot()
    msg = gr.Textbox(label="EventplannerGPT",placeholder='Enter your event details, such as services required, theme, type of event, etc...')
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = checking_output(history[-1][0])  # Call check_output method with user message
        history[-1][1] = ""
        for character in bot_message:
            history[-1][1] += character
            yield history

    msg.submit(user, [msg, eventPlannerGPT], [msg, eventPlannerGPT], queue=False).then(
        bot, eventPlannerGPT, eventPlannerGPT
    )
    clear.click(lambda: None, None, eventPlannerGPT, queue=False)

demo.queue()
demo.launch()
