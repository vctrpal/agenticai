import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

os.environ["GOOGLE_API_KEY"] = "AIzaSyCnGPFtXBmhbo_rYRBCVEtTzuYb0VGXd7o"

# 1. Define the Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert technical writer."),
    ("human", "Explain the concept of {topic} in two sentences.")
])



model = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.3)

# 3. Define the Output Parser
parser = StrOutputParser()

# 4. Compose the Chain using LCEL (|)
chain = prompt | model | parser

# 5. Invoke the Chain
response = chain.invoke({"topic": "Vector Databases"})
print(response)

