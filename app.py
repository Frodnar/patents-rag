from langchain.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl


api_key = "YOUR-API-KEY-HERE"

graph = Neo4jGraph(
    url="YOUR-DB-URL-HERE", username="neo4j", password="PASSWORD-HERE", database="neo4j"
)

schema = graph.get_schema

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Always search for assignee names, inventor names, and phrases in title and abstract using a case-insensitive search.
Never search for assignees names or inventor names using the curly braces syntax.
Always search for assignee names and inventor names using the toLower() and CONTAINS functions.
Always search for phrases about technology in both title and abstract using the toLower() and CONTAINS functions.
Unless told otherwise, always use lots of synonyms to search for technology concepts but DO NOT use acronyms such as AI for artificial intelligence.
If you are unsure about the direction of a relationship arrow in the query, use an undirected relationship without a < or > character.
Examples: Here are a few examples of generated Cypher statements for particular questions:

MATCH (a:Assignee)<-[:ASSIGNED_TO]-(d:Document)
RETURN a.name AS assignee, count(d) AS numberOfDocuments
ORDER BY numberOfDocuments DESC
LIMIT 1

MATCH (i:Inventor)-[:WORKS_AT]->(a:Assignee)
WHERE toLower(i.name) = "john smith"
RETURN a.name AS Workplace

MATCH (a:Assignee)<-[:ASSIGNED_TO]-(p:Publication)
WHERE toLower(a.name) CONTAINS 'samsung'
RETURN COUNT(DISTINCT p)

MATCH (a:Assignee)<-[:ASSIGNED_TO]-(p:Publication)
WHERE toLower(p.title) CONTAINS 'self-driving' OR toLower(p.title) CONTAINS 'autonomous vehicle' OR toLower(p.title) CONTAINS 'driverless car' OR toLower(p.title) CONTAINS 'robotic car'
OR toLower(p.abstract) CONTAINS 'self-driving' OR toLower(p.abstract) CONTAINS 'autonomous vehicle' OR toLower(p.abstract) CONTAINS 'driverless car' OR toLower(p.abstract) CONTAINS 'robotic car'
RETURN a.name AS assignee, COUNT(DISTINCT p) AS numberOfDocuments
ORDER BY numberOfDocuments DESC
LIMIT 5

MATCH (i:Inventor)-[:INVENTED_BY]-(p:Publication)-[:ASSIGNED_TO]-(a:Assignee)
WHERE toLower(a.name) CONTAINS '3m' AND (toLower(p.title) CONTAINS 'adhesive' OR toLower(p.abstract) CONTAINS 'adhesive' OR toLower(p.title) CONTAINS 'glue' OR toLower(p.abstract) CONTAINS 'glue')
RETURN i.name AS Inventor, COUNT(DISTINCT p) AS numberOfInventions
ORDER BY numberOfInventions DESC
LIMIT 1

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)


@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    llm_chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0, openai_api_key=api_key, model='gpt-4'), graph=graph, verbose=True, cypher_prompt=CYPHER_GENERATION_PROMPT
)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain synchronously in a different thread
    res = await cl.make_async(llm_chain)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res['result']).send()
    return llm_chain
