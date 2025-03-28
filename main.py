from fastapi import FastAPI
import os
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from py2neo import Graph
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
import ast
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from fastapi.middleware.cors import CORSMiddleware

# Define Policies
policies = """
1. **Access to employee compensation details including salaries, bonuses, and allowances**
   - **Access Levels**: [1, 2, 3]
   - **Keywords**: paycheck, bonus, payments, salary, salary cut, pay slip, allowance

2. **Access to legal department data and corporate legal documents**
   - **Access Levels**: [1, 5]
   - **Keywords**: bylaws, legal, lawyer, legal code

3. **Access to confidential product development data and project plans**
   - **Access Levels**: [1, 2, 3, 7, 15]
   - **Keywords**: Product design, prototypes, product specs, development roadmap, R&D

4. **Access to customer personal data and service records**
   - **Access Levels**: [1, 2, 3, 5, 6, 11]
   - **Keywords**: Customer data, personal information, contact details, customer records

5. **Access to company financial reports, statements, and budget plans**
   - **Access Levels**: [1, 2, 5]
   - **Keywords**: Financial reports, balance sheets, income statements, cash flow, budget

6. **Access to marketing strategies, campaigns, and market research data**
   - **Access Levels**: [1, 2, 3, 9]
   - **Keywords**: Marketing plan, campaign, branding, advertising, market research

7. **Access to IT infrastructure details and system configurations**
   - **Access Levels**: [1, 2, 12, 18]
   - **Keywords**: Servers, network configuration, system settings, IT infrastructure

8. **Access to human resources employee records and personal data**
   - **Access Levels**: [1, 2, 6]
   - **Keywords**: Employee records, personal data, HR files, personnel files

9. **Access to company strategic plans and executive decision documents**
   - **Access Levels**: [1, 2, 3]
   - **Keywords**: Strategic plan, executive decisions, company direction, board meetings

10. **Access to compliance reports and regulatory documentation**
    - **Access Levels**: [1, 2, 3, 16]
    - **Keywords**: Compliance, regulations, audit reports, regulatory filings

11. **Access to sales data, performance metrics, and revenue reports**
    - **Access Levels**: [1, 2, 3, 13]
    - **Keywords**: Sales data, sales performance, sales reports, revenue

12. **Access to quality assurance reports and product testing results**
    - **Access Levels**: [1, 2, 14]
    - **Keywords**: QA reports, testing results, quality metrics, defect reports

13. **Access to research data and development insights for innovation**
    - **Access Levels**: [1, 2, 3, 15]
    - **Keywords**: Research data, development insights, lab results, innovation

14. **Access to operations data, logistics, and supply chain information**
    - **Access Levels**: [1, 2, 17]
    - **Keywords**: Operations data, logistics, supply chain, inventory management

15. **Access to customer support tickets and interaction histories**
    - **Access Levels**: [1, 2, 11]
    - **Keywords**: Support tickets, customer interactions, help desk, issue tracking

16. **Access to IT security protocols, incident reports, and threat analyses**
    - **Access Levels**: [1, 2, 12]
    - **Keywords**: Security protocols, incident reports, cybersecurity, threat analysis

17. **Access to internal communications, company memos, and announcements**
    - **Access Levels**: [1, 2, 3, 4]
    - **Keywords**: Internal communications, memos, company announcements, emails

18. **Access to employee training materials and professional development programs**
    - **Access Levels**: [1, 2, 6]
    - **Keywords**: Training materials, onboarding, professional development, e-learning

19. **Access to software source code and development repositories**
    - **Access Levels**: [1, 2, 7]
    - **Keywords**: Source code, repositories, codebase, software development

20. **Access to vendor contracts, agreements, and procurement documents**
    - **Access Levels**: [1, 2, 5, 17]
    - **Keywords**: Vendor contracts, agreements, procurement, suppliers

21. **Access to inventory data, stock levels, and warehouse management systems**
    - **Access Levels**: [1, 2, 17]
    - **Keywords**: Inventory, stock levels, warehouse, supply chain

22. **Access to executive meeting minutes and confidential board discussions**
    - **Access Levels**: [1, 2]
    - **Keywords**: Meeting minutes, executive meetings, confidential discussions, board notes
"""

# Few-shot examples
few_shot_examples = """
Text: "In the second quarter of fiscal year 2023, the company experienced a significant increase in revenue, totaling $125 million, which represents a 15% growth compared to the same quarter last year. This growth is primarily attributed to the successful launch of the new product line and expansion into emerging markets. The gross profit margin improved to 48.5%, up from 46% in the previous quarter, while operating expenses rose by 8% to $45 million due to increased investment in research and development. Net income reached $20 million, marking a 25% year-over-year increase, and earnings per share climbed to $1.50 from $1.20 last year. Key highlights include $30 million in revenue from the new XYZ product line, significant market share gains in Indonesia and Vietnam as part of the Southeast Asian market expansion, and a 5% reduction in overhead costs through a cost-saving initiative. Looking ahead to the next quarter, the company anticipates continued growth driven by the holiday season and the planned release of version 2.0 of its flagship software, while remaining cautious about potential supply chain disruptions."
Access Levels: [1, 2, 3,4,5]

Text: "Both parties acknowledge that during the term of this agreement, they may have access to confidential and proprietary information, referred to as "Confidential Information," which includes business plans, customer data, financial records, trade secrets, and any other information designated as confidential. Each party agrees not to disclose any Confidential Information to third parties without the prior written consent of the disclosing party, and this obligation shall survive the termination of the agreement for a period of five years. Confidential Information does not include information that is or becomes publicly available without breach of this agreement, was known to the receiving party before disclosure by the disclosing party, is received from a third party without restriction on disclosure, or is independently developed by the receiving party without reference to the Confidential Information. If either party is required by law to disclose Confidential Information, they must promptly notify the other party and cooperate to seek a protective order or other appropriate remedy."
Access Levels: [1, 2, 3, 8]

Text: "The marketing plan for the third quarter aims to increase brand awareness and drive a 10% increase in lead generation for the new ABC product line by the end of the quarter. The primary target audience consists of tech-savvy professionals aged 25 to 40 in urban areas, while the secondary audience includes small to medium-sized enterprises seeking scalable solutions. Key strategies include launching targeted digital advertising campaigns on LinkedIn, Google, and industry-specific websites, and implementing retargeting campaigns to engage previous website visitors. Content marketing efforts will involve publishing thought leadership articles and whitepapers on emerging industry trends, as well as hosting webinars featuring industry experts to position the company as a market leader. Social media engagement will be increased by sharing behind-the-scenes content and employee spotlights on Twitter and Instagram, and running a user-generated content campaign encouraging customers to share their success stories. Email marketing will deploy a drip campaign to nurture leads collected from webinars and website sign-ups, with personalized email content based on user behavior and preferences. The campaign's success will be measured by key performance indicators such as a 20% increase in website traffic, a 10% growth in qualified leads, a 15% increase in social media followers across all platforms, and achieving a 25% open rate on campaign emails."
Access Levels: [1, 2, 3,4,7]

Text: "Project Phoenix aims to develop a scalable, cloud-based platform that leverages a microservices architecture to improve system reliability and flexibility. The system comprises an API gateway that serves as the single entry point for all client requests, handling routing, authentication, and rate limiting. Independent microservices are responsible for specific functionalities such as user authentication, data processing, and notification handling. The database layer utilizes a combination of SQL and NoSQL databases to optimize both transactional integrity and scalability. The technology stack includes Node.js with the Express framework for backend microservices, React.js for web applications, and React Native for mobile applications. PostgreSQL is used for relational data, and MongoDB for unstructured data, while RabbitMQ serves as the messaging queue for inter-service communication. Key design considerations focus on scalability, implementing containerization using Docker and orchestration with Kubernetes to allow horizontal scaling; security, incorporating OAuth 2.0 for authentication and JWT for secure token management; data integrity, utilizing ACID-compliant transactions where necessary and eventual consistency models for non-critical data; and monitoring and logging, integrating Prometheus for monitoring and the ELK stack (Elasticsearch, Logstash, Kibana) for logging and analytics. API specifications include the User Service API with endpoints for user operations and the Data Service API for data access, both requiring authentication."
Access Levels: [1, 2, 3, 4,7,14]

Text: "Effective January 1, 2023, the Data Protection and Privacy Policy ensures compliance with international data protection regulations, including GDPR and CCPA. The policy mandates that personal data must be collected lawfully, transparently, and with explicit consent, used only for specified purposes, and minimized to what is strictly necessary for business operations. Data security measures require that all personal data be encrypted both at rest and in transit, with role-based access controls to restrict data access to authorized personnel only. An incident response protocol is established for data breach notifications within 72 hours of detection. Individuals have rights to access, erasure, and portability of their personal data, with the company providing data in a structured, commonly used, and machine-readable format upon request. International data transfers must comply with GDPR transfer mechanisms, and due diligence is required to ensure that third-party processors meet data protection standards. Compliance monitoring involves regular internal audits and mandatory data protection training for all employees handling personal data. Enforcement of the policy states that non-compliance may result in disciplinary action, up to and including termination of employment."
Access Levels: [1, 2, 3, 8, 16]
"""


app = FastAPI()




# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Driven Access Control API",
    description="API for processing and storing text chunks with access control in Neo4j.",
    version="1.0.0",
)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration - Replace these with environment variables in production
OPENAI_API_KEY = "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"
NEO4J_URI = "neo4j@neo4j+s://3be32e52.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "qWl_dJfF6TQUxBX7Pxse3W4tOJzpjJc24dvq8Uuog-4"

# Initialize OpenAI
llm = OpenAI(api_key=OPENAI_API_KEY )

# Initialize Embedding Provider
embedding_provider = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-large",  # Update model as needed
    dimensions=1536
)

#////////////////////////////////////////////////////////////////
#Initialize Neo4j Graph
try:
    graph = Neo4jGraph(
    url="neo4j+s://3be32e52.databases.neo4j.io",
    username="neo4j",
    password="qWl_dJfF6TQUxBX7Pxse3W4tOJzpjJc24dvq8Uuog-4"
    )
except Exception as e:
     print(f"Failed to connect to Neo4j: {e}")
     raise

#/////////////////////////////////////


# Pydantic model for input validation
class TextUpload(BaseModel):
    text: str

class QueryInput(BaseModel):
    userQuestion: str
    user_access_level: int

# Function to construct the prompt for LLM
def construct_prompt(user_input: str) -> str:
    prompt = f"""
You are an assistant that assigns access level labels to text chunks based on predefined company policies. Below are the company policies:

{policies}

**Instructions**: Given a text chunk, identify which policies it relates to and assign the corresponding access levels. Provide the access levels as a list of numbers. You are only supposed to give the output in the format of an array: []

Refer to the below few-shot example:

{few_shot_examples}

Text: "{user_input}"
Access Levels:
"""
    return prompt

# Function to get access levels from LLM
def get_access_levels(user_text: str) -> list:
    try:
        response = llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You have to figure out the access level and return only the array of access levels. No other descriptions. Stick to the format."},
                {"role": "user", "content": construct_prompt(user_text)}
            ],
            temperature=0  # Ensure deterministic output
        )
        output = response.choices[0].message.content
        # Extract list from the output
        access_levels = output  # Caution: Using eval can be dangerous. Ensure trusted input.
      
        return access_levels
    except Exception as e:
        print(f"Error in get_access_levels: {e}")
        raise

# Function to store access levels in Neo4j
def store_access_levels(user_text: str) -> str:
    try:
        access_levels = get_access_levels(user_text)
        chunk_id = str(uuid.uuid4())
        chunk_embedding = embedding_provider.embed_query(user_text)

        properties = {
            "chunk_id": chunk_id,
            "text": user_text,
            "textEmbedding": chunk_embedding,
            "access_level": access_levels
        }

        query = """
        MERGE (c:Chunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.access_level = $access_level
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $textEmbedding)
        RETURN c
        """
        # //////////////////////////////////////////////////////////////////

        graph.query(query, properties)

        # ////////////////////////////////////////////////////////////////////////////////////////////////
        return access_levels
    except Exception as e:
        print(f"Error in store_access_levels: {e}")
        raise

# API Endpoint to upload and process text
@app.post("/upload", summary="Upload and process a text chunk")
async def upload_text(text_upload: TextUpload):
    try:
        # Read uploaded file content
        user_text = text_upload.text

        # Store access levels in Neo4j
        chunk_id = store_access_levels(user_text)

        return JSONResponse(status_code=200, content={
            "message": "Text chunk processed and stored successfully.",
            "chunk_id": chunk_id
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/query", summary="Get an access-controlled answer based on your question and access level.", response_description="LLM-generated response.")
async def query_answer(query_input: QueryInput):
    try:
        User_query = query_input.userQuestion
        user_level = query_input.user_access_level

        # Embed the user question
        embedding1 = embedding_provider.embed_query(User_query)

        response = graph.query(f"CALL db.index.vector.queryNodes('Vector3', 2, {embedding1}) YIELD node, score RETURN node.text,node.access_level")
        allowed_nodes  = []
        print(response)
        for node in response:
            levels = node['node.access_level']
            levels = ast.literal_eval(levels)
            exists = user_level in levels

        if exists:
            allowed_nodes.append(node['node.text'])

        allowed_nodes

        # Make the API call

        prompt = f"answer the following question ${User_query} based on this given information ${allowed_nodes}. Dont generate creative content. just answer from the information given"
        completion = llm.chat.completions.create(
            model="gpt-3.5-turbo",  # The model to use (you can also use other models like gpt-4)
            messages=[{"role": "system", "content": "You are a helpful assistant for company knowledge management system."},
                    {"role": "user", "content": prompt}]
        )

        # Extract and print the response
        print(completion.choices[0].message.content)

        llm_response = completion.choices[0].message.content

        return JSONResponse(status_code=200, content={
            "answer": llm_response
        })
    except Exception as e:
        print(f"Error in /query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    





# Optional: Endpoint to create vector index (if not created already)
# @app.post("/create-vector-index", summary="Create vector index in Neo4j")
# def create_vector_index():
#     try:
#         query = """
#             CREATE VECTOR INDEX `Vector3`
#             FOR (c:Chunk) ON (c.textEmbedding)
#             OPTIONS {indexConfig: {
#             `vector.dimensions`: 1536,
#             `vector.similarity_function`: 'cosine'
#             }};
#         """
#         graph.run(query)
#         return {"message": "Vector index created successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
