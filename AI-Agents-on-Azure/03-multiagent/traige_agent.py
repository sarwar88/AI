import os
from dotenv import load_dotenv

# Add references
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ConnectedAgentTool, MessageRole, ListSortOrder, ToolSet, FunctionTool
from azure.identity import DefaultAzureCredential

# Clear the console
os.system('cls' if os.name=='nt' else 'clear')

# Load environment variables from .env file
load_dotenv()
project_endpoint = os.getenv("PROJECT_ENDPOINT")
model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")


# Issue classifier agent definition
issue_classifier_agent_name = "issue_classifier_agent"
issue_classifier_agent_instructions = """
Classify the type of issue described in the ticket.

Choose one of the following categories:
- Bug
- Infra Issue
- Hardware Issue
- Feature Request
- Other

Respond only with the category and a brief explanation.
"""

# Priority agent definition
priority_agent_name = "priority_agent"
priority_agent_instructions = """
Assess how urgent a ticket is based on its description.

Respond with one of the following levels:
- High: User-facing or blocking issues
- Medium: Time-sensitive but not breaking anything
- Low: Cosmetic or non-urgent tasks

Only output the urgency level and a very brief explanation.
"""

# Team agent definition
team_agent_name = "team_agent"
team_agent_instructions = """
Decide which team should own each ticket.

Choose from the following teams:
- Frontend
- Backend
- Infrastructure
- Marketing
- Other

Base your answer on the content of the ticket. Respond with the team name and a very brief explanation.
"""

# Instructions for the primary agent
triage_agent_instructions = """
Triage the given ticket. Use the connected tools to determine the ticket's issue type, priority, 
and which team it should be assigned to.
"""

# Connect to the agents client
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True, 
        exclude_managed_identity_credential=True
    ),
)

# Agent initializer
with agents_client:

    # Create the issue classifier agent and tool
    issue_classifier_agent = agents_client.create_agent(
        model=model_deployment,
        name=issue_classifier_agent_name,
        instructions=issue_classifier_agent_instructions
    )

    issue_classifier_tool = ConnectedAgentTool(
        id=issue_classifier_agent.id, 
        name=issue_classifier_agent_name, 
        description="Classifies the issue type of the ticket"
    )


    # Create the priority agent and tool
    priority_agent = agents_client.create_agent(
        model=model_deployment,
        name=priority_agent_name,
        instructions=priority_agent_instructions
    )
    priority_agent_tool = ConnectedAgentTool(
    id=priority_agent.id, 
    name=priority_agent_name, 
    description="Assesses the priority of a ticket"
    )

    
    # Create the team agent and tool
    team_agent = agents_client.create_agent(
        model=model_deployment,
        name=team_agent_name,
        instructions=team_agent_instructions
    )

    team_agent_tool = ConnectedAgentTool(
        id=team_agent.id, 
        name=team_agent_name, 
        description="Determines which team should take the ticket"
    )

    # Create the main triage agent
    agent = agents_client.create_agent(
        model=model_deployment,
        name="triage-agent",
        instructions=triage_agent_instructions,
        tools=[
            issue_classifier_tool.definitions[0],
            priority_agent_tool.definitions[0],
            team_agent_tool.definitions[0]
        ]
    )

    # Create thread for the chat session
    print("Creating agent thread.")
    thread = agents_client.threads.create()


    # Create the ticket prompt
    prompt = "Users can't reset their password from the mobile app."

    # Send a prompt to the agent
    message = agents_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=prompt,
    )

    # Create and process Agent run in thread with tools
    print("Processing agent thread. Please wait.")
    run = agents_client.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Fetch and log all messages
    messages = agents_client.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    for message in messages:
        if message.text_messages:
            last_msg = message.text_messages[-1]
            print(f"{message.role}:\n{last_msg.text.value}\n")

    # Delete the agent when done
    print("Cleaning up agents:")
    agents_client.delete_agent(agent.id)
    print("Deleted triage agent.")

    # Delete the connected agents when done
    agents_client.delete_agent(issue_classifier_agent.id)
    print("Deleted issue classifier agent.")
    agents_client.delete_agent(priority_agent.id)
    print("Deleted priority agent.")
    agents_client.delete_agent(team_agent.id)
    print("Deleted team agent.")
