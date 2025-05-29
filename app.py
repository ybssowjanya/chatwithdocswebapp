from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import CodeInterpreterTool, FilePurpose, OpenApiTool, OpenApiAnonymousAuthDetails
from azure.storage.blob import BlobServiceClient
import json
import os
import shutil
import base64
from io import BytesIO
import time
import logging
from typing import Generator, Dict, Any
from pydantic import BaseModel
from jsonref import JsonRef
from fastapi import UploadFile, File, Form

#Intialize Logger
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

BASE_OPENAPI_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "Question DB API",
        "version": "1.0.0"
    },
    "paths": {
        "/questiondb": {
            "post": {
                "summary": "Ask a question to the DB",
                "operationId": "askQuestion",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                        "type": "string",
                                        "description": "The natural language question to query the database."
                                    }
                                },
                                "required": ["question"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "Final Results": {
                                            "type": "array",
                                            "items": {
                                                "type": "object"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "servers": [
        {
            "url": "https://databaseserver-g6cxg6fuffbqf7eb.eastus-01.azurewebsites.net"
        }
    ]
}
# Azure Blob Storage configuration
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=chatdocstorage;AccountKey=nmW8RvkMGU4O63dppdRQzzH08wNBS0gswaTdwo+Zx/SUVC4YVwnW03U3fVu13+qrEG7/hAJst4cE+AStduBNVA==;EndpointSuffix=core.windows.net"  # Replace with your secure key in production
BLOB_CONTAINER = "agentmappings"
BLOB_FILENAME = "agent_threads1.json"
THREAD_NAMES_FILENAME = "thread_names1.json"

# Pydantic model for file upload requests
class UploadRequest(BaseModel):
    agent_id: str
    file_path: str  # Path to file uploaded to App Service local storage

# Handler class that manages agent lifecycle, threads, and file uploads
class AgentHandler:
    def __init__(self):
        # Initialize AIProject client and Blob service
        self.client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str="eastus.api.azureml.ms;2cc4fb79-4f6a-4eeb-9fe2-c51dc1165e0e;chatwithdoc;chatwithdocs-proj"
        )
        self.blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        self.blob_client = self.blob_service.get_blob_client(container=BLOB_CONTAINER, blob=BLOB_FILENAME)

        # Load agent-thread mapping from Blob Storage
        self.agent_thread_map = self.load_agent_thread_map()

        self.agents_dict = {}  # Maps agent names to IDs
        self.thread_id = None  # Currently selected thread

    # Load agent-thread mappings from blob
    def load_agent_thread_map(self):
        try:
            if self.blob_client.exists():
                blob_data = self.blob_client.download_blob().readall()
                return json.loads(blob_data)
            return {}
        except Exception as e:
            print(f"⚠️ Load failed: {str(e)}")
            return {}

    # Save updated agent-thread mappings to blob
    def save_agent_thread_map(self):
        try:
            updated_json = json.dumps(self.agent_thread_map, indent=2)
            self.blob_client.upload_blob(updated_json, overwrite=True)
        except Exception as e:
            print(f"❌ Save failed: {str(e)}")
    def set_thread_name(self, thread_id: str, thread_name: str):
        try:
            # Ensure thread exists in agent_threads.json
            self.agent_thread_map = self.load_agent_thread_map()
            found = any(thread_id in threads for threads in self.agent_thread_map.values())
            if not found:
                return {"status": "error", "message": f"Thread ID {thread_id} not found in any agent mapping."}
    
            thread_names_blob = self.blob_service.get_blob_client(container=BLOB_CONTAINER, blob=THREAD_NAMES_FILENAME)
    
            # Load existing mappings
            if thread_names_blob.exists():
                data = thread_names_blob.download_blob().readall()
                thread_name_map = json.loads(data)
            else:
                thread_name_map = {}
    
            # Update or set new thread name
            thread_name_map[thread_id] = thread_name
    
            # Save updated map back to blob
            updated_data = json.dumps(thread_name_map, indent=2)
            thread_names_blob.upload_blob(updated_data, overwrite=True)
    
            print(f"✅ Set name '{thread_name}' for thread_id '{thread_id}'")
            return {"status": "success", "message": f"Thread name set to '{thread_name}'"}
    
        except Exception as e:
            print(f"❌ Failed to set thread name: {str(e)}")
            return {"status": "error", "message": str(e)}



    # Create a new thread for a given agent
    def create_new_thread(self, agent_id: str):
        self.agent_thread_map = self.load_agent_thread_map()

        try:
            new_thread = self.client.agents.create_thread()
            thread_id = new_thread.id

            if agent_id not in self.agent_thread_map:
                self.agent_thread_map[agent_id] = []

            if thread_id not in self.agent_thread_map[agent_id]:
                self.agent_thread_map[agent_id].append(thread_id)

            self.save_agent_thread_map()
            print(f"✅ Created new thread for agent {agent_id} and updated agent_threads.json.")
            return thread_id
        except Exception as e:
            print(f"❌ Error creating thread for agent {agent_id}: {str(e)}")
            return None

    # Create a new agent or return existing one
    def create_or_login_agent(self, agent_name):
        self.agent_thread_map = self.load_agent_thread_map()
        agents = self.client.agents.list_agents()
        for agent in agents["data"]:
            self.agents_dict[agent["name"]] = agent["id"]

        if agent_name in self.agents_dict:
            return self.agents_dict[agent_name]

        # Create new agent with code interpreter tool
        code_interpreter = CodeInterpreterTool(file_ids=[])
        new_agent = self.client.agents.create_agent(
            model="gpt-4o",
            name=agent_name,
            instructions=(
                "You're an intelligent AI assistant. "
                "Read the csv file(s) uploaded. "
                "Your task is to analyze knowledge base and provide accurate, context-aware answers based solely on the data in that file."
                "If any general questions are asked, you can answer them as well but check if the query is about files uploaded first."
            ),
            tools=code_interpreter.definitions,
            tool_resources=code_interpreter.resources,
        )
        agent_id = new_agent.id
        self.agents_dict[agent_name] = agent_id
        return agent_id
    def get_thread_name(self, thread_id: str):
        try:
            thread_names_blob = self.blob_service.get_blob_client(container=BLOB_CONTAINER, blob="thread_names1.json")
    
            if thread_names_blob.exists():
                data = thread_names_blob.download_blob().readall()
                thread_name_map = json.loads(data) if data else {}
            else:
                return {"thread_name": None, "message": "Thread name mapping file does not exist."}
    
            thread_name = thread_name_map.get(thread_id)
            return {"thread_name": thread_name}
        except Exception as e:
            print(f"❌ Failed to get thread name: {str(e)}")
            return {"status": "error", "message": str(e)}


    # List all threads associated with a given agent
    def list_threads(self, agent_id):
        agent_thread_map = self.load_agent_thread_map()
        thread_ids = agent_thread_map.get(agent_id, [])
        valid_thread_ids = []
        
        # Validate each thread ID
        for tid in thread_ids:
            try:
                # Check if thread exists in Azure AI Project
                self.client.agents.get_thread(thread_id=tid)
                valid_thread_ids.append(tid)
            except Exception as e:
                print(f"⚠️ Thread {tid} is invalid or deleted: {str(e)}")
                # Remove invalid thread from agent_thread_map
                agent_thread_map[agent_id].remove(tid)
        
        # Save updated agent_thread_map if any threads were removed
        if len(valid_thread_ids) < len(thread_ids):
            self.save_agent_thread_map()
            print(f"✅ Removed invalid thread IDs from agent_threads.json for agent {agent_id}.")
        
        # Load thread name map
        thread_name_map = {}
        try:
            thread_names_blob = self.blob_service.get_blob_client(container=BLOB_CONTAINER, blob=THREAD_NAMES_FILENAME)
            if thread_names_blob.exists():
                data = thread_names_blob.download_blob().readall()
                thread_name_map = json.loads(data) if data else {}
        except Exception as e:
            print(f"⚠️ Error loading thread name map: {str(e)}")
        
        # Remove invalid thread IDs from thread_name_map
        invalid_thread_names = [tid for tid in thread_name_map if tid not in valid_thread_ids]
        if invalid_thread_names:
            for tid in invalid_thread_names:
                del thread_name_map[tid]
            try:
                updated_data = json.dumps(thread_name_map, indent=2)
                thread_names_blob.upload_blob(updated_data, overwrite=True)
                print(f"✅ Removed invalid thread IDs from {THREAD_NAMES_FILENAME}.")
            except Exception as e:
                print(f"❌ Failed to update {THREAD_NAMES_FILENAME}: {str(e)}")
        
        # Create thread_id: thread_name mapping
        threads_with_names = {tid: thread_name_map.get(tid, "New Chat") for tid in valid_thread_ids}
        return threads_with_names

    #To find if agent exists or not
    def is_user_exist(self, agent_name: str):
        try:
            agents = self.client.agents.list_agents()
            self.agents_dict = {agent["name"]: agent["id"] for agent in agents["data"]}
            agent_names = list(self.agents_dict.keys())

            if agent_name:
                if agent_name in agent_names:
                    return HTTPException(status_code=200, detail="Agent found")
                else:
                    return HTTPException(status_code=404, detail="Agent not found")
            

        except Exception as e:
            print(f" Error listing agents: {str(e)}")
            return {"error": str(e)}

    # Select a specific thread to work with
    def select_thread(self, thread_id):
        self.thread_id = thread_id
        return thread_id

    # Upload a file and attach it to the agent as a tool resource
    def upload_file(self, agent_id: str, upload_file: UploadFile):
        try:
            # Save file to /tmp on App Service
            temp_dir = "/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, upload_file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)

            # Upload the temp file to Azure AI Project
            uploaded_file = self.client.agents.upload_file_and_poll(
                file_path=temp_path,
                purpose=FilePurpose.AGENTS
            )

            # Fetch existing agent configuration
            agent = self.client.agents.get_agent(agent_id=agent_id)
            existing_file_ids = agent.get("tool_resources", {}).get("code_interpreter", {}).get("file_ids", [])
            existing_tools = agent.get("tools", [])
            existing_instructions = agent.get("instructions", "")

            # Append new file ID to the list
            updated_file_ids = list(set(existing_file_ids + [uploaded_file.id]))

            # Initialize code interpreter tool
            code_interpreter = CodeInterpreterTool(file_ids=updated_file_ids)
            updated_tools = code_interpreter.definitions

            # Preserve existing tools (e.g., OpenAPI tool)
            for tool in existing_tools:
                if tool.get("type") == "code_interpreter":
                    continue  # Skip, as we are updating it
                elif tool.get("type") == "openapi":
                    openapi_tool = OpenApiTool(
                        name=tool.get("name", "database"),
                        spec=tool.get("spec", BASE_OPENAPI_SPEC),
                        description=tool.get("description", "Use this tool for information about students"),
                        auth=OpenApiAnonymousAuthDetails()
                    )
                    updated_tools.extend(openapi_tool.definitions)

            # Update the agent with combined tools and resources
            self.client.agents.update_agent(
                agent_id=agent_id,
                instructions=existing_instructions,  # Preserve existing instructions
                tools=updated_tools,
                tool_resources=code_interpreter.resources,
            )

            print(f"✅ File uploaded and agent {agent_id} updated with preserved tools.")
            return uploaded_file.id
            
        except Exception as e:
            print(f"❌ Error uploading file: {str(e)}")
            raise e


    
    def delete_file(self, agent_id: str, file_id: str):
        try:
            self.client.agents.delete_file(file_id=file_id)
            # Fetch existing agent configuration
            agent = self.client.agents.get_agent(agent_id=agent_id)
            existing_file_ids = agent.get("tool_resources", {}).get("code_interpreter", {}).get("file_ids", [])
            existing_tools = agent.get("tools", [])
            existing_instructions = agent.get("instructions", "")

            # Remove the deleted file ID
            updated_file_ids = [fid for fid in existing_file_ids if fid != file_id]

            # Initialize code interpreter tool
            code_interpreter = CodeInterpreterTool(file_ids=updated_file_ids)
            updated_tools = code_interpreter.definitions

            # Preserve existing tools (e.g., OpenAPI tool)
            for tool in existing_tools:
                if tool.get("type") == "code_interpreter":
                    continue  # Skip, as we are updating it
                elif tool.get("type") == "openapi":
                    openapi_tool = OpenApiTool(
                        name=tool.get("name", "database"),
                        spec=tool.get("spec", BASE_OPENAPI_SPEC),
                        description=tool.get("description", "Use this tool for information about students"),
                        auth=OpenApiAnonymousAuthDetails()
                    )
                    updated_tools.extend(openapi_tool.definitions)

            # Update the agent with combined tools and resources
            self.client.agents.update_agent(
                agent_id=agent_id,
                instructions=existing_instructions,  # Preserve existing instructions
                tools=updated_tools,
                tool_resources=code_interpreter.resources,
            )
            return {"status": "success", "message": f"File {file_id} deleted successfully."}
        except Exception as e:
            return {"status": "error", "message": str(e)}


    # Delete a thread and update the JSON blob
    def delete_thread(self, thread_id):
        try:
            # Delete the thread using Azure AI Project API
            self.client.agents.delete_thread(thread_id=thread_id)
            print(f"✅ Deleted thread {thread_id} from Azure AI Project.")
        
            # Remove thread from agent_threads.json
            found = False
            for agent_id, threads in self.agent_thread_map.items():
                if thread_id in threads:
                    threads.remove(thread_id)
                    found = True
        
            # Always save agent_thread_map to ensure consistency
            self.save_agent_thread_map()
            if found:
                print(f"✅ Removed thread {thread_id} from agent_threads.json.")
            else:
                print(f"⚠️ Thread {thread_id} not found in agent_threads.json.")
        
            # Remove thread name from thread_names.json
            thread_names_blob = self.blob_service.get_blob_client(container=BLOB_CONTAINER, blob=THREAD_NAMES_FILENAME)
            thread_name_map = {}
            if thread_names_blob.exists():
                data = thread_names_blob.download_blob().readall()
                thread_name_map = json.loads(data) if data else {}
        
            if thread_id in thread_name_map:
                del thread_name_map[thread_id]
                updated_data = json.dumps(thread_name_map, indent=2)
                thread_names_blob.upload_blob(updated_data, overwrite=True)
                print(f"✅ Removed thread name for thread_id {thread_id} from {THREAD_NAMES_FILENAME}.")
            else:
                print(f"⚠️ Thread {thread_id} not found in {THREAD_NAMES_FILENAME}.")
                
            return {"status": "success", "message": f"Thread {thread_id} deleted."}
        
        except Exception as e:
            print(f"❌ Failed to delete thread: {str(e)}")
            return {"status": "error", "message": str(e)}


    def ask_question(self, agent_id: str, question: str) -> Generator[Dict[str, Any], None, None]:
        try:
            print(f"Starting ask_question for agent_id={agent_id}, thread_id={self.thread_id}")

            if not agent_id or not self.thread_id:
                print("Agent or thread not initialized.")
                yield {"error": "Agent or thread not initialized."}
                return

            # Step 1: Get agent config and preserve existing tools
            agent = self.client.agents.get_agent(agent_id=agent_id)
            print("Fetched agent config.")
            print(f"Agent tools: {agent.get('tools', [])}")
            print(f"Agent tool_resources: {agent.get('tool_resources', {})}")

            # Get file_ids for the agent's code interpreter tool, if any
            file_ids = agent.get("tool_resources", {}).get("code_interpreter", {}).get("file_ids", [])

            # Initialize tools and resources
            code_interpreter = CodeInterpreterTool(file_ids=file_ids)
            existing_tools = agent.get("tools", [])
            existing_tool_definitions = []

            # Preserve existing tools (e.g., OpenAPI tool)
            for tool in existing_tools:
                if tool.get("type") == "code_interpreter":
                    continue
                elif tool.get("type") == "openapi":
                    openapi_tool = OpenApiTool(
                        name=tool.get("name", "database"),
                        spec=tool.get("spec", BASE_OPENAPI_SPEC),
                        description=tool.get("description", "Use this tool for information about students"),
                        auth=OpenApiAnonymousAuthDetails()
                    )
                    existing_tool_definitions.extend(openapi_tool.definitions)

            # Combine code interpreter and existing tool definitions
            updated_tools = code_interpreter.definitions + existing_tool_definitions

            # Update agent with combined tools and resources
            if file_ids or existing_tool_definitions:
                self.client.agents.update_agent(
                    agent_id=agent_id,
                    tools=updated_tools,
                    tool_resources=code_interpreter.resources
                )
                print("Updated agent with combined tools (code interpreter and existing tools).")
            else:
                print("No code interpreter files or additional tools found. Using existing agent configuration.")

            # Step 2: Send user message
            self.client.agents.create_message(
                thread_id=self.thread_id,
                role="user",
                content=question
            )
            print("User message sent.")

            # Step 3: Stream the agent response
            attempts = 0
            max_attempts = 3
            content_received = False
            base64_image = None  # Store image for yielding at the end

            while attempts < max_attempts:
                print(f"Stream attempt {attempts + 1}")
                with self.client.agents.create_stream(thread_id=self.thread_id, agent_id=agent_id) as stream:
                    current_response = {"response": ""}  # Initialize structured response for text
                    for event_type, event_data, _ in stream:
                        print(f"Event Type: {event_type}, Data: {event_data}")

                        # Handle text delta events for token-by-token streaming
                        if (event_type == "message.delta" or "delta" in event_type.lower()) and hasattr(event_data, "text"):
                            content_received = True
                            token = event_data.text
                            current_response["response"] += token  # Accumulate text for final response
                            yield token  # Stream each text token immediately

                        # Handle image content parts (store for yielding later)
                        elif (event_type == "message.created" or "message" in event_type.lower()) and hasattr(event_data, "content"):
                            content_received = True
                            for part in event_data.content:
                                if part.get("type") == "image_file":
                                    file_id = part.get("image_file", {}).get("file_id")
                                    if file_id:
                                        try:
                                            stream = self.client.agents.get_file_content(file_id)
                                            buffer = BytesIO()
                                            for chunk in stream:
                                                if isinstance(chunk, (bytes, bytearray)):
                                                    buffer.write(chunk)
                                                else:
                                                    raise TypeError("Invalid chunk type")
                                            buffer.seek(0)
                                            base64_image = base64.b64encode(buffer.read()).decode("utf-8")
                                        except Exception as e:
                                            error_msg = f"Failed to retrieve image (file_id: {file_id}): {str(e)}"
                                            current_response["response"] += error_msg
                                            yield  error_msg  # Stream error message as a token

                        # Handle error events
                        elif event_type == "error" or "error" in event_type.lower():
                            yield {"error": f"Stream error: {str(event_data)}"}
                            return

                        # Handle stream completion
                        elif event_type == "done":
                            if content_received:
                                print("Stream completed with content.")
                                break

                    # Retry only if no content was received
                    if not content_received:
                        print(f"Attempt {attempts + 1} received no content. Retrying...")
                        attempts += 1
                        if attempts < max_attempts:
                            time.sleep(2)
                        else:
                            print("No response generated after 3 attempts.")
                            yield {"error": "No response generated by the assistant after 3 attempts."}
                            return
                    else:
                        print("Content received, no further retries needed.")
                        break

            # Yield the image after streaming text
            if base64_image:
                yield {"image-file": base64_image}

        except Exception as e:
            print(f"Exception occurred in ask_question: {e}")
            yield {"error": f"Error: {str(e)}"}
            
    # Delete agent along with its associated threads
    def delete_agent(self, agent_id: str):
        try:
            # Delete associated threads
            if agent_id in self.agent_thread_map:
                thread_ids = self.agent_thread_map[agent_id]
                for thread_id in thread_ids:
                    try:
                        self.client.agents.delete_thread(thread_id=thread_id)
                    except Exception as te:
                        print(f"⚠️ Failed to delete thread {thread_id}: {te}")
 
                # Remove threads from mapping
                del self.agent_thread_map[agent_id]
                self.save_agent_thread_map()
 
            # Delete the agent from Azure
            self.client.agents.delete_agent(agent_id=agent_id)
 
            # Remove agent from local mapping
            agent_name_to_remove = None
            for name, aid in self.agents_dict.items():
                if aid == agent_id:
                    agent_name_to_remove = name
                    break
            if agent_name_to_remove:
                del self.agents_dict[agent_name_to_remove]
 
            return {"status": "success", "message": f"Agent {agent_id} and all associated threads deleted successfully."}
 
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
    # List all files associated with the given agent
    def list_files(self, agent_id: str):
        try:
            # Retrieve the agent details
            agent = self.client.agents.get_agent(agent_id=agent_id)
            file_ids = agent["tool_resources"]["code_interpreter"]["file_ids"]
            
            files_info = []
            for file_id in file_ids:
                try:
                    # Attempt to retrieve file details (hypothetical method)
                    file_details = self.client.agents.get_file(file_id=file_id)
                    
                    # Initialize metadata dictionary
                    file_metadata = {
                        "file_id": file_id,
                        "file_name": "Unknown",
                        "file_size": "0 kb",  # Size in kilobytes
                        "status": "Unknown"
                    
                    }
                    
                    # Extract attributes from OpenAIFile object
                    try:
                        # Try common attributes
                        file_metadata["file_name"] = getattr(file_details, "filename", "Unknown")
                        file_size_bytes = getattr(file_details, "bytes", 0)
                        # Convert bytes to kilobytes (1 KB = 1024 bytes)
                        file_size_kb = round(file_size_bytes / 1024, 2) if file_size_bytes else 0
                        file_metadata["file_size"] = f"{file_size_kb} kb"
                        file_metadata["status"] = getattr(file_details, "status", "Unknown")
                    except Exception as e:
                        print(f"⚠️ Failed to extract attributes for file {file_id}: {str(e)}")
                        continue
                    
                    files_info.append(file_metadata)
                except Exception as e:
                    print(f"⚠️ Failed to process file {file_id}: {str(e)}")
                    continue
            
            return {"files": files_info}
        except Exception as e:
            print(f"❌ Failed to list files for agent {agent_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
                    
    

    
        
    def remove_db_connection(self, agent_id: str):
        try:
            # Fetch existing agent configuration
            agent = self.client.agents.get_agent(agent_id=agent_id)
            existing_file_ids = agent.get("tool_resources", {}).get("code_interpreter", {}).get("file_ids", [])
            existing_tools = agent.get("tools", [])
            existing_instructions = agent.get("instructions", "")

            # Initialize updated tools, keeping only non-OpenAPI tools
            code_interpreter = CodeInterpreterTool(file_ids=existing_file_ids)
            updated_tools = code_interpreter.definitions  # Start with code interpreter tool

            # Preserve other non-OpenAPI tools (if any)
            for tool in existing_tools:
                if tool.get("type") == "code_interpreter":
                    continue  # Skip, as we already added it
                elif tool.get("type") == "openapi":
                    continue  # Explicitly skip OpenAPI tool to remove database access
                else:
                    # Preserve any other tools (if applicable in the future)
                    updated_tools.append(tool)

            # Update instructions to remove database-related guidance
            updated_instructions = """
             "You're an intelligent AI assistant. "
                "Read the csv file(s) uploaded. "
                "Your task is to analyze knowledge base and provide accurate, context-aware answers based solely on the data in that file."
                "If any general questions are asked, you can answer them as well but check if the query is about files uploaded first."
            """.strip()

            # Update the agent with new tools, resources, and instructions
            self.client.agents.update_agent(
                agent_id=agent_id,
                instructions=updated_instructions,
                tools=updated_tools,
                tool_resources=code_interpreter.resources,
            )

            print(f"✅ Database access (OpenAPI tool) removed for agent {agent_id}. Agent updated with preserved code interpreter files and new instructions.")
            return {"status": "success", "message": f"Database access removed for agent {agent_id}."}

        except Exception as e:
            print(f"❌ Failed to remove database access for agent {agent_id}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def is_db_active(self, agent_id: str):
        try:
            # Fetch the agent's configuration
            agent = self.client.agents.get_agent(agent_id=agent_id)
            existing_tools = agent.get("tools", [])

            # Check if an OpenAPI tool is present
            has_openapi_tool = any(tool.get("type") == "openapi" for tool in existing_tools)

            if has_openapi_tool:
                return {"status_code": 200, "detail": "Database connection is active for agent."}
            else:
                return {"status_code": 404, "detail": "No database connection found for agent."}

        except Exception as e:
            print(f"❌ Failed to check database connection for agent {agent_id}: {str(e)}")
            return {"status_code": 500, "detail": f"Error checking database connection: {str(e)}"}

    def update_agent_with_openapi_tool(self, agent_id: str):
        try:
            # Check if the database connection is already active
            db_status = self.is_db_active(agent_id)
            if db_status["status_code"] == 200:
                print(f"Agent {agent_id} already has an active database connection.")
                return {"status": "success", "message": f"Agent {agent_id} already has an active database connection."}

            # Proceed with database connection and agent update if no active connection 
            if db_status["status_code"] == 404:
                # Load OpenAPI spec from BASE_OPENAPI_SPEC
                openapi_spec = JsonRef.replace_refs(BASE_OPENAPI_SPEC)

                # Create OpenAPI tool
                auth = OpenApiAnonymousAuthDetails()
                openapi_tool = OpenApiTool(
                    name="database",
                    spec=openapi_spec,
                    description="Use this tool for information about students",
                    auth=auth
                )

                # Fetch agent to preserve existing file_ids
                agent = self.client.agents.get_agent(agent_id=agent_id)
                existing_file_ids = agent.get("tool_resources", {}).get("code_interpreter", {}).get("file_ids", [])
                code_interpreter_tool = CodeInterpreterTool(file_ids=existing_file_ids)

                # Compose new instructions
                #schema = self.get_db_schema()
                updated_instructions = f"""
                You're an intelligent AI assistant that can choose between files and analyze them to answer user queries.
                Read both the database and csv files.
                If the query is about Students,departments use the database tool.
                If the query is about files, use the code interpreter tool.
                """.strip()

                # Final update call
                self.client.agents.update_agent(
                    agent_id=agent_id,
                    instructions=updated_instructions,
                    tools=code_interpreter_tool.definitions + openapi_tool.definitions,
                    tool_resources=code_interpreter_tool.resources,
                )

                print(f"✅ Agent {agent_id} successfully updated with OpenAPI tool, preserved code interpreter files, and updated instructions.")
                return {"status": "success", "message": f"Agent {agent_id} successfully updated with OpenAPI tool."}

            # Handle unexpected status codes from is_db_active
            print(f"⚠️ Unexpected status code from is_db_active: {db_status['status_code']}")
            return {"status": "error", "message": f"Unexpected status code: {db_status['status_code']}"}

        except Exception as e:
            print(f"❌ Failed to update agent with OpenAPI tool: {str(e)}")
            return {"status": "error", "message": str(e)}



    
# Create the shared handler instance
handler = AgentHandler()

# API endpoint to create or login an agent
@app.post("/createorloginagent")
def create_or_login_agent(agent_name: str):
    agent_id = handler.create_or_login_agent(agent_name)
    return {"agent_id": agent_id}

# API endpoint to find if agent exists or not
@app.post("/isuserexist")
def is_user_exists(agent_name: str):
    response_code = handler.is_user_exist(agent_name)
    return {"response_code": response_code}
    
# API endpoint to list all threads for a specific agent
@app.get("/listthreads")
def list_threads(agent_id: str):
    handler.agent_thread_map = handler.load_agent_thread_map()
    threads = handler.list_threads(agent_id)
    return {"threads": threads}

# API endpoint to select a thread and return its chat history
@app.post("/selectthread")
def select_thread(thread_id: str):
    try:
        handler.select_thread(thread_id)
        all_messages = []
        continuation = None

        # 1. Fetch complete message history using pagination
        while True:
            if continuation:
                response = handler.client.agents.list_messages(thread_id=thread_id, after=continuation)
            else:
                response = handler.client.agents.list_messages(thread_id=thread_id)

            if not response or not response.data:
                break

            all_messages.extend(response.data)

            if not getattr(response, "has_more", False):
                break

            continuation = response.last_id

        # 2. Parse messages into structured history
        parsed_history = []
        for msg in all_messages:
            for part in msg.content:
                if part["type"] == "text":
                    parsed_history.append({
                        "role": msg.role,
                        "type": "text",
                        "content": part["text"]["value"]
                    })

                elif part["type"] == "image_file":
                    file_id = part.get("image_file", {}).get("file_id")
                    if file_id:
                        try:
                            stream = handler.client.agents.get_file_content(file_id)
                            buffer = BytesIO()
                            for chunk in stream:
                                if isinstance(chunk, (bytes, bytearray)):
                                    buffer.write(chunk)
                                else:
                                    raise TypeError("Invalid chunk type while reading image content")
                            buffer.seek(0)
                            base64_image = base64.b64encode(buffer.read()).decode("utf-8")
                            parsed_history.append({
                                "role": msg.role,
                                "type": "image_base64",
                                "content": base64_image
                            })
                        except Exception as e:
                            parsed_history.append({
                                "role": msg.role,
                                "type": "error",
                                "content": f"Failed to retrieve image (file_id: {file_id}): {str(e)}"
                            })

        return {
            "selected_thread": thread_id,
            "chat_history": parsed_history[::-1]  # Reverse to show most recent messages first
        }

    except Exception as e:
        return {"error": f"Error selecting thread {thread_id}: {str(e)}"}

# API endpoint to upload a file (referenced by path on server)


@app.post("/upload_file")
def upload_file(file: UploadFile = File(...), agent_id: str = Form(...)):
    try:
        file_id = handler.upload_file(agent_id=agent_id, upload_file=file)
        return {"file_id": file_id}
    except Exception as e:
        return {"error": str(e)}



# API endpoint to ask a question in a given thread
@app.post("/ask_question")
def ask_question(agent_id: str, thread_id: str, question: str):
    try:
        handler.agent_id = agent_id
        handler.thread_id = thread_id
        
        def response_generator():
            responses = handler.ask_question(agent_id, question)
            for response in responses:
                yield json.dumps(response)
        
        return StreamingResponse(
            response_generator(),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        return {"error": str(e)}

# API endpoint to delete a thread and update blob mapping
@app.delete("/delete_thread")
def delete_thread(thread_id: str):
    try:
        result = handler.delete_thread(thread_id)
        return result
    except Exception as e:
        return {"error": str(e)}

#API endpoint to create new thread
@app.post("/create_thread")
def create_thread(agent_id: str):
    thread_id = handler.create_new_thread(agent_id=agent_id)
    handler.set_thread_name(thread_id, "New Chat")
    return {"thread_id": thread_id}

#API endpoint to set the thread name
@app.post("/set_thread_name")
def set_thread_name(thread_id: str , thread_name: str):
    return handler.set_thread_name(thread_id, thread_name)

#API endpoint to get the thread name
@app.get("/get_thread_name")
def get_thread_name(thread_id: str):
    return handler.get_thread_name(thread_id)

#API endpoint to delete the file
@app.delete("/delete_file")
def delete_file(agent_id: str, file_id: str):
    try:
        result = handler.delete_file(agent_id, file_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#API endpoint to delete the agent
@app.delete("/delete_agent")
def delete_agent(agent_id: str):
    try:
        result = handler.delete_agent(agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to list all files for a specific agent
@app.get("/list_files")
def list_files(agent_id: str):
    try:
        result = handler.list_files(agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
@app.post("/remove_db_connection")
def remove_db_connection(agent_id: str):
    try:
        result = handler.remove_db_connection(agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/is_db_active")
def is_db_active(agent_id: str):
    try:
        result = handler.is_db_active(agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking database connection: {str(e)}")
        
@app.post("/update_agent_with_openapi_tool")
def update_openapi_tool(agent_id: str):
    try:
        result = handler.update_agent_with_openapi_tool(agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
