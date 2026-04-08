from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from crewai.llm import LLM
from tools.memory_tool import MemoryTool
from tools.remote_exec_tool import RemoteExecutionTool
from tools.chatopenai import CustomChatOpenAI

@CrewBase
class RemoteExecutionCrew:
    agents_config="config/remote_execute_agents.yaml"
    tasks_config="config/remote_execute_tasks.yaml"
    llm = LLM(
        model="your-local-model",
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty" 
    )

    @agent 
    def remote_execution_agent(self)->Agent:
        remote_execution_tool=RemoteExecutionTool.execute_on_remote_server
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key
        return Agent(
            config=self.agents_config["remote_execution_agent"],
            tools=[remote_execution_tool,memory_store_tool,memory_retrieve_tool,memory_get_key_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def remote_execution(self)->Task:
        return Task(config=self.tasks_config["remote_execution_task"])

    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )