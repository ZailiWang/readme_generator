from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task

from tools.memory_tool import MemoryTool
from tools.remote_exec_tool import RemoteExecutionTool
from tools.chatopenai import CustomChatOpenAI

@CrewBase
class RemoteExecutionCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")

    @agent 
    def remote_execution_agent(self)->Agent:
        remote_execution_tool=RemoteExecutionTool()
        memory_tool=MemoryTool()
        return Agent(
            config=self.agents_config["remote_execution_agent"],
            tools=[remote_execution_tool,memory_tool],
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