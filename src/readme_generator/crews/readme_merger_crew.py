from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task

from tools.merge_content_tool import MergeContentTool
from tools.memory_tool import MemoryTool
from tools.chatopenai import CustomChatOpenAI


@CrewBase
class ReadmeMergerCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")

    @agent
    def merge_readme_agent(self)->Agent:
        merge_readme_tool=MergeContentTool()
        memory_tool=MemoryTool()
        return Agent(
            config=self.agents_config["merge_readme_agent"],
            tools=[merge_readme_tool,memory_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def readme_merge(self)->Task:
        return Task(config=self.tasks_config["readme_merge_task"])

    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )