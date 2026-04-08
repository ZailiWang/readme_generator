from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task

from tools.merge_content_tool import MergeContentTool
from tools.memory_tool import MemoryTool
from tools.chatopenai import CustomChatOpenAI


@CrewBase
class ReadmeMergerCrew:
    agents_config="config/readme_merge_agents.yaml"
    tasks_config="config/readme_merge_tasks.yaml"
    llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")

    @agent
    def merge_readme_agent(self)->Agent:
        merge_readme_tool=MergeContentTool.merge_readme
        memory_store_tool=MemoryTool.store_memory
        memory_retrieve_tool=MemoryTool.retrieve_memory
        memory_get_key_tool=MemoryTool.get_memory_key
        return Agent(
            config=self.agents_config["merge_readme_agent"],
            tools=[merge_readme_tool,memory_store_tool,memory_get_key_tool,memory_retrieve_tool],
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