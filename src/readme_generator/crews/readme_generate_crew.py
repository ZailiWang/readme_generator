from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task

from tools.memory_tool import MemoryTool
from tools.chatopenai import CustomChatOpenAI


@CrewBase
class ReadmeGeneratorCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")

    @agent
    def readme_generator_agent(self)->Agent:
        memory_tool=MemoryTool()
        return Agent(
            config=self.agents_config["readme_generator_agent"],
            llm=self.llm,
            tools=[memory_tool],
            verbose=True,
            allow_delegation=True
        )
    
    @task
    def readme_generate(self)->Task:
        return Task(config=self.tasks_config["readme_generate_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )