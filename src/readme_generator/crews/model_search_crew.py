from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from tools.model_search_tool import ModelSearchTool
from tools.memory_tool import MemoryTool
from tools.chatopenai import CustomChatOpenAI

@CrewBase
class ModelSearchCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")

    @agent
    def model_search_agent(self)->Agent:
        model_search_tool=ModelSearchTool()
        memory_tool=MemoryTool()
        return Agent(
            config=self.agents_config["model_search_agent"],
            tools=[model_search_tool,memory_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True,
        )
    
    @task
    def model_search(self)->Task:
        import pdb;pdb.set_trace()
        return Task(config=self.tasks_config["model_search_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )