from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task
from tools.chatopenai import CustomChatOpenAI
from tools.github_pr_tool import GithubPRTool
from tools.memory_tool import MemoryTool


@CrewBase
class GithubPRCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=CustomChatOpenAI(base_url="http://10.54.34.78:30000/v1",password="empty")

    @agent 
    def github_agent(self)->Agent:
        github_tool=GithubPRTool()
        memory_tool=MemoryTool()
        return Agent(
            config=self.agents_config["github_agent"],
            tools=[github_tool,memory_tool],
            llm=self.llm,
            verbose=True
        )
    
    @task 
    def github_pr(self)->Task:
        return Task(config=self.tasks_config["github_pr_task"])
    
    @crew
    def crew(self)->Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )