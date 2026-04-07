from crewai import Agent,Crew,Process,Task
from crewai.project import CrewBase,agent,crew,task

from readme_generator.tools.model_search_tool import ModelSearchTool
from readme_generator.tools.github_pr_tool import GithubPRTool
from readme_generator.tools.merge_content_tool import MergeContentTool
from readme_generator.tools.remote_exec_tool import RemoteExecutionTool
from readme_generator.tools.memory_tool import MemoryTool
from langchain_openai import ChatOpenAI

@CrewBase
class GithubCrew:
    agents_config="config/agents.yaml"
    tasks_config="config/tasks.yaml"
    llm=ChatOpenAI(model="gpt-4o")

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
    
    @agent 
    def model_search_agent(self)->Agent:
        model_search_tool=ModelSearchTool()
        memory_tool=MemoryTool()
        return Agent(
            config=self.agents_config["model_search_agent"],
            tools=[model_search_tool,memory_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
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
    def github_pr(self)->Task:
        return Task(config=self.tasks_config["github_pr_task"])
    
    @task
    def readme_generate(self)->Task:
        return Task(config=self.tasks_config["readme_generate_task"])
    
    @task
    def model_search(self)->Task:
        return Task(config=self.tasks_config["model_search_task"])

    @task
    def remote_execution(self)->Task:
        return Task(config=self.tasks_config["remote_execution_task"])

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