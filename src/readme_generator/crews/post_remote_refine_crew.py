import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_path not in sys.path:
    sys.path.append(root_path)

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from readme_generator.tools.post_remote_refine_tool import PostRemoteRefineTool

LLM_BASE_URL = os.getenv("README_GENERATOR_LLM_BASE_URL", "http://10.54.34.78:30000/v1")
LLM_MODEL = os.getenv("README_GENERATOR_LLM_MODEL", "your-local-model")
LLM_API_KEY = os.getenv("README_GENERATOR_LLM_API_KEY", "empty")


@CrewBase
class PostRemoteRefineCrew:
    agents_config = "config/post_remote_refine_agents.yaml"
    tasks_config = "config/post_remote_refine_tasks.yaml"
    llm = LLM(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )

    def __init__(self, global_memory):
        self.global_memory = global_memory
        PostRemoteRefineTool.global_memory = global_memory

    @agent
    def post_remote_refine_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["post_remote_refine_agent"],
            tools=[
                PostRemoteRefineTool.memory_retrieve_post_remote_context,
                PostRemoteRefineTool.memory_store_refined_family_artifacts,
            ],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    @task
    def refine_after_remote_task(self) -> Task:
        return Task(
            config=self.tasks_config["refine_after_remote_task"],
            agent=self.post_remote_refine_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            stream=True,
        )

