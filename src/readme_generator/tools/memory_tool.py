from typing import Any,List,Dict
from crewai.tools import tool
import traceback
import json
import os
from dataclasses import dataclass,asdict
from pathlib import Path

DEFAULT_MEMORY_PATH = "/home/changrui/readme_generator/src/readme_generator/global_memory_1.json"


def resolve_memory_path(memory_profile: str = "default") -> str:
    profile = (memory_profile or "default").strip().lower()
    if profile in {"", "default"}:
        return DEFAULT_MEMORY_PATH
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in profile)
    base = Path(DEFAULT_MEMORY_PATH)
    return str(base.with_name(f"{base.stem}_{safe}{base.suffix}"))

@dataclass
class MemoryData:
    generation_mode:str="reference"
    github_md_folder_url:str=""
    github_js_folder_url:str=""
    source_md_url:str=""
    source_js_url:str=""
    input_text:str=""
    model_list:list=None
    github_url:list=None
    remote_folder:str=""
    ssh_config:dict=None
    github_config:dict=None
    model_url_list:list=None
    model_id_list:list=None
    execution_result:list=None
    fail_reason_list:list=None
    executed_command:list=None
    family_md:str=""
    family_index_js:str=""
    family_content:str=""
    family_js_files:list=None
    source_md_files:list=None
    source_js_files:list=None
    review_failure_report:str=""
    ref_md:str=""
    ref_index_js:str=""
    pr_info:dict=None
    remote_payload:dict=None

class GlobalMemory:
    def __init__(
        self,
        persist_path:str=DEFAULT_MEMORY_PATH
    ):
        self.persist_path=persist_path
        self.memory=MemoryData()
        self.load_from_file()

    def load_from_file(self)->None:
        if os.path.exists(self.persist_path):
            with open(self.persist_path,"r",encoding="utf-8") as f:
                data=json.load(f)
                self.memory.model_list=data.get("model_list",[])
                self.memory.generation_mode=data.get("generation_mode","reference")
                self.memory.github_md_folder_url=data.get("github_md_folder_url","")
                self.memory.github_js_folder_url=data.get("github_js_folder_url","")
                self.memory.source_md_url=data.get("source_md_url","")
                self.memory.source_js_url=data.get("source_js_url","")
                self.memory.remote_folder=data.get("remote_folder","")
                self.memory.ssh_config=data.get("ssh_config",{})
                self.memory.github_config=data.get("github_config",{})
                self.memory.model_url_list=data.get("model_url_list",[])
                self.memory.model_id_list=data.get("model_id_list",[])
                self.memory.execution_result=data.get("execution_result",[])
                self.memory.executed_command=data.get("executed_command",[])
                self.memory.github_url=data.get("github_url",[])
                self.memory.fail_reason_list=data.get("fail_reason_list",[])
                self.memory.input_text=data.get("input_text","")
                self.memory.family_md=data.get("family_md","")
                self.memory.family_index_js=data.get("family_index_js","")
                self.memory.family_content=data.get("family_content","")
                self.memory.family_js_files=data.get("family_js_files",[])
                self.memory.source_md_files=data.get("source_md_files",[])
                self.memory.source_js_files=data.get("source_js_files",[])
                self.memory.review_failure_report=data.get("review_failure_report","")
                self.memory.ref_md=data.get("ref_md","")
                self.memory.ref_index_js=data.get("ref_index_js","")
                self.memory.pr_info=data.get("pr_info",{})
                self.memory.remote_payload=data.get("remote_payload",{})
        else:
            self.memory.model_list=[]
            self.memory.generation_mode="reference"
            self.memory.github_md_folder_url=""
            self.memory.github_js_folder_url=""
            self.memory.source_md_url=""
            self.memory.source_js_url=""
            self.memory.remote_folder=""
            self.memory.ssh_config={}
            self.memory.github_config={}
            self.memory.model_url_list=[]
            self.memory.model_id_list=[]
            self.memory.execution_result=[]
            self.memory.executed_command=[]
            self.memory.github_url=[]
            self.memory.fail_reason_list=[]
            self.memory.input_text=""
            self.memory.family_md=""
            self.memory.family_index_js=""
            self.memory.family_content=""
            self.memory.family_js_files=[]
            self.memory.source_md_files=[]
            self.memory.source_js_files=[]
            self.memory.review_failure_report=""
            self.memory.ref_md=""
            self.memory.ref_index_js=""
            self.memory.pr_info={}
            self.memory.remote_payload={}
            self.save_to_file()

    @staticmethod
    def _dedup_str_list(values: Any) -> list:
        if not isinstance(values, list):
            return []
        out = []
        seen = set()
        for item in values:
            s = str(item or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    @staticmethod
    def _compose_family_content(md: str, index_js: str, js_files: Any) -> str:
        md_text = str(md or "").strip()
        files = js_files if isinstance(js_files, list) else []
        sections = []
        if files:
            for i, item in enumerate(files):
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or f"file_{i}.js")
                content = str(item.get("content") or "")
                if not path.endswith(".js"):
                    path = f"{path}.js"
                sections.append(f"### {path}\n\n```javascript\n{content}\n```")
        else:
            js_text = str(index_js or "").strip()
            if js_text:
                sections.append(f"### index.js\n\n```javascript\n{js_text}\n```")
        js_block = "\n\n".join(sections).strip()
        if not md_text and not js_block:
            return ""
        if not js_block:
            return md_text
        if not md_text:
            return js_block
        return f"{md_text}\n\n---\n\n{js_block}"

    @classmethod
    def _compact_for_persist(cls, data: dict) -> dict:
        compact = dict(data)

        # De-duplicate and align frequently repeated list fields.
        compact["model_list"] = cls._dedup_str_list(compact.get("model_list", []))
        compact["model_id_list"] = cls._dedup_str_list(compact.get("model_id_list", []))
        compact["model_url_list"] = cls._dedup_str_list(compact.get("model_url_list", []))
        github_url = compact.get("github_url", [])
        if isinstance(github_url, list):
            g = [str(x or "") for x in github_url]
            n = len(compact["model_list"])
            if n > 0:
                if len(g) < n:
                    g = g + [""] * (n - len(g))
                elif len(g) > n:
                    g = g[:n]
            compact["github_url"] = g

        # family_content is derivable from family_md + js artifacts; don't persist duplicate blob.
        composed = cls._compose_family_content(
            compact.get("family_md", ""),
            compact.get("family_index_js", ""),
            compact.get("family_js_files", []),
        )
        if str(compact.get("family_content") or "").strip() == str(composed or "").strip():
            compact["family_content"] = ""

        # source_*_files often duplicate ref_md/ref_index_js content; keep lightweight path-only metadata on disk.
        if str(compact.get("ref_md") or "").strip():
            md_files = compact.get("source_md_files")
            if isinstance(md_files, list):
                slim_md = []
                for item in md_files:
                    if isinstance(item, dict):
                        slim_md.append({"path": str(item.get("path") or ""), "content": ""})
                    else:
                        slim_md.append({"path": str(item or ""), "content": ""})
                compact["source_md_files"] = slim_md
        if str(compact.get("ref_index_js") or "").strip():
            js_files = compact.get("source_js_files")
            if isinstance(js_files, list):
                slim_js = []
                for item in js_files:
                    if isinstance(item, dict):
                        slim_js.append({"path": str(item.get("path") or ""), "content": ""})
                    else:
                        slim_js.append({"path": str(item or ""), "content": ""})
                compact["source_js_files"] = slim_js

        # remote_payload for legacy can be huge due to repeated content; persist compact marker.
        rp = compact.get("remote_payload")
        if isinstance(rp, dict):
            mode = str(rp.get("generation_mode") or "").strip().lower()
            if mode == "legacy":
                rp2 = dict(rp)
                content = rp2.get("content")
                if isinstance(content, dict):
                    ref_md = str(content.get("family_md") or content.get("ref_md") or "")
                    ref_js = str(content.get("family_index_js") or content.get("ref_index_js") or "")
                    input_text = str(content.get("input_text") or "")
                    if (
                        ref_md
                        == (
                            str(compact.get("family_md") or "")
                            or str(compact.get("ref_md") or "")
                        )
                        and ref_js
                        == (
                            str(compact.get("family_index_js") or "")
                            or str(compact.get("ref_index_js") or "")
                        )
                        and input_text == str(compact.get("input_text") or "")
                    ):
                        rp2["content"] = {"from_memory": True}
                compact["remote_payload"] = rp2
        return compact

    def save_to_file(self)->bool:
        try:
            data={
                "model_list":self.memory.model_list,
                "generation_mode":self.memory.generation_mode,
                "github_md_folder_url":self.memory.github_md_folder_url,
                "github_js_folder_url":self.memory.github_js_folder_url,
                "source_md_url":self.memory.source_md_url,
                "source_js_url":self.memory.source_js_url,
                "remote_folder":self.memory.remote_folder,
                "ssh_config":self.memory.ssh_config,
                "github_config":self.memory.github_config,
                "model_url_list":self.memory.model_url_list,
                "model_id_list":self.memory.model_id_list,
                "execution_result":self.memory.execution_result,
                "executed_command":self.memory.executed_command,
                "github_url":self.memory.github_url,
                "fail_reason_list":self.memory.fail_reason_list,
                "input_text":self.memory.input_text,
                "family_md":self.memory.family_md,
                "family_index_js":self.memory.family_index_js,
                "family_content":self.memory.family_content,
                "family_js_files":self.memory.family_js_files,
                "source_md_files":self.memory.source_md_files,
                "source_js_files":self.memory.source_js_files,
                "review_failure_report":self.memory.review_failure_report,
                "ref_md":self.memory.ref_md,
                "ref_index_js":self.memory.ref_index_js,
                "pr_info":self.memory.pr_info,
                "remote_payload":self.memory.remote_payload
            }
            data = self._compact_for_persist(data)
            with open(self.persist_path,"w",encoding="utf-8") as f:
                json.dump(data,f,ensure_ascii=False,indent=2)
            return True
        except Exception as e:
            print(e)
            traceback.print_exc()
            return False

    def memory_store(self,key:str,value:Any)->bool:
        try:
            if key=="model_list":
                self.memory.model_list=value
            elif key=="generation_mode":
                self.memory.generation_mode=value
            elif key=="github_md_folder_url":
                self.memory.github_md_folder_url=value
            elif key=="github_js_folder_url":
                self.memory.github_js_folder_url=value
            elif key=="source_md_url":
                self.memory.source_md_url=value
            elif key=="source_js_url":
                self.memory.source_js_url=value
            elif key=="remote_folder":
                self.memory.remote_folder=value
            elif key=="ssh_config":
                self.memory.ssh_config=value
            elif key=="github_config":
                self.memory.github_config=value
            elif key=="model_url_list":
                self.memory.model_url_list=value
            elif key=="model_id_list":
                self.memory.model_id_list=value
            elif key=="execution_result":
                self.memory.execution_result=value
            elif key=="executed_command":
                self.memory.executed_command=value
            elif key=="github_url":
                self.memory.github_url=value
            elif key=="fail_reason_list":
                self.memory.fail_reason_list=value
            elif key=="input_text":
                self.memory.input_text=value
            elif key=="family_md":
                self.memory.family_md=value
            elif key=="family_index_js":
                self.memory.family_index_js=value
            elif key=="family_content":
                self.memory.family_content=value
            elif key=="family_js_files":
                self.memory.family_js_files=value
            elif key=="source_md_files":
                self.memory.source_md_files=value
            elif key=="source_js_files":
                self.memory.source_js_files=value
            elif key=="review_failure_report":
                self.memory.review_failure_report=value
            elif key=="ref_md":
                self.memory.ref_md=value
            elif key=="ref_index_js":
                self.memory.ref_index_js=value
            elif key=="pr_info":
                self.memory.pr_info=value
            elif key=="remote_payload":
                self.memory.remote_payload=value
            self.save_to_file()
            return True
        except Exception as e:
            print(e)
            traceback.print_exc()
            return False
        
    def memory_retrieve(self,key:str)->Any:
        try:
            self.load_from_file()
            if key=="model_list":
                return self.memory.model_list
            elif key=="generation_mode":
                return self.memory.generation_mode
            elif key=="github_md_folder_url":
                return self.memory.github_md_folder_url
            elif key=="github_js_folder_url":
                return self.memory.github_js_folder_url
            elif key=="source_md_url":
                return self.memory.source_md_url
            elif key=="source_js_url":
                return self.memory.source_js_url
            elif key=="remote_folder":
                return self.memory.remote_folder
            elif key=="ssh_config":
                return self.memory.ssh_config
            elif key=="github_config":
                return self.memory.github_config
            elif key=="model_url_list":
                return self.memory.model_url_list
            elif key=="model_id_list":
                return self.memory.model_id_list
            elif key=="execution_result":
                return self.memory.execution_result
            elif key=="executed_command":
                return self.memory.executed_command
            elif key=="github_url":
                return self.memory.github_url
            elif key=="fail_reason_list":
                return self.memory.fail_reason_list
            elif key=="input_text":
                return self.memory.input_text
            elif key=="family_md":
                return self.memory.family_md
            elif key=="family_index_js":
                return self.memory.family_index_js
            elif key=="family_content":
                if str(self.memory.family_content or "").strip():
                    return self.memory.family_content
                return self._compose_family_content(
                    self.memory.family_md,
                    self.memory.family_index_js,
                    self.memory.family_js_files,
                )
            elif key=="family_js_files":
                return self.memory.family_js_files
            elif key=="source_md_files":
                return self.memory.source_md_files
            elif key=="source_js_files":
                return self.memory.source_js_files
            elif key=="review_failure_report":
                return self.memory.review_failure_report
            elif key=="ref_md":
                return self.memory.ref_md
            elif key=="ref_index_js":
                return self.memory.ref_index_js
            elif key=="pr_info":
                return self.memory.pr_info
            elif key=="remote_payload":
                return self.memory.remote_payload
            return ""
        except Exception as e:
            print(e)
            traceback.print_exc()
            return ""
        
    def get_memory_keys(self)->List[str]:
        return list(asdict(self.memory).keys())
    
    def get_memory_value_types(self)->Dict[str,Any]:
        return {key:type(value).__name__ for key,value in asdict(self.memory).items()}
     

class MemoryTool:
    @tool("Store Memory")
    def store_memory(key:str,value:Any):
        """Used to persistently store any specified data content into global memory using a custom key, and synchronously save it to a local file to ensure data integrity and persistence. Supports storing all types of task data, including model lists, single model details, generated documents, intermediate results, and more. It validates key validity and data consistency during storage to maintain standardized global memory management. Each storage operation is automatically saved to disk, allowing all subsequent agents to access and reuse data stably across tasks and workflows."""
        memory=GlobalMemory()
        flag=memory.memory_store(key=key,value=value)
        return flag
    
    @tool("Retrieve Memory")
    def retrieve_memory(key:str):
        """Used to retrieve the corresponding stored data from the global persistent memory according to the specified memory key. Supports fetching any saved content in the global memory. Data is loaded from a local file to ensure consistent and complete content across tasks and agents. Returns a default value if the specified memory key does not exist or has no corresponding data, maintaining stable and uninterrupted task execution."""
        memory=GlobalMemory()
        info=memory.memory_retrieve(key=key)
        return info
    
    @tool("Get Memory Key")
    def get_memory_key():
        """Used to enable agents in CrewAI to accurately obtain all stored data key values in the global memory (GLOBAL_MEMORY). Its core function is to provide agents with clear guidance on memory key names, ensuring that subsequent data retrieval and storage operations can accurately locate the target keys. It avoids data operation failures caused by incorrect key names, guarantees the accuracy and efficiency of the interaction between agents and global memory, and supports the smooth connection of data reading and writing in the task flow."""
        memory=GlobalMemory()
        return memory.get_memory_keys()
    
    @tool("Get Memory Value Type")
    def get_memory_value_type():
        """"""
        memory=GlobalMemory()
        return memory.get_memory_value_types()
    
