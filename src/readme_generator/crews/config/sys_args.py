from dataclasses import dataclass,field,asdict
from typing import List,Literal,Optional,Dict,Any
from transformers import HfArgumentParser

@dataclass
class RemoteArguments:
    host:str=field(default="10.239.60.71")
    user_name:str=field(default="yuchangrui")
    password:str=field(default="Ycr2wy1314")
    def get(self,key,default=None):
        return getattr(self,key,default)
    
@dataclass
class ModelArguments:
    pass

remote_args,model_args=HfArgumentParser(
    (RemoteArguments,ModelArguments)
).parse_args_into_dataclasses()
