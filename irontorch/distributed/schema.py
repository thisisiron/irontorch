from typing import Union, Any, Optional
from pydantic import BaseModel, StrictBool

class Config(BaseModel):
    class Config:
        extra = 'allow'
    launch_config: Any
    distributed: Optional[StrictBool]


