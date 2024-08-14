from typing import Union, Any, Optional
from pydantic import BaseModel, StrictBool


class MainConfig(BaseModel):
    class Config:
        extra = 'allow'
    launch_config: Any = None
    distributed: Optional[StrictBool] = False

