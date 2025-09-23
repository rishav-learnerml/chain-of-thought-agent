from typing import Literal
from typing import Optional, Dict
from pydantic import BaseModel, Field

class ToolSchema(BaseModel):
    name: str = Field(...,description="Name of the tool to call")
    input: Dict[str, str] = Field(...,description="Dictionary of input parameters for the tool")

class OutputSchema(BaseModel):
    step: Literal["START", "PLAN", "OUTPUT", "TOOL"] = Field(description="The Current Step of the Chain of Thought")
    content: str = Field(description="The content of the output")
    tool: Optional[ToolSchema] = Field(
        default=None,
        description="Tool call details if step is TOOL, else None."
    )