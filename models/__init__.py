from .TimeLLM import Model as TimeLLM
from .logicgate import Model as logicgate

# Backward-compatible export for older code paths.
RuleGatingTIMELLM = logicgate

__all__ = ['TimeLLM', 'logicgate', 'RuleGatingTIMELLM']
