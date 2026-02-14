"""
Two-track rule optimisation system

The package achieves the smart rule optimization framework based on a two-track evaluation system，
By CallQwen APIAutomatic optimization and security replacement of rules。

Module Structure：
- evaluation_analyzer: Core analysis module（Phase I）
- config_manager: Configure Management Module（Phase II）
- api_client: APIInteractive module（Phase III）
- data_processor: Data processing module（Phase IV）
- rule_manager: Rule management module（Phase V）
- executor: Implementation and monitoring module（Phase VI）
- orchestrator: Integration and optimization module（Phase VII）
- enhanced_optimizer: System integration module（Phase VIII）
"""

from .evaluation_analyzer import EvaluationAnalyzer 
from .configuration_manager import (
ConfigurationManager ,
APIConfig ,
ThresholdConfig ,
ValidationConfig 
)
from .prompt_engineering import PromptEngineering 
from .prompt_templates import PromptTemplates 
from .gpt_api_client import GPTAPIClient 
from .api_response_parser import APIResponseParser ,ParseResult 
from .structured_converter import StructuredConverter 
from .rule_version_manager import RuleVersionManager ,VersionInfo ,RollbackResult 
from .replacement_decision_engine import (
ReplacementDecisionEngine ,ReplacementAction ,ReplacementDecision ,RuleMetrics 
)
from .rule_replacement_executor import RuleReplacementExecutor ,ExecutionResult ,PhaseResult 
from .performance_validator import (
PerformanceValidator ,BaselineMetrics ,ValidationResult ,PerformanceAlert 
)
from .optimization_orchestrator import (
OptimizationOrchestrator ,OrchestrationConfig 
)
from .data_structures import (
QualityReport ,
OptimizationContext ,
APIResponse ,
StructuredRuleConfig ,
ReplacementResult ,
OptimizationSession ,
ValidationError ,
ParsedResponse ,
create_optimization_context ,
create_api_response ,
create_replacement_result ,
validate_quality_report 
)

__version__ ="1.0.0"
__author__ ="Dual Track Optimization Team"

__all__ =[
# Core analysis
'EvaluationAnalyzer',

# Configuration Management
'ConfigurationManager',
'APIConfig',
'ThresholdConfig',
'ValidationConfig',

# APIInteractive
'PromptEngineering',
'PromptTemplates',
'GPTAPIClient',

# Data processing
'APIResponseParser',
'ParseResult',
'StructuredConverter',

# Rule management
'RuleVersionManager',
'VersionInfo',
'RollbackResult',
'ReplacementDecisionEngine',
'ReplacementAction',
'ReplacementDecision',
'RuleMetrics',

# Implementation and monitoring
'RuleReplacementExecutor',
'ExecutionResult',
'PhaseResult',
'PerformanceValidator',
'BaselineMetrics',
'ValidationResult',
'PerformanceAlert',

# Integration and optimization
'OptimizationOrchestrator',
'OrchestrationConfig',
# Data structure
'QualityReport',
'OptimizationContext',
'APIResponse',
'StructuredRuleConfig',
'ReplacementResult',
'OptimizationSession',
'ValidationError',
'ParsedResponse',
'create_optimization_context',
'create_api_response',
'create_replacement_result',
'validate_quality_report'
]
