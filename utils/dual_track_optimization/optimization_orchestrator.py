"""
Optimizing organiser

Coordination of the entire two-track rule optimization process，Integration of all components of the first six phases into a complete end-to-end system。
Provide a unified interface，Manage optimized sessions，Coordination of workflows for modules。
"""

import logging 
import time 
from datetime import datetime 
from typing import Dict ,List ,Any ,Optional 
from dataclasses import dataclass 

from .data_structures import (
OptimizationSession ,ReplacementResult ,ValidationError ,APIResponse 
)
from .evaluation_analyzer import EvaluationAnalyzer 
from .configuration_manager import ConfigurationManager 
from .prompt_engineering import PromptEngineering 
from .gpt_api_client import GPTAPIClient 
from .api_response_parser import APIResponseParser 
from .structured_converter import StructuredConverter 
from .rule_version_manager import RuleVersionManager 
from .replacement_decision_engine import ReplacementDecisionEngine ,ReplacementAction 
from .rule_replacement_executor import RuleReplacementExecutor 
from .performance_validator import PerformanceValidator 


@dataclass 
class OrchestrationConfig :
    """Organisation"""
    max_concurrent_optimizations :int =3 
    optimization_timeout :int =3600 # sec
    enable_performance_validation :bool =True 
    enable_automatic_rollback :bool =True 
    batch_size :int =5 
    retry_attempts :int =2 




class OptimizationOrchestrator :
    """
    Optimizing organiser
    
    Coordination of the entire two-track rule optimization process，Integration of all modules，
    Provide a uniform optimized interface and session management functionality。
    """

    def __init__ (self ,config :Dict [str ,Any ],model =None ):
        self .config =config 
        self .logger =logging .getLogger (__name__ )
        self .model =model # Add Model Reference，For reloading rules

        # Organisation
        self .orchestration_config =OrchestrationConfig (**config .get ('orchestration',{}))

        # Initialize all components
        self ._initialize_components ()

        # Session Management
        self .active_sessions :Dict [str ,OptimizationSession ]={}
        self .session_history :List [OptimizationSession ]=[]

    def _initialize_components (self ):
        """Initialize all components"""
        try :
        # Phase I：Core analysis
            self .analyzer =EvaluationAnalyzer (self .config .get ('thresholds',{}))

            # Phase II：Infrastructure
            self .config_manager =ConfigurationManager ()

            # Phase III：APIInteractive
            self .prompt_engineering =PromptEngineering ()
            self .gpt_client =GPTAPIClient (self .config_manager )

            # Phase IV：Data processing
            self .response_parser =APIResponseParser ()
            self .structured_converter =StructuredConverter ()

            # Phase V：Rule management
            self .version_manager =RuleVersionManager ()
            self .decision_engine =ReplacementDecisionEngine (self .config )

            # Phase VI：Implementation and monitoring
            self .replacement_executor =RuleReplacementExecutor (self .version_manager ,self .config ,self .model )
            self .performance_validator =PerformanceValidator (self .config )

            self .logger .info ("Initialization of all components completed")

        except Exception as e :
            self .logger .error (f"Component initialization failed: {e }")
            raise ValidationError (f"Orchestrator initialization failed: {str (e )}")

    def create_optimization_session (self ,epoch :int ,target_rules :Optional [List [int ]]=None )->str :
        """
        Create optimized session
        
        Args:
            epoch: Training rotations
            target_rules: List of destination rules（NoneFor automatic selection）
            
        Returns:
            str: SessionID
        """
        session_id =f"opt_session_{epoch }_{int (time .time ())}"

        session =OptimizationSession (
        session_id =session_id ,
        epoch =epoch ,
        start_time =datetime .now (),
        status ="CREATED"
        )

        self .active_sessions [session_id ]=session 

        self .logger .info (f"Create optimization session: {session_id }")
        return session_id 

    def run_full_optimization (self ,session_id :str ,model_outputs :Optional [Dict ]=None )->OptimizationSession :
        """
        Run complete optimization process

        Args:
            session_id: SessionID
            model_outputs: Model output data，Organisationoptimization_contextsWait.

        Returns:
            OptimizationSession: Completed Optimization Session
        """
        if session_id not in self .active_sessions :
            raise ValidationError (f"Session does not exist: {session_id }")

        if not model_outputs or 'optimization_contexts'not in model_outputs :
            raise ValidationError ("Lack of optimised context generated in phase I，Please run Phase 1 analysis first.")

        session =self .active_sessions [session_id ]
        session .status ="RUNNING"

        try :
            self .logger .info (f"Start running an optimization session: {session_id }")
            self .logger .info ("Use optimised context generated by Phase 1")

            optimization_contexts =model_outputs ['optimization_contexts']
            quality_report =model_outputs .get ('quality_report')

            # Implement the complete four to seven-phase process
            optimization_results =self ._execute_full_pipeline (session ,optimization_contexts ,quality_report )
            session .results .extend (optimization_results )

            # Validation phase：Performance Authentication
            if self .orchestration_config .enable_performance_validation :
                self .logger .info ("Phase4: Performance Authentication")
                self ._validate_optimizations (session ,session .results )

                # Finish Session
            session .mark_completed ()
            session .status ="COMPLETED"

            # Move to History
            self .session_history .append (session )
            del self .active_sessions [session_id ]

            self .logger .info (f"Optimization session completed: {session_id }, time consuming: {session .get_duration ():.2f}Second")
            return session 

        except Exception as e :
            session .status ="FAILED"
            session .metadata ['error']=str (e )
            self .logger .error (f"Optimization session failed: {session_id }, mistake: {e }")

            # If autorolling is enabled，Try rolling back all changes
            if self .orchestration_config .enable_automatic_rollback :
                self ._rollback_session_changes (session )

            raise ValidationError (f"Optimization session execution failed: {str (e )}")

    def _execute_full_pipeline (self ,session :OptimizationSession ,optimization_contexts :List ,quality_report )->List [ReplacementResult ]:
        """
        Implement the complete four to seven-phase process

        Args:
            session: Optimizing Sessions
            optimization_contexts: Optimized context generated in phase I
            quality_report: Quality reports

        Returns:
            List[ReplacementResult]: Optimizing Results List
        """
        results =[]

        try :
            self .logger .info (f"Start processing {len (optimization_contexts )} optimization context")

            for i ,context in enumerate (optimization_contexts ):
                self .logger .info (f"Handle optimization context {i +1 }/{len (optimization_contexts )}: rule{context .rule_idx }")

                try :
                # Phase II：APICall（Tip Project + APIInteractive）
                    self .logger .info ("  Phase2: APICall and tip project")
                    api_response =self ._call_optimization_api (context )

                    if not api_response .success :
                    # APICall Failed
                        result =ReplacementResult (
                        rule_idx =context .rule_idx ,
                        optimization_type =context .optimization_type ,
                        original_rule =str (context .current_rule ),
                        new_rule ="",
                        success =False ,
                        strategy ="FAILED",
                        performance_change ={'improvement':0.0 },
                        metadata ={'error':api_response .error ,'stage':'api_call'}
                        )
                        results .append (result )
                        continue 

                        # Phase IV：Data processing（Parsing and Converting）
                    self .logger .info ("  Phase4: Data processing")
                    parse_result =self .response_parser .parse_and_clean (api_response ,context .optimization_type )

                    if not parse_result .success :
                    # Parsing failed
                        result =ReplacementResult (
                        rule_idx =context .rule_idx ,
                        optimization_type =context .optimization_type ,
                        original_rule =str (context .current_rule ),
                        new_rule ="",
                        success =False ,
                        strategy ="FAILED",
                        performance_change ={'improvement':0.0 },
                        metadata ={'error':parse_result .errors ,'stage':'parsing'}
                        )
                        results .append (result )
                        continue 

                        # Convert to Structure Configuration
                    structured_config =self .structured_converter .convert_to_config (parse_result .data )

                    # Phase V：Rule management（Version control and decision-making）
                    self .logger .info ("  Phase5: Rule management")

                    # Backup Current Rules
                    try :
                        backup_path =self .version_manager .create_backup (
                        operation_type ="rule_optimization",
                        rule_idx =context .rule_idx ,
                        description =f"rule{context .rule_idx }Backup before optimization"
                        )
                        self .logger .info (f"  ✅ rule{context .rule_idx }Backup successful: {backup_path }")
                    except Exception as e :
                        self .logger .warning (f"  ⚠️ rule{context .rule_idx }Backup failed: {e }，Continue execution")

                        # Phase V：Replace Decision Configuration
                        # Attention.：No more redundant health here./Performance judgement，
                        # Candidate rules have been identified for optimization at the pre-identification stage，
                        # Focus here on configuration of rollback thresholds and backup parameters
                    replacement_decision =self .decision_engine .calculate_replacement_decision (context )

                    # Phase VI：Implementation and monitoring
                    self .logger .info (f"  stage6: Perform replacement (decision making: {replacement_decision .action .value })")
                    self .logger .info (f"  Reasons for decision: {replacement_decision .rationale }")

                    # Only decision-makingREPLACEOrganisation
                    if replacement_decision .action ==ReplacementAction .REPLACE :
                    # Send decision information to implementer，Ensure that the roll-back mechanism works properly
                        enhanced_strategy_params ={
                        'context':context .optimization_requirements ,
                        'rollback_threshold':replacement_decision .rollback_threshold ,
                        'backup_required':replacement_decision .backup_required ,
                        'decision_rationale':replacement_decision .rationale ,
                        'optimization_type':context .optimization_type 
                        }

                        execution_result =self .replacement_executor .execute_replacement (
                        rule_idx =context .rule_idx ,
                        new_config =structured_config ,
                        strategy ="REPLACE",
                        strategy_params =enhanced_strategy_params 
                        )
                    else :
                    # Theoretically, this branch should not be executed.，Because all candidates should be.REPLACE
                    # Keep this logic for abnormal situations
                        self .logger .warning (f"  ⚠️ candidate rules{context .rule_idx }The decision isKEEP，This usually shouldn't happen")
                        from .rule_replacement_executor import ExecutionResult 
                        execution_result =ExecutionResult (
                        success =True ,
                        execution_time =0.0 ,
                        backup_path ="",
                        error_message ="",
                        metadata ={
                        'action':'KEEP',
                        'reason':replacement_decision .rationale ,
                        'rollback_threshold':replacement_decision .rollback_threshold 
                        }
                        )

                    if execution_result .success :
                    # Create final result（Remove Immediate Performance Authentication）
                        result =ReplacementResult (
                        rule_idx =context .rule_idx ,
                        optimization_type =context .optimization_type ,
                        original_rule =str (context .current_rule ),
                        new_rule =structured_config .description ,
                        success =True ,# Successful implementation is considered success，The two-track monitoring system is responsible for assessing the impact.
                        strategy =str (replacement_decision .action .value ),
                        performance_change ={'improvement':0.0 },# Initial value [v]，assessed by two-track monitoring system
                        metadata ={
                        'execution_time':execution_result .execution_time ,
                        'replacement_timestamp':datetime .now ().isoformat (),
                        'replacement_source':'dual_track_optimization'
                        }
                        )

                        self .logger .info (f"  ✅ rule{context .rule_idx }Replacement successful，Will be naturally assessed through a dual-track monitoring system")
                    else :
                    # Implementation Failed
                        result =ReplacementResult (
                        rule_idx =context .rule_idx ,
                        optimization_type =context .optimization_type ,
                        original_rule =str (context .current_rule ),
                        new_rule =structured_config .description ,
                        success =False ,
                        strategy =replacement_decision .action .value ,
                        performance_change ={'improvement':0.0 },
                        metadata ={'error':execution_result .error_message ,'stage':'execution'}
                        )

                        self .logger .error (f"  ❌ rule{context .rule_idx }Execution failed: {execution_result .error_message }")

                    results .append (result )

                except Exception as e :
                # Single rule failed
                    self .logger .error (f"  ❌ rule{context .rule_idx }Processing failed: {e }")
                    result =ReplacementResult (
                    rule_idx =context .rule_idx ,
                    optimization_type =context .optimization_type ,
                    original_rule =str (context .current_rule ),
                    new_rule ="",
                    success =False ,
                    strategy ="FAILED",
                    performance_change ={'improvement':0.0 },
                    metadata ={'error':str (e ),'stage':'unknown'}
                    )
                    results .append (result )

            self .logger .info (f"The complete process is completed，success: {sum (1 for r in results if r .success )}/{len (results )}")
            return results 

        except Exception as e :
            self .logger .error (f"Complete process execution failed: {e }")
            raise 

    def _validate_optimizations (self ,session :OptimizationSession ,results :List [ReplacementResult ]):
        """Record optimization results and statistical information"""
        try :
        # Statistical optimization results
            successful_count =sum (1 for result in results if result .success )
            total_count =len (results )
            current_time =datetime .now ().isoformat ()

            # Record replacement time and source information
            for result in results :
                if result .success :
                    result .metadata ['replacement_timestamp']=current_time 
                    result .metadata ['replacement_source']='dual_track_optimization'
                    result .metadata ['optimization_epoch']=session .epoch if hasattr (session ,'epoch')else 'unknown'
                    self .logger .info (f"rule{result .rule_idx }Replacement completed，Natural assessment will be carried out through a dual-track monitoring system")
                else :
                    self .logger .warning (f"rule{result .rule_idx }Replacement failed")

            session .metadata ['optimization_completed']=current_time 
            session .metadata ['optimization_summary']={
            'total_rules':total_count ,
            'successful_replacements':successful_count ,
            'success_rate':successful_count /total_count if total_count >0 else 0 ,
            'completed_replacements':successful_count 
            }

            self .logger .info (f"Optimization batch completed: {successful_count }/{total_count } Rule replacement successful，Leave it to the dual-track monitoring system for natural assessment")

        except Exception as e :
            self .logger .error (f"Optimization result recording failed: {e }")
            session .metadata ['optimization_error']=str (e )

    def _rollback_session_changes (self ,session :OptimizationSession ):
        """Roll back all changes in sessions"""
        try :
            rollback_count =0 

            for result in session .results :
                if result .success and result .metadata and 'backup_path'in result .metadata :
                    backup_path =result .metadata ['backup_path']
                    rollback_result =self .version_manager .rollback_to_version (
                    backup_path ,
                    f"session{session .session_id }Automatic rollback"
                    )

                    if rollback_result .success :
                        rollback_count +=1 
                        self .logger .info (f"rule{result .rule_idx }Rollback successful")
                    else :
                        self .logger .error (f"rule{result .rule_idx }Rollback failed")

            session .metadata ['rollback_completed']=datetime .now ().isoformat ()
            session .metadata ['rollback_count']=rollback_count 

        except Exception as e :
            self .logger .error (f"Session rollback failed: {e }")
            session .metadata ['rollback_error']=str (e )

    def get_session_status (self ,session_id :str )->Dict [str ,Any ]:
        """Get Session Status"""
        if session_id in self .active_sessions :
            session =self .active_sessions [session_id ]
        else :
        # Find in History
            session =next ((s for s in self .session_history if s .session_id ==session_id ),None )
            if not session :
                raise ValidationError (f"Session does not exist: {session_id }")

        return {
        'session_id':session .session_id ,
        'status':session .status ,
        'epoch':session .epoch ,
        'start_time':session .start_time .isoformat (),
        'duration':session .get_duration (),
        'total_results':len (session .results ),
        'successful_optimizations':sum (1 for r in session .results if r .success ),
        'metadata':session .metadata 
        }

    def get_orchestrator_stats (self )->Dict [str ,Any ]:
        """Retrieving statistical information on the organiser"""
        total_sessions =len (self .active_sessions )+len (self .session_history )
        completed_sessions =len (self .session_history )

        if self .session_history :
            avg_duration =sum (s .get_duration ()or 0 for s in self .session_history )/len (self .session_history )
            total_optimizations =sum (len (s .results )for s in self .session_history )
            successful_optimizations =sum (sum (1 for r in s .results if r .success )for s in self .session_history )
            success_rate =successful_optimizations /total_optimizations if total_optimizations >0 else 0 
        else :
            avg_duration =0 
            total_optimizations =0 
            success_rate =0 

        return {
        'total_sessions':total_sessions ,
        'active_sessions':len (self .active_sessions ),
        'completed_sessions':completed_sessions ,
        'average_session_duration':avg_duration ,
        'total_optimizations':total_optimizations ,
        'optimization_success_rate':success_rate ,
        'components_initialized':True 
        }

    def _call_optimization_api (self ,context )->'APIResponse':
        """
        Call OptimizationAPI（Phase III）

        Args:
            context: Optimizing Context

        Returns:
            APIResponse: APIResponse Results
        """
        try :
        # Generate optimization tips using enhanced tips
            full_prompt =self .prompt_engineering .create_enhanced_optimization_prompt (context )

            # Separate System Hint and User Hint
            system_prompt ="You're a power market analyst.，Rules specifically designed to optimize the power price forecasting system。Please generate high-quality recommendations based on the information provided by users。"
            user_prompt =full_prompt 

            # CallAPI
            api_response =self .gpt_client .generate_rule_optimization (
            system_prompt =system_prompt ,
            user_prompt =user_prompt ,
            optimization_type =context .optimization_type 
            )

            return api_response 

        except Exception as e :
        # Return failedAPIResponse
            return APIResponse (
            success =False ,
            data =None ,
            error =str (e ),
            metadata ={'context_rule_idx':context .rule_idx }
            )
