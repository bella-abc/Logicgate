"""
Two-track rule optimisation system - Integrated Manager

Coordination of the first phase（Core analysis）Phase II（APIInteractive）Connection，
Manage complete optimization processes。
"""

import logging 
from typing import Dict ,List ,Optional ,Tuple ,Any 
from dataclasses import dataclass 
from datetime import datetime 

from .evaluation_analyzer import EvaluationAnalyzer 
from .data_structures import QualityReport ,OptimizationContext 
from ..optimized_rule_quality_monitor import DifferentiatedRuleQualityMonitor 


@dataclass 
class IntegrationConfig :
    """Integrated Configuration"""
    # Phase 1 Configuration
    health_threshold :float =0.4 
    effectiveness_threshold :float =0.3 
    min_batches_for_analysis :int =50 

    # Trigger Conditional Configuration
    analysis_interval_epochs :int =1 
    min_epochs_before_analysis :int =1 
    max_optimizations_per_session :int =3 

    # Security Configuration
    enable_automatic_optimization :bool =False 
    require_human_approval :bool =True 
    enable_rollback :bool =True 

    # Log Configuration
    enable_detailed_logging :bool =True 
    log_optimization_contexts :bool =True 


class OptimizationSession :
    """Optimizing Sessions，Record complete information for single optimization"""

    def __init__ (self ,session_id :str ,epoch :int ):
        self .session_id =session_id 
        self .epoch =epoch 
        self .start_time =datetime .now ()
        self .end_time =None 

        # Analysis
        self .quality_report :Optional [QualityReport ]=None 
        self .replacement_candidates :List [Dict ]=[]
        self .enhancement_candidates :List [Dict ]=[]
        self .optimization_contexts :List [OptimizationContext ]=[]

        # Execute Status
        self .status ="INITIALIZED"# INITIALIZED, ANALYZING, READY_FOR_API, COMPLETED, FAILED
        self .error_message :Optional [str ]=None 

        # Results statistics
        self .total_candidates =0 
        self .replacement_candidates =0 
        self .enhancement_candidates =0 

        # Generate Rule Track
        self .generated_rules_in_session :List [Dict [str ,Any ]]=[]

    def add_generated_rule (self ,rule_info :Dict [str ,Any ]):
        """Add newly generated rules to session track Medium"""
        self .generated_rules_in_session .append (rule_info )

    def complete (self ,status :str ="COMPLETED",error_message :Optional [str ]=None ):
        """Finish Session"""
        self .end_time =datetime .now ()
        self .status =status 
        self .error_message =error_message 

    @property 
    def duration (self )->float :
        """Duration of sessions（sec）"""
        if self .end_time :
            return (self .end_time -self .start_time ).total_seconds ()
        return (datetime .now ()-self .start_time ).total_seconds ()


class DualTrackIntegrationManager :
    """
    Two-track rule optimized integration manager
    
    Core responsibilities：
    1. Coordination of quality monitors and evaluation analysers
    2. Management optimization trigger logic
    3. Generate context optimization for phase II
    4. Record and manage optimized sessions
    """

    def __init__ (self ,
    quality_monitor :DifferentiatedRuleQualityMonitor ,
    config :Optional [IntegrationConfig ]=None ):
        """
        Initial Integration Manager
        
        Args:
            quality_monitor: Examples of quality monitors
            config: Integrated Configuration
        """
        self .quality_monitor =quality_monitor 
        self .config =config or IntegrationConfig ()

        # Initialization evaluation analyser
        self .evaluation_analyzer =EvaluationAnalyzer ({
        'health_threshold':self .config .health_threshold ,
        'effectiveness_threshold':self .config .effectiveness_threshold 
        })

        # Session Management
        self .current_session :Optional [OptimizationSession ]=None 
        self .session_history :List [OptimizationSession ]=[]
        self .session_counter =0 

        # Organisation
        self .last_analysis_epoch =-1 
        self .total_optimizations =0 

        # Log Configuration
        self .logger =logging .getLogger (__name__ )
        if self .config .enable_detailed_logging :
            self .logger .setLevel (logging .INFO )

    def should_trigger_analysis (self ,epoch :int ,val_loss :float )->Tuple [bool ,Dict [str ,Any ]]:
        """
        To determine whether the first stage of analysis should be triggered
        
        Args:
            epoch: Currentepoch
            val_loss: Validation of loss
            
        Returns:
            (should_trigger, decision_info)
        """
        decision_info ={
        'epoch':epoch ,
        'val_loss':val_loss ,
        'last_analysis_epoch':self .last_analysis_epoch ,
        'total_batches':self .quality_monitor .total_batches ,
        'reasons':[]
        }

        # Check basic conditions
        if epoch <self .config .min_epochs_before_analysis :#Minimum number of training rounds checked3
            decision_info ['reasons'].append (f"Epoch {epoch } < min_epochs_before_analysis {self .config .min_epochs_before_analysis }")
            return False ,decision_info 

        if self .quality_monitor .total_batches <self .config .min_batches_for_analysis :#Minimum batch check50
            decision_info ['reasons'].append (f"Total batches {self .quality_monitor .total_batches } < min_batches_for_analysis {self .config .min_batches_for_analysis }")
            return False ,decision_info 

            # Check interval conditions  It's the distance between training wheels.（epoch）5
        epochs_since_last_analysis =epoch -self .last_analysis_epoch # Time interval control，Prevention of excessive analysis，Give the rules enough time to observe.
        if epochs_since_last_analysis <self .config .analysis_interval_epochs :
            decision_info ['reasons'].append (f"Epochs since last analysis {epochs_since_last_analysis } < interval {self .config .analysis_interval_epochs }")
            return False ,decision_info 

            # Check optimization limit 3Prevent over-optimization in single training sessions，Avoiding Rule Bank Instability
        if self .total_optimizations >=self .config .max_optimizations_per_session :
            decision_info ['reasons'].append (f"Total optimizations {self .total_optimizations } >= max_optimizations_per_session {self .config .max_optimizations_per_session }")
            return False ,decision_info 

            # Check for ongoing sessions
        if self .current_session and self .current_session .status in ["ANALYZING","READY_FOR_API"]:
            decision_info ['reasons'].append (f"Current session {self .current_session .session_id } is in progress")
            return False ,decision_info 

        decision_info ['reasons'].append ("All conditions met for analysis")
        return True ,decision_info 

    def run_stage1_analysis (self ,epoch :int )->OptimizationSession :
        """
        Phase I analysis of operations
        
        Args:
            epoch: Currentepoch
            
        Returns:
            OptimizationSession: Optimizing Session Object
        """
        # Create a new optimised session
        self .session_counter +=1 
        session_id =f"opt_session_{self .session_counter :03d}_epoch_{epoch }"
        session =OptimizationSession (session_id ,epoch )
        self .current_session =session 

        self .logger .info (f"🔍 Begin the first phase of analysis - sessionID: {session_id }")

        try :
            session .status ="ANALYZING"

            # 1. Access to quality reports
            self .logger .info ("   📊 Generate quality reports...")
            report_data =self .quality_monitor .get_comprehensive_quality_report ()

            session .quality_report =QualityReport (
            library_health_scores =report_data ['library_health_scores'],
            effectiveness_scores =report_data ['effectiveness_scores'],
            optimization_strategies =report_data ['optimization_strategies'],
            statistics =report_data ['statistics']
            )

            # 2. Analysis of findings            self.logger.info("   🔍 Analysis of findings...")
            analysis_result =self .evaluation_analyzer .parse_evaluation_result (session .quality_report )

            session .replacement_candidates =analysis_result ['replacement_candidates']
            session .enhancement_candidates =analysis_result ['enhancement_candidates']
            session .total_candidates =len (session .replacement_candidates )+len (session .enhancement_candidates )

            # 3. Context of access to health rules
            self .logger .info ("   📋 Context of access to health rules...")
            healthy_rules =self .evaluation_analyzer .get_healthy_rules (session .quality_report .library_health_scores )
            self .logger .info (f"   turn up {len (healthy_rules )} health rules")

            # 4. Generate optimized context（Take the following perception.）
            self .logger .info ("   📝 Generate optimized context...")
            all_candidates =session .replacement_candidates +session .enhancement_candidates 

            # Initialization counter
            replacement_count =0 
            enhancement_count =0 

            for candidate in all_candidates :
                context =self .evaluation_analyzer .generate_optimization_context (
                candidate ,
                existing_healthy_rules =healthy_rules ,
                generated_rules_in_session =session .generated_rules_in_session 
                )
                session .optimization_contexts .append (context )

                # Statistical type
                if context .optimization_type =="REPLACEMENT":
                    replacement_count +=1 
                elif context .optimization_type =="ENHANCEMENT":
                    enhancement_count +=1 

                    # 4. Apply threshold policy
            if all_candidates :
                ordered_candidates =self .evaluation_analyzer .apply_threshold_strategy (all_candidates )
                self .logger .info (f"   ⚡ Threshold strategy sorting completed，common{len (ordered_candidates )}candidates")

                # Calculate priority（Simplified version：The first half is of high priority.，The second half is medium priority.）
                total_count =len (ordered_candidates )
                session .high_priority_candidates =min (3 ,total_count //2 +1 )# At least.1individual，Most3individual
                session .medium_priority_candidates =total_count -session .high_priority_candidates 
            else :
                session .high_priority_candidates =0 
                session .medium_priority_candidates =0 

                # 5. Update Status
            session .status ="READY_FOR_API"
            self .last_analysis_epoch =epoch 

            # 6. Record results
            self .logger .info ("   ✅ Phase 1 analysis completed")
            self .logger .info (f"      Need to replace: {replacement_count } rules")
            self .logger .info (f"      Need to enhance: {enhancement_count } rules")

            if self .config .log_optimization_contexts :
                self ._log_optimization_contexts (session )

        except Exception as e :
            self .logger .error (f"   ❌ First stage analysis failed: {e }")
            session .complete ("FAILED",str (e ))
            raise 

        return session 

    def get_optimization_contexts_for_api (self )->List [OptimizationContext ]:
        """
        Get ready optimal context，For phase IIAPICall
        
        Returns:
            List[OptimizationContext]: Optimizing Context List
        """
        if not self .current_session or self .current_session .status !="READY_FOR_API":
            return []

        return self .current_session .optimization_contexts 

    def complete_current_session (self ,success :bool =True ,error_message :Optional [str ]=None ):
        """
        Finish the current optimized session
        
        Args:
            success: Success
            error_message: Error message（If you fail,）
        """
        if self .current_session :
            status ="COMPLETED"if success else "FAILED"
            self .current_session .complete (status ,error_message )

            # Add to History
            self .session_history .append (self .current_session )

            if success :
                self .total_optimizations +=1 
                self .logger .info (f"✅ Optimize session {self .current_session .session_id } Finish")
            else :
                self .logger .error (f"❌ Optimize session {self .current_session .session_id } fail: {error_message }")

            self .current_session =None 

    def get_session_summary (self )->Dict [str ,Any ]:
        """Get Session Summary Information"""
        return {
        'total_sessions':len (self .session_history ),
        'successful_sessions':len ([s for s in self .session_history if s .status =="COMPLETED"]),
        'failed_sessions':len ([s for s in self .session_history if s .status =="FAILED"]),
        'total_optimizations':self .total_optimizations ,
        'last_analysis_epoch':self .last_analysis_epoch ,
        'current_session_status':self .current_session .status if self .current_session else None ,
        'current_session_id':self .current_session .session_id if self .current_session else None 
        }

    def _log_optimization_contexts (self ,session :OptimizationSession ):
        """Record the best context details"""
        self .logger .info ("   📋 Optimizing context details:")
        for i ,context in enumerate (session .optimization_contexts ,1 ):
            self .logger .info (f"      {i }. rule{context .rule_idx } - {context .optimization_type }")
            self .logger .info (f"         health: {context .performance_metrics .get ('health_score','N/A'):.3f}")
            self .logger .info (f"         efficacy: {context .performance_metrics .get ('effectiveness_score','N/A'):.3f}")


def create_integration_manager (quality_monitor :DifferentiatedRuleQualityMonitor ,
config_overrides :Optional [Dict ]=None )->DualTrackIntegrationManager :
    """
    A convenient function to create an integrated manager
    
    Args:
        quality_monitor: Examples of quality monitors
        config_overrides: Configure Overwrite
        
    Returns:
        DualTrackIntegrationManager: Examples of integrated manager
    """
    config =IntegrationConfig ()

    if config_overrides :
        for key ,value in config_overrides .items ():
            if hasattr (config ,key ):
                setattr (config ,key ,value )

    return DualTrackIntegrationManager (quality_monitor ,config )
