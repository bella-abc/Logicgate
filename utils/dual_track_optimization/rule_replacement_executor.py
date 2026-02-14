"""
Rules Replace Executor

Responsible for accurate enforcement of rule replacement operations，Deal with different replacement strategies，
Make sure every move is safe and manageable.，And can roll back quickly when there's a problem.。
"""

import json 
import logging 
import time 
from datetime import datetime 
from pathlib import Path 
from typing import Dict ,List ,Any ,Optional 
from dataclasses import dataclass ,asdict 

from .data_structures import StructuredRuleConfig ,ValidationError 
from .rule_version_manager import RuleVersionManager 


@dataclass 
class ExecutionResult :
    """Implementation results data category"""
    success :bool 
    strategy :str 
    rule_idx :int 
    backup_path :str 
    execution_time :float 
    progress :float 
    phase :Optional [str ]=None 
    errors :List [str ]=None 
    warnings :List [str ]=None 
    metadata :Dict [str ,Any ]=None 

    def __post_init__ (self ):
        if self .errors is None :
            self .errors =[]
        if self .warnings is None :
            self .warnings =[]
        if self .metadata is None :
            self .metadata ={}


@dataclass 
class PhaseResult :
    """Phase implementation results"""
    phase_id :int 
    success :bool 
    duration :float 
    traffic_ratio :float 
    metrics :Dict [str ,float ]
    errors :List [str ]=None 

    def __post_init__ (self ):
        if self .errors is None :
            self .errors =[]


class RuleReplacementExecutor :
    """
    Rules Replace Executor
    
    Exactly implement rule replacement operations，Support immediate replacement、Progressive replacement、A/BTesting and other strategies.。
    Make sure every move is safe and manageable.，And can roll back quickly when there's a problem.。
    """

    def __init__ (self ,version_manager :RuleVersionManager ,config :Dict [str ,Any ],model =None ):
        self .version_manager =version_manager 
        self .config =config 
        self .logger =logging .getLogger (__name__ )
        self .model =model # Add Model Reference，For reloading rules

        # Profile Path
        self .config_file =Path (config .get ('config_file','config/rule_patterns.json'))

        # Implementation status tracking
        self .current_execution =None 
        self .execution_history =[]

    def execute_replacement (self ,rule_idx :int ,new_config :StructuredRuleConfig ,
    strategy :str ,strategy_params :Dict [str ,Any ])->ExecutionResult :
        """
        Replacement of implementing rules
        
        Args:
            rule_idx: Rule Index
            new_config: New rule configuration
            strategy: Replace Policy
            strategy_params: Policy parameters
            
        Returns:
            ExecutionResult: Implementation results
        """
        start_time =time .time ()

        try :
            self .logger .info (f"Start enforcing rules{rule_idx }of{strategy }replace")

            # Simplified replacement logic：OnlyREPLACEA strategy.
            if strategy =="REPLACE":
                result =self ._execute_simple_replacement (rule_idx ,new_config ,strategy_params )
            else :
                raise ValidationError (f"Unsupported replacement strategy: {strategy }")

                # Update implementation time
            result .execution_time =time .time ()-start_time 

            # Record implementation history
            self .execution_history .append (result )

            self .logger .info (f"rule{rule_idx }Replacement execution completed，time consuming{result .execution_time :.2f}Second")
            return result 

        except Exception as e :
            self .logger .error (f"rule{rule_idx }Replacement execution failed: {e }")
            return ExecutionResult (
            success =False ,
            strategy =strategy if isinstance (strategy ,str )else strategy .value ,
            rule_idx =rule_idx ,
            backup_path ="",
            execution_time =time .time ()-start_time ,
            progress =0.0 ,
            errors =[f"Execution exception: {str (e )}"]
            )

    def _execute_simple_replacement (self ,rule_idx :int ,new_config :StructuredRuleConfig ,
    params :Dict [str ,Any ])->ExecutionResult :
        """Execute simple replacement policy"""
        try :
        # 1. Create Backup
            backup_path =self .version_manager .create_backup (
            "immediate_replacement",
            rule_idx ,
            f"rule{rule_idx }Backup before replacing immediately"
            )

            # 2. Verify new configuration
            self ._validate_new_config (new_config )

            # 3. Update Profile
            self ._update_rule_patterns_json (rule_idx ,new_config )

            # 4. Verify updated configuration
            if not self ._validate_updated_config ():
                raise ValidationError ("Updated configuration validation failed")

                # 5. Reload Model Rules（Critical repairs）
            if self .model is not None :
                self ._reload_model_rules ()
                self .logger .info (f"✅ Model rules reloaded，rule{rule_idx }The new configuration has taken effect")
            else :
                self .logger .warning ("⚠️ Model references as empty，Could not reload rule，The new rules may not come into effect immediately.")

            return ExecutionResult (
            success =True ,
            strategy ="IMMEDIATE",
            rule_idx =rule_idx ,
            backup_path =backup_path ,
            execution_time =0.0 ,# Setup on the outer layer
            progress =1.0 ,
            metadata ={
            'validation_required':params .get ('validation_required',True ),
            'monitoring_duration':params .get ('monitoring_duration',24 ),
            'rollback_threshold':params .get ('rollback_threshold',0.1 )
            }
            )

        except Exception as e :
        # If there's a backup,，Try rolling back
            if 'backup_path'in locals ():
                try :
                    self .version_manager .rollback_to_version (
                    backup_path ,
                    f"immediate_replacement_failed: {e }"
                    )
                except :
                    pass # Rollback failure does not affect the bug report

            raise ValidationError (f"Simple replacement execution failed: {str (e )}")

    def _validate_new_config (self ,new_config :StructuredRuleConfig ):
        """Verify new configuration"""
        if not new_config .description :
            raise ValidationError ("Rule description cannot be empty")

        if not new_config .pattern_features :
            raise ValidationError ("Mode characteristics cannot be empty")

    def _update_rule_patterns_json (self ,rule_idx :int ,new_config :StructuredRuleConfig ):
        """Updaterule_patterns.jsonDocumentation"""
        if not self .config_file .exists ():
            raise ValidationError (f"Configuration file does not exist: {self .config_file }")

            # Read Current Configuration
        with open (self .config_file ,'r',encoding ='utf-8')as f :
            current_config =json .load (f )

            # Update designation rules
        rule_id =str (rule_idx )
        if 'rule_patterns'not in current_config :
            current_config ['rule_patterns']={}

        current_config ['rule_patterns'][rule_id ]=new_config .to_rule_patterns_format ()

        # Update metadata
        if 'metadata'not in current_config :
            current_config ['metadata']={}

        current_config ['metadata']['last_updated']=datetime .now ().isoformat ()
        current_config ['metadata']['last_updated_rule']=rule_idx 

        # Save updated configuration
        with open (self .config_file ,'w',encoding ='utf-8')as f :
            json .dump (current_config ,f ,indent =2 ,ensure_ascii =False )

        self .logger .info (f"Rules updated{rule_idx }configuration")

    def _reload_model_rules (self ):
        """Reload Model Rules（Critical rehabilitation methods）"""
        try :
        # 1. Reread rules from profile
            with open (self .config_file ,'r',encoding ='utf-8')as f :
                rules_config =json .load (f )

                # 2. Extract Rule Text List
            new_rules_list =[]
            rule_patterns =rules_config .get ('rule_patterns',{})

            # Rebuild rule list in rule index order
            for rule_idx in sorted (rule_patterns .keys (),key =int ):
                rule_data =rule_patterns [rule_idx ]
                # UsedescriptionAs Rule Text（This is the standard format for the current profile.）
                rule_text =rule_data .get ('description','')
                if not rule_text :
                    self .logger .warning (f"rule{rule_idx }LackdescriptionField，Use empty string")
                new_rules_list .append (rule_text )

                # 3. List of rules for updating models
            old_rules_count =len (self .model .rules_list )if hasattr (self .model ,'rules_list')else 0 
            self .model .rules_list =new_rules_list 
            self .model .num_rules =len (new_rules_list )

            # 4. Smart re-initiation rule quality monitoring Device（Maintain History Statistics）
            if hasattr (self .model ,'rule_quality_monitor')and self .model .rule_quality_monitor is not None :
            # Keep old statistics (Safe cloning，Dealing with possible anomalies)
                old_total_batches =self .model .rule_quality_monitor .total_batches 
                old_selection_counts =self ._safe_clone_tensor (self .model .rule_quality_monitor .selection_counts )
                old_attention_weights_sum =self ._safe_clone_tensor (self .model .rule_quality_monitor .attention_weights_sum )
                old_attention_counts =self ._safe_clone_tensor (self .model .rule_quality_monitor .attention_counts )
                old_similarity_scores_sum =self ._safe_clone_tensor (self .model .rule_quality_monitor .similarity_scores_sum )
                old_similarity_counts =self ._safe_clone_tensor (self .model .rule_quality_monitor .similarity_counts )

                from utils .optimized_rule_quality_monitor import DifferentiatedRuleQualityMonitor 
                self .model .rule_quality_monitor =DifferentiatedRuleQualityMonitor (
                num_rules =len (new_rules_list ),
                device =self .model .device 
                )

                # Restore statistical information（For unmodified rules）
                self ._safe_restore_statistics (
                self .model .rule_quality_monitor ,
                old_selection_counts ,
                old_attention_weights_sum ,
                old_attention_counts ,
                old_similarity_scores_sum ,
                old_similarity_counts ,
                old_total_batches ,
                len (new_rules_list )
                )

                self .logger .info ("✅ Rules quality monitor re-initiated smartly，Maintain History Statistics")

                # 5. Reinitiation Rules Filter（If it exists）
            if hasattr (self .model ,'rule_filter')and self .model .rule_filter is not None :
            # The rule filter usually does not need to be re-initiated.，But we need to clear the cache.
                if hasattr (self .model .rule_filter ,'clear_cache'):
                    self .model .rule_filter .clear_cache ()
                    self .logger .info ("✅ Rules filter cache cleared")

            self .logger .info (f"✅ Model rules reloading completed: {old_rules_count } -> {len (new_rules_list )} rules")

        except Exception as e :
            self .logger .error (f"❌ Model rule reload failed: {e }")
            raise ValidationError (f"Model rule reload failed: {str (e )}")

    def _safe_clone_tensor (self ,tensor_or_value ):
        """
        Safe clone scale or value
        
        Dealing with possible anomalies，As integer、List etc.
        """
        import torch 

        if tensor_or_value is None :
            return None 
        elif isinstance (tensor_or_value ,torch .Tensor ):
            return tensor_or_value .clone ()
        elif isinstance (tensor_or_value ,(int ,float )):
        # For the Standard Value，Direct Return
            return tensor_or_value 
        elif isinstance (tensor_or_value ,(list ,tuple )):
        # For List or Form，Convert to Scale Reclon
            try :
                return torch .tensor (tensor_or_value ).clone ()
            except :
            # If you can't convert to grid，Return directly to original value
                return tensor_or_value 
        else :
        # For other types，Try returning directly
            self .logger .warning (f"Unknown types of data cannot be safely cloned: {type (tensor_or_value )}")
            return tensor_or_value 

    def _safe_restore_statistics (self ,monitor ,old_selection_counts ,old_attention_weights_sum ,
    old_attention_counts ,old_similarity_scores_sum ,old_similarity_counts ,
    old_total_batches ,new_rules_count ):
        """
        Statistical information on security restoration
        
        Address possible tension dimensions or type inconsistencies
        """
        try :
        # Number of restorations
            monitor .total_batches =old_total_batches 

            # Security restoration of statistical information
            if old_selection_counts is not None and hasattr (monitor ,'selection_counts'):
                self ._safe_copy_to_tensor (monitor .selection_counts ,old_selection_counts ,new_rules_count )

            if old_attention_weights_sum is not None and hasattr (monitor ,'attention_weights_sum'):
                self ._safe_copy_to_tensor (monitor .attention_weights_sum ,old_attention_weights_sum ,new_rules_count )

            if old_attention_counts is not None and hasattr (monitor ,'attention_counts'):
                self ._safe_copy_to_tensor (monitor .attention_counts ,old_attention_counts ,new_rules_count )

            if old_similarity_scores_sum is not None and hasattr (monitor ,'similarity_scores_sum'):
                self ._safe_copy_to_tensor (monitor .similarity_scores_sum ,old_similarity_scores_sum ,new_rules_count )

            if old_similarity_counts is not None and hasattr (monitor ,'similarity_counts'):
                self ._safe_copy_to_tensor (monitor .similarity_counts ,old_similarity_counts ,new_rules_count )

        except Exception as e :
            self .logger .warning (f"Exception occurred while restoring statistics，Use default initialization: {e }")

    def _safe_copy_to_tensor (self ,target_tensor ,source_data ,max_size ):
        """
        Copy the data safely to target size
        """
        import torch 

        try :
            if source_data is None :
                return 

                # Ensuresource_dataIt's the scale.
            if not isinstance (source_data ,torch .Tensor ):
                if isinstance (source_data ,(int ,float )):
                # Mark Value，Fill to the full size
                    target_tensor .fill_ (source_data )
                    return 
                else :
                # Try to convert to a grid
                    source_data =torch .tensor (source_data )

                    # Calculate the minimum length to copy
            copy_size =min (len (source_data ),len (target_tensor ),max_size )

            if copy_size >0 :
                target_tensor [:copy_size ]=source_data [:copy_size ]

        except Exception as e :
            self .logger .warning (f"Copying tensor data failed: {e }")
            # Silence failed.，Use Default

    def _validate_updated_config (self )->bool :
        """Verify updated profile"""
        try :
            with open (self .config_file ,'r',encoding ='utf-8')as f :
                config =json .load (f )

                # Basic Structure Validation
            if 'rule_patterns'not in config :
                return False 

                # Verify the basic fields of each rule
            for rule_id ,rule_config in config ['rule_patterns'].items ():
                required_fields =['description','direction','category']
                for field in required_fields :
                    if field not in rule_config :
                        return False 

            return True 

        except Exception as e :
            self .logger .error (f"Configuration verification failed: {e }")
            return False 

    def get_execution_status (self )->Dict [str ,Any ]:
        """Get Execute Status"""
        return {
        'current_execution':asdict (self .current_execution )if self .current_execution else None ,
        'execution_history_count':len (self .execution_history ),
        'recent_executions':[asdict (ex )for ex in self .execution_history [-5 :]],
        'success_rate':self ._calculate_success_rate ()
        }

    def _calculate_success_rate (self )->float :
        """Calculation of implementation success rate"""
        if not self .execution_history :
            return 0.0 

        successful =sum (1 for ex in self .execution_history if ex .success )
        return successful /len (self .execution_history )
