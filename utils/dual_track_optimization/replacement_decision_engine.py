"""
Replace Policy Decision Engine

Rules-based health scores，Simple decision whether you need a replacement rule。
Focus on rule improvement in an experimental environment，Avoid complex strategies at the industrial level。
"""

import logging 
from typing import Dict ,List ,Any 
from dataclasses import dataclass 
from enum import Enum 

from .data_structures import OptimizationContext 


class ReplacementAction (Enum ):
    """Replace Action Counting"""
    REPLACE ="REPLACE"# Replacement rules
    KEEP ="KEEP"# Reservations rules


@dataclass 
class ReplacementDecision :
    """Replace Decision Data Category"""
    action :ReplacementAction 
    rationale :str 
    rollback_threshold :float =0.05 
    backup_required :bool =True 

    def to_dict (self )->Dict [str ,Any ]:
        """Convert to Dictionary Format"""
        return {
        'action':self .action .value ,
        'rationale':self .rationale ,
        'rollback_threshold':self .rollback_threshold ,
        'backup_required':self .backup_required 
        }


@dataclass 
class RuleMetrics :
    """Standard indicator data category"""
    rule_idx :int 
    health_score :float 
    effectiveness_score :float =0.0 


class ReplacementDecisionEngine :
    """
    Replace Policy Decision Engine (Optimization)

    Removed redundant health/Performance judgement logic，Focus on established candidate rules
    Configure rolling back and backup parameters。The identification of candidate rules has been completed at an earlier stage。
    
    Core functions：
    1. Configure replacement parameters for identified candidate rules
    2. Set a different rollback threshold according to the optimal type  
    3. Ensure that backup mechanisms work properly to support rollback
    4. Generate clear decision reasons for log tracking
    """

    def __init__ (self ,config :Dict [str ,Any ]):
        self .config =config 
        self .logger =logging .getLogger (__name__ )

        # Simplified threshold configuration
        self .health_threshold =config .get ('thresholds',{}).get ('health_threshold',0.4 )
        self .effectiveness_threshold =config .get ('thresholds',{}).get ('effectiveness_threshold',0.3 )

    def calculate_replacement_decision (self ,context :OptimizationContext )->ReplacementDecision :
        """
        Create replacement decision for identified optimized candidates
        
        Remove redundant health/Performance judgement，Focus on rollback and backup parameter configuration。
        Candidate rules have been identified for optimization through the pre-identification phase，Configure execution parameters directly here。

        Args:
            context: Optimizing Context，Include identified optimization types and rule information

        Returns:
            ReplacementDecision: Replacement decision-making（Include Rollback Configuration）
        """
        try :
        # Set rollback thresholds based on optimized type
            rollback_threshold =self ._get_rollback_threshold (context .optimization_type )

            # Generate rationale for decision-making
            rationale =self ._generate_rationale (context )

            return ReplacementDecision (
            action =ReplacementAction .REPLACE ,# The candidate rule must be replaced.
            rationale =rationale ,
            rollback_threshold =rollback_threshold ,
            backup_required =True # Always create backup to support rollback
            )

        except Exception as e :
            self .logger .error (f"Failed to create replacement decision: {e }")
            # Returns conservative default decisions
            return ReplacementDecision (
            action =ReplacementAction .REPLACE ,# Go on even if it's wrong.，Because it's a confirmed candidate.
            rationale =f"Decision configuration fails but execution continues: {str (e )}",
            rollback_threshold =0.05 ,# Use conservative threshold
            backup_required =True 
            )

    def evaluate_multiple_rules (self ,optimization_contexts :List [OptimizationContext ])->List [ReplacementDecision ]:
        """
        Assessment of multi-rule replacement decision-making (New Version)

        Args:
            optimization_contexts: Optimizing Context List

        Returns:
            List[ReplacementDecision]: Replace Decision List，Sort by priority
        """
        decisions =[]
        for context in optimization_contexts :
            decision =self .calculate_replacement_decision (context )
            decisions .append ((context ,decision ))

            # Sort by rule，Consistency in order of treatment
        decisions .sort (key =lambda x :x [0 ].rule_idx )

        return [decision for _ ,decision in decisions ]

    def evaluate_multiple_rules_legacy (self ,rules_metrics :List [RuleMetrics ])->List [ReplacementDecision ]:
        """
        Assessment of multi-rule replacement decision-making (Compatible with old versions)

        Args:
            rules_metrics: List of rule indicators

        Returns:
            List[ReplacementDecision]: Replace Decision List，Sort by Health（Worst priority）
        """
        self .logger .warning ("Used abandonedevaluate_multiple_rules_legacyMethodology，Suggested migration to new version")

        decisions =[]
        for rule_metric in rules_metrics :
            decision =self .calculate_replacement_decision_legacy (rule_metric )
            decisions .append ((rule_metric ,decision ))

            # Sort by Health，Worst priority
        decisions .sort (key =lambda x :x [0 ].health_score )

        return [decision for _ ,decision in decisions ]

    def _get_rollback_threshold (self ,optimization_type :str )->float :
        """
        Set rollback thresholds based on optimized type
        
        Args:
            optimization_type: Optimization Type ('REPLACEMENT' or 'ENHANCEMENT')
            
        Returns:
            float: Rollback threshold
        """
        if optimization_type =='REPLACEMENT':
        # Replacement rules use stricter thresholds，Because there's a higher risk of replacement.
            return 0.03 
        else :# ENHANCEMENT
        # Use of standard thresholds for enhanced rules
            return 0.05 

    def _generate_rationale (self ,context :OptimizationContext )->str :
        """
        Generate rationale for decision-making
        
        Args:
            context: Optimizing Context
            
        Returns:
            str: Reasons for decision-making
        """
        if context .optimization_type =='REPLACEMENT':
            return f"rule{context .rule_idx }Identified as a replacement candidate through health screening，Trigger condition does not match data pattern"
        else :# ENHANCEMENT
            return f"rule{context .rule_idx }Identified as an enhancement candidate through performance analysis，Forecast accuracy needs improvement"

    def calculate_replacement_decision_legacy (self ,rule_metrics :RuleMetrics )->ReplacementDecision :
        """
        Alternative decision-making methods compatible with the old version（Keep to prevent other modules from calling）
        
        Args:
            rule_metrics: Rule-based indicators
            
        Returns:
            ReplacementDecision: Replacement decision-making
        """
        self .logger .warning ("Used abandonedcalculate_replacement_decision_legacyMethodology，Suggested migration to new version")

        try :
            health_score =rule_metrics .health_score 
            effectiveness_score =rule_metrics .effectiveness_score 

            # Simplified decision-making logic
            if health_score <self .health_threshold :
                action =ReplacementAction .REPLACE 
                rationale =f"low health({health_score :.3f} < {self .health_threshold })，Trigger condition does not match data，Need to replace"
            elif health_score >=self .health_threshold and effectiveness_score >0 and effectiveness_score <self .effectiveness_threshold :
                action =ReplacementAction .REPLACE 
                rationale =f"Health is normal({health_score :.3f})But the performance is too low({effectiveness_score :.3f} < {self .effectiveness_threshold })，Need to improve causality"
            else :
                action =ReplacementAction .KEEP 
                rationale =f"good health({health_score :.3f})，Rules behave normally，reserved for use"

            return ReplacementDecision (
            action =action ,
            rationale =rationale ,
            rollback_threshold =0.05 ,
            backup_required =True 
            )

        except Exception as e :
            self .logger .error (f"Calculate replacement decision failed: {e }")
            return ReplacementDecision (
            action =ReplacementAction .KEEP ,
            rationale =f"Decision calculation failed，Adopt a conservative strategy: {str (e )}",
            rollback_threshold =0.05 ,
            backup_required =True 
            )


