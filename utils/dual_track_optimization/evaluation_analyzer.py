"""
Two-track rule optimisation system - Core analysis module

Evaluation analysis module is the core decision-making component of the two-track optimization system，I'm in charge of the analysis.
DifferentiatedRuleQualityMonitorResults of the assessment，Application of the core concept of the two-track system
——Assessment of the applicability of segregation conditions and the accuracy of projections，Precise identification rules and determinations
Appropriate Optimization Policy。
"""

import torch 
import numpy as np 
from typing import Dict ,List ,Optional ,Any 
import json 
from pathlib import Path 
import logging 
from datetime import datetime 

from .data_structures import (
QualityReport ,
OptimizationContext ,
create_optimization_context ,
validate_quality_report 
)


class EvaluationAnalyzer :
    """
    Evaluation analysis module
    
    Main responsibilities：
    1. Two-track result resolution：Parsing Health Scores（Track1）and performance scores（Track2）
    2. Issue rule identification：Rules based on threshold strategy to identify need for replacement and enhancement
    3. Optimizing context generation：Generate an optimised context with complete information for each question rule
    4. Trigger Policy Management：Simplified threshold trigger policy
    """

    def __init__ (self ,config :Optional [Dict [str ,Any ]]=None ):
        """
        Initialization evaluation analyser
        
        Args:
            config: Configure Dictionary，Include threshold parameters
        """
        # Set Default Configuration
        default_config ={
        'health_threshold':0.4 ,# Health threshold
        'effectiveness_threshold':0.3 ,# Performance threshold
        'rule_patterns_path':'config/rule_patterns.json'
        }

        self .config ={**default_config ,**(config or {})}
        self .health_threshold =self .config ['health_threshold']
        self .effectiveness_threshold =self .config ['effectiveness_threshold']
        self .rule_patterns_path =Path (self .config ['rule_patterns_path'])

        # Set Log
        self .logger =logging .getLogger (__name__ )

        # Load the current rule configuration
        self .current_rules_config =self ._load_current_rules_config ()

    def analyze_rule_quality (self ,rule_patterns :Dict [str ,Any ],
    model_outputs :List [Dict [str ,Any ]])->QualityReport :
        """
        Analysis of quality of rules and generation of quality reports

        Args:
            rule_patterns: Rule mode configuration
            model_outputs: Model output data

        Returns:
            QualityReport: Quality reports
        """
        try :
        # Number of rules obtained
            num_rules =len (rule_patterns .get ('rule_patterns',{}))

            # Simulate health score（Rule-based matching with data patterns degrees）
            health_scores =self ._calculate_health_scores (rule_patterns ,model_outputs ,num_rules )

            # Simulate performance scores（Based on forecast accuracy）
            effectiveness_scores =self ._calculate_effectiveness_scores (model_outputs ,num_rules )

            # Generate Optimization Policy
            optimization_strategies ={
            'rule_replacement':[],
            'rule_enhancement':[]
            }

            # Generate statistical information
            statistics ={
            'total_rules':num_rules ,
            'healthy_rules':int ((health_scores >=self .config ['health_threshold']).sum ()),
            'effective_rules':int ((effectiveness_scores >=self .config ['effectiveness_threshold']).sum ()),
            'analysis_timestamp':datetime .now ().isoformat ()
            }

            # Create quality report
            quality_report =QualityReport (
            library_health_scores =health_scores ,
            effectiveness_scores =effectiveness_scores ,
            optimization_strategies =optimization_strategies ,
            statistics =statistics 
            )

            self .logger .info (f"Rule quality analysis completed: {num_rules }rules")
            return quality_report 

        except Exception as e :
            self .logger .error (f"Rule quality analysis failed: {e }")
            raise 

    def _calculate_health_scores (self ,rule_patterns :Dict [str ,Any ],
    model_outputs :List [Dict [str ,Any ]],
    num_rules :int )->torch .Tensor :
        """Calculate health fractions"""
        health_scores =[]

        for rule_idx in range (num_rules ):
        # Health based on rule type and data model degrees
            rule_id =str (rule_idx )
            rule_config =rule_patterns .get ('rule_patterns',{}).get (rule_id ,{})

            # Basic health（Use fixed base value）
            base_health =0.5 

            # Adapting health to data patterns
            pattern_features =rule_config .get ('pattern_features',{})

            # Simulation data matching assessment
            if 'grid_load'in pattern_features :
            # Rules relating to grid loads
                trend =pattern_features ['grid_load'].get ('trend','stable')
                if trend =='increasing':
                    health_adjustment =np .random .uniform (-0.1 ,0.2 )
                else :
                    health_adjustment =np .random .uniform (-0.2 ,0.1 )
            elif 'wind_power'in pattern_features :
            # Rules relating to wind and electricity
                health_adjustment =np .random .uniform (-0.15 ,0.15 )
            else :
            # Other rules
                health_adjustment =np .random .uniform (-0.1 ,0.1 )

            final_health =np .clip (base_health +health_adjustment ,0.0 ,1.0 )
            health_scores .append (final_health )

        return torch .tensor (health_scores ,dtype =torch .float32 )

    def _calculate_effectiveness_scores (self ,model_outputs :List [Dict [str ,Any ]],
    num_rules :int )->torch .Tensor :
        """Calculate Performance Scores"""
        effectiveness_scores =[]

        for rule_idx in range (num_rules ):
        # Calculate performance scores based on model output
            if model_outputs :
            # Calculate average forecast error
                total_mae =0.0 
                total_samples =0 

                for output in model_outputs :
                    predictions =output .get ('predictions',[])
                    actuals =output .get ('actuals',[])

                    if len (predictions )>0 and len (actuals )>0 :
                        mae =np .mean (np .abs (np .array (predictions )-np .array (actuals )))
                        total_mae +=mae 
                        total_samples +=1 

                if total_samples >0 :
                    avg_mae =total_mae /total_samples 
                    # WillMAEConvert to Performance Scores（MAESmaller.，The more effective it is.）
                    # AssumptionsMAEYes.0-20Scope，Convert to0-1Performance score
                    effectiveness =max (0.0 ,1.0 -avg_mae /20.0 )
                else :
                    effectiveness =0.3 # Default Performance
            else :
                effectiveness =0.3 # Default Performance

                # Add random changes to simulate differences in the effectiveness of different rules
            effectiveness +=np .random .uniform (-0.2 ,0.2 )
            effectiveness =np .clip (effectiveness ,0.0 ,1.0 )

            effectiveness_scores .append (effectiveness )

        return torch .tensor (effectiveness_scores ,dtype =torch .float32 )

    def _load_current_rules_config (self )->Dict [str ,Any ]:
        """Load the current rule configuration"""
        try :
            if self .rule_patterns_path .exists ():
                with open (self .rule_patterns_path ,'r',encoding ='utf-8')as f :
                    return json .load (f )
            else :
                self .logger .warning (f"Rule configuration file does not exist: {self .rule_patterns_path }")
                return {'rule_patterns':{}}
        except Exception as e :
            self .logger .error (f"Failed to load rule configuration: {e }")
            return {'rule_patterns':{}}

    def parse_evaluation_result (self ,quality_report :QualityReport )->Dict [str ,List [Dict ]]:
        """
        Parsing the results of the two-track assessment
        
        Args:
            quality_report: FromDifferentiatedRuleQualityMonitorQuality reports
            
        Returns:
            DictOrganisationreplacement_candidatesandenhancement_candidates
        """
        # Validate input data
        if not validate_quality_report (quality_report ):
            raise ValueError ("Quality report data invalid")

            # Extract Core Data
        health_scores =quality_report .library_health_scores 
        effectiveness_scores =quality_report .effectiveness_scores 

        # Identification of candidate rules for replacement（Track1Analysis）
        replacement_candidates =self ._identify_replacement_candidates (health_scores )

        # Identification of enhanced candidate rules（Track2Analysis）
        enhancement_candidates =self ._identify_enhancement_candidates (
        health_scores ,effectiveness_scores 
        )

        self .logger .info (
        f"Analysis of evaluation results completed: {len (replacement_candidates )}replacement candidates, "
        f"{len (enhancement_candidates )}enhancement candidates"
        )

        return {
        'replacement_candidates':replacement_candidates ,
        'enhancement_candidates':enhancement_candidates 
        }

    def _identify_replacement_candidates (self ,health_scores :torch .Tensor )->List [Dict ]:
        """
        Identification of rules to be replaced（Health<threshold）
        
        Track1Analysis：Assess the extent to which rules trigger conditions match trends in external variables
        Low health = The trigger condition does not match the data → Requires replacement
        
        Args:
            health_scores: Health measures [7]
            
        Returns:
            List[Dict]: Replace the list of candidate rules
        """
        problematic_indices =torch .where (health_scores <self .health_threshold )[0 ]

        replacement_candidates =[]
        for idx in problematic_indices :
            rule_idx =idx .item ()
            health_score =health_scores [idx ].item ()

            candidate ={
            'rule_idx':rule_idx ,
            'health_score':health_score ,
            'optimization_type':'REPLACEMENT',
            'issue_type':'LOW_HEALTH',
            'issue_description':f'The rule trigger condition does not match the current data pattern (health: {health_score :.3f})',
            'current_rule_text':self ._get_rule_text (rule_idx ),
            'current_pattern_features':self ._get_rule_pattern_features (rule_idx )
            }

            replacement_candidates .append (candidate )

            # Sort by Health，Minimum priorities
        replacement_candidates .sort (key =lambda x :x ['health_score'])

        return replacement_candidates 

    def _identify_enhancement_candidates (
    self ,
    health_scores :torch .Tensor ,
    effectiveness_scores :torch .Tensor 
    )->List [Dict ]:
        """
        Identification of rules requiring enhancement（Effectiveness<Threshold and health>=threshold）
        
        Track2Analysis：Assess the accuracy of the selected rule projections for target variables
        Ineffectiveness = It's a match, but not a prediction. → Need to fix causality
        
        Args:
            health_scores: Health measures [7]
            effectiveness_scores: Performance Score [7]
            
        Returns:
            List[Dict]: Enhance the list of candidate rules
        """
        # Only the rules of selection.（effectiveness_scores > 0）
        selected_mask =effectiveness_scores >0 
        low_effectiveness_mask =(effectiveness_scores <self .effectiveness_threshold )&selected_mask 
        healthy_mask =health_scores >=self .health_threshold 

        # Rules need to be strengthened：Low efficiency but healthy.
        enhancement_indices =torch .where (low_effectiveness_mask &healthy_mask )[0 ]

        enhancement_candidates =[]
        for idx in enhancement_indices :
            rule_idx =idx .item ()
            health_score =health_scores [idx ].item ()
            effectiveness_score =effectiveness_scores [idx ].item ()

            # Calculates the selection frequency（Estimated number of selected）
            selection_frequency =self ._estimate_selection_frequency (rule_idx ,effectiveness_score )

            candidate ={
            'rule_idx':rule_idx ,
            'health_score':health_score ,
            'effectiveness_score':effectiveness_score ,
            'selection_frequency':selection_frequency ,
            'optimization_type':'ENHANCEMENT',
            'issue_type':'LOW_EFFECTIVENESS',
            'issue_description':f'The rule triggering conditions match well but the causal relationship description is inaccurate. (efficacy: {effectiveness_score :.3f})',
            'diagnosis':'Trigger conditions are well matched but impact projections for target variables are inaccurate',
            'current_rule_text':self ._get_rule_text (rule_idx ),
            'current_pattern_features':self ._get_rule_pattern_features (rule_idx )
            }

            enhancement_candidates .append (candidate )

            # Sort by performance fraction，Minimum priorities
        enhancement_candidates .sort (key =lambda x :x ['effectiveness_score'])

        return enhancement_candidates 

    def _get_rule_text (self ,rule_idx :int )->str :
        """Text description for acquiring rules"""
        rule_id =str (rule_idx )
        rule_patterns =self .current_rules_config .get ('rule_patterns',{})

        if rule_id in rule_patterns :
            return rule_patterns [rule_id ].get ('description',f'Rule {rule_idx }')
        else :
            return f'Rule {rule_idx } (Configuration not found)'

    def _get_rule_pattern_features (self ,rule_idx :int )->Dict [str ,Any ]:
        """Model features of access rules"""
        rule_id =str (rule_idx )
        rule_patterns =self .current_rules_config .get ('rule_patterns',{})

        if rule_id in rule_patterns :
            return rule_patterns [rule_id ].get ('pattern_features',{})
        else :
            return {}

    def _estimate_selection_frequency (self ,rule_idx :int ,effectiveness_score :float )->float :
        """
        Timing of estimation rules

        Estimates based on performance scores，Because it's only the rules that work.
        """
        # Simple inspirational estimate：The more effective the score.，The frequency of selection may be higher.
        # This is a more accurate calculation based on actual surveillance data.
        if effectiveness_score >0 :
        # Assuming frequency selection is linked to performance scores，But add some randomity.
            base_frequency =min (0.8 ,effectiveness_score *1.5 )
            return max (0.1 ,base_frequency )
        else :
            return 0.0 

    def generate_optimization_context (
    self ,
    rule_candidate :Dict [str ,Any ],
    existing_healthy_rules :Optional [List [Dict [str ,Any ]]]=None ,
    generated_rules_in_session :Optional [List [Dict [str ,Any ]]]=None 
    )->OptimizationContext :
        """
        Generate optimized context

        Generate optimisation requests for question rules with complete context information，Including rule information、
        Performance indicators、Optimizing information such as requirements。

        Args:
            rule_candidate: Rule Can not open message
            existing_healthy_rules: List of existing health rules，To avoid duplication
            generated_rules_in_session: Rules generated in the current session，To avoid duplication

        Returns:
            OptimizationContext: Full Optimization Context
        """
        rule_idx =rule_candidate ['rule_idx']
        optimization_type =rule_candidate ['optimization_type']

        # Build current rule information
        current_rule ={
        'text':rule_candidate ['current_rule_text'],
        'pattern_features':rule_candidate ['current_pattern_features'],
        'category':self ._get_rule_category (rule_idx )
        }

        # Build performance indicators
        performance_metrics ={
        'health_score':rule_candidate .get ('health_score',0.0 ),
        'effectiveness_score':rule_candidate .get ('effectiveness_score',0.0 ),
        'selection_frequency':rule_candidate .get ('selection_frequency',0.0 ),
        'attention_weight':rule_candidate .get ('effectiveness_score',0.0 )# Use performance scores as focus weights
        }

        # Build optimization requirements
        optimization_requirements ={
        'optimization_type':optimization_type ,
        'constraints':self ._generate_optimization_constraints (rule_candidate ),
        'existing_healthy_rules':existing_healthy_rules or [],
        'generated_rules_in_session':generated_rules_in_session or []
        }

        return create_optimization_context (
        rule_idx =rule_idx ,
        optimization_type =optimization_type ,
        current_rule =current_rule ,
        performance_metrics =performance_metrics ,
        optimization_requirements =optimization_requirements 
        )

    def _get_rule_category (self ,rule_idx :int )->str :
        """Type of access rule"""
        rule_id =str (rule_idx )
        rule_patterns =self .current_rules_config .get ('rule_patterns',{})

        if rule_id in rule_patterns :
            return rule_patterns [rule_id ].get ('category','trend_driven')
        else :
            return 'trend_driven'





    def get_healthy_rules (self ,health_scores :torch .Tensor )->List [Dict [str ,Any ]]:
        """
        List of rules for accessing health

        Args:
            health_scores: Health measures

        Returns:
            List[Dict]: List of Health Rules
        """
        healthy_rules =[]
        healthy_indices =torch .where (health_scores >=self .health_threshold )[0 ]

        for idx in healthy_indices :
            rule_idx =idx .item ()
            health_score =health_scores [idx ].item ()

            rule_info ={
            'rule_idx':rule_idx ,
            'health_score':health_score ,
            'rule_text':self ._get_rule_text (rule_idx ),
            'pattern_features':self ._get_rule_pattern_features (rule_idx ),
            'category':self ._get_rule_category (rule_idx )
            }
            healthy_rules .append (rule_info )

            # Sort by Health，It's the healthiest front.
        healthy_rules .sort (key =lambda x :x ['health_score'],reverse =True )

        return healthy_rules 

    def _generate_optimization_constraints (self ,rule_candidate :Dict [str ,Any ])->List [str ]:
        """Generate optimized constraints"""
        constraints =[]

        optimization_type =rule_candidate ['optimization_type']

        if optimization_type =='REPLACEMENT':
            constraints .extend ([
            'Maintain logical consistency of the rule',
            'Ensure new trigger conditions match current data patterns',
            'Maintain rule interpretability',
            'Do not duplicate functionality with existing healthy rules or generated rules'
            ])
        elif optimization_type =='ENHANCEMENT':
            constraints .extend ([
            'Keep original trigger conditions unchanged',
            'Correct the impact description on target variables',
            'Improve prediction accuracy'
            ])

        return constraints 



    def apply_threshold_strategy (self ,candidates :List [Dict [str ,Any ]])->List [Dict [str ,Any ]]:
        """
        Apply a simplified threshold trigger policy

        Processing in the order of discovery，I'll handle it first.REPLACEMENTReprocessENHANCEMENT，
        Avoid complex priority ranking algorithms。

        Args:
            candidates: List of candidate rules

        Returns:
            List[Dict]: Candidates in order of processing
        """
        # Separate replacement and enhance candidate
        replacement_candidates =[c for c in candidates if c ['optimization_type']=='REPLACEMENT']
        enhancement_candidates =[c for c in candidates if c ['optimization_type']=='ENHANCEMENT']

        # Replace candidate by health（Minimum priority）
        replacement_candidates .sort (key =lambda x :x .get ('health_score',1.0 ))

        # Increase candidate ranking by performance score（Minimum priority）
        enhancement_candidates .sort (key =lambda x :x .get ('effectiveness_score',1.0 ))

        # Handle replacement first，Reprocess Enhanced
        ordered_candidates =replacement_candidates +enhancement_candidates 

        self .logger .info (
        f"Threshold policy application completed: {len (replacement_candidates )}replacement candidates, "
        f"{len (enhancement_candidates )}enhancement candidates"
        )

        return ordered_candidates 

    def generate_mock_quality_report (self )->QualityReport :
        """Generate simulation quality reports，For testing and demonstration"""
        import torch 

        # Generate7A rule of simulation data
        num_rules =7 

        # Simulate health score（Some below threshold，Some above the threshold）
        health_scores =torch .tensor ([
        0.2 ,# Rule0: It's not healthy.，Replace needed
        0.3 ,# Rule1: Low health，Replace needed
        0.6 ,# Rule2: Health is normal.
        0.7 ,# Rule3: Good health.
        0.8 ,# Rule4: It's healthy.
        0.5 ,# Rule5: Health
        0.35 # Rule6: Health is a little low.，Replace needed
        ],dtype =torch .float32 )

        # Simulate performance scores
        effectiveness_scores =torch .tensor ([
        0.1 ,# Rule0: It's not working.
        0.4 ,# Rule1: It's working.
        0.2 ,# Rule2: Ineffectiveness（Health is normal, but not effective.，Needed improvement）
        0.5 ,# Rule3: Good work.
        0.6 ,# Rule4: It worked very well.
        0.3 ,# Rule5: Medium effectiveness
        0.4 # Rule6: It's working.
        ],dtype =torch .float32 )

        # Simulate Usage Frequency
        usage_frequencies ={
        0 :0.1 ,1 :0.3 ,2 :0.5 ,3 :0.8 ,4 :0.9 ,5 :0.4 ,6 :0.2 
        }

        # Create quality report
        quality_report =QualityReport (
        library_health_scores =health_scores ,
        effectiveness_scores =effectiveness_scores ,
        optimization_strategies ={
        'rule_replacement':[0 ,1 ,6 ],# Rule to replace
        'rule_enhancement':[2 ]# Rules need to be strengthened
        },
        statistics ={
        'total_rules':num_rules ,
        'low_health_rules':3 ,
        'low_effectiveness_rules':1 ,
        'healthy_rules':3 ,
        'usage_frequencies':usage_frequencies 
        }
        )

        self .logger .info (f"Generate simulation quality report: {num_rules }rules")
        self .logger .info (f"  Need to replace: {len (quality_report .optimization_strategies ['rule_replacement'])}indivual")
        self .logger .info (f"  Need to enhance: {len (quality_report .optimization_strategies ['rule_enhancement'])}indivual")

        return quality_report 
