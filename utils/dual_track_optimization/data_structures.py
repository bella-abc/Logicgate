"""
Two-track rule optimisation system - Data Structure Definition Module

Core system-wide data structure and interface specifications defined，Ensure between modules
The ability to exchange data in a standardized manner。
"""

import torch 
from dataclasses import dataclass ,field 
from typing import Dict ,List ,Optional ,Any 
from datetime import datetime 


@dataclass 
class QualityReport :
    """Two-track assessment data structure"""
    library_health_scores :torch .Tensor # [7] Health scores per rule
    effectiveness_scores :torch .Tensor # [7] Performance score per rule
    optimization_strategies :Dict [str ,List [Dict ]]# Optimization strategy recommendations
    statistics :Dict [str ,Any ]# Statistical information
    recommendations :Optional [Dict [str ,List [Dict ]]]=None # Recommended Operations

    def to_dict (self )->Dict [str ,Any ]:
        """Convert to Dictionary Format"""
        return {
        'library_health_scores':self .library_health_scores .tolist (),
        'effectiveness_scores':self .effectiveness_scores .tolist (),
        'optimization_strategies':self .optimization_strategies ,
        'statistics':self .statistics ,
        'recommendations':self .recommendations 
        }


@dataclass 
class OptimizationContext :
    """Optimizing context data structure"""
    rule_idx :int # Rule Index
    optimization_type :str # 'REPLACEMENT' | 'ENHANCEMENT'
    current_rule :Dict [str ,Any ]=field (default_factory =dict )# Current rule information
    performance_metrics :Dict [str ,float ]=field (default_factory =dict )# Performance indicators

    optimization_requirements :Dict [str ,Any ]=field (default_factory =dict )# Optimization requirements

    def validate (self )->bool :
        """Verify the integrity of context data"""
        required_fields =['rule_idx','optimization_type']
        return all (hasattr (self ,field )and getattr (self ,field )is not None 
        for field in required_fields )


@dataclass 
class APIResponse :
    """APIResponse data structure"""
    success :bool # Successful call
    data :Dict [str ,Any ]# Response Data
    metadata :Dict [str ,Any ]=field (default_factory =dict )# Metadata Information
    error :Optional [str ]=None # Error message

    def is_valid (self )->bool :
        """Check for response effectiveness"""
        if not self .success :
            return False 

        if not self .data :
            return False 

            # Check required data fields
        required_keys =['analysis']
        if self .data .get ('optimized_rule'):# Replace scene
            required_keys .extend (['optimized_rule','pattern_features'])
        elif self .data .get ('enhanced_rule'):# Enhance scene
            required_keys .extend (['enhanced_rule','enhanced_features'])
        else :
            return False 

        return all (key in self .data for key in required_keys )

    def get_rule_data (self )->Optional [Dict [str ,Any ]]:
        """Access to rule data"""
        if not self .is_valid ():
            return None 

        if 'optimized_rule'in self .data :
            return self .data ['optimized_rule']
        elif 'enhanced_rule'in self .data :
            return self .data ['enhanced_rule']
        else :
            return None 

    def get_pattern_features (self )->Optional [Dict [str ,Any ]]:
        """Get Mode Features"""
        if not self .is_valid ():
            return None 

        if 'pattern_features'in self .data :
            return self .data ['pattern_features']
        elif 'enhanced_features'in self .data :
            return self .data ['enhanced_features']
        else :
            return None 


@dataclass 
class StructuredRuleConfig :
    """Structured rules configuration data structure"""
    description :str # Rule Description
    direction :float # Impact direction (1.0/-1.0)
    direction_description :str # Direction Description
    pattern_features :Dict [str ,Any ]# Feature Configuration
    category :str # Rule category
    metadata :Dict [str ,Any ]# Metadata

    def to_rule_patterns_format (self )->Dict [str ,Any ]:
        """Convert torule_patterns.jsonFormat"""
        return {
        'description':self .description ,
        'direction':self .direction ,
        'direction_description':self .direction_description ,
        'pattern_features':self .pattern_features ,
        'category':self .category 
        }

    def validate (self )->bool :
        """Validate configuration"""
        # Check Required Fields
        if not all ([self .description ,self .pattern_features ,self .category ]):
            return False 

            # Check value ranges
        if not (-1.0 <=self .direction <=1.0 ):
            return False 

            # Check category validity
        valid_categories =[
        'trend_driven','temporal_pattern'
        ]
        if self .category not in valid_categories :
            return False 

        return True 


@dataclass 
class ReplacementResult :
    """Replace the result data structure"""
    rule_idx :int # Rule Index
    optimization_type :str # Optimization Type (REPLACEMENT/ENHANCEMENT)
    original_rule :str # Original rules
    new_rule :str # New rules
    success :bool # Successful replacement
    strategy :str ="REPLACE"# Replace Action
    backup_path :Optional [str ]=None # Backup file path
    backup_created :bool =True # Whether to create backup
    validation_passed :bool =True # Validation
    performance_change :Dict [str ,float ]=field (default_factory =dict )# Performance Change
    metadata :Dict [str ,Any ]=field (default_factory =dict )# Metadata
    errors :List [str ]=field (default_factory =list )# Error message
    timestamp :str =field (default_factory =lambda :datetime .now ().isoformat ())

    def to_dict (self )->Dict [str ,Any ]:
        """Convert to Dictionary Format"""
        return {
        'success':self .success ,
        'rule_idx':self .rule_idx ,
        'optimization_type':self .optimization_type ,
        'original_rule':self .original_rule ,
        'new_rule':self .new_rule ,
        'strategy':self .strategy ,
        'backup_path':self .backup_path ,
        'backup_created':self .backup_created ,
        'validation_passed':self .validation_passed ,
        'performance_change':self .performance_change ,
        'metadata':self .metadata ,
        'errors':self .errors ,
        'timestamp':self .timestamp 
        }


        # Auxiliary Functions
def create_optimization_context (
rule_idx :int ,
optimization_type :str ,
current_rule :Dict [str ,Any ],
performance_metrics :Dict [str ,float ],
optimization_requirements :Optional [Dict [str ,Any ]]=None 
)->OptimizationContext :
    """Create auxiliary function to optimize context"""
    if optimization_requirements is None :
        optimization_requirements ={
        'optimization_type':optimization_type ,
        'constraints':[]
        }

    return OptimizationContext (
    rule_idx =rule_idx ,
    optimization_type =optimization_type ,
    current_rule =current_rule ,
    performance_metrics =performance_metrics ,
    optimization_requirements =optimization_requirements 
    )


@dataclass 
class OptimizationSession :
    """Optimizing session data structure"""
    session_id :str # SessionID
    epoch :int # Training rotations
    start_time :datetime =field (default_factory =datetime .now )# Start Time
    end_time :Optional [datetime ]=None # End Time
    status :str ="INITIALIZED"# Session Status
    quality_report :Optional [QualityReport ]=None # Quality reports
    optimization_contexts :List [OptimizationContext ]=field (default_factory =list )# Optimizing Context
    results :List [ReplacementResult ]=field (default_factory =list )# Optimizing results
    metadata :Dict [str ,Any ]=field (default_factory =dict )# Metadata

    # List of candidate rules（Dynamic added field）
    replacement_candidates :List [Dict [str ,Any ]]=field (default_factory =list )# Replace Candidate Rule
    enhancement_candidates :List [Dict [str ,Any ]]=field (default_factory =list )# Enhance the candidacy rule
    total_candidates :int =0 # Total number of candidates
    high_priority_candidates :int =0 # Number of high-priority candidates
    medium_priority_candidates :int =0 # Number of candidates with medium priority
    generated_rules_in_session :List [Dict [str ,Any ]]=field (default_factory =list )# Rules generated in sessions

    def get_duration (self )->Optional [float ]:
        """Get Session Duration（sec）"""
        if self .end_time :
            return (self .end_time -self .start_time ).total_seconds ()
        return None 

    def mark_completed (self )->None :
        """Mark Session Completed"""
        self .end_time =datetime .now ()
        self .status ="COMPLETED"


class ValidationError (Exception ):
    """Authentication error abnormal category"""
    def __init__ (self ,message :str ,field_name :str =None ,error_type :str =None ):
        super ().__init__ (message )
        self .field_name =field_name 
        self .error_type =error_type 
        self .message =message 


@dataclass 
class ParsedResponse :
    """Response structure after resolution"""
    validated :bool # Validation pass
    cleaned_rule :Optional [Dict [str ,Any ]]=None # Clean-up rule data
    cleaned_features :Optional [Dict [str ,Any ]]=None # Cleanup feature data
    analysis :Optional [Dict [str ,Any ]]=None # Analysis of data
    validation_errors :List [ValidationError ]=field (default_factory =list )# Authentication error

    def has_errors (self )->bool :
        """Could not close temporary folder: %s"""
        return len (self .validation_errors )>0 


def validate_quality_report (report :QualityReport )->bool :
    """Validation of quality reports"""
    try :
    # Check the dimensions of the tension
        if report .library_health_scores .shape [0 ]!=6 :
            return False 
        if report .effectiveness_scores .shape [0 ]!=6 :
            return False 

            # Check value ranges
        if not torch .all ((report .library_health_scores >=0 )&(report .library_health_scores <=1 )):
            return False 
        if not torch .all ((report .effectiveness_scores >=0 )&(report .effectiveness_scores <=1 )):
            return False 

            # Check required policy fields
        required_strategy_keys =['rule_replacement','rule_enhancement']
        if not all (key in report .optimization_strategies for key in required_strategy_keys ):
            return False 

        return True 
    except Exception :
        return False 


def create_api_response (success :bool ,data :Dict [str ,Any ]=None ,
error :str =None ,metadata :Dict [str ,Any ]=None )->APIResponse :
    """CreateAPIAuxiliary function to respond"""
    return APIResponse (
    success =success ,
    data =data or {},
    metadata =metadata or {},
    error =error 
    )


def create_replacement_result (rule_idx :int ,optimization_type :str ,
original_rule :str ,new_rule :str ,
success :bool ,**kwargs )->ReplacementResult :
    """Auxiliary function to create replacement result"""
    return ReplacementResult (
    rule_idx =rule_idx ,
    optimization_type =optimization_type ,
    original_rule =original_rule ,
    new_rule =new_rule ,
    success =success ,
    **kwargs 
    )
