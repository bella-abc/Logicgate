"""
Two-track rule optimisation system - Tip Project Module

The hint project module isAPIInteractive"Translator"，Responsible for optimizing the context of the structuralization
Convert to high quality natural language hints，EnsureAIModels can accurately understand optimization needs
and generate recommendations for rule optimization that meet the requirements。
"""

import logging 
from typing import Dict ,Any ,List 
from datetime import datetime 

from .data_structures import OptimizationContext ,QualityReport 
from .prompt_templates import PromptTemplates 


class PromptEngineering :
    """
    Tip Project Category

    Main responsibilities：
    1. Context Analysis：Depth analysis optimize context，Can not open message
    2. Template Selection：Select the right hint mode according to the optimal type Board
    3. Dynamic Fill：Convert structured data into natural language descriptions
    4. Quality control：Ensuring quality and completeness of generated hints
    5. Format standardization：Harmonized hint format and structure
    """

    def __init__ (self ):
        """Initialise Phrasing Projector"""
        self .logger =logging .getLogger (__name__ )
        self .templates =PromptTemplates ()

        # Data variables supported by dynamic acquisition
        self .supported_variables =self ._get_supported_variables ()

    def _get_supported_variables (self ):
        """Data variables to obtain default support（For initialization only）"""
        # This is only the default value，When actually used fromcontextdynamic extraction
        return ['variable_1','variable_2']

    def _extract_variables_from_context (self ,context :OptimizationContext )->list :
        """FromOptimizationContextin which to extract the variable actually used First Name"""
        variables =set ()

        # Extract variables from the current rule
        pattern_features =context .current_rule .get ('pattern_features',{})

        # Fromtrend_patternMiddle extraction variable
        if 'trend_pattern'in pattern_features :
            trend_pattern =pattern_features ['trend_pattern']
            if isinstance (trend_pattern ,dict )and 'variable'in trend_pattern :
                variables .add (trend_pattern ['variable'])

                # Fromexisting_healthy_rulesMiddle extraction variable
        existing_rules =context .optimization_requirements .get ('existing_healthy_rules',[])
        for rule in existing_rules :
            rule_features =rule .get ('pattern_features',{})
            if 'trend_pattern'in rule_features :
                trend_pattern =rule_features ['trend_pattern']
                if isinstance (trend_pattern ,dict )and 'variable'in trend_pattern :
                    variables .add (trend_pattern ['variable'])

                    # Fromsessionin which to generate a variable
        session_rules =context .optimization_requirements .get ('generated_rules_in_session',[])
        for rule in session_rules :
            rule_features =rule .get ('pattern_features',{})
            if 'trend_pattern'in rule_features :
                trend_pattern =rule_features ['trend_pattern']
                if isinstance (trend_pattern ,dict )and 'variable'in trend_pattern :
                    variables .add (trend_pattern ['variable'])

                    # Convert to List and Sort，Ensuring coherence
        result =sorted (list (variables ))if variables else ['variable_1','variable_2']

        self .logger .info (f"fromcontextvariables extracted from: {result }")
        return result 

    def generate_prompt (self ,context :OptimizationContext ,
    quality_report :QualityReport )->Dict [str ,str ]:
        """
        Generate optimized hints

        Args:
            context: Optimizing Context
            quality_report: Quality reports

        Returns:
            Organisationsystem_promptanduser_promptDictionary
        """
        try :
        # Validate input
            if not context .validate ():
                raise ValueError ("Optimizing context validation failed")

                # Use enhanced hint generation method，Make full use of all available information
            user_prompt =self .create_enhanced_optimization_prompt (context )
            system_prompt =self .templates .get_system_prompt ()

            # Validation generated hints
            self ._validate_prompt (user_prompt ,context .optimization_type )

            self .logger .info (f"Successfully generated{context .optimization_type }Type of enhanced prompt words")

            return {
            'system_prompt':system_prompt ,
            'user_prompt':user_prompt ,
            'metadata':{
            'rule_idx':context .rule_idx ,
            'optimization_type':context .optimization_type ,
            'priority':getattr (context ,'priority','MEDIUM'),
            'generated_at':datetime .now ().isoformat (),
            'method':'enhanced_optimization_prompt',
            'features_used':{
            'health_score':context .performance_metrics .get ('health_score',0.0 ),
            'effectiveness_score':context .performance_metrics .get ('effectiveness_score',0.0 ),
            'selection_frequency':context .performance_metrics .get ('selection_frequency',0.0 ),
            'attention_weight':context .performance_metrics .get ('attention_weight',0.0 ),
            'rule_category':context .current_rule .get ('category','unknown'),
            'has_pattern_features':bool (context .current_rule .get ('pattern_features')),
            'has_constraints':bool (context .optimization_requirements .get ('constraints')),
            'existing_rules_count':len (context .optimization_requirements .get ('existing_healthy_rules',[])),
            'session_rules_count':len (context .optimization_requirements .get ('generated_rules_in_session',[]))
            }
            }
            }

        except Exception as e :
            self .logger .error (f"Failed to generate prompt word: {e }")
            raise 




    def _format_existing_rules_context (self ,context :OptimizationContext )->str :
        """Format the context of existing health rules"""
        existing_rules =context .optimization_requirements .get ('existing_healthy_rules',[])

        if not existing_rules :
            return "There is no information available on health rules.。"

            # Limit the number of rules displayed，AvoidpromptToo long.
        max_rules_to_show =5 
        rules_to_show =existing_rules [:max_rules_to_show ]

        context_parts =["Health rules in the current system（Please avoid creating duplication of functions.）："]

        for i ,rule in enumerate (rules_to_show ,1 ):
            rule_text =rule .get ('rule_text',f"rule{rule .get ('rule_idx','unknown')}")
            health_score =rule .get ('health_score',0.0 )
            category =rule .get ('category','unknown')

            context_parts .append (
            f"{i }. 【{category }】{rule_text } (health: {health_score :.2f})"
            )

        if len (existing_rules )>max_rules_to_show :
            context_parts .append (f"... besides {len (existing_rules )-max_rules_to_show } health rules not shown")

        return "\n".join (context_parts )

    def _format_session_generated_rules (self ,context :OptimizationContext )->str :
        """Format the rules generated from the current session"""
        generated_rules =context .optimization_requirements .get ('generated_rules_in_session',[])

        if not generated_rules :
            return "No new rules have been generated in the current session。"

        context_parts =["Optimizing rules generated from this session（Please avoid duplication.）："]

        for i ,rule in enumerate (generated_rules ,1 ):
            rule_text =rule .get ('description',rule .get ('rule_text',f"new rules{i }"))
            category =rule .get ('category','unknown')

            context_parts .append (
            f"{i }. 【{category }】{rule_text }"
            )

        return "\n".join (context_parts )

    def _format_pattern_features (self ,pattern_features :Dict [str ,Any ])->str :
        """Formatting Mode Characteristic Information"""
        if not pattern_features :
            return "There is no specific mode profile for the current rule。"

        feature_parts =["Pattern characterization of the current rule："]

        for feature_name ,feature_config in pattern_features .items ():
            if isinstance (feature_config ,dict ):
                config_details =[]
                for key ,value in feature_config .items ():
                    config_details .append (f"{key }: {value }")
                feature_parts .append (f"- {feature_name }: {{{', '.join (config_details )}}}")
            else :
                feature_parts .append (f"- {feature_name }: {feature_config }")

        return "\n".join (feature_parts )

    def _format_optimization_constraints (self ,context :OptimizationContext )->str :
        """Formatting Optimizing Constraints"""
        constraints =context .optimization_requirements .get ('constraints',[])

        if not constraints :
            return "No specific constraints。"

        constraint_parts =["Optimizing constraints："]
        for i ,constraint in enumerate (constraints ,1 ):
            constraint_parts .append (f"{i }. {constraint }")

        return "\n".join (constraint_parts )

    def _generate_data_pattern_analysis (self ,context :OptimizationContext )->str :
        """Based onpattern_featuresData generation model analysis"""
        pattern_features =context .current_rule .get ('pattern_features',{})

        if not pattern_features :
            return "Data model analysis：The current rule lacks specific mode profiles，Suggested analysis based on type of rules and performance indicators。"

        analysis_parts =["Data model analysis："]

        # Generate analysis according to different signature types
        for feature_name ,feature_config in pattern_features .items ():
            if feature_name =='grid_load'and isinstance (feature_config ,dict ):
                trend =feature_config .get ('trend','unknown')
                analysis_parts .append (f"- Grid load pattern：The trend is{trend }")

            elif feature_name =='wind_power'and isinstance (feature_config ,dict ):
                variability =feature_config .get ('variability','unknown')
                analysis_parts .append (f"- Wind power output mode：The variability is{variability }")

            elif feature_name =='time_pattern'and isinstance (feature_config ,dict ):
                pattern_type =feature_config .get ('type','unknown')
                hours =feature_config .get ('hours',[])
                analysis_parts .append (f"- time pattern：The type is{pattern_type }，critical period{hours }")

            elif feature_name =='temperature'and isinstance (feature_config ,dict ):
                sensitivity =feature_config .get ('sensitivity','unknown')
                analysis_parts .append (f"- temperature mode：The sensitivity is{sensitivity }")

            else :
                analysis_parts .append (f"- {feature_name }：{feature_config }")

        return "\n".join (analysis_parts )




    def _validate_prompt (self ,prompt :str ,optimization_type :str )->None :
        """Verify the quality of the hint generated"""
        if not prompt or len (prompt .strip ())<100 :
            raise ValueError ("The resulting hint is too short，Possible lack of critical information")

            # Check essential keywords
        required_keywords =['Rule','Optimization','JSON']
        missing_keywords =[kw for kw in required_keywords if kw not in prompt ]

        if missing_keywords :
            raise ValueError (f"Prompt word is missing keywords: {missing_keywords }")

            # Check for a specific type of optimised keyword
        if optimization_type =='REPLACEMENT':
            if 'Replace'not in prompt :
                raise ValueError ("Missing replacement type hint'Replace'keyword")
        elif optimization_type =='ENHANCEMENT':
            if 'Enhanced'not in prompt :
                raise ValueError ("Enhancement type hint missing'Enhanced'keyword")

        self .logger .debug ("Plugin Validation")

    def get_expected_response_schema (self ,optimization_type :str )->Dict [str ,Any ]:
        """Get the desired responseSchema"""
        schemas =self .templates .get_response_schema ()

        if optimization_type =='REPLACEMENT':
            return schemas ['replacement_response']
        elif optimization_type =='ENHANCEMENT':
            return schemas ['enhancement_response']
        else :
            raise ValueError (f"Unsupported optimization type: {optimization_type }")

    def validate_response_format (self ,response_data :Dict [str ,Any ],
    optimization_type :str )->List [str ]:
        """
        AuthenticationAPIResponse Format
        
        Args:
            response_data: APIResponse Data
            optimization_type: Optimization Type
            
        Returns:
            Authentication Error List，Empty list indicates authentication pass
        """
        errors =[]
        schema =self .get_expected_response_schema (optimization_type )

        try :
        # SimpleSchemaAuthentication
            self ._validate_schema (response_data ,schema ,errors ,"")
        except Exception as e :
            errors .append (f"SchemaValidation exception: {e }")

        return errors 

    def _validate_schema (self ,data :Any ,schema :Dict [str ,Any ],
    errors :List [str ],path :str )->None :
        """Recursive ValidationSchema"""
        if schema .get ('type')=='object':
            if not isinstance (data ,dict ):
                errors .append (f"{path }: expected object type，Actually{type (data ).__name__ }")
                return 

                # Check Required Fields
            required =schema .get ('required',[])
            for field in required :
                if field not in data :
                    errors .append (f"{path }.{field }: Missing required field")

                    # Authentication Properties
            properties =schema .get ('properties',{})
            for field ,field_schema in properties .items ():
                if field in data :
                    self ._validate_schema (data [field ],field_schema ,errors ,f"{path }.{field }")

        elif schema .get ('type')=='string':
            if not isinstance (data ,str ):
                errors .append (f"{path }: Expected string type，Actually{type (data ).__name__ }")

        elif schema .get ('type')=='number':
            if not isinstance (data ,(int ,float )):
                errors .append (f"{path }: Expected numeric type，Actually{type (data ).__name__ }")
            else :
            # Check value ranges
                if 'minimum'in schema and data <schema ['minimum']:
                    errors .append (f"{path }: value{data }less than minimum{schema ['minimum']}")
                if 'maximum'in schema and data >schema ['maximum']:
                    errors .append (f"{path }: value{data }greater than maximum{schema ['maximum']}")

        elif schema .get ('type')=='array':
            if not isinstance (data ,list ):
                errors .append (f"{path }: expected array type，Actually{type (data ).__name__ }")


    def create_enhanced_optimization_prompt (self ,context :OptimizationContext )->str :
        """
        Create enhanced optimization prompt based on complete OptimizationContext

        This is the main prompt generation method that fully utilizes all available information

        Args:
            context: Complete optimization context

        Returns:
            str: Generated prompt
        """
        # Extract all available information
        rule_text =context .current_rule .get ('text','Rule description missing')
        optimization_type =context .optimization_type 
        performance_metrics =context .performance_metrics 

        # Basic rule information
        rule_category =context .current_rule .get ('category','trend_driven')
        pattern_features =context .current_rule .get ('pattern_features',{})

        # Performance metrics
        health_score =performance_metrics .get ('health_score',0.0 )
        effectiveness_score =performance_metrics .get ('effectiveness_score',0.0 )
        selection_frequency =performance_metrics .get ('selection_frequency',0.0 )
        attention_weight =performance_metrics .get ('attention_weight',0.0 )

        # Optimization requirements
        constraints =context .optimization_requirements .get ('constraints',[])
        existing_rules =context .optimization_requirements .get ('existing_healthy_rules',[])
        session_rules =context .optimization_requirements .get ('generated_rules_in_session',[])

        # Format feature information
        pattern_features_text =self ._format_pattern_features_detailed (pattern_features )
        constraints_text =self ._format_constraints_detailed (constraints )
        existing_rules_text =self ._format_existing_rules_detailed (existing_rules )
        session_rules_text =self ._format_session_rules_detailed (session_rules )

        # Extract actual variables from context
        actual_variables =self ._extract_variables_from_context (context )

        # Extract original rule's variables and time granularity information
        if rule_category =="trend_driven":
        # For trend_driven type, extract variable from trend_pattern
            trend_pattern =pattern_features .get ('trend_pattern',{})
            variable =trend_pattern .get ('variable','Unknown variable')
            original_variables =f'"{variable }"'
            original_time_granularity ="No specific time granularity"
        elif rule_category =="temporal_pattern":
        # For temporal_pattern type, extract type from time_pattern as time granularity
            time_pattern =pattern_features .get ('time_pattern',{})
            time_type =time_pattern .get ('type','Unknown time granularity')
            original_variables ="Time features"
            original_time_granularity =time_type 
        else :
            original_variables ="Unknown variable"
            original_time_granularity ="Unknown type"

            # Build conditional constraint information based on rule category
        if rule_category =="trend_driven":
            constraint_info =f"**Original Rule Variables**: {original_variables }"
            replacement_constraint_rule =f"- **Current rule is trend_driven type**: New rule must only optimize for the same exogenous variables ({original_variables }) as the original rule, cannot switch to other variables"
        else :# temporal_pattern
            constraint_info =f"**Original Rule Time Granularity**: {original_time_granularity }"
            replacement_constraint_rule =f"- **Current rule is temporal_pattern type**: New rule must maintain the same time granularity type ({original_time_granularity }), cannot change time granularity"

        if optimization_type =="REPLACEMENT":
            prompt =f"""
# Time Series Prediction Rule Replacement Task - Complete Analysis

## Task Background
You are a data analysis expert who needs to optimize rules for a time series prediction system. Current rule {context .rule_idx } has health score issues with low matching degree to current data patterns, requiring generation of new rules to replace it.

**Important Note**: A health score of {health_score :.3f} indicates that the rule's trigger conditions do not match actual data patterns, leading to low attention weights, selection frequency and other indicators. This is because the system excludes it during the health screening stage, not due to prediction performance issues.

## Complete Analysis of Current Problem Rule
**Rule Index**: {context .rule_idx }
**Rule Category**: {rule_category }
**Current Description**: {rule_text }

### Performance Metrics Details
- **Health Score**: {health_score :.3f} (Data pattern matching degree - Serious problem)
- **Effectiveness Score**: {effectiveness_score :.3f} (Prediction accuracy)
- **Selection Frequency**: {selection_frequency :.3f} (Frequency of rule being selected for use)
- **Attention Weight**: {attention_weight :.3f} (Weight assigned by system)

### Current Rule Feature Configuration
{pattern_features_text }

## Root Cause Analysis
Based on health score {health_score :.3f}, this rule has the following problems:
- Trigger conditions severely mismatch current data patterns
- Rule parameters may be outdated and not adapted to current market environment

## Optimization Constraints
{constraints_text }

## Existing Healthy Rules Reference
{existing_rules_text }

## Rules Generated in Current Session
{session_rules_text }

### REPLACEMENT Rule Special Constraints:
**CRITICAL CONSTRAINT**: Since this is rule REPLACEMENT, the new rule MUST maintain exactly the same category and variable as the original rule:

**MANDATORY REQUIREMENTS**:
- **Original Rule Category**: {rule_category } → **New Rule Category MUST BE**: {rule_category }
- **Original Rule Variable(s)**: {' and '.join (actual_variables )} → **New Rule Variable(s) MUST BE**: {' and '.join (actual_variables )}

{constraint_info }

#### Replacement Constraint Rules:
{replacement_constraint_rule }

**IMPORTANT**: The new rule can ONLY optimize the trigger conditions, thresholds, or impact descriptions within the SAME category and variable scope. Category changes (trend_driven ↔ temporal_pattern) are FORBIDDEN.

### CRITICAL Variable and Impact Requirements:

#### Variable Name Constraints (MANDATORY):
- **MUST use EXACT variable names**: {' and '.join (actual_variables )}
- **DO NOT modify, simplify, or abbreviate variable names**
- **Correct usage**: Use the EXACT names as provided: {' and '.join (actual_variables )}

#### Impact Direction Constraints (MANDATORY):
- **MUST specify ONE clear impact direction only**
- **FORBIDDEN ambiguous descriptions**:
  ❌ "stabilize or decrease" (multiple effects)
  ❌ "increase or remain stable" (unclear direction)
  ❌ "tend to vary" (no clear direction)
  ❌ "may increase or decrease" (ambiguous)
- **REQUIRED clear descriptions**:
  ✅ "tend to increase" (clear positive impact)
  ✅ "tend to decrease" (clear negative impact)
  ✅ "remain stable" (clear neutral impact, but rare)

#### Direction Value and Description Consistency (MANDATORY):
- **MUST ensure direction value matches description**:
  ✅ direction: 1.0 + "tend to increase" (consistent)
  ✅ direction: -1.0 + "tend to decrease" (consistent)
  ❌ direction: 1.0 + "tend to decrease" (inconsistent)
  ❌ direction: -1.0 + "stabilize or decrease" (ambiguous)

### Rule Description and Feature Configuration Matching Requirements:
- **For trend_driven category**: MUST use trend_pattern configuration with the exact variable(s): {' and '.join (actual_variables )}
- **For temporal_pattern category**: MUST use time_pattern configuration and follow these STRICT field requirements:

#### Temporal Pattern Field Requirements (CRITICAL):
- **If type is "peak_hours"**: Must provide "hours" array (e.g., [7,8,9,18,19,20,21])
- **If type is "seasonal"**: Must provide "months" array (e.g., [6,7,8] for summer, [12,1,2] for winter)  
- **If type is "weekend"**: Must provide "weekdays" array [5,6] for Saturday+Sunday, NOT "hours"

#### IMPORTANT: Weekday Encoding (CRITICAL):
**System uses 0-6 encoding**: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
- **Weekend is [5,6]**: 5=Saturday, 6=Sunday
- **DO NOT use [6,7]**: 7 is invalid (only 0-6 exist)
- **DO NOT use [1,7]**: This is wrong encoding

#### FORBIDDEN Combinations:
- ❌ NEVER use "hours" field with "weekend" type
- ❌ NEVER use "months" field with "peak_hours" type
- ❌ NEVER use "weekdays" field with "seasonal" or "peak_hours" type
- ❌ NEVER use weekday values outside 0-6 range



## Output Requirements
Please return optimization results in JSON format, ensuring complete compliance with rule_patterns.json standard format:

{{
    "analysis": {{
        "current_rule_issues": "Detailed analysis of specific problems with current rule",
        "data_pattern_insights": "Data pattern insights based on feature configuration and performance metrics",
        "optimization_strategy": "Targeted optimization strategy explanation",
        "expected_improvements": "Expected performance improvement effects"
    }},
    "optimized_rule": {{
        "description": "Detailed description of new rule, must completely match configuration in pattern_features",
        "direction": "Impact direction, must be one of -1.0(negative), 0.0(neutral), or 1.0(positive)",
        "direction_description": "English description of impact direction, corresponding to direction value: 'Positive impact - price increase'(1.0), 'Negative impact - price decrease'(-1.0), 'Neutral impact - unclear price effect'(0.0)",
        "category": "Rule category, must be: 'trend_driven'(trend-driven) or 'temporal_pattern'(time pattern)"
    }},
    "pattern_features": {{
        "Must strictly configure features according to category requirements, completely matching description and category"
    }},
    "improvement_explanation": "Detailed explanation of improvements and expected effects of new rule compared to original rule"
}}

## Standard Feature Configuration Examples

### Complete trend_driven Rule Example:
```json
"pattern_features": {{
    "trend_pattern": {{
        "variable": "{actual_variables [0 ]if actual_variables else 'variable_1'}",
        "trend": "increasing",
        "adaptive_threshold": true,
        "impact_direction": "positive"
    }}
}}
```

### Complete temporal_pattern Rule Examples:

#### Seasonal Pattern:
```json
"pattern_features": {{
    "time_pattern": {{
        "type": "seasonal",
        "months": [6, 7, 8],
        "seasonal_context": "summer"
    }}
}}
```

#### Peak Hours Pattern:
```json
"pattern_features": {{
    "time_pattern": {{
        "type": "peak_hours",
        "hours": [7, 8, 9, 18, 19, 20, 21],
        "peak_context": "morning_evening"
    }}
}}
```

#### Weekend Pattern:
```json
"pattern_features": {{
    "time_pattern": {{
        "type": "weekend",
        "weekdays": [5, 6]
    }}
}}
```
**Note**: weekdays uses 0-6 encoding where 5=Saturday, 6=Sunday

Please generate a high-quality replacement rule based on the above strict constraints.
"""
        else :# ENHANCEMENT
            prompt =f"""
# Time Series Prediction Rule Enhancement Task - Complete Analysis

## Task Background
You are a data analysis expert who needs to enhance rules for a time series prediction system. Current rule {context .rule_idx } has a typical "right condition, wrong conclusion" problem.

**Core Issue**: This rule's trigger conditions match data patterns well (can be selected for use), but its impact prediction on target values is inaccurate (low model trust). This is not a trigger condition problem, but a causal relationship description problem.

**Important Understanding**: The effectiveness score {effectiveness_score :.3f} reflects the model's trust in the rule's predictive ability. A low score indicates the model finds that the rule's causal logic does not align with actual data patterns, requiring correction of the target value impact description, not modification of trigger conditions.

## Complete Analysis of Current Rule
**Rule Index**: {context .rule_idx }
**Rule Category**: {rule_category }
**Current Description**: {rule_text }

### Performance Metrics Details
- **Health Score**: {health_score :.3f} (Data pattern matching degree - Good)
- **Effectiveness Score**: {effectiveness_score :.3f} (Prediction accuracy - Needs improvement)
- **Selection Frequency**: {selection_frequency :.3f} (Frequency of rule being selected for use)
- **Attention Weight**: {attention_weight :.3f} (Weight assigned by system)

### Current Rule Feature Configuration
{pattern_features_text }

## Root Cause Analysis of Effectiveness Issues
Based on effectiveness score {effectiveness_score :.3f} and selection frequency {selection_frequency :.3f}, this rule has a typical "right condition, wrong conclusion" problem:

### Core Problem Diagnosis
- ✅ **Trigger conditions match well**: Rule can be selected (selection frequency {selection_frequency :.3f}), indicating trigger conditions match data patterns
- ❌ **Causal relationship description inaccurate**: Model finds rule provides little help for target value prediction (attention weight {attention_weight :.3f}), indicating prediction logic problems



### Key Insights
**No need to modify trigger conditions**, focus should be on correcting target value impact descriptions:
- Impact direction may be wrong (incorrect rise/fall judgment)
- Impact magnitude may be inaccurate (overestimation or underestimation)
- Time lag characteristics may be missing (immediate impact vs delayed impact)
- Synergistic impact factors may not be considered (interaction with other factors)

## Optimization Constraints
{constraints_text }

## Existing Healthy Rules Reference
{existing_rules_text }

## Rules Generated in Current Session
{session_rules_text }

### ENHANCEMENT Rule Special Constraints:
**Key Note**: Since this is rule ENHANCEMENT, new rules must maintain the same variable scope and type as original rules while optimizing impact mechanisms and parameters:

**Original Rule Category**: {rule_category }
{constraint_info }

### Rule Description and Feature Configuration Matching Requirements:
- If description involves variable trends (changes in {' or '.join (actual_variables )}), must choose trend_driven category and provide trend_pattern configuration
- If description involves time patterns, must choose temporal_pattern category and follow these STRICT field requirements:

#### Temporal Pattern Field Requirements (CRITICAL):
- **If type is "peak_hours"**: Must provide "hours" array (e.g., [7,8,9,18,19,20,21])
- **If type is "seasonal"**: Must provide "months" array (e.g., [6,7,8] for summer, [12,1,2] for winter)  
- **If type is "weekend"**: Must provide "weekdays" array [5,6] for Saturday+Sunday, NOT "hours"

#### IMPORTANT: Weekday Encoding (CRITICAL):
**System uses 0-6 encoding**: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday
- **Weekend is [5,6]**: 5=Saturday, 6=Sunday
- **DO NOT use [6,7]**: 7 is invalid (only 0-6 exist)
- **DO NOT use [1,7]**: This is wrong encoding

#### FORBIDDEN Combinations:
- ❌ NEVER use "hours" field with "weekend" type
- ❌ NEVER use "months" field with "peak_hours" type
- ❌ NEVER use "weekdays" field with "seasonal" or "peak_hours" type
- ❌ NEVER use weekday values outside 0-6 range



## Output Requirements
Please return enhancement results in JSON format, ensuring complete compliance with rule_patterns.json standard format:

{{
    "analysis": {{
        "current_rule_strengths": "Analyze current rule advantages: high trigger condition matching degree, able to accurately identify relevant market states",
        "effectiveness_issues": "Detailed analysis of specific reasons for insufficient effectiveness: descriptions of impact direction/magnitude/time characteristics do not match reality",
        "enhancement_strategy": "Keep trigger conditions unchanged, focus on correcting causal relationships: adjust impact direction, quantify impact magnitude, consider time lag",
        "expected_improvements": "Expected to improve model trust in rules, increasing attention weight from {attention_weight :.3f} to above 0.6"
    }},
    "enhanced_rule": {{
        "description": "Enhanced rule description, must completely match configuration in enhanced_features",
        "direction": "Impact direction, must be one of -1.0(negative), 0.0(neutral), or 1.0(positive)",
        "direction_description": "English description of impact direction, corresponding to direction value: 'Positive impact - price increase'(1.0), 'Negative impact - price decrease'(-1.0), 'Neutral impact - unclear price effect'(0.0)",
        "category": "Rule category, must be: 'trend_driven'(trend-driven) or 'temporal_pattern'(time pattern)"
    }},
    "enhanced_features": {{
        "Must strictly configure features according to category requirements, completely matching description and category"
    }},
    "enhancement_details": {{
        "adjusted_parameters": ["List adjusted parameters"],
        "mechanism_improvements": "Detailed explanation of impact mechanism improvements"
    }},
    "effectiveness_improvement": "Detailed explanation of how enhancement improves prediction accuracy and expected effects"
}}

## Standard Feature Configuration Examples

### Complete trend_driven Rule Example:
```json
"enhanced_features": {{
    "trend_pattern": {{
        "variable": "{actual_variables [0 ]if actual_variables else 'variable_1'}",
        "trend": "increasing",
        "adaptive_threshold": true,
        "impact_direction": "positive"
    }}
}}
```

### Complete temporal_pattern Rule Example:
```json
"enhanced_features": {{
    "time_pattern": {{
        "type": "seasonal",
        "months": [6, 7, 8],
        "seasonal_context": "summer"
    }}
}}
```

Please generate a high-quality enhanced rule based on the above strict constraints.
"""

        return prompt 

    def _format_pattern_features_detailed (self ,pattern_features :Dict [str ,Any ])->str :
        """Format detailed pattern feature information"""
        if not pattern_features :
            return "Current rule has no specific pattern feature configuration, which may be one reason for low health score."

        feature_parts =["Current rule's detailed feature configuration:"]

        for feature_name ,feature_config in pattern_features .items ():
            if isinstance (feature_config ,dict ):
                feature_parts .append (f"\n**{feature_name } Feature**:")
                for key ,value in feature_config .items ():
                    feature_parts .append (f"  - {key }: {value }")
            else :
                feature_parts .append (f"- {feature_name }: {feature_config }")

        return "\n".join (feature_parts )

    def _format_constraints_detailed (self ,constraints :List [str ])->str :
        """Format detailed constraint conditions"""
        if not constraints :
            return "No specific constraints, can optimize freely."

        constraint_parts =["Optimization constraints that must be followed:"]
        for i ,constraint in enumerate (constraints ,1 ):
            constraint_parts .append (f"{i }. {constraint }")

        return "\n".join (constraint_parts )

    def _format_existing_rules_detailed (self ,existing_rules :List [Dict [str ,Any ]])->str :
        """Format detailed existing healthy rules"""
        if not existing_rules :
            return "No existing healthy rule information available for reference."

        max_rules =3 # Limit display count
        rules_to_show =existing_rules [:max_rules ]

        rule_parts =["Existing healthy rules (please avoid generating functionally duplicate rules):"]

        for i ,rule in enumerate (rules_to_show ,1 ):
            rule_text =rule .get ('rule_text',f"Rule {rule .get ('rule_idx','unknown')}")
            health_score =rule .get ('health_score',0.0 )
            category =rule .get ('category','unknown')

            rule_parts .append (f"\n{i }. **{category } Rule** (Health Score: {health_score :.2f})")
            rule_parts .append (f"   Description: {rule_text }")

        if len (existing_rules )>max_rules :
            rule_parts .append (f"\n... {len (existing_rules )-max_rules } more healthy rules not shown")

        return "\n".join (rule_parts )

    def _format_session_rules_detailed (self ,session_rules :List [Dict [str ,Any ]])->str :
        """Format detailed rules generated in current session"""
        if not session_rules :
            return "No new rules generated in current session yet."

        rule_parts =["Rules generated in current optimization session (please avoid duplication):"]

        for i ,rule in enumerate (session_rules ,1 ):
            rule_text =rule .get ('description',rule .get ('rule_text',f"New rule {i }"))
            category =rule .get ('category','unknown')

            rule_parts .append (f"\n{i }. **{category } Rule**")
            rule_parts .append (f"   Description: {rule_text }")

        return "\n".join (rule_parts )

    def create_debug_prompt (self ,context :OptimizationContext )->str :
        """Create simplified prompt for debugging"""
        return f"""
Debug Mode - Rule Optimization Request

Rule Index: {context .rule_idx }
Optimization Type: {context .optimization_type }
Priority: {context .priority }
Health Score: {context .performance_metrics .get ('health_score','N/A')}
Effectiveness Score: {context .performance_metrics .get ('effectiveness_score','N/A')}

Please return a simple JSON response for testing.
"""
