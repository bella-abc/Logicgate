"""
Structured Converter Module

It's for cleaning up.APIConvert Response to Matchrule_patterns.jsonFormatted Standard Rules Configuration。
Characteristic structure according to different rule categories，Convert the corresponding format，Ensure that the configuration generated is fully compatible with the existing system。
"""

import logging 
from datetime import datetime 
from typing import Dict ,Any 

from .data_structures import StructuredRuleConfig ,ValidationError 


class StructuredConverter :
    """
    Structured Converter
    
    After cleaningAPIResponse to the rule configuration format converted to standard，
    Achieving specific features according to the category of rules，Ensuring compatibility with existing systems。
    """

    def __init__ (self ):
        self .logger =logging .getLogger (__name__ )

        # Feature Ripper Map
        self .feature_extractors ={
        'trend_driven':self ._extract_trend_features ,
        'temporal_pattern':self ._extract_temporal_features 
        }

    def convert_to_config (self ,cleaned_response :Dict [str ,Any ])->StructuredRuleConfig :
        """
        Convert to Standard Rules Configuration
        
        Args:
            cleaned_response: After clearanceAPIResponse Data
            
        Returns:
            StructuredRuleConfig: Standardized rule configuration object
            
        Raises:
            ValidationError: When data validation failed
        """
        try :
            rule_data =cleaned_response .get ('cleaned_rule',{})
            features_data =cleaned_response .get ('cleaned_features',{})
            optimization_type =cleaned_response .get ('optimization_type','UNKNOWN')

            # Validate input data
            self ._validate_input_data (rule_data ,features_data )

            # Extract features by category
            category =rule_data ['category']
            extractor =self .feature_extractors .get (category ,self ._generic_feature_extraction )
            normalized_features =extractor (features_data )

            # Build Standard Configuration
            structured_config =StructuredRuleConfig (
            description =rule_data ['description'],
            direction =rule_data ['direction'],
            direction_description =rule_data ['direction_description'],
            pattern_features =normalized_features ,
            category =rule_data ['category'],
            metadata ={
            'generated_at':datetime .now ().isoformat (),
            'optimization_type':optimization_type ,
            'analysis_summary':cleaned_response .get ('analysis',{}),
            'converter_version':'1.0.0'
            }
            )

            # Verify Configuration Structure
            self ._validate_rule_structure (structured_config )

            self .logger .info (f"成功转换规则配置，类别: {category }")
            return structured_config 

        except Exception as e :
            self .logger .error (f"转换过程出错: {e }")
            raise ValidationError (f"配置转换失败: {str (e )}")

    def _validate_input_data (self ,rule_data :Dict ,features_data :Dict ):
        """Validate input data"""
        required_fields =['description','direction','direction_description','category']

        for field in required_fields :
            if field not in rule_data :
                raise ValidationError (f"规则数据缺少必需字段: {field }")

        if not isinstance (features_data ,dict ):
            raise ValidationError ("Characteristic data must be in dictionary format")

    def _extract_trend_features (self ,features_data :Dict )->Dict :
        """Extract trend drivers - For any external variable"""
        features ={}

        # Check if there's any.trend_patternDirect definition
        if 'trend_pattern'in features_data :
            trend_data =features_data ['trend_pattern']
            features ['trend_pattern']={
            'variable':self ._safe_get (trend_data ,'variable','grid_load'),
            'trend':self ._safe_get (trend_data ,'trend','stable'),
            'adaptive_threshold':self ._safe_get (trend_data ,'adaptive_threshold',True ),
            'impact_direction':self ._safe_get (trend_data ,'impact_direction','positive')
            }
        else :
        # Through all possible external variables，Looking for trend characteristics
            for var_name in ['grid_load','wind_power','generation','demand']:
                if var_name in features_data :
                    var_data =features_data [var_name ]
                    if isinstance (var_data ,dict )and 'trend'in var_data :
                    # Infer direction of impact based on variable name
                        impact_direction ='positive'
                        if var_name in ['wind_power','generation']:
                            impact_direction ='negative'# Increased supply usually leads to lower prices

                        features ['trend_pattern']={
                        'variable':var_name ,
                        'trend':self ._safe_get (var_data ,'trend','stable'),
                        'adaptive_threshold':self ._safe_get (var_data ,'adaptive_threshold',True ),
                        'impact_direction':self ._safe_get (var_data ,'impact_direction',impact_direction )
                        }
                        break 

                        # If no trend characteristic is found，Create default feature
        if not features :
            features ={
            'trend_pattern':{
            'variable':'grid_load',
            'trend':'stable',
            'adaptive_threshold':True ,
            'impact_direction':'positive'
            }
            }

        return features 

    def _extract_temporal_features (self ,features_data :Dict )->Dict :
        """Extract Time Mode Characteristics"""
        features ={}

        # Process Time Mode
        if 'time_pattern'in features_data :
            time_data =features_data ['time_pattern']
            features ['time_pattern']={
            'type':self ._safe_get (time_data ,'type','peak_hours'),
            'hours':self ._safe_get_list (time_data ,'hours',[18 ,19 ,20 ,21 ])
            }

            # Processing seasonal features
        if 'seasonal'in features_data :
            seasonal_data =features_data ['seasonal']
            features ['seasonal']={
            'season':self ._safe_get (seasonal_data ,'season','summer'),
            'intensity':self ._safe_get_float (seasonal_data ,'intensity',0.6 )
            }

        if not features :
            features ={
            'time_pattern':{
            'type':'peak_hours',
            'hours':[18 ,19 ,20 ,21 ]
            }
            }

        return features 



    def _generic_feature_extraction (self ,features_data :Dict )->Dict :
        """Generic feature extraction"""
        if features_data :
            return features_data 
        else :
            return {
            'generic_pattern':{
            'type':'unknown',
            'description':'Generic Mode Characteristics'
            }
            }

    def _validate_rule_structure (self ,config :StructuredRuleConfig ):
        """Structure of certification rules"""
        # Validate Required Fields
        if not config .description :
            raise ValidationError ("Rule description cannot be empty")

        if not config .direction_description :
            raise ValidationError ("Direction description cannot be empty")

        if not config .pattern_features :
            raise ValidationError ("Mode characteristics cannot be empty")

        if config .category not in ['trend_driven','temporal_pattern']:
            raise ValidationError (f"无效的规则类别: {config .category }")

            # Auxiliary approach
    def _safe_get (self ,data :Dict ,key :str ,default :Any )->Any :
        """Secure access to dictionary values"""
        if isinstance (data ,dict ):
            return data .get (key ,default )
        return default 

    def _safe_get_float (self ,data :Dict ,key :str ,default :float )->float :
        """Securely capture floating point values"""
        value =self ._safe_get (data ,key ,default )
        try :
            return float (value )
        except (ValueError ,TypeError ):
            return default 

    def _safe_get_int (self ,data :Dict ,key :str ,default :int )->int :
        """Secure access to integer values"""
        value =self ._safe_get (data ,key ,default )
        try :
            return int (value )
        except (ValueError ,TypeError ):
            return default 

    def _safe_get_list (self ,data :Dict ,key :str ,default :list )->list :
        """Secure Access List Value"""
        value =self ._safe_get (data ,key ,default )
        if isinstance (value ,list ):
            return value 
        return default 
