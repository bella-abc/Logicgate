"""
Two-track rule optimisation system - Configure Management Module

The configuration management module is a system"Configure Center"，Management of all system configurations、Environmental variables、
APISet and run parameters。It provides a unified configuration interface，Ensure that all modules are consistent
How to access the required configuration information。
"""

import os 
import json 
import logging 
from pathlib import Path 
from typing import Dict ,Any 
from dataclasses import dataclass 
from datetime import datetime 


@dataclass 
class APIConfig :
    """APIConfigure Data Classes"""
    api_key :str 
    base_url :str 
    model_name :str 
    max_tokens :int 
    temperature :float 
    top_p :float 
    timeout :int 
    retry_attempts :int 
    retry_delay :float 


@dataclass 
class ThresholdConfig :
    """Threshold Configuration Data Category"""
    health_threshold :float 
    effectiveness_threshold :float 
    performance_degradation :float 
    critical_health_threshold :float 
    high_priority_threshold :float 


@dataclass 
class ValidationConfig :
    """Authenticate configuration data class"""
    tolerance_thresholds :Dict [str ,float ]
    monitoring_duration :int 
    validation_sample_size :int 


class ConfigurationManager :
    """
    Configure Manager
    
    Main responsibilities：
    1. Configure File Management：Load Settlementoptimization_config.jsonProfile
    2. Environmental variables processing：Read and validate environmental variables safely（LikeAPIKey）
    3. APIConfigure Provision：YesAPIClient provides complete connection configuration
    4. Threshold parameter management：Manage judgement thresholds and policy parameters
    5. Configure Authentication：Ensure that all necessary configurations are set correctly
    """

    def __init__ (self ,config_path :str ="config/optimization_config.json"):
        """
        Initialisation Configuration Manager
        
        Args:
            config_path: Profile Path
        """
        self .config_path =Path (config_path )
        self .logger =logging .getLogger (__name__ )

        # Load Configuration
        self .config =self ._load_config ()

        # Verify Environmental Variables
        self ._validate_environment ()

        # Cache Configuration Object
        self ._api_config =None 
        self ._threshold_config =None 
        self ._validation_config =None 

    def _load_config (self )->Dict [str ,Any ]:
        """Load optimized profile"""
        try :
            if not self .config_path .exists ():
                raise FileNotFoundError (f"Configuration file does not exist: {self .config_path }")

            with open (self .config_path ,'r',encoding ='utf-8')as f :
                config =json .load (f )

            self .logger .info (f"Configuration file loaded successfully: {self .config_path }")
            return config 

        except Exception as e :
            self .logger .error (f"Failed to load configuration file: {e }")
            raise 

    def _validate_environment (self )->None :
        """Verify Environmental Variables"""
        required_vars =self .config .get ('security',{}).get ('required_env_vars',['OPENAI_API_KEY'])
        missing_vars =[]

        for var in required_vars :
            if not os .getenv (var ):
                missing_vars .append (var )

        if missing_vars :
            error_msg =f"Required environment variables are missing: {missing_vars }"
            self .logger .error (error_msg )
            raise EnvironmentError (error_msg )

        self .logger .info ("Environmental variables validated")

    def get_api_config (self )->APIConfig :
        """AccessAPIConfigure"""
        if self ._api_config is None :
            api_config_dict =self .config ['api']
            api_key_env_var =self .config .get ('security',{}).get ('api_key_env_var','OPENAI_API_KEY')

            self ._api_config =APIConfig (
            api_key =os .getenv (api_key_env_var ),
            base_url =api_config_dict ['base_url'],
            model_name =api_config_dict ['model_name'],
            max_tokens =api_config_dict ['max_tokens'],
            temperature =api_config_dict ['temperature'],
            top_p =api_config_dict ['top_p'],
            timeout =api_config_dict ['timeout'],
            retry_attempts =api_config_dict ['retry_attempts'],
            retry_delay =api_config_dict ['retry_delay']
            )

        return self ._api_config 

    def get_threshold_config (self )->ThresholdConfig :
        """Get Threshold Configuration"""
        if self ._threshold_config is None :
            threshold_dict =self .config ['thresholds']

            self ._threshold_config =ThresholdConfig (
            health_threshold =threshold_dict ['health_threshold'],
            effectiveness_threshold =threshold_dict ['effectiveness_threshold'],
            performance_degradation =threshold_dict ['performance_degradation'],
            critical_health_threshold =threshold_dict ['critical_health_threshold'],
            high_priority_threshold =threshold_dict ['high_priority_threshold']
            )

        return self ._threshold_config 

    def get_validation_config (self )->ValidationConfig :
        """Get Authentication Configuration"""
        if self ._validation_config is None :
            validation_dict =self .config ['validation']

            self ._validation_config =ValidationConfig (
            tolerance_thresholds =validation_dict ['tolerance_thresholds'],
            monitoring_duration =validation_dict ['monitoring_duration'],
            validation_sample_size =validation_dict ['validation_sample_size']
            )

        return self ._validation_config 

    def get_strategy_config (self )->Dict [str ,Any ]:
        """Get Policy Configuration"""
        return self .config ['strategies']

    def get_optimization_config (self )->Dict [str ,Any ]:
        """Get Optimised Configuration"""
        return self .config ['optimization']

    def get_logging_config (self )->Dict [str ,Any ]:
        """Get Log Configuration"""
        return self .config ['logging']

    def get_security_config (self )->Dict [str ,Any ]:
        """Get Security Configuration"""
        return self .config ['security']

    def get_config_value (self ,key_path :str ,default :Any =None )->Any :
        """
        Fetch Configuration Value（Support embedded keys）
        
        Args:
            key_path: Configure Key Path，Like 'api.timeout' or 'thresholds.health_threshold'
            default: Default value
            
        Returns:
            Configure Values
        """
        keys =key_path .split ('.')
        value =self .config 

        try :
            for key in keys :
                value =value [key ]
            return value 
        except (KeyError ,TypeError ):
            return default 

    def update_config_value (self ,key_path :str ,value :Any )->None :
        """
        Update Configuration Values（Support embedded keys）
        
        Args:
            key_path: Configure Key Path
            value: New Value
        """
        keys =key_path .split ('.')
        config_ref =self .config 

        # Navigation to Parent Configuration
        for key in keys [:-1 ]:
            if key not in config_ref :
                config_ref [key ]={}
            config_ref =config_ref [key ]

            # Settings
        config_ref [keys [-1 ]]=value 

        # Clear Cache
        self ._clear_config_cache ()

        self .logger .info (f"Configuration values ​​updated: {key_path } = {value }")

    def _clear_config_cache (self )->None :
        """Clear Configuration Cache"""
        self ._api_config =None 
        self ._threshold_config =None 
        self ._validation_config =None 

    def save_config (self )->None :
        """Save Profile to File"""
        try :
        # Update metadata
            if 'metadata'in self .config :
                self .config ['metadata']['last_updated']=datetime .now ().isoformat ()

            with open (self .config_path ,'w',encoding ='utf-8')as f :
                json .dump (self .config ,f ,indent =2 ,ensure_ascii =False )

            self .logger .info (f"Configuration saved to: {self .config_path }")

        except Exception as e :
            self .logger .error (f"Failed to save configuration: {e }")
            raise 

    def validate_config (self )->bool :
        """
        Verify the integrity and validity of the configuration
        
        Returns:
            bool: The configuration is effective
        """
        try :
        # Check required top key
            required_keys =['api','thresholds','strategies','optimization','validation']
            for key in required_keys :
                if key not in self .config :
                    self .logger .error (f"Missing required configuration key: {key }")
                    return False 

                    # AuthenticationAPIConfigure
            api_config =self .get_api_config ()
            if not api_config .api_key :
                self .logger .error ("APIKey not set")
                return False 

                # Validation threshold range
            threshold_config =self .get_threshold_config ()
            if not (0.0 <=threshold_config .health_threshold <=1.0 ):
                self .logger .error ("The health threshold is beyond the range. [0.0, 1.0]")
                return False 

            if not (0.0 <=threshold_config .effectiveness_threshold <=1.0 ):
                self .logger .error ("Performance threshold beyond validity [0.0, 1.0]")
                return False 

            self .logger .info ("Configure Authentication Passed")
            return True 

        except Exception as e :
            self .logger .error (f"Configuration verification failed: {e }")
            return False 

    def get_config_summary (self )->Dict [str ,Any ]:
        """Get Profile Summary"""
        return {
        'config_path':str (self .config_path ),
        'api_model':self .config ['api']['model_name'],
        'health_threshold':self .config ['thresholds']['health_threshold'],
        'effectiveness_threshold':self .config ['thresholds']['effectiveness_threshold'],
        'automatic_optimization':self .config ['optimization']['enable_automatic_optimization'],
        'analysis_interval':self .config ['optimization']['analysis_interval_epochs'],
        'version':self .config .get ('metadata',{}).get ('version','unknown'),
        'last_updated':self .config .get ('metadata',{}).get ('last_updated','unknown')
        }
