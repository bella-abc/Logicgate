"""
Two-track rule optimisation system - GPT APIClient

GPT APIClient is system and externalAIServices"Bridges"，Responsible for and coordinatedGPT APIYes.
All communications，Including requests for dispatch、Response processing、Error Retry、Core functions such as timeout management。
"""

import json 
import time 
import logging 
import requests 
from typing import Dict ,Any ,Optional 
from datetime import datetime 

from .configuration_manager import ConfigurationManager 
from .data_structures import APIResponse ,create_api_response 


class GPTAPIClient :
    """
    GPT APIClient
    
    Main responsibilities：
    1. APICommunications：Processing andGPT APIYes.HTTPCommunications
    2. Retry Mechanism：Achieving smart retry policy，Deal with temporary errors
    3. Error Handling：Harmonization of treatmentAPIError
    4. Reply Parsing：Parsing and ValidatingAPIResponse Data
    5. Performance monitoring：RecordsAPICall performance indicators
    """

    def __init__ (self ,config_manager :Optional [ConfigurationManager ]=None ):
        """
        InitializeGPT APIClient
        
        Args:
            config_manager: Profile Manager Example，If forNoneOther Organiser
        """
        self .config_manager =config_manager or ConfigurationManager ()
        self .api_config =self .config_manager .get_api_config ()
        self .logger =logging .getLogger (__name__ )

        # Performance statistics
        self .stats ={
        'total_requests':0 ,
        'successful_requests':0 ,
        'failed_requests':0 ,
        'total_retry_attempts':0 ,
        'average_response_time':0.0 
        }

        # Authentication Configuration
        self ._validate_config ()

    def _validate_config (self )->None :
        """AuthenticationAPIConfigure"""
        if not self .api_config .api_key :
            raise ValueError ("APIKey not set，Check the environment variableOPENAI_API_KEY")

        if not self .api_config .base_url :
            raise ValueError ("APIBasisURLNot set")

        self .logger .info ("APIConfigure Authentication Passed")

    def generate_rule_optimization (self ,system_prompt :str ,user_prompt :str ,
    optimization_type :str )->APIResponse :
        """
        Generate rule optimization recommendations
        
        Args:
            system_prompt: System Hint
            user_prompt: User hint
            optimization_type: Optimization Type (REPLACEMENT/ENHANCEMENT)
            
        Returns:
            APIResponse Object
        """
        start_time =time .time ()
        self .stats ['total_requests']+=1 

        try :
        # Prepare to request data
            request_data =self ._prepare_request_data (system_prompt ,user_prompt )

            # Send Request（With a retest mechanism）
            response_data =self ._send_request_with_retry (request_data )

            # Parsing Response
            parsed_response =self ._parse_response (response_data ,optimization_type )

            # Update statistical information
            response_time =time .time ()-start_time 
            self ._update_stats (True ,response_time )

            self .logger .info (f"APICall successful，time consuming{response_time :.2f}Second")

            return create_api_response (
            success =True ,
            data =parsed_response ,
            metadata ={
            'optimization_type':optimization_type ,
            'response_time':response_time ,
            'timestamp':datetime .now ().isoformat ()
            }
            )

        except Exception as e :
        # Update statistical information
            response_time =time .time ()-start_time 
            self ._update_stats (False ,response_time )

            self .logger .error (f"APIcall failed: {e }")

            return create_api_response (
            success =False ,
            error =str (e ),
            metadata ={
            'optimization_type':optimization_type ,
            'response_time':response_time ,
            'timestamp':datetime .now ().isoformat ()
            }
            )

    def _prepare_request_data (self ,system_prompt :str ,user_prompt :str )->Dict [str ,Any ]:
        """Ready.APIRequest data"""
        return {
        "model":self .api_config .model_name ,
        "messages":[
        {
        "role":"system",
        "content":system_prompt 
        },
        {
        "role":"user",
        "content":user_prompt 
        }
        ],
        "max_tokens":self .api_config .max_tokens ,
        "temperature":self .api_config .temperature ,
        "top_p":self .api_config .top_p 
        }

    def _send_request_with_retry (self ,request_data :Dict [str ,Any ])->Dict [str ,Any ]:
        """Send it with a retry mechanism.APIRequest"""
        headers ={
        "Authorization":f"Bearer {self .api_config .api_key }",
        "Content-Type":"application/json"
        }

        last_exception =None 

        for attempt in range (self .api_config .retry_attempts +1 ):
            try :
                if attempt >0 :
                # Retry delay
                    delay =self .api_config .retry_delay *(2 **(attempt -1 ))# Index Refuse
                    self .logger .info (f"No.{attempt }retries，Delay{delay :.1f}Second")
                    time .sleep (delay )
                    self .stats ['total_retry_attempts']+=1 

                    # Send Request
                    # Check to disableSSLAuthentication（For debugging only）
                verify_ssl =self .config_manager .config .get ('security',{}).get ('enable_ssl_verification',True )

                response =requests .post (
                self .api_config .base_url ,
                headers =headers ,
                json =request_data ,
                timeout =self .api_config .timeout ,
                verify =verify_ssl 
                )

                # InspectionHTTPStatus Code
                if response .status_code ==200 :
                    try :
                        response_data =response .json ()
                        # Validate basic integrity of response data
                        if not response_data or 'choices'not in response_data :
                            raise ValueError ("APIIncomplete response data structure")
                        return response_data 
                    except (json .JSONDecodeError ,ValueError )as parse_error :
                    # Response format error，Process as a network error，Keep trying.
                        self .logger .warning (f"Response parsing failed (try{attempt +1 }/{self .api_config .retry_attempts +1 }): {parse_error }")
                        if attempt ==self .api_config .retry_attempts :
                            raise requests .exceptions .RequestException (f"Response format error: {parse_error }")
                        continue # Keep trying.
                elif response .status_code ==429 :
                # Rate limit，You can try again.
                    raise requests .exceptions .RequestException (f"rate limit: {response .status_code }")
                elif response .status_code >=500 :
                # Server error，You can try again.
                    raise requests .exceptions .RequestException (f"Server error: {response .status_code }")
                else :
                # Client Error，Do Not Try Again
                    response .raise_for_status ()

            except (requests .exceptions .Timeout ,
            requests .exceptions .ConnectionError ,
            requests .exceptions .RequestException )as e :
                last_exception =e 
                self .logger .warning (f"APIRequest failed (try{attempt +1 }/{self .api_config .retry_attempts +1 }): {e }")

                # If it's the last attempt,，Throw abnormal.
                if attempt ==self .api_config .retry_attempts :
                    break 

            except Exception as e :
            # Other abnormally untried
                self .logger .error (f"APIA non-retryable error occurred in the request: {e }")
                raise 

                # All retests failed.
        raise Exception (f"APIRequest failed，Retried{self .api_config .retry_attempts }Second-rate: {last_exception }")

    def _parse_response (self ,response_data :Dict [str ,Any ],
    optimization_type :str )->Dict [str ,Any ]:
        """ParsingAPIResponse"""
        try :
        # Check response structure
            if 'choices'not in response_data :
                raise ValueError ("APIResponse missingchoicesFields")

            choices =response_data ['choices']
            if not choices :
                raise ValueError ("APIResponsechoicesEmpty")

                # Fetch first selected message contents
            choice =choices [0 ]
            if 'message'not in choice :
                raise ValueError ("APIResponse missingmessageFields")

            message_content =choice ['message'].get ('content','')
            if not message_content :
                raise ValueError ("APIReply message is empty")

                # ParsingJSONContents
            try :
            # Try Direct ParsingJSON
                parsed_content =json .loads (message_content )
            except json .JSONDecodeError :
            # If Direct Parsing Failed，Try extractionJSONPart
                parsed_content =self ._extract_json_from_text (message_content )

                # Verify Parsing Results
            self ._validate_parsed_content (parsed_content ,optimization_type )

            return parsed_content 

        except Exception as e :
            self .logger .error (f"parseAPIResponse failed: {e }")
            raise ValueError (f"APIResponse parsing failed: {e }")

    def _extract_json_from_text (self ,text :str )->Dict [str ,Any ]:
        """Extract from TextJSONContents"""
        # FindJSONCode Block
        import re 

        # Try Match```json...```Format
        json_pattern =r'```json\s*(.*?)\s*```'
        match =re .search (json_pattern ,text ,re .DOTALL |re .IGNORECASE )

        if match :
            json_text =match .group (1 )
        else :
        # Try Match{}Format
            brace_pattern =r'\{.*\}'
            match =re .search (brace_pattern ,text ,re .DOTALL )
            if match :
                json_text =match .group (0 )
            else :
                raise ValueError ("Could not extract from response textJSONContents")

        try :
            return json .loads (json_text )
        except json .JSONDecodeError as e :
            raise ValueError (f"extractedJSONContent parsing failed: {e }")

    def _validate_parsed_content (self ,content :Dict [str ,Any ],
    optimization_type :str )->None :
        """Validate parsed content"""
        if not isinstance (content ,dict ):
            raise ValueError ("The resolution is not a dictionary type")

            # Check required top level fields
        if optimization_type =='REPLACEMENT':
            required_fields =['analysis','optimized_rule','pattern_features']
        elif optimization_type =='ENHANCEMENT':
            required_fields =['analysis','enhanced_rule','enhanced_features']
        else :
            raise ValueError (f"Unsupported optimization type: {optimization_type }")

        missing_fields =[field for field in required_fields if field not in content ]
        if missing_fields :
            raise ValueError (f"Response is missing a required field: {missing_fields }")

        self .logger .debug ("APIResponse content verified")

    def _update_stats (self ,success :bool ,response_time :float )->None :
        """Update statistical information"""
        if success :
            self .stats ['successful_requests']+=1 
        else :
            self .stats ['failed_requests']+=1 

            # Update average response time
        total_requests =self .stats ['successful_requests']+self .stats ['failed_requests']
        if total_requests >0 :
            current_avg =self .stats ['average_response_time']
            self .stats ['average_response_time']=(
            (current_avg *(total_requests -1 )+response_time )/total_requests 
            )

    def get_stats (self )->Dict [str ,Any ]:
        """AccessAPICall for statistical information"""
        return self .stats .copy ()

    def reset_stats (self )->None :
        """Reset Statistical Information"""
        self .stats ={
        'total_requests':0 ,
        'successful_requests':0 ,
        'failed_requests':0 ,
        'total_retry_attempts':0 ,
        'average_response_time':0.0 
        }
        self .logger .info ("APIStatistical information reset")

    def test_connection (self )->bool :
        """TestAPIConnection"""
        try :
            test_data =self ._prepare_request_data (
            "You're a test assistant.。",
            "Please respond.'Connection test successful'。"
            )

            response_data =self ._send_request_with_retry (test_data )

            # Check Response Format
            if 'choices'in response_data and response_data ['choices']:
                choice =response_data ['choices'][0 ]
                if 'message'in choice and choice ['message'].get ('content'):
                    self .logger .info ("APIConnection test successful")
                    return True 
                else :
                    self .logger .warning ("APIConnection test failed：Response format abnormal")
                    return False 
            else :
                self .logger .warning ("APIConnection test failed：Response missingchoicesFields")
                return False 

        except Exception as e :
            self .logger .error (f"APIConnection test failed: {e }")
            return False 
