"""
APIRespond to parser module

I'm in charge of analysis and cleaning.Qwen APIReturning Natural Language Response，Convert it to structured data。
ProcessingJSONFormat error、Fields Missing、Questions like type mismatches，Through smart resolution and automatic repair，
Maximize FromAPIDraw useful information in response。
"""

import json 
import re 
import logging 
from typing import Dict ,List ,Any ,Optional 
from dataclasses import dataclass 

from .data_structures import APIResponse 


@dataclass 
class ParseResult :
    """Parsing Result Data Class"""
    success :bool 
    data :Optional [Dict [str ,Any ]]=None 
    errors :List [str ]=None 
    warnings :List [str ]=None 

    def __post_init__ (self ):
        if self .errors is None :
            self .errors =[]
        if self .warnings is None :
            self .warnings =[]


class APIResponseParser :
    """
    APIRespond to parser
    
    Deal with uncertainties in the output of large language models，ResponseJSONFormat error、Fields Missing、Type mismatch, etc.，
    Through smart resolution and automatic repair，Maximize FromAPIDraw useful information in response。
    """

    def __init__ (self ):
        self .logger =logging .getLogger (__name__ )

        # Define Required Fields
        self .required_fields ={
        'replacement':['analysis','optimized_rule','pattern_features'],
        'enhancement':['analysis','enhanced_rule','enhanced_features']
        }

        # Define rule field requirements
        self .rule_required_fields =[
        'description','direction','direction_description',
        'category'
        ]

        # A valid category value
        self .valid_categories =[
        'trend_driven','temporal_pattern'
        ]

    def parse_and_clean (self ,api_response :APIResponse ,optimization_type :str )->ParseResult :
        """
        Parsing and cleaningAPIResponse
        
        Args:
            api_response: APIResponse Object
            optimization_type: Optimization Type ('REPLACEMENT' or 'ENHANCEMENT')
            
        Returns:
            ParseResult: Parsing Results
        """
        if not api_response .success :
            return ParseResult (
            success =False ,
            errors =[f"APIcall failed: {api_response .error }"]
            )

        try :
        # 1. ExtractJSONContents
            json_content =self ._extract_json_content (api_response .data )
            if json_content is None :
                return ParseResult (
                success =False ,
                errors =["Could not extract validity from responseJSONContents"]
                )

                # 2. Validate Required Fields
            validation_errors =self ._validate_required_fields (
            json_content ,optimization_type .lower ()
            )
            if validation_errors :
            # Try Autofix
                repaired_data =self ._attempt_auto_repair (
                json_content ,optimization_type ,validation_errors 
                )
                if repaired_data :
                    json_content =repaired_data 
                    validation_errors =[]
                else :
                    return ParseResult (
                    success =False ,
                    errors =validation_errors 
                    )

                    # 3. Clear and standardize data
            cleaned_data =self ._clean_response_data (json_content ,optimization_type )

            # 4. Final Authentication
            final_errors =self ._final_validation (cleaned_data ,optimization_type )
            if final_errors :
                return ParseResult (
                success =False ,
                errors =final_errors 
                )

            return ParseResult (
            success =True ,
            data =cleaned_data ,
            warnings =[]
            )

        except Exception as e :
            self .logger .error (f"An error occurred during parsing: {e }")
            return ParseResult (
            success =False ,
            errors =[f"parsing exception: {str (e )}"]
            )

    def _extract_json_content (self ,response_data :Any )->Optional [Dict ]:
        """Extract from ResponseJSONContents"""
        if isinstance (response_data ,dict ):
            return response_data 

        if isinstance (response_data ,str ):
        # Try Direct Parsing
            try :
                return json .loads (response_data )
            except json .JSONDecodeError :
                pass 

                # Try extracting from a code block
            json_match =re .search (r'```json\s*\n(.*?)\n```',response_data ,re .DOTALL )
            if json_match :
                try :
                    return json .loads (json_match .group (1 ))
                except json .JSONDecodeError :
                    pass 

                    # Try extracting from textJSONObject
            json_match =re .search (r'\{.*\}',response_data ,re .DOTALL )
            if json_match :
                try :
                    return json .loads (json_match .group (0 ))
                except json .JSONDecodeError :
                    pass 

        return None 

    def _validate_required_fields (self ,data :Dict ,optimization_type :str )->List [str ]:
        """Validate Required Fields"""
        errors =[]
        required =self .required_fields .get (optimization_type ,[])

        for field in required :
            if field not in data :
                errors .append (f"Missing required field: {field }")
            elif not data [field ]:
                errors .append (f"Field is empty: {field }")

        return errors 

    def _attempt_auto_repair (self ,data :Dict ,optimization_type :str ,errors :List [str ])->Optional [Dict ]:
        """Trying to automatically fix common errors"""
        repaired_data =data .copy ()

        # Fixing MissinganalysisFields
        if 'analysis'not in repaired_data :
            repaired_data ['analysis']={
            'current_rule_issues':'Automatically generated analysis',
            'optimization_rationale':'Optimization based on performance indicators'
            }

            # Fix Rules Fields
        if optimization_type =='replacement':
            if 'optimized_rule'not in repaired_data :
                return None # Could not fix core field missing
            if 'pattern_features'not in repaired_data :
                repaired_data ['pattern_features']={}
        elif optimization_type =='enhancement':
            if 'enhanced_rule'not in repaired_data :
                return None 
            if 'enhanced_features'not in repaired_data :
                repaired_data ['enhanced_features']={}

        return repaired_data 

    def _clean_response_data (self ,data :Dict ,optimization_type :str )->Dict :
        """Clear Response Data"""
        cleaned_data ={}

        # Clean-up analysis component - Processing strings or dictionaries
        analysis_data =data .get ('analysis','')
        if isinstance (analysis_data ,str ):
            cleaned_data ['analysis']={
            'current_rule_issues':analysis_data ,
            'improvement_suggestions':analysis_data 
            }
        else :
            cleaned_data ['analysis']=analysis_data 

            # Processing rule data according to optimal type
        if optimization_type .upper ()=='REPLACEMENT':
            rule_data =data .get ('optimized_rule','')
            features_data =data .get ('pattern_features',[])
        else :# ENHANCEMENT
            rule_data =data .get ('enhanced_rule','')
            features_data =data .get ('enhanced_features',[])

            # Clear Rule Data - Processing strings or dictionaries
        if isinstance (rule_data ,str ):
            cleaned_rule ={
            'description':rule_data ,
            'direction':1.0 ,# Default Positive
            'direction_description':'Price increases',
            'category':'trend_driven'# Default Category（When?rule_dataWhen a string）
            }
        else :
            cleaned_rule =self ._clean_rule_data (rule_data )

            # Clear feature data - Process list or dictionary
        if isinstance (features_data ,list ):
            cleaned_features ={
            'primary_features':features_data [:3 ]if len (features_data )>=3 else features_data ,
            'secondary_features':features_data [3 :]if len (features_data )>3 else [],
            'feature_count':len (features_data )
            }
        else :
            cleaned_features =self ._clean_features_data (features_data )

        cleaned_data ['cleaned_rule']=cleaned_rule 
        cleaned_data ['cleaned_features']=cleaned_features 
        cleaned_data ['optimization_type']=optimization_type .upper ()

        return cleaned_data 

    def _clean_rule_data (self ,rule_data :Dict )->Dict :
        """Clear Rule Data"""
        cleaned ={}

        # Clear text fields
        cleaned ['description']=self ._clean_text (
        rule_data .get ('description','')
        )
        cleaned ['direction_description']=self ._clean_text (
        rule_data .get ('direction_description','')
        )

        # Standardized Numeric Fields
        try :
            cleaned ['direction']=float (rule_data .get ('direction',0.0 ))
        except (ValueError ,TypeError ):
            cleaned ['direction']=0.0 



            # Validate and clear category fields
        category =rule_data .get ('category')
        if not category :
        # If not providedcategory，Try to extrapolate based on a description of content
            description =cleaned .get ('description','').lower ()
            inferred_category =self ._infer_category_from_description (description )
            self .logger .warning (f"LLMResponse missingcategoryField，Based on the description it is inferred that: {inferred_category }")
            cleaned ['category']=inferred_category 
        elif category not in self .valid_categories :
        # If you provide invalidcategory，Log errors and try smart extrapolations
            description =cleaned .get ('description','').lower ()
            inferred_category =self ._infer_category_from_description (description )
            self .logger .error (f"LLMThe response contains an invalidcategory: {category }, Valid values: {self .valid_categories }, Inferred as: {inferred_category }")
            cleaned ['category']=inferred_category 
        else :
            cleaned ['category']=category 

        return cleaned 

    def _infer_category_from_description (self ,description :str )->str :
        """Smart extrapolation based on rule description of contentcategoryType"""
        # Time-related keywords
        time_keywords =[
        'hour','hours','time','morning','evening','night','day',
        'peak','off-peak','weekend','weekday','daily','hourly',
        'season','seasonal','summer','winter','spring','autumn','fall',
        'month','months','january','february','march','april','may','june',
        'july','august','september','october','november','december',
        'Hours','Time','Morning.','Evening','Day','Night','Peak','Low Peak',
        'Chile','Weekend','Season','Summer','Winter','Spring','Fall','Month'
        ]

        # Trend-related keywords  
        trend_keywords =[
        'increase','increases','increasing','rise','rising','grow','growing',
        'decrease','decreases','decreasing','fall','falling','decline','declining',
        'trend','trending','load','demand','supply','generation','power',
        'wind','grid','forecast','variable','change','changing',
        'Increase','Up','Down','Trends','Load','Requirements','Supply','Power generation','Wind','Electricity grid','Change'
        ]

        # Number of occurrence of statistical keywords
        time_count =sum (1 for keyword in time_keywords if keyword in description )
        trend_count =sum (1 for keyword in trend_keywords if keyword in description )

        # Based on the number of keywords
        if time_count >trend_count :
            return 'temporal_pattern'
        elif trend_count >time_count :
            return 'trend_driven'
        else :
        # If you can't judge，Default Usetrend_driven（Because it's more common.）
            return 'trend_driven'

    def _clean_features_data (self ,features_data :Dict )->Dict :
        """Clear feature data"""
        if not isinstance (features_data ,dict ):
            return {}

        cleaned_features ={}
        for key ,value in features_data .items ():
            if isinstance (value ,dict ):
                cleaned_features [key ]=value 
            else :
            # Try converting to dictionary format
                cleaned_features [key ]={'value':value }

        return cleaned_features 

    def _clean_text (self ,text :str )->str :
        """Clear text content"""
        if not isinstance (text ,str ):
            return str (text )if text is not None else ''

            # Remove an extra blank character
        text =re .sub (r'\s+',' ',text .strip ())

        # Remove Special Characters（Keep Basic Points）
        text =re .sub (r'[^\w\s\u4e00-\u9fff.,;:!?()-]','',text )

        return text 

    def _final_validation (self ,cleaned_data :Dict ,optimization_type :str )->List [str ]:
        """Final Authentication"""
        errors =[]

        rule_data =cleaned_data .get ('cleaned_rule',{})

        # Validate Required Fields
        for field in self .rule_required_fields :
            if field not in rule_data :
                errors .append (f"Rule data is missing fields: {field }")
            elif not rule_data [field ]and field !='direction':# directionYes.0
                errors .append (f"Rule data field is empty: {field }")

                # Validation range removedconfidenceInspection

        return errors 
