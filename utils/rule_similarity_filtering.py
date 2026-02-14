import numpy as np 
import torch 
from typing import Dict ,List ,Tuple ,Optional ,Any 
import warnings 
import hashlib 
import pickle 
import json 
import os 
from functools import lru_cache 

class FuzzyRuleSimilarityFilter :
    """
    Based on Fuzzy Logic+A rule similarity filter for dynamic thresholds
    
    AchievedREADME_Rule_Similarity_Filtering.mdcomplete function as described：
    1. Dynamic threshold calculation
    2. Fuzzy attribute function
    3. Data Mode Characteristic Extract
    4. Rule Similarity Calculation
    5. Smart Rules Filter
    """

    def __init__ (self ,
    percentile :int =75 ,
    window_size :int =24 ,
    membership_type :str ='adaptive',
    debug :bool =False ,
    rule_config_path :str ='config/rule_patterns.json'):
        """
        Initialise Filter

        Args:
            percentile: Dynamic threshold percentage (Recommendations75)
            window_size: Slide Window Size (Recommendations24)
            membership_type: Type of reporting function ('adaptive')
            debug: Whether to output debug information
            rule_config_path: Rule ConfigurationJSONFile Path
        """
        self .percentile =percentile 
        self .window_size =window_size 
        self .membership_type =membership_type 
        self .debug =debug 
        self .rule_config_path =rule_config_path 

        # FromJSONFile Load Rule Mode Features
        self .rule_patterns =self ._load_rule_config ()

        # External variable name map（Based on actual data set characteristics）
        self .exog_names =['grid_load','wind_power']

        # Performance optimization：Cache mechanism
        self .feature_cache ={}
        self .cache_enabled =True 
        self .max_cache_size =1000 

    def _load_rule_config (self )->Dict :
        """
        FromJSONFile Loading Rules Configuration

        Returns:
            rule_patterns: Rule mode characteristics
        """
        try :
        # Try loadingJSONDocumentation
            if os .path .exists (self .rule_config_path ):
                with open (self .rule_config_path ,'r',encoding ='utf-8')as f :
                    config =json .load (f )

                rule_patterns ={}

                # Parsing Rule Configuration
                for rule_id_str ,rule_config in config ['rule_patterns'].items ():
                    rule_id =int (rule_id_str )

                    # Extract Mode Features
                    rule_patterns [rule_id ]=rule_config ['pattern_features']

                if self .debug :
                    print (f"✅ successfully from {self .rule_config_path } load {len (rule_patterns )} rule configuration")

                return rule_patterns 

            else :
                if self .debug :
                    print (f"⚠️  Configuration file {self .rule_config_path } does not exist，Use default configuration")

        except Exception as e :
            if self .debug :
                print (f"❌ Failed to load configuration file: {e }，Use default configuration")

                # If loading failed，Use default configuration
        return self ._get_default_config ()

    def _get_default_config (self )->Dict :
        """Get Default Rule Configuration"""
        rule_patterns ={
        0 :{'trend_pattern':{'variable':'grid_load','trend':'increasing','adaptive_threshold':True ,'impact_direction':'positive'}},
        1 :{'trend_pattern':{'variable':'wind_power','trend':'increasing','adaptive_threshold':True ,'impact_direction':'negative'}},
        2 :{'time_pattern':{'type':'peak_hours','hours':[8 ,9 ,18 ,19 ,20 ]}},
        3 :{'time_pattern':{'type':'weekend','days':[5 ,6 ]}},
        4 :{'time_pattern':{'type':'seasonal','months':[6 ,7 ,8 ]}},
        5 :{'time_pattern':{'type':'seasonal','months':[12 ,1 ,2 ]}},
        6 :{
        'trend_patterns':[
        {'variable':'grid_load','trend':'decreasing','adaptive_threshold':True ,'impact_direction':'negative'},
        {'variable':'wind_power','trend':'increasing','adaptive_threshold':True ,'impact_direction':'negative'}
        ],
        'combination':True 
        }
        }

        return rule_patterns 

    def calculate_dynamic_threshold (self ,time_series_data :np .ndarray )->Tuple [float ,Dict ]:
        """
        Calculate dynamic thresholds
        
        Args:
            time_series_data: Time series data [seq_len]
            
        Returns:
            (dynamic_threshold, trend_distribution)
        """
        if len (time_series_data )<self .window_size :
        # If the data is insufficient，Use simple standard deviation as a threshold
            return np .std (time_series_data )*0.5 ,{'insufficient_data':True }

            # Slide Window Analysis
        sliding_windows =[]
        window_trends =[]

        for i in range (len (time_series_data )-self .window_size +1 ):
            window =time_series_data [i :i +self .window_size ]
            sliding_windows .append (window )

            # Calculating trend intensity in windows
            early_mean =window [:6 ].mean ()
            late_mean =window [-6 :].mean ()
            trend =(late_mean -early_mean )/(abs (early_mean )+1e-6 )
            window_trends .append (trend )

            # Calculate dynamic thresholds
        trend_strengths =np .abs (window_trends )
        dynamic_threshold =np .percentile (trend_strengths ,self .percentile )

        # Trends in distribution statistics
        trend_distribution ={
        'mean_trend':np .mean (window_trends ),
        'std_trend':np .std (window_trends ),
        'trend_range':(np .min (window_trends ),np .max (window_trends )),
        'dynamic_threshold':dynamic_threshold ,
        'num_windows':len (window_trends )
        }

        # Comment Debug Output，Avoid too much console information
        # if self.debug:
        #     print(f"Dynamic threshold calculation: {dynamic_threshold:.6f}")
        #     print(f"Trend distribution: {trend_distribution}")

        return dynamic_threshold ,trend_distribution 

    def adaptive_increasing_membership (self ,x :float ,dynamic_threshold :float )->float :
        """Self-adapted Incremental Dependencies"""
        lower_bound =-dynamic_threshold 
        upper_bound =dynamic_threshold 

        if x <=lower_bound :
            return 0.0 
        elif x >=upper_bound :
            return 1.0 
        else :
            return (x -lower_bound )/(upper_bound -lower_bound )

    def adaptive_decreasing_membership (self ,x :float ,dynamic_threshold :float )->float :
        """Self-adapted Declination Reporting Function"""
        return self .adaptive_increasing_membership (-x ,dynamic_threshold )

    def adaptive_stable_membership (self ,x :float ,dynamic_threshold :float )->float :
        """Self-According to Stable Dependencies"""
        inc_score =self .adaptive_increasing_membership (x ,dynamic_threshold )
        dec_score =self .adaptive_decreasing_membership (x ,dynamic_threshold )
        return 1.0 -max (inc_score ,dec_score )

    def calculate_trend_strength (self ,time_series_data :np .ndarray )->float :
        """
        Calculate trend intensity
        
        Args:
            time_series_data: Time series data [seq_len]
            
        Returns:
            Trends intensity values
        """
        if len (time_series_data )<72 :
        # Use simple methods when data is insufficient
            if len (time_series_data )<12 :
                return 0.0 
            recent_mean =time_series_data [-6 :].mean ()
            early_mean =time_series_data [:6 ].mean ()
            return (recent_mean -early_mean )/(abs (early_mean )+1e-6 )

            # Calculate the average short- and long-term movement
        short_ma =np .convolve (time_series_data ,np .ones (24 )/24 ,mode ='valid')
        long_ma =np .convolve (time_series_data ,np .ones (72 )/72 ,mode ='valid')

        # Get Recent12Average of points
        if len (short_ma )>=12 and len (long_ma )>=12 :
            recent_short =short_ma [-12 :].mean ()
            recent_long =long_ma [-12 :].mean ()
            trend_strength =(recent_short -recent_long )/(abs (recent_long )+1e-6 )
        else :
        # Back to a simple way.
            recent_mean =time_series_data [-12 :].mean ()
            early_mean =time_series_data [:12 ].mean ()
            trend_strength =(recent_mean -early_mean )/(abs (early_mean )+1e-6 )

        return trend_strength 

    def _get_data_hash (self ,data :np .ndarray )->str :
        """Hash to generate data for cache"""
        return hashlib .md5 (data .tobytes ()).hexdigest ()

    def _cache_key (self ,data_hash :str ,feature_type :str )->str :
        """Generate Cache Keys"""
        return f"{feature_type }_{data_hash }_{self .percentile }_{self .window_size }"

    def extract_exog_features (self ,x_exog :torch .Tensor )->List [Dict [str ,Dict ]]:
        """
        Extract external variable characteristics - Batch Support

        Args:
            x_exog: External variable data [batch_size, seq_len, num_exog]

        Returns:
            Identity dictionary list，EachbatchA dictionary
        """
        batch_features =[]

        # Convert tonumpy
        if isinstance (x_exog ,torch .Tensor ):
            x_exog_np =x_exog .detach ().cpu ().numpy ()# [batch_size, seq_len, num_exog]
        else :
            x_exog_np =x_exog 

        batch_size =x_exog_np .shape [0 ]

        # Handle Everybatch
        for batch_idx in range (batch_size ):
            features ={}
            batch_data =x_exog_np [batch_idx ]# [seq_len, num_exog]

            for i ,var_name in enumerate (self .exog_names ):
                if i >=batch_data .shape [1 ]:
                    break 

                var_data =batch_data [:,i ]# [seq_len]

                # Calculate dynamic thresholds
                dynamic_threshold ,trend_dist =self .calculate_dynamic_threshold (var_data )

                # Calculate trend intensity
                trend_strength =self .calculate_trend_strength (var_data )

                # Calculating Fuzzy Substance Scores
                membership_scores ={
                'increasing':self .adaptive_increasing_membership (trend_strength ,dynamic_threshold ),
                'decreasing':self .adaptive_decreasing_membership (trend_strength ,dynamic_threshold ),
                'stable':self .adaptive_stable_membership (trend_strength ,dynamic_threshold )
                }

                # Calculate statistical characteristics
                mean_val =float (np .mean (var_data ))
                std_val =float (np .std (var_data ))
                recent_mean =float (np .mean (var_data [-int (len (var_data )*0.25 ):]))
                volatility =std_val /(abs (mean_val )+1e-6 )

                # Level of judgement
                if recent_mean >mean_val +0.5 *std_val :
                    level ='high'
                elif recent_mean <mean_val -0.5 *std_val :
                    level ='low'
                else :
                    level ='normal'

                features [var_name ]={
                'trend_strength':trend_strength ,
                'raw_data':var_data ,
                'dynamic_threshold':dynamic_threshold ,
                'membership_scores':membership_scores ,
                'mean_val':mean_val ,
                'std_val':std_val ,
                'recent_mean':recent_mean ,
                'volatility':volatility ,
                'level':level ,
                'trend_distribution':trend_dist 
                }

            batch_features .append (features )

        return batch_features 

    def extract_exog_features_fast (self ,x_exog :torch .Tensor ,use_cache :bool =True )->List [Dict [str ,Dict ]]:
        """
        Efficient version of external variable characterization - Supporting cache and quantification

        Args:
            x_exog: External variable data [batch_size, seq_len, num_exog]
            use_cache: Whether to use the cache

        Returns:
            Identity dictionary list，EachbatchA dictionary
        """
        # Convert tonumpy
        if isinstance (x_exog ,torch .Tensor ):
            x_exog_np =x_exog .detach ().cpu ().numpy ()
        else :
            x_exog_np =x_exog 

        batch_size ,seq_len ,num_exog =x_exog_np .shape 
        batch_features =[]

        # Quantified AllbatchStatistical characteristics
        means =np .mean (x_exog_np ,axis =1 )# [batch_size, num_exog]
        stds =np .std (x_exog_np ,axis =1 )# [batch_size, num_exog]
        recent_len =max (1 ,int (seq_len *0.25 ))
        recent_means =np .mean (x_exog_np [:,-recent_len :,:],axis =1 )# [batch_size, num_exog]

        # Quantified trend intensity
        if seq_len >=12 :
            early_means =np .mean (x_exog_np [:,:6 ,:],axis =1 )# [batch_size, num_exog]
            late_means =np .mean (x_exog_np [:,-6 :,:],axis =1 )# [batch_size, num_exog]
            trend_strengths =(late_means -early_means )/(np .abs (early_means )+1e-6 )
        else :
            trend_strengths =np .zeros ((batch_size ,num_exog ))

            # Handle Everybatch
        for batch_idx in range (batch_size ):
            features ={}

            for i ,var_name in enumerate (self .exog_names ):
                if i >=num_exog :
                    break 

                var_data =x_exog_np [batch_idx ,:,i ]

                # Check Cache
                if use_cache and self .cache_enabled :
                    data_hash =self ._get_data_hash (var_data )
                    cache_key =self ._cache_key (data_hash ,f"exog_{var_name }")

                    if cache_key in self .feature_cache :
                        features [var_name ]=self .feature_cache [cache_key ]
                        continue 

                        # Calculate dynamic thresholds（This is the most time-consuming part.）
                dynamic_threshold ,trend_dist =self .calculate_dynamic_threshold (var_data )

                # Use precalculated values
                trend_strength =trend_strengths [batch_idx ,i ]
                mean_val =float (means [batch_idx ,i ])
                std_val =float (stds [batch_idx ,i ])
                recent_mean =float (recent_means [batch_idx ,i ])
                volatility =std_val /(abs (mean_val )+1e-6 )

                # Calculating Fuzzy Substance Scores
                membership_scores ={
                'increasing':self .adaptive_increasing_membership (trend_strength ,dynamic_threshold ),
                'decreasing':self .adaptive_decreasing_membership (trend_strength ,dynamic_threshold ),
                'stable':self .adaptive_stable_membership (trend_strength ,dynamic_threshold )
                }

                # Level of judgement
                if recent_mean >mean_val +0.5 *std_val :
                    level ='high'
                elif recent_mean <mean_val -0.5 *std_val :
                    level ='low'
                else :
                    level ='normal'

                feature_dict ={
                'trend_strength':trend_strength ,
                'raw_data':var_data ,
                'dynamic_threshold':dynamic_threshold ,
                'membership_scores':membership_scores ,
                'mean_val':mean_val ,
                'std_val':std_val ,
                'recent_mean':recent_mean ,
                'volatility':volatility ,
                'level':level ,
                'trend_distribution':trend_dist 
                }

                features [var_name ]=feature_dict 

                # Cache Result
                if use_cache and self .cache_enabled :
                    if len (self .feature_cache )<self .max_cache_size :
                        self .feature_cache [cache_key ]=feature_dict 

            batch_features .append (features )

        return batch_features 

    def extract_time_features (self ,x_mark_enc :torch .Tensor ,pred_len :Optional [int ]=None )->List [Dict [str ,Any ]]:
        """
        Extract Time Mode Characteristics - Batch Support

        Args:
            x_mark_enc: Timemark [batch_size, seq_len, 4] - [month, day, weekday, hour]
            pred_len: Projected length，If provided, use only the time characteristic of the pure projection period

        Returns:
            Time Character Dictionary List，EachbatchA dictionary
        """
        batch_time_features =[]

        # Convert tonumpy
        if isinstance (x_mark_enc ,torch .Tensor ):
            time_data_all =x_mark_enc .detach ().cpu ().numpy ()# [batch_size, seq_len, 4]
        else :
            time_data_all =x_mark_enc 

        batch_size =time_data_all .shape [0 ]

        # If providedpred_len，Use only the time characteristic of the pure projection period（Backpred_lenA moment.）
        if pred_len is not None and pred_len >0 :
            if time_data_all .shape [1 ]>=pred_len :
                time_data_all =time_data_all [:,-pred_len :,:]# Only afterpred_lenA moment.
                if not hasattr (self ,'_pred_len_debug_printed'):
                    print (f"[DEBUG] Use pure forecast period time features: from{time_data_all .shape [1 ]+pred_len }After extraction in hours{pred_len }Hour")
                    self ._pred_len_debug_printed =True 
            else :
                print (f"[WARNING] Time data length({time_data_all .shape [1 ]})less than predicted length({pred_len })，Use all time data")
        else :
            if not hasattr (self ,'_full_time_debug_printed'):
                print (f"[DEBUG] Use all time features: {time_data_all .shape [1 ]}Hour")
                self ._full_time_debug_printed =True 

                # Debug Output：Check the type of time data（Output only on first call）
        if not hasattr (self ,'_debug_printed'):
            print (f"[DEBUG] extract_time_features: Time data shape = {time_data_all .shape }")
            print (f"[DEBUG] extract_time_features: Time data sample = {time_data_all [0 ,:3 ]}")

            # Disconnection or continuous time characteristics.
            sample_values =time_data_all [0 ,0 ]# FirstbatchFirst point of time.
            is_discrete =all (abs (val -round (val ))<1e-6 for val in sample_values )
            print (f"[DEBUG] extract_time_features: Is it a discrete time characteristic? = {is_discrete }")
            self ._debug_printed =True 

            # Handle Everybatch
        for batch_idx in range (batch_size ):
            time_data =time_data_all [batch_idx ]# [seq_len, 4]

            # Attention.：recent_time Fields removed，Because it's not used in follow-up.
            # All time characteristics are based on statistical analysis.（mode/average）Not a single point.

            # Amendments：Using the dominant mode method instead of the average method（Time characteristics for the projection period）
            from scipy import stats 

            # Lead mode judgement
            if time_data .shape [0 ]>1 :
            # Use statistics to find the most frequent values
                mode_month =float (stats .mode (time_data [:,0 ],keepdims =True )[0 ][0 ])
                mode_weekday =float (stats .mode (time_data [:,2 ],keepdims =True )[0 ][0 ])
                mode_hour =float (stats .mode (time_data [:,3 ],keepdims =True )[0 ][0 ])
            else :
            # Status of individual points of time
                mode_month =float (time_data [0 ,0 ])
                mode_weekday =float (time_data [0 ,2 ])
                mode_hour =float (time_data [0 ,3 ])

                # Based on the dominant model
            is_weekend =mode_weekday >=5 # LeadweekdayIs it a weekend?
            # Attention.：is_peak_hourIt should be judged by the configuration dynamics of specific rules，Not hard code.
            # Keep fields here for backward compatibility，But...calculate_rule_similarityto use a rule-specific peak period
            is_peak_hour =False # Default value，The actual judgment is...calculate_rule_similarityIn progress

            # Calculate average（Maintain backward compatibility）
            avg_month =float (np .mean (time_data [:,0 ]))
            avg_day =float (np .mean (time_data [:,1 ]))
            avg_weekday =float (np .mean (time_data [:,2 ]))
            avg_hour =float (np .mean (time_data [:,3 ]))

            # Attention.：season Fields removed，Because... calculate_rule_similarity Unused
            # Seasonal judgment passes directly. mode_month and avg_month Deal with in similarity calculations

            batch_time_features .append ({
            # Attention.：recent_time Fields removed，Not here. calculate_rule_similarity Use
            # Add：Main Mode Features
            'mode_month':mode_month ,
            'mode_weekday':mode_weekday ,
            'mode_hour':mode_hour ,
            # Keep it compatible.：mean characteristics
            'avg_month':avg_month ,
            'avg_day':avg_day ,
            'avg_weekday':avg_weekday ,
            'avg_hour':avg_hour ,
            # Revised findings（Based on the dominant model）
            'is_weekend':is_weekend ,
            'is_peak_hour':is_peak_hour 
            # Attention.：season Fields removed，Seasonal pass mode_month/avg_month Processing
            })

        return batch_time_features 


    def calculate_rule_similarity (self ,
    exog_features :Dict [str ,Dict ],
    time_features :Dict [str ,Any ],
    rule_idx :int )->float :
        """
        Calculating Fuzzy Similarities of Individual Rules

        Args:
            exog_features: Characteristics of external variables
            time_features: Time Features
            rule_idx: Rule Index

        Returns:
            Similarity Scores [0, 1]
        """
        if rule_idx not in self .rule_patterns :
            return 0.0 

        rule_pattern =self .rule_patterns [rule_idx ]
        total_score =0.0 
        feature_count =0 

        # Process external variable signature matching
        # for var_name, requirements in rule_pattern.items():
        #     if var_name in exog_features and isinstance(requirements, dict) and 'trend' in requirements:
        #         var_feature = exog_features[var_name]
        #         trend_requirement = requirements['trend']

        #         if trend_requirement in var_feature['membership_scores']:
        #             score = var_feature['membership_scores'][trend_requirement]
        #             total_score += score
        #             feature_count += 1

        # Comment Debug Output
        # if self.debug:
        #     print(f"Rule{rule_idx} - {var_name}({trend_requirement}): {score:.3f}")

        # Process trend pattern matching
        if 'trend_pattern'in rule_pattern :
            trend_pattern =rule_pattern ['trend_pattern']
            variable =trend_pattern ['variable']
            expected_trend =trend_pattern ['trend']

            if variable in exog_features :
                var_feature =exog_features [variable ]
                if expected_trend in var_feature ['membership_scores']:
                    score =var_feature ['membership_scores'][expected_trend ]
                    total_score +=score 
                    feature_count +=1 

                    # Comment Debug Output
                    # if self.debug:
                    #     print(f"Rule{rule_idx} - trend_pattern({variable}, {expected_trend}): {score:.3f}")

                    # Process Time Mode Match
        if 'time_pattern'in rule_pattern :
            time_pattern =rule_pattern ['time_pattern']

            if time_pattern ['type']=='peak_hours':
            # Peak Time Match - Based on the specific time frame of the rule configuration
            # Usemode_hour（Lead hours）Not...avg_hour（Average hours）To judge.
            # mode_hourThe most frequent hour of presence.，Usually the integer.，Fits better for discrete peak periods
                current_hour =time_features .get ('mode_hour',time_features .get ('avg_hour',0 ))
                peak_hours =time_pattern ['hours']

                # Dynamically judge whether the current time period is the peak period defined by the rule
                is_current_peak =int (current_hour )in peak_hours 

                if is_current_peak :
                    score =1.0 
                else :
                # Calculate distance from peak time（Smooth calculation using average hours）
                    avg_hour =time_features .get ('avg_hour',current_hour )
                    min_distance =min (abs (avg_hour -h )for h in peak_hours )
                    score =max (0.0 ,1.0 -min_distance /12.0 )# 12Maximum distance for hour

                total_score +=score 
                feature_count +=1 

                # Comment Debug Output
                # if self.debug:
                #     print(f"Rule{rule_idx} - peak_hours: {score:.3f}")

            elif time_pattern ['type']=='weekend':
            # Weekend Match - Support new weekdays Field Configuration
                current_weekday =time_features .get ('mode_weekday',time_features .get ('avg_weekday',0 ))

                # Get the target working day（New Format vs Backward compatibility）
                if 'weekdays'in time_pattern :
                    target_weekdays =time_pattern ['weekdays']
                else :
                # Backward compatibility：Default weekend is Saturday
                    target_weekdays =[5 ,6 ]

                    # Exact Match Check
                is_target_weekday =int (current_weekday )in target_weekdays 

                if is_target_weekday :
                    score =1.0 
                else :
                # Calculate the distance to the target working day
                    avg_weekday =time_features .get ('avg_weekday',current_weekday )
                    min_distance =float ('inf')

                    for target_weekday in target_weekdays :
                    # Calculating Cycle Distance（A week.7days）
                        distance =min (
                        abs (avg_weekday -target_weekday ),
                        7 -abs (avg_weekday -target_weekday )
                        )
                        min_distance =min (min_distance ,distance )

                        # Maximum distance is3.5days (Half a week.)
                    score =max (0.0 ,1.0 -min_distance /3.5 )

                total_score +=score 
                feature_count +=1 

                # Comment Debug Output
                # if self.debug:
                #     print(f"Rule{rule_idx} - weekend: {score:.3f}")

            elif time_pattern ['type']=='seasonal':
            # Seasonal Match
            # Usemode_month（Main month）Not...avg_month（Average months）To judge.
            # mode_monthIt represents the most frequent month.，Usually the integer.，More suitable to match the separated month list
                current_month =time_features .get ('mode_month',time_features .get ('avg_month',1 ))
                target_months =time_pattern ['months']

                if int (current_month )in target_months :
                    score =1.0 
                else :
                # Calculate minimum distance from target month（Consider annual cycles）
                # Use average month for smooth distance calculations
                    avg_month =time_features .get ('avg_month',current_month )
                    min_distance =float ('inf')
                    for target_month in target_months :
                    # Calculating Cycle Distance
                        distance =min (
                        abs (avg_month -target_month ),
                        12 -abs (avg_month -target_month )
                        )
                        min_distance =min (min_distance ,distance )

                        # Closer.，The higher the score（Maximum distance is6Month）
                    score =max (0.0 ,1.0 -min_distance /6.0 )

                total_score +=score 
                feature_count +=1 



                # Calculate average similarity
        if feature_count >0 :
            similarity =total_score /feature_count 
        else :
            similarity =0.0 

            # Comment Debug Output
            # if self.debug:
            #     print(f"Rule{rule_idx} Total Similarity: {similarity:.3f} (Number of features: {feature_count})")

        return similarity 

    def select_relevant_rules (self ,
    x_exog :Optional [torch .Tensor ]=None ,
    x_mark_enc :Optional [torch .Tensor ]=None ,
    rules_list :Optional [List [str ]]=None ,
    fuzzy_config :Optional [Dict ]=None ,
    pred_len :Optional [int ]=None )->Tuple [List [str ],List [int ],Dict ]:
        """
        Based on Fuzzy Logic+Rules for dynamic threshold filtering

        Args:
            x_exog: External variable data [batch_size, seq_len, num_exog]
            x_mark_enc: Timemark [batch_size, seq_len, 4]
            rules_list: Rule List
            fuzzy_config: Fuzzy Logic Configuration
            pred_len: Projected length，Time characteristics for extracting pure projection periods

        Returns:
            (selected_rules, selected_indices, selection_info)
        """
        if fuzzy_config :
            self .percentile =fuzzy_config .get ('percentile',self .percentile )
            self .window_size =fuzzy_config .get ('window_size',self .window_size )

            # Default Rule List
        if rules_list is None :
            rules_list =[
            "When grid_load increases, electricity prices tend to increase due to higher demand pressure on the electricity grid",
            "When wind_power increases, electricity prices tend to decrease as more renewable energy enters the grid at lower marginal costs",
            "During peak demand hours (typically morning and evening), electricity prices are generally higher due to increased consumption",
            "On weekends, electricity prices tend to be lower due to reduced industrial and commercial activity",
            "During summer months, electricity prices tend to increase due to air conditioning demand",
            "During winter months, electricity prices tend to increase due to heating demand",
            "When both grid_load decreases and wind_power increases simultaneously, electricity prices tend to significantly decrease"
            ]

            # If no data is entered，Return all rules
        if x_exog is None and x_mark_enc is None :
            return rules_list ,list (range (len (rules_list ))),{
            'fuzzy_similarities':[1.0 ]*len (rules_list ),
            'dynamic_thresholds':{},
            'threshold_used':0.0 ,
            'total_rules_considered':len (rules_list ),
            'rules_selected':len (rules_list ),
            'selection_reason':'No input data provided'
            }

            # Extract Character - Batch Support
        batch_exog_features =[]
        batch_time_features =[]

        if x_exog is not None :
            batch_exog_features =self .extract_exog_features (x_exog )

        if x_mark_enc is not None :
            batch_time_features =self .extract_time_features (x_mark_enc ,pred_len =pred_len )

            # Calculating Similarity of All Rules - For each one.batchCalculated separately，Then take the average.
        batch_size =len (batch_exog_features )if batch_exog_features else len (batch_time_features )if batch_time_features else 1 

        # For eachbatchCalculate rule similarity
        all_batch_similarities =[]
        for batch_idx in range (batch_size ):
            exog_features =batch_exog_features [batch_idx ]if batch_exog_features else {}
            time_features =batch_time_features [batch_idx ]if batch_time_features else {}

            batch_rule_similarities =[]
            for rule_idx in range (len (rules_list )):
                similarity =self .calculate_rule_similarity (
                exog_features ,time_features ,rule_idx 
                )
                batch_rule_similarities .append (similarity )
            all_batch_similarities .append (batch_rule_similarities )

            # Calculate average similarity
        if all_batch_similarities :
            avg_similarities =np .mean (all_batch_similarities ,axis =0 )
            rule_similarities =[(rule_idx ,avg_similarities [rule_idx ])for rule_idx in range (len (rules_list ))]
        else :
            rule_similarities =[(rule_idx ,0.0 )for rule_idx in range (len (rules_list ))]

            # Sort by Similarity
        rule_similarities .sort (key =lambda x :x [1 ],reverse =True )

        # Self-adaptation threshold filter
        similarity_scores =[sim for _ ,sim in rule_similarities ]
        if len (similarity_scores )>0 :
            adaptive_threshold =np .percentile (similarity_scores ,60 )# Use60%Bits
        else :
            adaptive_threshold =0.3 

            # Filter rules
        relevant_rules =[]
        relevant_indices =[]
        fuzzy_similarities =[]

        for rule_idx ,similarity in rule_similarities :
            if similarity >=adaptive_threshold :
                relevant_rules .append (rules_list [rule_idx ])
                relevant_indices .append (rule_idx )
                fuzzy_similarities .append (similarity )

                # Minimum guaranteed amount
        if len (relevant_rules )<3 :
        # Add the highest degree of similarity before3Rule
            for rule_idx ,similarity in rule_similarities [:3 ]:
                if rule_idx not in relevant_indices :
                    relevant_rules .append (rules_list [rule_idx ])
                    relevant_indices .append (rule_idx )
                    fuzzy_similarities .append (similarity )

                    # Collect dynamic threshold information - Use firstbatchas representative of
        dynamic_thresholds ={}
        if batch_exog_features :
            for var_name ,features in batch_exog_features [0 ].items ():
                dynamic_thresholds [var_name ]=features ['dynamic_threshold']

                # Build a list of similarities for all rules（For quality monitoring）
        all_rule_similarities =[0.0 ]*len (rules_list )
        for rule_idx ,similarity in rule_similarities :
            all_rule_similarities [rule_idx ]=similarity 

            # Build Selection Information
        selection_info ={
        'fuzzy_similarities':all_rule_similarities ,# Similarity of all rules
        'selected_similarities':fuzzy_similarities ,# Similarity of selected rules only
        'dynamic_thresholds':dynamic_thresholds ,
        'threshold_used':adaptive_threshold ,
        'total_rules_considered':len (rules_list ),
        'rules_selected':len (relevant_rules ),
        'selection_reason':f'Fuzzy logic filtering with adaptive threshold {adaptive_threshold :.3f}'
        }

        # Comment Debug Output，Avoid too much console information
        # if self.debug:
        #     print(f"\n=== Rule Filter Results ===")
        #     print(f"Filter Out {len(relevant_rules)} Relevant rules: {relevant_indices}")
        #     print(f"Filter threshold: {adaptive_threshold:.3f}")
        #     print(f"Similarity Scores: {[f'{s:.3f}' for s in fuzzy_similarities]}")

        return relevant_rules ,relevant_indices ,selection_info 

    def select_relevant_rules_fast (self ,
    x_exog :Optional [torch .Tensor ]=None ,
    x_mark_enc :Optional [torch .Tensor ]=None ,
    rules_list :Optional [List [str ]]=None ,
    fuzzy_config :Optional [Dict ]=None ,
    use_cache :bool =True ,
    simplified :bool =False ,
    pred_len :Optional [int ]=None )->Tuple [List [str ],List [int ],Dict ]:
        """
        Efficient version of the rules filter - Supporting caches and simplified models

        Args:
            x_exog: External variable data [batch_size, seq_len, num_exog]
            x_mark_enc: Timemark [batch_size, seq_len, 4]
            rules_list: Rule List
            fuzzy_config: Fuzzy Logic Configuration
            use_cache: Whether to use the cache
            simplified: Whether to use a simplified model（Only the first one.batch，Raise Speed）
            pred_len: Projected length，Time characteristics for extracting pure projection periods

        Returns:
            (selected_rules, selected_indices, selection_info)
        """
        if fuzzy_config :
            self .percentile =fuzzy_config .get ('percentile',self .percentile )
            self .window_size =fuzzy_config .get ('window_size',self .window_size )

            # Default Rule List
        if rules_list is None :
            rules_list =[
            "When grid_load increases, electricity prices tend to increase due to higher demand pressure on the electricity grid",
            "When wind_power increases, electricity prices tend to decrease as more renewable energy enters the grid at lower marginal costs",
            "During peak demand hours (typically morning and evening), electricity prices are generally higher due to increased consumption",
            "On weekends, electricity prices tend to be lower due to reduced industrial and commercial activity",
            "During summer months, electricity prices tend to increase due to air conditioning demand",
            "During winter months, electricity prices tend to increase due to heating demand",
            "When both grid_load decreases and wind_power increases simultaneously, electricity prices tend to significantly decrease"
            ]

            # If no data is entered，Return all rules
        if x_exog is None and x_mark_enc is None :
            return rules_list ,list (range (len (rules_list ))),{
            'fuzzy_similarities':[1.0 ]*len (rules_list ),
            'dynamic_thresholds':{},
            'threshold_used':0.0 ,
            'total_rules_considered':len (rules_list ),
            'rules_selected':len (rules_list ),
            'selection_reason':'No input data provided'
            }

            # Simplified Mode：Only the first one.batch
        if simplified :
            if x_exog is not None :
                x_exog =x_exog [:1 ]# Only the first one.batch
            if x_mark_enc is not None :
                x_mark_enc =x_mark_enc [:1 ]# Only the first one.batch

                # Extract Character - Use efficient version
        batch_exog_features =[]
        batch_time_features =[]

        if x_exog is not None :
            batch_exog_features =self .extract_exog_features_fast (x_exog ,use_cache =use_cache )

        if x_mark_enc is not None :
            batch_time_features =self .extract_time_features (x_mark_enc ,pred_len =pred_len )

            # Calculate rule similarity
        batch_size =len (batch_exog_features )if batch_exog_features else len (batch_time_features )if batch_time_features else 1 

        # Simplified Mode or SinglebatchProcessing
        if simplified or batch_size ==1 :
            exog_features =batch_exog_features [0 ]if batch_exog_features else {}
            time_features =batch_time_features [0 ]if batch_time_features else {}

            rule_similarities =[]
            for rule_idx in range (len (rules_list )):
                similarity =self .calculate_rule_similarity (
                exog_features ,time_features ,rule_idx 
                )
                rule_similarities .append ((rule_idx ,similarity ))
        else :
        # Full batch
            all_batch_similarities =[]
            for batch_idx in range (batch_size ):
                exog_features =batch_exog_features [batch_idx ]if batch_exog_features else {}
                time_features =batch_time_features [batch_idx ]if batch_time_features else {}

                batch_rule_similarities =[]
                for rule_idx in range (len (rules_list )):
                    similarity =self .calculate_rule_similarity (
                    exog_features ,time_features ,rule_idx 
                    )
                    batch_rule_similarities .append (similarity )
                all_batch_similarities .append (batch_rule_similarities )

                # Calculate average similarity
            avg_similarities =np .mean (all_batch_similarities ,axis =0 )
            rule_similarities =[(rule_idx ,avg_similarities [rule_idx ])for rule_idx in range (len (rules_list ))]

            # Sort by Similarity
        rule_similarities .sort (key =lambda x :x [1 ],reverse =True )

        # Self-adaptation threshold filter
        similarity_scores =[sim for _ ,sim in rule_similarities ]
        if len (similarity_scores )>0 :
            adaptive_threshold =np .percentile (similarity_scores ,60 )
        else :
            adaptive_threshold =0.3 

            # Filter rules
        relevant_rules =[]
        relevant_indices =[]
        fuzzy_similarities =[]

        for rule_idx ,similarity in rule_similarities :
            if similarity >=adaptive_threshold :
                relevant_rules .append (rules_list [rule_idx ])
                relevant_indices .append (rule_idx )
                fuzzy_similarities .append (similarity )

                # Minimum guaranteed amount
        if len (relevant_rules )<3 :
            for rule_idx ,similarity in rule_similarities [:3 ]:
                if rule_idx not in relevant_indices :
                    relevant_rules .append (rules_list [rule_idx ])
                    relevant_indices .append (rule_idx )
                    fuzzy_similarities .append (similarity )

                    # Collect dynamic threshold information
        dynamic_thresholds ={}
        if batch_exog_features :
            for var_name ,features in batch_exog_features [0 ].items ():
                dynamic_thresholds [var_name ]=features ['dynamic_threshold']

                # Build a list of similarities for all rules（For quality monitoring）
        all_rule_similarities =[0.0 ]*len (rules_list )
        for rule_idx ,similarity in rule_similarities :
            all_rule_similarities [rule_idx ]=similarity 

            # Build Selection Information
        selection_info ={
        'fuzzy_similarities':all_rule_similarities ,# Similarity of all rules
        'selected_similarities':fuzzy_similarities ,# Similarity of selected rules only
        'dynamic_thresholds':dynamic_thresholds ,
        'threshold_used':adaptive_threshold ,
        'total_rules_considered':len (rules_list ),
        'rules_selected':len (relevant_rules ),
        'selection_reason':f'Fast fuzzy logic filtering (simplified={simplified }, cached={use_cache }) with threshold {adaptive_threshold :.3f}',
        'cache_hits':len ([k for k in self .feature_cache .keys ()])if use_cache else 0 
        }

        # Comment Debug Output，Avoid too much console information
        # if self.debug:
        #     print(f"\n=== Efficient rules filter results ===")
        #     print(f"Filter Out {len(relevant_rules)} Relevant rules: {relevant_indices}")
        #     print(f"Filter threshold: {adaptive_threshold:.3f}")
        #     print(f"Simplified Mode: {simplified}, Cache: {selection_info['cache_hits']}")

        return relevant_rules ,relevant_indices ,selection_info 

    def clear_cache (self ):
        """Clear Cache"""
        self .feature_cache .clear ()

    def get_cache_info (self )->Dict :
        """Fetch Cache Information"""
        return {
        'cache_size':len (self .feature_cache ),
        'max_cache_size':self .max_cache_size ,
        'cache_enabled':self .cache_enabled 
        }
