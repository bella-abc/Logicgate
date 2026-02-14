"""
Performance Validator

It's responsible for monitoring changes in performance following the replacement of rules.，Ensure that optimization actually leads to improvement rather than deterioration。
By comparing baseline and current indicators，Timely detection of performance problems and trigger of corresponding mechanisms。
"""

import logging 
import numpy as np 
from datetime import datetime 
from typing import Dict ,List ,Any ,Optional 
from dataclasses import dataclass 
from scipy import stats 

from .data_structures import ValidationError 


@dataclass 
class BaselineMetrics :
    """Baseline indicator data category"""
    timestamp :str 
    prediction_metrics :Dict [str ,float ]
    rule_usage_metrics :Dict [str ,float ]
    system_metrics :Dict [str ,float ]
    sample_size :int 
    metadata :Dict [str ,Any ]=None 

    def __post_init__ (self ):
        if self .metadata is None :
            self .metadata ={}


@dataclass 
class ValidationResult :
    """Verify Result Data Class"""
    approved :bool 
    overall_score :float 
    metrics_comparison :Dict [str ,Dict [str ,Any ]]
    warnings :List [str ]
    risk_assessment :Dict [str ,str ]
    recommendation :str 
    statistical_significance :Dict [str ,Any ]
    timestamp :str 

    def __post_init__ (self ):
        if not hasattr (self ,'warnings')or self .warnings is None :
            self .warnings =[]


@dataclass 
class PerformanceAlert :
    """Performance Alert Data"""
    alert_type :str # 'degradation', 'improvement', 'anomaly'
    severity :str # 'low', 'medium', 'high', 'critical'
    metric_name :str 
    current_value :float 
    baseline_value :float 
    change_ratio :float 
    message :str 
    timestamp :str 


class PerformanceValidator :
    """
    Performance Validator
    
    Control changes in performance after replacement of rules，Successful replacement through statistical analysis。
    Provision of multilevel certification mechanisms，Ensuring systemic stability and improved effectiveness。
    """

    def __init__ (self ,config :Dict [str ,Any ]):
        self .config =config 
        self .logger =logging .getLogger (__name__ )

        # Tolerance threshold configuration
        self .tolerance_thresholds =config .get ('validation',{}).get ('tolerance_thresholds',{
        'mse_degradation':0.05 ,
        'mae_degradation':0.03 ,
        'r2_degradation':0.02 ,
        'rule_usage_change':0.1 
        })

        # Statistical profile
        self .significance_level =0.05 
        self .minimum_sample_size =config .get ('validation',{}).get ('validation_sample_size',100 )

        # Storage of baseline indicators
        self .baseline_metrics =None 
        self .monitoring_history =[]

    def capture_baseline_metrics (self ,model_outputs :Optional [Dict ]=None )->BaselineMetrics :
        """
        Baseline capture performance indicators
        
        Args:
            model_outputs: Model output data（Optional，For actual integration）
            
        Returns:
            BaselineMetrics: Benchmark indicators
        """
        try :
        # In reality，Here's the real indicators from the model surveillance system.
        # Now use simulation data for demonstration

            prediction_metrics ={
            'mse':self ._get_current_mse (model_outputs ),
            'mae':self ._get_current_mae (model_outputs ),
            'r2_score':self ._get_current_r2 (model_outputs ),
            'mape':self ._get_current_mape (model_outputs )
            }

            rule_usage_metrics ={
            'selection_frequency':self ._get_rule_selection_frequency (model_outputs ),
            'average_attention_weights':self ._get_attention_weights (model_outputs ),
            'rule_diversity':self ._calculate_rule_diversity (model_outputs )
            }

            system_metrics ={
            'inference_time':self ._get_inference_time (model_outputs ),
            'memory_usage':self ._get_memory_usage (model_outputs ),
            'throughput':self ._get_throughput (model_outputs )
            }

            self .baseline_metrics =BaselineMetrics (
            timestamp =datetime .now ().isoformat (),
            prediction_metrics =prediction_metrics ,
            rule_usage_metrics =rule_usage_metrics ,
            system_metrics =system_metrics ,
            sample_size =self .minimum_sample_size ,
            metadata ={
            'capture_method':'simulated',# It's going to be real. 'real_time'
            'model_version':'current',
            'data_source':'validation_set'
            }
            )

            self .logger .info ("Baseline indicator capture completed")
            return self .baseline_metrics 

        except Exception as e :
            self .logger .error (f"Failed to capture benchmark metrics: {e }")
            raise ValidationError (f"Benchmark metric capture failed: {str (e )}")

    def validate_replacement (self ,current_metrics :Dict [str ,Any ],rule_idx :int ,
    strategy :str ="UNKNOWN")->ValidationResult :
        """
        Validate replacement performance
        
        Args:
            current_metrics: Current performance indicators
            rule_idx: Rule Index
            strategy: Replace Policy
            
        Returns:
            ValidationResult: Authentication Results
        """
        if not self .baseline_metrics :
            raise ValidationError ("No baseline indicator set，Please call first.capture_baseline_metrics")

        try :
            validation_result =ValidationResult (
            approved =True ,
            overall_score =0.0 ,
            metrics_comparison ={},
            warnings =[],
            risk_assessment ={'risk_level':'LOW'},
            recommendation ="",
            statistical_significance ={},
            timestamp =datetime .now ().isoformat ()
            )

            # 1. Comparative projections
            prediction_comparison =self ._compare_prediction_metrics (
            current_metrics .get ('prediction_metrics',{}),
            self .baseline_metrics .prediction_metrics 
            )
            validation_result .metrics_comparison ['prediction']=prediction_comparison 

            # 2. Use of indicators for comparative rules
            usage_comparison =self ._compare_usage_metrics (
            current_metrics .get ('rule_usage_metrics',{}),
            self .baseline_metrics .rule_usage_metrics 
            )
            validation_result .metrics_comparison ['rule_usage']=usage_comparison 

            # 3. Comparative System Indicators
            system_comparison =self ._compare_system_metrics (
            current_metrics .get ('system_metrics',{}),
            self .baseline_metrics .system_metrics 
            )
            validation_result .metrics_comparison ['system']=system_comparison 

            # 4. Statistical highlights test
            validation_result .statistical_significance =self ._perform_statistical_tests (
            current_metrics ,self .baseline_metrics 
            )

            # 5. Calculating Combined Scores
            validation_result .overall_score =self ._calculate_overall_score (
            validation_result .metrics_comparison 
            )

            # 6. Risk assessment
            validation_result .risk_assessment =self ._assess_risk (
            validation_result .metrics_comparison ,
            validation_result .statistical_significance 
            )

            # 7. Generate recommendations
            validation_result .recommendation =self ._generate_recommendation (
            validation_result ,strategy ,rule_idx 
            )

            # 8. Final approval of decision-making
            validation_result .approved =self ._make_approval_decision (validation_result )

            # Record validation history
            self .monitoring_history .append (validation_result )

            self .logger .info (f"rule{rule_idx }Performance verification completed，Approval status: {validation_result .approved }")
            return validation_result 

        except Exception as e :
            self .logger .error (f"Performance verification failed: {e }")
            return ValidationResult (
            approved =False ,
            overall_score =0.0 ,
            metrics_comparison ={},
            warnings =[f"Verification process exception: {str (e )}"],
            risk_assessment ={'risk_level':'HIGH'},
            recommendation ="Because it's an anomaly.，Recommended rollback",
            statistical_significance ={},
            timestamp =datetime .now ().isoformat ()
            )

    def _compare_prediction_metrics (self ,current :Dict [str ,float ],
    baseline :Dict [str ,float ])->Dict [str ,Any ]:
        """Comparative projections"""
        comparison ={}

        for metric_name ,baseline_value in baseline .items ():
            current_value =current .get (metric_name ,baseline_value )
            change_ratio =(current_value -baseline_value )/baseline_value if baseline_value !=0 else 0 

            # For error indicators（MSE, MAE），Lower is good.
            # For accuracy indicators（R²），It's good to raise it.
            if metric_name in ['mse','mae','mape']:
                improvement =change_ratio <0 # Lower error is improvement.
                threshold =self .tolerance_thresholds .get (f"{metric_name }_degradation",0.05 )
                acceptable =change_ratio <=threshold 
            else :# r2_scoreWait.
                improvement =change_ratio >0 # Increased scores are improvements
                threshold =self .tolerance_thresholds .get (f"{metric_name }_degradation",0.02 )
                acceptable =change_ratio >=-threshold 

            comparison [metric_name ]={
            'baseline':baseline_value ,
            'current':current_value ,
            'change_ratio':change_ratio ,
            'improvement':improvement ,
            'acceptable':acceptable ,
            'threshold':threshold 
            }

        return comparison 

    def _compare_usage_metrics (self ,current :Dict [str ,float ],
    baseline :Dict [str ,float ])->Dict [str ,Any ]:
        """Use of indicators for comparative rules"""
        comparison ={}

        for metric_name ,baseline_value in baseline .items ():
            current_value =current .get (metric_name ,baseline_value )
            change_ratio =(current_value -baseline_value )/baseline_value if baseline_value !=0 else 0 

            threshold =self .tolerance_thresholds .get ('rule_usage_change',0.1 )
            acceptable =abs (change_ratio )<=threshold 

            comparison [metric_name ]={
            'baseline':baseline_value ,
            'current':current_value ,
            'change_ratio':change_ratio ,
            'acceptable':acceptable ,
            'threshold':threshold 
            }

        return comparison 

    def _compare_system_metrics (self ,current :Dict [str ,float ],
    baseline :Dict [str ,float ])->Dict [str ,Any ]:
        """Comparative System Indicators"""
        comparison ={}

        for metric_name ,baseline_value in baseline .items ():
            current_value =current .get (metric_name ,baseline_value )
            change_ratio =(current_value -baseline_value )/baseline_value if baseline_value !=0 else 0 

            # For Time and Memory Usage，Lower is good.
            if metric_name in ['inference_time','memory_usage']:
                improvement =change_ratio <0 
                acceptable =change_ratio <=0.2 # Allow20%It's falling.
            else :# throughputWait.
                improvement =change_ratio >0 
                acceptable =change_ratio >=-0.1 # Allow10%It's falling.

            comparison [metric_name ]={
            'baseline':baseline_value ,
            'current':current_value ,
            'change_ratio':change_ratio ,
            'improvement':improvement ,
            'acceptable':acceptable 
            }

        return comparison 

    def _perform_statistical_tests (self ,current_metrics :Dict ,
    baseline_metrics :BaselineMetrics )->Dict [str ,Any ]:
        """Conduct statistical visibility tests"""
        # In reality，We'll use real sample data here.tTest
        # Now use simulation data demonstration

        results ={}

        for metric_name in baseline_metrics .prediction_metrics .keys ():
        # Simulate sample data
            baseline_samples =np .random .normal (
            baseline_metrics .prediction_metrics [metric_name ],
            baseline_metrics .prediction_metrics [metric_name ]*0.1 ,
            baseline_metrics .sample_size 
            )

            current_value =current_metrics .get ('prediction_metrics',{}).get (
            metric_name ,baseline_metrics .prediction_metrics [metric_name ]
            )
            current_samples =np .random .normal (
            current_value ,
            current_value *0.1 ,
            baseline_metrics .sample_size 
            )

            # ImplementationtTest
            t_stat ,p_value =stats .ttest_ind (baseline_samples ,current_samples )

            results [metric_name ]={
            't_statistic':float (t_stat ),
            'p_value':float (p_value ),
            'significant':p_value <self .significance_level ,
            'sample_size':baseline_metrics .sample_size 
            }

        return results 

    def _calculate_overall_score (self ,metrics_comparison :Dict [str ,Dict ])->float :
        """Calculating Combined Scores"""
        scores =[]
        weights ={'prediction':0.6 ,'rule_usage':0.3 ,'system':0.1 }

        for category ,weight in weights .items ():
            if category in metrics_comparison :
                category_metrics =metrics_comparison [category ]
                acceptable_count =sum (1 for m in category_metrics .values ()if m .get ('acceptable',False ))
                total_count =len (category_metrics )

                if total_count >0 :
                    category_score =acceptable_count /total_count 
                    scores .append (category_score *weight )

        return sum (scores )if scores else 0.0 

    def _assess_risk (self ,metrics_comparison :Dict ,statistical_significance :Dict )->Dict [str ,str ]:
        """Assessing risk levels"""
        risk_factors =[]

        # Check for forecast indicators to deteriorate
        prediction_metrics =metrics_comparison .get ('prediction',{})
        for metric_name ,comparison in prediction_metrics .items ():
            if not comparison .get ('acceptable',True ):
                if abs (comparison .get ('change_ratio',0 ))>0.1 :
                    risk_factors .append (f"{metric_name }Significant deterioration")

                    # Check for statistical visibility
        significant_degradations =[
        name for name ,test in statistical_significance .items ()
        if test .get ('significant',False )and test .get ('t_statistic',0 )<0 
        ]

        if significant_degradations :
            risk_factors .append (f"Statistical significance worsens: {', '.join (significant_degradations )}")

            # Determination of risk level
        if len (risk_factors )>=3 :
            risk_level ="CRITICAL"
        elif len (risk_factors )>=2 :
            risk_level ="HIGH"
        elif len (risk_factors )>=1 :
            risk_level ="MEDIUM"
        else :
            risk_level ="LOW"

        return {
        'risk_level':risk_level ,
        'risk_factors':risk_factors ,
        'factor_count':len (risk_factors )
        }

    def _generate_recommendation (self ,validation_result :ValidationResult ,
    strategy :str ,rule_idx :int )->str :
        """Generate recommendations"""
        if validation_result .overall_score >=0.8 :
            return f"rule{rule_idx }Replacement works well，It is recommended to keep the current configuration"
        elif validation_result .overall_score >=0.6 :
            return f"rule{rule_idx }The replacement effect is average，It is recommended to continue monitoring"
        elif validation_result .overall_score >=0.4 :
            return f"rule{rule_idx }Substitution is risky，It is recommended to consider adjusting or rolling back"
        else :
            return f"rule{rule_idx }Poor replacement，It is strongly recommended to roll back immediately"

    def _make_approval_decision (self ,validation_result :ValidationResult )->bool :
        """Decision on approval"""
        # Consolidated score threshold
        if validation_result .overall_score <0.5 :
            return False 

            # Risk level threshold
        if validation_result .risk_assessment .get ('risk_level')in ['CRITICAL','HIGH']:
            return False 

            # Check for serious deterioration.
        prediction_metrics =validation_result .metrics_comparison .get ('prediction',{})
        for metric_name ,comparison in prediction_metrics .items ():
            if not comparison .get ('acceptable',True ):
                change_ratio =abs (comparison .get ('change_ratio',0 ))
                if change_ratio >0.15 :# Over15%The deterioration
                    return False 

        return True 

        # Simulation method（The actual implementation will connect to a real surveillance system.）
    def _get_current_mse (self ,model_outputs =None )->float :
        return np .random .uniform (0.1 ,0.5 )

    def _get_current_mae (self ,model_outputs =None )->float :
        return np .random .uniform (0.05 ,0.3 )

    def _get_current_r2 (self ,model_outputs =None )->float :
        return np .random .uniform (0.7 ,0.95 )

    def _get_current_mape (self ,model_outputs =None )->float :
        return np .random .uniform (0.05 ,0.2 )

    def _get_rule_selection_frequency (self ,model_outputs =None )->float :
        return np .random .uniform (0.3 ,0.8 )

    def _get_attention_weights (self ,model_outputs =None )->float :
        return np .random .uniform (0.4 ,0.9 )

    def _calculate_rule_diversity (self ,model_outputs =None )->float :
        return np .random .uniform (0.5 ,0.9 )

    def _get_inference_time (self ,model_outputs =None )->float :
        return np .random .uniform (0.01 ,0.1 )

    def _get_memory_usage (self ,model_outputs =None )->float :
        return np .random .uniform (100 ,500 )

    def _get_throughput (self ,model_outputs =None )->float :
        return np .random .uniform (50 ,200 )

    def get_monitoring_summary (self )->Dict [str ,Any ]:
        """Get Control Summary"""
        if not self .monitoring_history :
            return {'total_validations':0 ,'approval_rate':0.0 }

        total_validations =len (self .monitoring_history )
        approved_count =sum (1 for v in self .monitoring_history if v .approved )
        approval_rate =approved_count /total_validations 

        avg_score =np .mean ([v .overall_score for v in self .monitoring_history ])

        return {
        'total_validations':total_validations ,
        'approved_count':approved_count ,
        'approval_rate':approval_rate ,
        'average_score':avg_score ,
        'baseline_captured':self .baseline_metrics is not None ,
        'last_validation':self .monitoring_history [-1 ].timestamp if self .monitoring_history else None 
        }
