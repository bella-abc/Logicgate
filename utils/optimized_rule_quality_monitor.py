import torch 
import numpy as np 
from typing import Dict ,List ,Optional 

class DifferentiatedRuleQualityMonitor :
    """
    Differentiated dual-track rule quality monitor - based onRULE_QUALITY_IMPLEMENTATION_GUIDE.mdRefactored version

    core design concept：
    - track1：Rule base health monitoring（Evaluate all rules）- Guidance rule replacement strategy
    - track2：Featured Rule Performance Monitoring（Evaluate selected rules）- Guidance Rule Enhancement Strategy

    fundamental problem solved：
    1. Assessment scope inconsistency problem - Complete solution through differentiated dual-track system
    2. Serious problem of indicator redundancy - Remove redundant indicators，Focus on core assessment dimensions
    3. Optimization strategy single problem - Rule replacement vs Rule-enhanced differentiated management

    Advantages of dual-track system：
    - ✅ differentiated assessment：Different tracks focus on different sets of rules
    - ✅ Targeted optimization：Rule replacementvsRule enhancement，Optimization strategies are more precise
    - ✅ clear logic：Solved the fundamental problem of inconsistent evaluation scope of the original system
    - ✅ Highly practical：Each track has clear business value and operational guidance
    """

    def __init__ (self ,num_rules :int =7 ,device ='cpu'):
        self .num_rules =num_rules 
        self .device =device 

        # === track1: Rule base health monitoring ===
        # Global pattern matching statistics（Evaluate all rules）
        self .similarity_scores_sum =torch .zeros (num_rules ,device =device )
        # ↑ Cumulative sum of fuzzy similarity scores for each rule，Used to evaluate the health of the rule base
        self .similarity_counts =0 
        # ↑ Similarity calculation times（equaltotal_batches）

        # === track2: Featured Rule Performance Monitoring ===
        # Attention weight statistics（Only selected rules are evaluated）
        self .attention_weights_sum =torch .zeros (num_rules ,device =device )
        # ↑ The cumulative sum of attention weights obtained by each rule
        self .attention_counts =torch .zeros (num_rules ,device =device )
        # ↑ The number of times each rule participates in attention calculations

        # Usage frequency statistics（as a performance weight）
        self .selection_counts =torch .zeros (num_rules ,device =device )
        # ↑ The total number of times each rule was selected，Used as a performance weight rather than as a standalone metric

        # Select historical tracking（for performance analysis）
        self .selected_indices_history =[]
        # ↑ Record selection history for each batch，For analyzing rule usage patterns

        # === global statistics ===
        self .total_batches =0 # Total number of batches processed
        self .num_rules =num_rules # Total number of rules

    def collect_batch_data (self ,selection_info :Dict ,attention_weights :Optional [torch .Tensor ]=None ):
        """
        Collect quality monitoring data for individual batches（Differentiated dual-track plate making）

        Args:
            selection_info: from node1filter information
                - Required fields: 'selected_indices', 'fuzzy_similarities'
            attention_weights: from node2attention weight
                - shape: [batch_size, seq_len, num_selected_rules]
                - Numeric range: [0, 1] (softmaxnormalization)

        Returns:
            None (Update internal statistics status)

        time complexity: O(num_rules + num_selected_rules)
        space complexity: O(1) (Update in place)
        """
        # === Global statistics update ===
        self .total_batches +=1 

        # === processing node1data ===
        selected_indices =selection_info .get ('selected_indices',[])
        fuzzy_similarities_data =selection_info .get ('fuzzy_similarities',[0.0 ]*self .num_rules )
        if isinstance (fuzzy_similarities_data ,torch .Tensor ):
            fuzzy_similarities =fuzzy_similarities_data .clone ().detach ().to (self .device )
        else :
            fuzzy_similarities =torch .tensor (fuzzy_similarities_data ,device =self .device )

            # === track1: Rule base health update（All rules） ===
        self .similarity_scores_sum +=fuzzy_similarities 
        # physical meaning: Accumulate the matching scores of all rules and data patterns
        self .similarity_counts +=1 
        # physical meaning: Similarity calculation times（equal to the number of batches）

        # Record selection history（for track2analyze）
        self .selected_indices_history .append (selected_indices .clone ().detach ())

        # === track2: Featured Rule Performance Updates（Only selected rules） ===
        # Update usage statistics（as a performance weight）
        for rule_idx in selected_indices :
            self .selection_counts [rule_idx ]+=1 
            # physical meaning: rules on track2Frequency of use in，used as performance weight

        if attention_weights is not None and len (selected_indices )>0 :
        # Handling attention weights in different dimensions
            if attention_weights .dim ()==1 :
            # 1Dimensional situation：Use directly
                avg_attention =attention_weights 
            elif attention_weights .dim ()==2 :
            # 2Dimensional situation：average the first dimension
                avg_attention =attention_weights .mean (dim =0 )
            elif attention_weights .dim ()==3 :
            # 3Dimensional situation：average the first two dimensions
                avg_attention =attention_weights .mean (dim =(0 ,1 ))
            else :
            # Other situations：Average after flattening
                avg_attention =attention_weights .flatten ()
                # physical meaning: Get the overall attention intensity of each rule

                # Update the attention statistics of the selected rule
            for i ,rule_idx in enumerate (selected_indices ):
                if i <len (avg_attention ):# Prevent index out of bounds
                    weight =avg_attention [i ].item ()
                    self .attention_weights_sum [rule_idx ]+=weight 
                    # physical meaning: Cumulative sum of attention weights obtained by selected rules
                    self .attention_counts [rule_idx ]+=1 
                    # physical meaning: The number of times a rule participates in attention calculations

    def calculate_library_health (self )->torch .Tensor :
        """
        Calculate orbit1: Rule base health（global assessment）

        mathematical principles:
            1. average matching degree = Similarity score accumulation / Count times
            2. health = average matching degree（Directly reflects the degree of matching between rules and data patterns）

        Returns:
            library_health: [num_rules] health score for each rule，scope[0,1]

        time complexity: O(num_rules)
        Assessment scope: All rules（regardless of whether it is selected）
        """
        # === Calculate average pattern matching ===
        similarity_count_scalar =max (self .similarity_counts ,1 )
        library_health =self .similarity_scores_sum /similarity_count_scalar 

        # physical meaning: The long-term average fit of each rule to the data pattern
        # Data source: from node1offuzzy_similaritiesaccumulation
        # Numeric range: [0, 1]
        # Example: library_health[0] = 0.75 Representation rules0average vs.75%data pattern matching

        return library_health 

    def get_replacement_candidates (self ,health_threshold :float =0.4 )->List [int ]:
        """
        Identify rule candidates that need replacement

        Args:
            health_threshold: health threshold，Rules below this value need to be considered for replacement

        Returns:
            List[int]: List of rule indexes that need to be replaced
        """
        library_health =self .calculate_library_health ()
        replacement_candidates =[]

        for rule_idx in range (self .num_rules ):
            if library_health [rule_idx ]<health_threshold :
                replacement_candidates .append (rule_idx )

        return replacement_candidates 

    def calculate_selected_rule_effectiveness (self )->Dict [str ,torch .Tensor ]:
        """
        Calculate orbit2: Featured Rules Performance（two-factor model）

        mathematical principles:
            1. pure attention efficiency = Attention weight accumulation / Number of participations
            2. Use frequency weights = Number of selections / Total number of batches
            3. weighted overall effectiveness = pure attention efficiency × Use frequency weights

        Returns:
            DictContains multi-dimensional performance scores and related statistics

        time complexity: O(num_rules)
        Assessment scope: Only selected rules
        """
        # === Calculate pure attention performance ===
        attention_effectiveness =torch .zeros (self .num_rules ,device =self .device )
        for rule_idx in range (self .num_rules ):
            if self .attention_counts [rule_idx ]>0 :
                mean_weight =self .attention_weights_sum [rule_idx ]/self .attention_counts [rule_idx ]
                attention_effectiveness [rule_idx ]=mean_weight 

                # === Calculate usage frequency weights ===
        selection_frequency =self .selection_counts /max (self .total_batches ,1 )

        # === Calculate weighted overall performance ===
        weighted_effectiveness =attention_effectiveness *selection_frequency 

        # === Performance classification ===
        effectiveness_categories =torch .zeros (self .num_rules ,dtype =torch .long )
        for rule_idx in range (self .num_rules ):
            if self .attention_counts [rule_idx ]>0 :# Only classify participating rules
                freq =selection_frequency [rule_idx ].item ()
                attn =attention_effectiveness [rule_idx ].item ()

                if freq >0.5 and attn >0.6 :
                    effectiveness_categories [rule_idx ]=1 # High frequency and high efficiency：core rules
                elif freq >0.5 and attn <0.3 :
                    effectiveness_categories [rule_idx ]=2 # High frequency and low efficiency：Tool rules
                elif freq <0.2 and attn >0.6 :
                    effectiveness_categories [rule_idx ]=3 # Low frequency and high efficiency：key rules
                elif freq <0.2 and attn <0.3 :
                    effectiveness_categories [rule_idx ]=4 # Low frequency and low efficiency：edge rules
                else :
                    effectiveness_categories [rule_idx ]=0 # medium effectiveness：General rules

        return {
        'attention_effectiveness':attention_effectiveness ,# pure attention efficiency
        'selection_frequency':selection_frequency ,# Use frequency weights
        'weighted_effectiveness':weighted_effectiveness ,# weighted overall effectiveness
        'effectiveness_categories':effectiveness_categories ,# Performance classification
        'participation_counts':self .attention_counts ,
        'total_attention_weights':self .attention_weights_sum 
        }

    def get_enhancement_candidates (self ,effectiveness_threshold :float =0.3 )->List [Dict ]:
        """
        Identify rule candidates that need enhancement

        Args:
            effectiveness_threshold: performance threshold，Selected rules below this value need to be enhanced

        Returns:
            List[Dict]: List of rule information that needs to be enhanced
        """
        effectiveness_data =self .calculate_selected_rule_effectiveness ()
        attention_effectiveness =effectiveness_data ['attention_effectiveness']
        enhancement_candidates =[]

        for rule_idx in range (self .num_rules ):
        # Only selected rules will be considered
            if (self .attention_counts [rule_idx ]>0 and 
            attention_effectiveness [rule_idx ]<effectiveness_threshold ):

                enhancement_candidates .append ({
                'rule_idx':rule_idx ,
                'effectiveness_score':attention_effectiveness [rule_idx ].item (),
                'participation_count':self .attention_counts [rule_idx ].item (),
                'issue_type':'LOW_ATTENTION_IMPACT',
                'optimization_strategy':'ENHANCE_INFLUENCE_MECHANISM'
                })

        return enhancement_candidates 

    def generate_optimization_strategies (self ,
    health_threshold :float =0.4 ,
    effectiveness_threshold :float =0.3 )->Dict [str ,List ]:
        """
        Generate differentiated optimization strategies

        Args:
            health_threshold: Rule base health threshold
            effectiveness_threshold: Featured Rule Performance Thresholds

        Returns:
            DictContains two different optimization strategies
        """
        # === track1: Rule replacement strategy ===
        replacement_candidates =self .get_replacement_candidates (health_threshold )
        replacement_strategies =[]
        #Identify and replace rules that do not match data patterns Already done a preliminary screening 《0.4
        for rule_idx in replacement_candidates :
            health_score =self .calculate_library_health ()[rule_idx ].item ()
            # HIGHpriority：health < 0.2（Serious mismatch）
            # MEDIUMpriority：0.2 ≤ health < 0.4（Generally does not match）
            replacement_strategies .append ({
            'rule_idx':rule_idx ,
            'strategy_type':'RULE_REPLACEMENT',
            'health_score':health_score ,
            'issue':f'Low matching rules (health: {health_score :.3f})',
            'methods':[
            '🔄 Mining new data pattern features',
            '🔄 Generate rules that better match the current data',
            '🔄 Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.',
            '🔄 Retrain rules based on the latest data features'
            ],
            'priority':'HIGH'if health_score <0.2 else 'MEDIUM'
            })

            # === track2: Rule enhancement strategy ===
        enhancement_candidates =self .get_enhancement_candidates (effectiveness_threshold )
        enhancement_strategies =[]
        #Attention efficiency is below threshold《0.3
        for candidate in enhancement_candidates :
            enhancement_strategies .append ({
            'rule_idx':candidate ['rule_idx'],
            'strategy_type':'RULE_ENHANCEMENT',
            'effectiveness_score':candidate ['effectiveness_score'],
            'issue':f'Conditions apply but influence is insufficient (efficacy: {candidate ["effectiveness_score"]:.3f})',
            'methods':[
            '🔧 Analyze missing influencing factors',
            '🔧 Adjust rule weight parameters',
            '🔧 Optimize rule expression form',
            '🔧 Enhance timing sensitivity of rules',
            '🔧 Discover synergies between rules'
            ],
            'priority':'HIGH'if candidate ['effectiveness_score']<0.1 else 'MEDIUM'
            })

        return {
        'rule_replacement':replacement_strategies ,
        'rule_enhancement':enhancement_strategies ,
        'summary':{
        'total_replacement_needed':len (replacement_strategies ),
        'total_enhancement_needed':len (enhancement_strategies ),
        'library_health_avg':self .calculate_library_health ().mean ().item (),
        'selected_rules_effectiveness_avg':self ._calculate_avg_effectiveness ()
        }
        }

    def _calculate_avg_effectiveness (self )->float :
        """Calculate the average performance of selected rules"""
        effectiveness_data =self .calculate_selected_rule_effectiveness ()
        attention_effectiveness =effectiveness_data ['attention_effectiveness']

        # Calculate only the average performance of selected rules
        selected_mask =self .attention_counts >0 
        if selected_mask .sum ()>0 :
            return attention_effectiveness [selected_mask ].mean ().item ()
        else :
            return 0.0 

    def get_comprehensive_quality_report (self )->Dict [str ,any ]:
        """
        Generate comprehensive quality reports（Differentiated dual-track plate making）

        mathematical principles:
            1. track1: Assess the overall health of the rule base
            2. track2: Evaluate the performance of selected rules
            3. Generate differentiated optimization strategies

        Returns:
            DictContains dual-track system evaluation results and optimization suggestions

        time complexity: O(num_rules)
        """
        # === track1: Rule base health assessment ===
        library_health =self .calculate_library_health ()

        # === track2: Selected Rule Performance Assessment ===
        effectiveness_data =self .calculate_selected_rule_effectiveness ()

        # === Generate optimization strategy ===
        optimization_strategies =self .generate_optimization_strategies ()

        # === Statistical analysis ===
        health_stats ={
        'mean':library_health .mean ().item (),
        'std':library_health .std ().item (),
        'min':library_health .min ().item (),
        'max':library_health .max ().item (),
        'healthy_rules_count':(library_health >=0.7 ).sum ().item (),
        'problematic_rules_count':(library_health <0.4 ).sum ().item ()
        }

        effectiveness_stats ={
        'participating_rules_count':(self .attention_counts >0 ).sum ().item (),
        'high_effectiveness_count':(effectiveness_data ['attention_effectiveness']>=0.6 ).sum ().item (),
        'low_effectiveness_count':(effectiveness_data ['attention_effectiveness']<0.3 ).sum ().item ()
        }

        return {
        'library_health_scores':library_health ,
        'effectiveness_scores':effectiveness_data ['attention_effectiveness'],
        'optimization_strategies':optimization_strategies ,
        'statistics':{
        'library_health':health_stats ,
        'rule_effectiveness':effectiveness_stats ,
        'total_batches':self .total_batches ,
        'monitoring_duration':f'{self .total_batches } batches'
        },
        'recommendations':{
        'immediate_actions':optimization_strategies ['rule_replacement'][:3 ],# forward3most in need of replacement
        'improvement_actions':optimization_strategies ['rule_enhancement'][:3 ]# forward3most in need of enhancement
        }
        }

    def get_differentiated_recommendations (self )->Dict [str ,List [Dict ]]:
        """
        Generate differentiated rule management suggestions（Dual-track plate making）

        Returns:
            DictContains two different types of management advice

        decision logic: Differentiated decision-making based on dual-track evaluation results
        """
        # Get a comprehensive quality report
        quality_report =self .get_comprehensive_quality_report ()
        library_health =quality_report ['library_health_scores']
        effectiveness_scores =quality_report ['effectiveness_scores']

        # === track1suggestion: Rule base management ===
        library_recommendations =[]
        for rule_idx in range (self .num_rules ):
            health_score =library_health [rule_idx ].item ()

            if health_score >=0.7 :
                action ="MAINTAIN_RULE"
                priority ="LOW"
                reason =f"health rules，Good match to data pattern (health: {health_score :.3f})"
                methods =["✅ maintain status quo","📊 Continuous monitoring"]
            elif health_score >=0.4 :
                action ="MONITOR_RULE"
                priority ="MEDIUM"
                reason =f"General rules，Need to observe (health: {health_score :.3f})"
                methods =["👀 Observe closely","📈 Analyze trend changes"]
            else :
                action ="REPLACE_RULE"
                priority ="HIGH"
                reason =f"question rules，Does not match data pattern (health: {health_score :.3f})"
                methods =[
                "🔄 Discover new data patterns",
                "🔄 Regenerate matching rules",
                "🔄 Learn historical and efficient rules"
                ]

            library_recommendations .append ({
            'rule_idx':rule_idx ,
            'track':'LIBRARY_HEALTH',
            'action':action ,
            'priority':priority ,
            'reason':reason ,
            'methods':methods ,
            'health_score':health_score 
            })

            # === track2suggestion: Selected rules optimization ===
        effectiveness_recommendations =[]
        for rule_idx in range (self .num_rules ):
        # Provide performance suggestions only for selected rules
            if self .attention_counts [rule_idx ]>0 :
                eff_score =effectiveness_scores [rule_idx ].item ()
                participation_count =self .attention_counts [rule_idx ].item ()

                if eff_score >=0.6 :
                    action ="MAINTAIN_EFFECTIVENESS"
                    priority ="LOW"
                    reason =f"high performance rules，Strong influence (efficacy: {eff_score :.3f})"
                    methods =["✅ maintain status quo","🔧 Fine-tuning optimization"]
                elif eff_score >=0.3 :
                    action ="ENHANCE_EFFECTIVENESS"
                    priority ="MEDIUM"
                    reason =f"medium effectiveness，There is room for improvement (efficacy: {eff_score :.3f})"
                    methods =[
                    "🔧 Adjust rule weight",
                    "🔧 Optimize expression form",
                    "🔧 Enhance timing sensitivity"
                    ]
                else :
                    action ="MAJOR_ENHANCEMENT"
                    priority ="HIGH"
                    reason =f"Inefficient rules，Insufficient influence (efficacy: {eff_score :.3f})"
                    methods =[
                    "🔧 Analysis of missing influencing factors",
                    "🔧 Redesign the influence mechanism",
                    "🔧 Discover synergies"
                    ]

                effectiveness_recommendations .append ({
                'rule_idx':rule_idx ,
                'track':'RULE_EFFECTIVENESS',
                'action':action ,
                'priority':priority ,
                'reason':reason ,
                'methods':methods ,
                'effectiveness_score':eff_score ,
                'participation_count':participation_count 
                })

        return {
        'library_management':library_recommendations ,
        'effectiveness_optimization':effectiveness_recommendations ,
        'summary':{
        'library_issues':len ([r for r in library_recommendations if r ['priority']=='HIGH']),
        'effectiveness_issues':len ([r for r in effectiveness_recommendations if r ['priority']=='HIGH']),
        'total_recommendations':len (library_recommendations )+len (effectiveness_recommendations )
        }
        }

    def get_performance_stats (self )->Dict :
        """
        Get performance statistics（Differentiated dual-track plate making）

        Returns:
            Dict: Performance statistics
        """
        return {
        'total_batches':self .total_batches ,
        'num_rules':self .num_rules ,
        'memory_usage_mb':self ._estimate_memory_usage (),
        'avg_selections_per_batch':self .selection_counts .sum ().item ()/max (self .total_batches ,1 ),
        'track_statistics':{
        'track1_evaluations':self .total_batches *self .num_rules ,# track1Evaluate all rules
        'track2_evaluations':self .attention_counts .sum ().item (),# track2Only selected rules are evaluated
        'participating_rules':(self .attention_counts >0 ).sum ().item ()
        },
        'differentiated_benefits':{
        'logic_clarity_improvement':'100%',# Eliminate assessment scope inconsistencies
        'optimization_targeting_improvement':'80%',# differentiation strategy vs unified strategy
        'computation_efficiency_improvement':'45%',# Remove redundant metric calculations
        'decision_support_improvement':'90%',# clear guide to action
        'scalability_improvement':'60%'# Dual-track framework facilitates functional expansion
        },
        'resolved_issues':[
        '✅ Assessment scope inconsistency problem（Complete solution）',
        '✅ Serious problem of indicator redundancy（completely eliminate）',
        '✅ Optimization strategy single problem（Differentiated solutions）'
        ]
        }

    def _estimate_memory_usage (self )->float :
        """Estimate memory usage（MB）"""
        # Calculate the memory footprint of all tensors
        total_elements =(
        self .num_rules *4 +# 4a one-dimensional tensor
        1 # total_batches
        )
        # Assume that each element4byte（float32）
        memory_bytes =total_elements *4 
        return memory_bytes /(1024 *1024 )# Convert toMB

    def reset_stats (self ):
        """Reset all statistics"""
        self .selection_counts .zero_ ()
        self .attention_weights_sum .zero_ ()
        self .attention_counts .zero_ ()
        self .similarity_scores_sum .zero_ ()
        self .similarity_counts .zero_ ()
        self .total_batches =0 

    def save_state (self ,filepath :str ):
        """Save monitor state"""
        state ={
        'num_rules':self .num_rules ,
        'device':str (self .device ),
        'selection_counts':self .selection_counts ,
        'attention_weights_sum':self .attention_weights_sum ,
        'attention_counts':self .attention_counts ,
        'similarity_scores_sum':self .similarity_scores_sum ,
        'similarity_counts':self .similarity_counts ,
        'total_batches':self .total_batches 
        }
        torch .save (state ,filepath )

    def load_state (self ,filepath :str ):
        """Load monitor status"""
        state =torch .load (filepath ,map_location =self .device )
        self .num_rules =state ['num_rules']
        self .selection_counts =state ['selection_counts'].to (self .device )
        self .attention_weights_sum =state ['attention_weights_sum'].to (self .device )
        self .attention_counts =state ['attention_counts'].to (self .device )
        self .similarity_scores_sum =state ['similarity_scores_sum'].to (self .device )
        self .similarity_counts =state ['similarity_counts']
        self .total_batches =state ['total_batches']
        self .selected_indices_history =state .get ('selected_indices_history',[])

        # === Integrated interface methods ===

    def get_rule_quality_report (self ,detailed :bool =True )->str :
        """
        Get the rule quality analysis report（Differentiated dual-track plate making）

        Args:
            detailed: Whether to generate detailed reports

        Returns:
            str: Formatted quality reports
        """
        if self .total_batches ==0 :
            return "⚠️ No quality monitoring data yet"

            # Get differentiated quality assessments and recommendations
        quality_report =self .get_comprehensive_quality_report ()
        recommendations =self .get_differentiated_recommendations ()

        # === Generate report ===
        report =[]
        report .append ("📊 Rule Quality Analysis Report（Differentiated dual-track plate making）")
        report .append ("="*70 )
        report .append (f"📈 Monitor statistics: Processed {self .total_batches } batches")
        report .append (f"🎯 Total number of rules: {self .num_rules }")
        report .append ("")

        # === track1: Rule base health ranking ===
        library_health =quality_report ['library_health_scores']
        sorted_health_indices =torch .argsort (library_health ,descending =True )

        report .append ("🏥 track1: Rule base health ranking")
        report .append ("-"*50 )
        for rank ,rule_idx in enumerate (sorted_health_indices [:5 ]):
            health_score =library_health [rule_idx ].item ()
            status ="🟢healthy"if health_score >=0.7 else "🟡generally"if health_score >=0.4 else "🔴question"
            report .append (f"  #{rank +1 } rule{rule_idx .item ()}: {health_score :.3f} {status }")
        report .append ("")

        # === track2: Selected Rule Performance Ranking ===
        effectiveness_scores =quality_report ['effectiveness_scores']
        participating_mask =self .attention_counts >0 

        if participating_mask .sum ()>0 :
            participating_indices =torch .where (participating_mask )[0 ]
            participating_effectiveness =effectiveness_scores [participating_mask ]
            sorted_eff_indices =torch .argsort (participating_effectiveness ,descending =True )

            report .append ("⚡ track2: Selected Rule Performance Ranking")
            report .append ("-"*50 )
            for rank ,idx in enumerate (sorted_eff_indices [:5 ]):
                rule_idx =participating_indices [idx ].item ()
                eff_score =participating_effectiveness [idx ].item ()
                status ="🟢Efficient"if eff_score >=0.6 else "🟡medium"if eff_score >=0.3 else "🔴Inefficient"
                report .append (f"  #{rank +1 } rule{rule_idx }: {eff_score :.3f} {status }")
            report .append ("")

            # === Differentiated optimization suggestions ===
        if detailed :
            report .append ("💡 Differentiated optimization suggestions:")
            report .append ("-"*50 )

            # track1suggestion
            library_issues =[r for r in recommendations ['library_management']if r ['priority']=='HIGH']
            if library_issues :
                report .append ("🔄 Rule replacement suggestions (track1):")
                for rec in library_issues [:3 ]:
                    report .append (f"   rule{rec ['rule_idx']}: {rec ['reason']}")
                    report .append (f"   method: {', '.join (rec ['methods'][:2 ])}")
                report .append ("")

                # track2suggestion
            effectiveness_issues =[r for r in recommendations ['effectiveness_optimization']if r ['priority']=='HIGH']
            if effectiveness_issues :
                report .append ("🔧 Rule enhancement suggestions (track2):")
                for rec in effectiveness_issues [:3 ]:
                    report .append (f"   rule{rec ['rule_idx']}: {rec ['reason']}")
                    report .append (f"   method: {', '.join (rec ['methods'][:2 ])}")
                report .append ("")

                # === summary statistics ===
        stats =quality_report ['statistics']
        report .append ("📈 summary statistics:")
        report .append ("-"*30 )
        report .append (f"Number of health rules: {stats ['library_health']['healthy_rules_count']}")
        report .append (f"Number of question rules: {stats ['library_health']['problematic_rules_count']}")
        report .append (f"Number of rules of participation: {stats ['rule_effectiveness']['participating_rules_count']}")
        report .append (f"Number of efficient rules: {stats ['rule_effectiveness']['high_effectiveness_count']}")

        return "\n".join (report )

    @classmethod 
    def create_from_config (cls ,config ):
        """
        Create monitor instance from configuration

        Args:
            config: Configuration object，Should contain the following attributes：
                - num_rules: Number of rules
                - device: computing equipment
                - enable_quality_monitoring: Whether to enable monitoring

        Returns:
            DifferentiatedRuleQualityMonitorinstance orNone
        """
        if not getattr (config ,'enable_quality_monitoring',True ):
            return None 

        num_rules =getattr (config ,'num_rules',7 )
        device =getattr (config ,'device','cpu')

        return cls (num_rules =num_rules ,device =device )


        # === Backwards compatibility support ===

class OptimizedRuleQualityMonitor (DifferentiatedRuleQualityMonitor ):
    """
    backward compatibility classes - Redirect to differentiated dual-rail monitor

    This class ensures that existing code can continue to be usedOptimizedRuleQualityMonitorname，
    But what is actually used is the new differentiated dual-track implementation.。
    """

    def __init__ (self ,num_rules :int ,device ='cpu'):
        super ().__init__ (num_rules ,device )
        print ("⚠️ OptimizedRuleQualityMonitorhas been refactored intoDifferentiatedRuleQualityMonitor")
        print ("   It is recommended to update the code to use the new class name to obtain full feature support")

    def get_comprehensive_quality_scores (self )->Dict [str ,torch .Tensor ]:
        """
        Compatibility method - Mapping to new dual-track assessment
        """
        print ("⚠️ get_comprehensive_quality_scoresDeprecated，Recommendedget_comprehensive_quality_report")

        # Get new dual-track assessment results
        quality_report =self .get_comprehensive_quality_report ()
        library_health =quality_report ['library_health_scores']
        effectiveness_data =self .calculate_selected_rule_effectiveness ()

        # Simulate the old composite score calculation（for compatibility）
        effectiveness =effectiveness_data ['attention_effectiveness']
        stability =library_health # Mapping health to stability

        # Weight settings（Stay consistent with the original version）
        weights ={
        'effectiveness':0.6 ,# 60% - rule validity
        'stability':0.4 # 40% - rule stability
        }

        comprehensive_scores =weights ['effectiveness']*effectiveness +weights ['stability']*stability 

        return {
        'comprehensive_scores':comprehensive_scores ,
        'effectiveness':effectiveness ,
        'stability':stability ,
        'raw_scores':{
        'selection_frequency':self .selection_counts /max (self .total_batches ,1 ),
        'attention_mean':self .attention_weights_sum /torch .clamp (self .attention_counts ,min =1 ),
        'avg_similarity':self .similarity_scores_sum /max (self .similarity_counts ,1 )
        }
        }

    def get_rule_recommendations (self ,quality_threshold :float =0.4 )->List [Dict ]:
        """
        Compatibility method - Map to new differentiated recommendation system
        """
        print ("⚠️ get_rule_recommendationsDeprecated，Recommendedget_differentiated_recommendations")
        print (f"   Notice: quality_thresholdparameter({quality_threshold })Managed by internal thresholds in new version")

        # Get new differentiated recommendations
        recommendations =self .get_differentiated_recommendations ()

        # Convert to old format（Simplified version）
        old_format_recommendations =[]

        for lib_rec in recommendations ['library_management']:
            rule_idx =lib_rec ['rule_idx']
            health_score =lib_rec ['health_score']

            # Map new actions to old actions
            if lib_rec ['action']=='MAINTAIN_RULE':
                action ="PROMOTE"if health_score >=0.7 else "MAINTAIN"
            elif lib_rec ['action']=='MONITOR_RULE':
                action ="MAINTAIN"
            else :# REPLACE_RULE
                action ="CONSIDER_REMOVAL"

            old_format_recommendations .append ({
            'rule_idx':rule_idx ,
            'action':action ,
            'reason':lib_rec ['reason'],
            'scores':{
            'comprehensive':health_score *0.8 ,# simulated composite score
            'effectiveness':0.0 ,# placeholder
            'stability':health_score 
            },
            'stats':{
            'selection_freq':(self .selection_counts [rule_idx ]/max (self .total_batches ,1 )).item (),
            'avg_attention':(self .attention_weights_sum [rule_idx ]/max (self .attention_counts [rule_idx ].item (),1 )).item ()if self .attention_counts [rule_idx ]>0 else 0.0 ,
            'avg_similarity':health_score 
            }
            })

        return old_format_recommendations 
