from .TimeLLM import Model as TimeLLMModel ,FlattenHead 
import torch 
import torch .nn as nn 
from layers .StandardNorm import Normalize 
from layers .Embed import PatchEmbedding 
from utils .rule_similarity_filtering import FuzzyRuleSimilarityFilter 
from utils .optimized_rule_quality_monitor import DifferentiatedRuleQualityMonitor 

class Model (TimeLLMModel ):
    """logicgate model implementation - simplified version"""
    def __init__ (self ,configs ,rules_list ,patch_len =16 ,stride =8 ):
        super ().__init__ (configs ,patch_len ,stride )
        self .rules_list =rules_list if isinstance (rules_list ,list )else [rules_list ]
        self .num_rules =len (self .rules_list )

        # Set up device
        self .device =configs .device if hasattr (configs ,'device')else torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')

        # use the correctembeddingDimensions
        self .embedding_dim =768 # GPT2/BERTDimensions

        # Initialize rule similarity filter - fromconfigsRead parameters
        self .rule_filter =FuzzyRuleSimilarityFilter (
        percentile =getattr (configs ,'rule_percentile',75 ),
        window_size =getattr (configs ,'rule_window_size',24 ),
        membership_type ='adaptive',
        debug =getattr (configs ,'rule_debug',False ),
        rule_config_path =getattr (configs ,'rule_config_path','config/rule_patterns.json')
        )

        # Rule filtering related configurations
        self .enable_rule_filtering =getattr (configs ,'enable_rule_filtering',True )
        self .rule_filter_method =getattr (configs ,'rule_filter_method','fast')
        self .rule_use_cache =getattr (configs ,'rule_use_cache',True )
        self .rule_simplified =getattr (configs ,'rule_simplified',False )

        # Notice：Do not overwrite already correctly processedrules_list
        # self.rules_listAlready set correctly above，No need to reassign here

        # Initialize rule quality monitoring based on optimization mode
        self .optimization_mode =getattr (configs ,'optimization_mode','baseline')

        # Quality monitoring is enabled for 'monitoring' and 'optimization' modes
        if self .optimization_mode in ['monitoring','optimization']and len (self .rules_list )>0 :
            self .rule_quality_monitor =DifferentiatedRuleQualityMonitor (
            num_rules =len (self .rules_list ),
            device =self .device 
            )
            self .enable_quality_monitoring =True 
            print (f"✅ Quality monitor is enabled (model: {self .optimization_mode })")
        else :
            self .rule_quality_monitor =None 
            self .enable_quality_monitoring =False 
            if self .optimization_mode =='baseline':
                print ("🎯 Quality monitor is disabled (baseline mode)")
            else :
                print (f"⚠️ Quality monitor is not enabled (model: {self .optimization_mode })")

                # Create an embedding layer for the target sequence
        self .target_patch_embedding =self .patch_embedding # Use parent classpatch_embeddingEmbedding layer as target sequence

        # Create new normalization and embedding layers for exogenous sequences
        if hasattr (configs ,'num_exog_vars')and configs .num_exog_vars >0 :
            self .exog_normalize_layers =Normalize (configs .num_exog_vars ,affine =False )
            self .exog_patch_embedding =PatchEmbedding (
            configs .d_model ,self .patch_len ,self .stride ,configs .dropout 
            )

            # Always add a cross-attention layer（for rule processing）
        self .cross_attention =nn .MultiheadAttention (
        embed_dim =self .d_llm ,# used_llmDimensions
        num_heads =8 ,
        batch_first =True 
        )

        # Add interactive encoder f_interact（If there are exogenous variables）
        if hasattr (configs ,'num_exog_vars')and configs .num_exog_vars >0 :
            self .interaction_encoder =nn .Sequential (
            nn .Linear (self .d_llm *2 ,self .d_llm ),
            nn .ReLU (),
            nn .Linear (self .d_llm ,self .d_llm )
            )



            # Recreate suitable for simplified versionoutput_projection
            # calculatepatch embeddingactual afterpatchquantity
        actual_patch_nums =int ((configs .seq_len -self .patch_len )/self .stride +2 )
        # # used_modelinstead ofd_ff，becausepatch_embeddingThe output isd_modelDimensions
        # self.simplified_head_nf = configs.d_model * actual_patch_nums
        # used_ffinstead ofd_llm，because we don't havereprogramminglayer
        # self.simplified_head_nf = self.d_ff * actual_patch_nums
        # used_llminstead ofd_ff，Because now we need to use all dimensions
        self .simplified_head_nf =self .d_llm *actual_patch_nums 

        if self .task_name =='long_term_forecast'or self .task_name =='short_term_forecast':
            self .output_projection =FlattenHead (configs .enc_in ,self .simplified_head_nf ,self .pred_len ,
            head_dropout =configs .dropout )

    def _process_rules_with_filtering (self ,batch_size ,x_exog =None ,x_mark_enc =None ,pred_len =None ):
        """
        Based on fuzzy logic+Dynamic threshold filtering related rules and processing

        Args:
            batch_size: batch size
            x_exog: exogenous variable data [batch_size, seq_len, num_exog]
            x_mark_enc: time stamp [batch_size, seq_len, 4]
            pred_len: prediction length，Used to extract pure prediction period time features

        Returns:
            rule_features: Filtered rule characteristics [batch_size, selected_num_rules, max_rule_length, embedding_dim]
            selection_info: filter information dictionary
        """
        # step1: Filter related rules - Choose method based on configuration
        if self .enable_rule_filtering and (x_exog is not None or x_mark_enc is not None ):
            fuzzy_config ={
            'percentile':getattr (self .rule_filter ,'percentile',75 ),
            'window_size':getattr (self .rule_filter ,'window_size',24 )
            }

            if self .rule_filter_method =='basic':
            # Basic version：Most accurate but slower
                relevant_rules ,relevant_indices ,selection_info =self .rule_filter .select_relevant_rules (
                x_exog =x_exog ,
                x_mark_enc =x_mark_enc ,
                rules_list =self .rules_list ,
                fuzzy_config =fuzzy_config ,
                pred_len =pred_len 
                )
            else :# 'fast'
            # Efficient version：Supports caching and simplified mode
                relevant_rules ,relevant_indices ,selection_info =self .rule_filter .select_relevant_rules_fast (
                x_exog =x_exog ,
                x_mark_enc =x_mark_enc ,
                rules_list =self .rules_list ,
                fuzzy_config =fuzzy_config ,
                use_cache =self .rule_use_cache ,
                simplified =self .rule_simplified ,
                pred_len =pred_len 
                )

                # Comment out debug output，Avoid too much console information
                # if self.rule_filter.debug:
                #     print(f"filter out {len(relevant_rules)} related rules: {relevant_indices}")
                #     print(f"filter threshold: {selection_info['threshold_used']:.3f}")
        else :
        # If filtering is not enabled or no data is entered，Use all rules
            relevant_rules =self .rules_list 
            relevant_indices =list (range (len (self .rules_list )))
            selection_info ={
            'fuzzy_similarities':[1.0 ]*len (self .rules_list ),
            'dynamic_thresholds':{},
            'threshold_used':0.0 ,
            'total_rules_considered':len (self .rules_list ),
            'rules_selected':len (self .rules_list ),
            'selection_reason':'Rule filtering disabled or no input data'
            }

            # Add filtered rule index toselection_infomiddle
        selection_info ['selected_rule_indices']=relevant_indices 
        # Add correctly formatted data for quality monitor
        selection_info ['selected_indices']=torch .tensor (relevant_indices ,dtype =torch .long )
        # make surefuzzy_similaritiesTootensorFormat
        if 'fuzzy_similarities'in selection_info and not isinstance (selection_info ['fuzzy_similarities'],torch .Tensor ):
            selection_info ['fuzzy_similarities']=torch .tensor (selection_info ['fuzzy_similarities'],dtype =torch .float32 )

            # keepselection_infofor quality control（in training mode）
        if self .training and self .rule_quality_monitor is not None :
            self ._current_selection_info ={
            'selected_indices':torch .tensor (relevant_indices ,dtype =torch .long ),
            'fuzzy_similarities':torch .tensor (selection_info ['fuzzy_similarities'],dtype =torch .float32 )
            }

            # step2: Encode filtered rules
        rule_features =[]

        for b in range (batch_size ):
            batch_rule_features =[]
            for rule in relevant_rules :
            # directly to the rulestokenization
                rule_tokens =self .tokenizer (rule ,return_tensors ="pt",padding =True ,truncation =True ,max_length =128 )

                # make sureinput_idsremain aslongtype and move to the correct device
                input_ids =rule_tokens .input_ids .long ().to (self .device )

                # make sureembeddinglayer on the correct device
                embedding_layer =self .llm_model .get_input_embeddings ()
                embedding_layer =embedding_layer .to (self .device )

                # Get rulesembedding
                rule_embedding =embedding_layer (input_ids )# [1, seq_len, embedding_dim]
                rule_embedding =rule_embedding .squeeze (0 )# [seq_len, embedding_dim]

                batch_rule_features .append (rule_embedding )

                # Align length of all rules
            if batch_rule_features :
                max_length =max (feat .size (0 )for feat in batch_rule_features )
                aligned_features =[]
                for feat in batch_rule_features :
                    if feat .size (0 )<max_length :
                    # Pad shorter rules
                        padding =torch .zeros (max_length -feat .size (0 ),feat .size (1 ),device =feat .device )
                        feat =torch .cat ([feat ,padding ],dim =0 )
                    aligned_features .append (feat )

                    # Stacking Rule Features [selected_num_rules, max_rule_length, embedding_dim]
                batch_rules =torch .stack (aligned_features ,dim =0 )
            else :
            # If there are no filtering rules，Create an empty feature tensor
                batch_rules =torch .zeros (1 ,1 ,self .embedding_dim ,device =self .device )

            rule_features .append (batch_rules )

            # Stack all batches [batch_size, selected_num_rules, max_rule_length, embedding_dim]
        rule_features =torch .stack (rule_features ,dim =0 )

        return rule_features ,selection_info 





    def _process_rules_directly (self ,batch_size ):
        """Direct handling of rules，No need to start frompromptExtract location in（Stay backwards compatible）"""
        rule_features ,_ =self ._process_rules_with_filtering (batch_size )
        return rule_features 

    def forward (self ,x_enc ,x_mark_enc ,x_dec ,x_mark_dec ,x_exog =None ,mask =None ):
        """
        Args:
            x_enc: Target sequence historical data [batch_size, seq_len, 1]
            x_mark_enc: Temporal feature encoding [batch_size, seq_len, num_time_features]
            x_dec: Decoder input
            x_mark_dec: Decoder temporal characteristics
            x_exog: exogenous sequence data [batch_size, seq_len + pred_len, num_exog]
            mask: optional mask
        """
        if self .task_name =='long_term_forecast'or self .task_name =='short_term_forecast':
        # Get prediction results and true gating vector
            dec_out ,gate_vector =self .forecast (x_enc ,x_mark_enc ,x_dec ,x_mark_dec ,x_exog )
            return dec_out [:,-self .pred_len :,:],gate_vector 
        return None ,None 

    def forecast (self ,x_enc ,x_mark_enc ,x_dec ,x_mark_dec ,x_exog =None ):
        """
        simplified versionforecastmethod：Only retain the core timing prediction function
        Args:
            x_enc: Target sequence historical data [batch_size, seq_len, 1]
            x_mark_enc: Temporal feature encoding [batch_size, seq_len, num_time_features]
            x_dec: Decoder input
            x_mark_dec: Decoder temporal characteristics
            x_exog: exogenous sequence data [batch_size, seq_len + pred_len, num_exog]
        """
        # 1. Data preprocessing
        x_enc =self .normalize_layers (x_enc ,'norm')# target sequence normalization

        # Normalize exogenous sequences
        if x_exog is not None and hasattr (self ,'exog_normalize_layers'):
            x_exog =self .exog_normalize_layers (x_exog ,'norm')

        B ,T ,N =x_enc .size ()

        # 2. Process target sequence
        target_history =x_enc .permute (0 ,2 ,1 ).contiguous ()# [B, N, T]
        target_embeddings ,n_vars =self .target_patch_embedding (target_history )# [B, N, patch_num, D]

        # 3. Handling exogenous sequences（if there is）
        if x_exog is not None and hasattr (self ,'exog_patch_embedding'):
        # Only use exogenous sequences from the historical part for feature extraction
            exog_history =x_exog [:,:T ].permute (0 ,2 ,1 ).contiguous ()# [B, num_exog, T]
            exog_embeddings ,n_exog =self .exog_patch_embedding (exog_history )# [B, num_exog, patch_num, D]

        else :
            exog_embeddings =None 


            # 4. Processing rule characteristics（for cross attention）- Filter using rules
            # correction：Use forecast period time featuresx_mark_decrather than historical time characteristicsx_mark_enc
            # further corrections：Use only time features for the pure forecast period（Does not include overlap）
        B =target_embeddings .size (0 )
        rule_features ,selection_info =self ._process_rules_with_filtering (
        B ,x_exog =x_exog ,x_mark_enc =x_mark_dec ,pred_len =self .pred_len 
        )# [batch_size, selected_num_rules, max_rule_length, embedding_dim]


        # Comment out debug output，Avoid too much console information
        # if hasattr(self.rule_filter, 'debug') and self.rule_filter.debug:
        #     print(f"Rule filter results: {selection_info['rules_selected']}/{selection_info['total_rules_considered']} rules")
        #     print(f"filter threshold: {selection_info['threshold_used']:.3f}")

        source_embeddings =self .mapping_layer (self .word_embeddings .permute (1 ,0 ))
        source_embeddings =source_embeddings .permute (1 ,0 )
        target_embeddings =self .reprogramming_layer (target_embeddings ,source_embeddings ,source_embeddings )

        # 4. Dealing with the interaction of exogenous variables and target variables（If there are exogenous variables）
        if exog_embeddings is not None :
        # Reprogram exogenous variables
            exog_embeddings =self .reprogramming_layer (exog_embeddings ,source_embeddings ,source_embeddings )

            # Get the feature dimensions after reprogramming
            B =target_embeddings .size (0 )
            target_patches =target_embeddings .size (1 )
            D_out =target_embeddings .size (-1 )

            # Reshape exogenous variable dimensions to maintain independence
            num_exog =exog_embeddings .size (0 )//B 
            patch_num =exog_embeddings .size (1 )
            exog_embeddings =exog_embeddings .view (B ,num_exog ,patch_num ,D_out )
            exog_embeddings =exog_embeddings .view (B ,num_exog *patch_num ,D_out )

            # Building lagging interactive queries Q_interact（The attenuating influence of exogenous variables on current and future time steps）
            num_exog_patches =exog_embeddings .size (1 )
            lag_steps =3 # Number of time steps affected by lag，Can be tuned as a hyperparameter

            q_interact_list =[]

            for i in range (num_exog_patches ):
            # Get the current exogenous variablepatch
                exog_patch =exog_embeddings [:,i :i +1 ,:]# [B, 1, D_out]

                # Calculate the exogenous variablepatchtarget variable to influencepatchscope
                # Start from current time step，Impact on current and futurelag_stepstime steps
                start_target_idx =i # Starting from the current corresponding time
                end_target_idx =min (i +lag_steps +1 ,target_patches )# affect the follow-uplag_stepstime steps

                if start_target_idx <target_patches :
                # Get the affected target variablepatches（current time + Lag time）
                    affected_target_patches =target_embeddings [:,start_target_idx :end_target_idx ,:]# [B, lag_range, D_out]

                    if affected_target_patches .size (1 )>0 :
                        lag_range =affected_target_patches .size (1 )

                        # Extended exogenous variablespatchto match the affected target variablepatchquantity
                        exog_expanded =exog_patch .expand (-1 ,lag_range ,-1 )# [B, lag_range, D_out]

                        # Splice target variables and exogenous variables（No attenuation is applied here）
                        concat_features =torch .cat ([affected_target_patches ,exog_expanded ],dim =-1 )# [B, lag_range, 2*D_out]

                        # Interactive query generation via interactive encoder
                        q_interact_patch =self .interaction_encoder (concat_features )# [B, lag_range, D_out]

                        q_interact_list .append (q_interact_patch )

                        # Splice all interactive queries
            if q_interact_list :
                q_interact =torch .cat (q_interact_list ,dim =1 )# [B, total_lag_interactions, D_out]
            else :
            # If there is no valid hysteresis interaction，Use the target variable itself
                q_interact =target_embeddings 

                # Cross-attention using real regular features
                # Prepare key-value vector：Rule representation K_rules, V_rules
            key_value =torch .mean (rule_features ,dim =2 )# [batch_size, num_rules, embedding_dim]

            # Make sure dimensions match：ifembedding_dim != D_out，Need mapping
            if key_value .size (-1 )!=D_out :
                if not hasattr (self ,'rule_projection'):
                    self .rule_projection =nn .Linear (key_value .size (-1 ),D_out ).to (key_value .device )
                key_value =self .rule_projection (key_value )

                # perform cross attention：Attention(Q=Q_interact, K=K_rules, V=V_rules)
            attn_output ,attn_weights =self .cross_attention (
            query =q_interact ,
            key =key_value ,
            value =key_value 
            )

            # Save attention weights and collect quality monitoring data instantly
            if self .training and self .rule_quality_monitor is not None :
            # Calculate average attention weight [num_selected_rules]
                if attn_weights .dim ()==3 :# [batch_size, seq_len, num_rules]
                    avg_attention =attn_weights .mean (dim =(0 ,1 ))
                elif attn_weights .dim ()==2 :# [batch_size, num_rules]
                    avg_attention =attn_weights .mean (dim =0 )
                else :
                    avg_attention =attn_weights .flatten ()

                self ._current_attention_weights =avg_attention .detach ().cpu ()

                # Collect quality monitoring data instantly（if there isselection_info）
                if hasattr (self ,'_current_selection_info'):
                    try :
                        self .rule_quality_monitor .collect_batch_data (
                        self ._current_selection_info ,
                        self ._current_attention_weights 
                        )
                    except Exception :
                    # Handle errors silently，Avoid affecting the main training process
                        pass 

                        # apply residual connection
            q_interact_enhanced =q_interact +attn_output 

            # Fusion of lagged interaction information back to the target variable（Consider the lag effect of decay weights）
            enhanced_target_embeddings =target_embeddings .clone ()

            # Create a weighted cumulative impact matrix，Record the impact of exogenous variables at each target time step
            influence_matrix =torch .zeros_like (target_embeddings )# [B, target_patches, D_out]
            influence_weights_sum =torch .zeros (target_patches ,device =target_embeddings .device )# Record the sum of the weights of each position

            # Reassign the enhanced interaction information to the corresponding target variable location（Consider attenuating weights）
            interaction_idx =0 
            for i in range (num_exog_patches ):
                start_target_idx =i # Starting from the current corresponding time
                end_target_idx =min (i +lag_steps +1 ,target_patches )# Affects subsequent time steps

                if start_target_idx <target_patches and interaction_idx <q_interact_enhanced .size (1 ):
                    lag_range =end_target_idx -start_target_idx 
                    if lag_range >0 :
                    # Obtain the enhanced interaction information corresponding to the current exogenous variable
                        interaction_end_idx =min (interaction_idx +lag_range ,q_interact_enhanced .size (1 ))
                        current_interaction =q_interact_enhanced [:,interaction_idx :interaction_end_idx ,:]

                        # Calculate decay weight（Only apply falloff once when blending）
                        actual_range =min (lag_range ,current_interaction .size (1 ))

                        # Apply attenuation weights to accumulate to the corresponding target variable position
                        for j in range (actual_range ):
                            target_pos =start_target_idx +j 
                            if target_pos <target_patches :
                            # Calculate the attenuation weight of the current position
                                lag_distance =j # jIt’s the lag distance
                                weight =1.0 /(lag_distance +1 )# decay weight：1.0, 0.5, 0.33, 0.25, ...

                                # Apply decay weight（only one decay）
                                influence_matrix [:,target_pos ,:]+=weight *current_interaction [:,j ,:]
                                influence_weights_sum [target_pos ]+=weight 

                        interaction_idx +=actual_range 

                        # Normalize weighted impact，Avoid certain locations being over-amplified due to the influence of multiple exogenous variables
                        # for t in range(target_patches):
                        #     if influence_weights_sum[t] > 0:
                        #         influence_matrix[:, t, :] = influence_matrix[:, t, :] / influence_weights_sum[t]

                        # final fusion
            alpha =0.8 # Smaller fusion weight，Avoid overly influencing the original target variable information
            enhanced_target_embeddings =target_embeddings +alpha *influence_matrix 
            combined_embeddings =enhanced_target_embeddings 
        else :
        # If there are no exogenous variables，Use the target variable directly as the query
            query =target_embeddings 
            D_out =query .size (-1 )

            # Prepare key-value vector：Rule representation
            key_value =torch .mean (rule_features ,dim =2 )# [batch_size, num_rules, embedding_dim]

            # Make sure dimensions match：ifembedding_dim != D_out，Need mapping
            if key_value .size (-1 )!=D_out :
                if not hasattr (self ,'rule_projection'):
                    self .rule_projection =nn .Linear (key_value .size (-1 ),D_out ).to (key_value .device )
                key_value =self .rule_projection (key_value )

                # perform cross attention
            attn_output ,attn_weights =self .cross_attention (
            query =query ,
            key =key_value ,
            value =key_value 
            )

            # Save attention weights and collect quality monitoring data instantly
            if self .training and self .rule_quality_monitor is not None :
            # Calculate average attention weight [num_selected_rules]
                if attn_weights .dim ()==3 :# [batch_size, seq_len, num_rules]
                    avg_attention =attn_weights .mean (dim =(0 ,1 ))
                elif attn_weights .dim ()==2 :# [batch_size, num_rules]
                    avg_attention =attn_weights .mean (dim =0 )
                else :
                    avg_attention =attn_weights .flatten ()

                self ._current_attention_weights =avg_attention .detach ().cpu ()

                # Collect quality monitoring data instantly（if there isselection_info）
                if hasattr (self ,'_current_selection_info'):
                    try :
                        self .rule_quality_monitor .collect_batch_data (
                        self ._current_selection_info ,
                        self ._current_attention_weights 
                        )
                    except Exception :
                    # Handle errors silently，Avoid affecting the main training process
                        pass 

                        # Update the representation of the target variable
            enhanced_target_embeddings =target_embeddings +attn_output # residual connection
            combined_embeddings =enhanced_target_embeddings 

            # 5. Make predictions using all dimensions（No more intercepting dimensions）
        dec_out =combined_embeddings # keep intactd_llmDimensions
        # Reshape：[batch_size, seq_len, d_llm] -> [batch_size, n_vars, seq_len, d_llm]
        dec_out =torch .reshape (dec_out ,(-1 ,n_vars ,dec_out .shape [-2 ],dec_out .shape [-1 ]))
        # Adjust dimension order：[batch_size, n_vars, seq_len, d_llm] -> [batch_size, n_vars, d_llm, seq_len]
        dec_out =dec_out .permute (0 ,1 ,3 ,2 ).contiguous ()

        # 4. Prediction via prediction header（Use the lastpatch_numstime steps）
        dec_out =self .output_projection (dec_out [:,:,:,-self .patch_nums :])
        dec_out =dec_out .permute (0 ,2 ,1 ).contiguous ()# [batch_size, seq_len, n_vars]

        # 6. denormalization
        dec_out =self .normalize_layers (dec_out ,'denorm')

        # If the forecast contains exogenous variables，Denormalization is also required
        if x_exog is not None and hasattr (self ,'exog_normalize_layers')and dec_out .shape [-1 ]>n_vars :
            exog_part =dec_out [:,:,n_vars :]
            exog_part =self .exog_normalize_layers (exog_part ,'denorm')
            dec_out =torch .cat ([dec_out [:,:,:n_vars ],exog_part ],dim =-1 )

            # create virtualgate_vector（Maintain a simplified gating mechanism）
        gate_vector =torch .ones (B ,self .num_rules ,device =dec_out .device )

        return dec_out ,gate_vector 

    def get_current_quality_data (self ):
        """Get currentbatchquality monitoring data"""
        if not hasattr (self ,'_current_selection_info')or not hasattr (self ,'_current_attention_weights'):
            return None 

        return {
        'selection_info':getattr (self ,'_current_selection_info',None ),
        'attention_weights':getattr (self ,'_current_attention_weights',None )
        }

    def clear_quality_data (self ):
        """clear currentbatchquality monitoring data"""
        if hasattr (self ,'_current_selection_info'):
            delattr (self ,'_current_selection_info')
        if hasattr (self ,'_current_attention_weights'):
            delattr (self ,'_current_attention_weights')
