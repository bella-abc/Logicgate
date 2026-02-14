"""
Enhance training cycle - integratedEpochLevel rule optimization

Seamlessly integrate rule optimization into training loops，Enable intelligent progressive optimization
"""

import torch 
import numpy as np 
from typing import Dict ,Optional 
from tqdm import tqdm 
import time 
import logging 
from sklearn .metrics import mean_squared_error ,mean_absolute_error ,r2_score 

from .optimized_rule_quality_monitor import DifferentiatedRuleQualityMonitor 
from .dual_track_optimization .integration_manager import DualTrackIntegrationManager ,IntegrationConfig 
from .tools import adjust_learning_rate 

class EnhancedTrainingLoop :
    """
    Enhanced training loop，Integrated rule optimization function
    
    Core features：
    1. in everyepochEvaluate rule quality after completion
    2. Whether intelligent decision-making triggers rule optimization
    3. Ensure training continuity and stability
    4. Provide detailed optimization logs and reports
    """

    def __init__ (self ,
    model ,
    train_loader ,
    val_loader ,
    optimizer ,
    criterion ,
    scheduler =None ,
    device ='cuda',
    enable_rule_optimization =True ,
    optimization_config =None ,
    config =None ,
    args =None ,
    checkpoint_dir :Optional [str ]=None ):
        """
        Initialize the boost training loop

        Args:
            model: RuleGatingTIMELLMModel
            train_loader: Training data loader
            val_loader: Verify data loader
            optimizer: optimizer
            criterion: loss function
            scheduler: Learning rate scheduler
            device: computing equipment
            enable_rule_optimization: Whether to enable rule optimization
            optimization_config: Rule optimization configuration
            config: Additional configuration parameters
            args: Command line parameter configuration
        """
        self .model =model 
        self .train_loader =train_loader 
        self .val_loader =val_loader 
        self .optimizer =optimizer 
        self .criterion =criterion 
        self .scheduler =scheduler 
        self .device =device 
        self .config =config or {}
        self .args =args 
        self .checkpoint_dir =checkpoint_dir 

        # fromargsGet the necessary configuration parameters from
        if args is not None :
            self .label_len =args .label_len 
            self .pred_len =args .pred_len 
            self .features =args .features 
        else :
        # default value，if not providedargs
            self .label_len =48 
            self .pred_len =96 
            self .features ='M'

            # Rule optimization related
        self .enable_rule_optimization =enable_rule_optimization 
        self .quality_monitor =None 

        if enable_rule_optimization and hasattr (model ,'rules_list'):
        # Using a quality monitor instance inside the model，Ensure data consistency
            if hasattr (model ,'rule_quality_monitor')and model .rule_quality_monitor is not None :
                self .quality_monitor =model .rule_quality_monitor 
            else :
            # If the model does not have a quality monitor，create a new
                self .quality_monitor =DifferentiatedRuleQualityMonitor (
                num_rules =len (model .rules_list ),
                device =device 
                )
                model .rule_quality_monitor =self .quality_monitor 

                # Initialize the dual-track integrated manager
                # Get parameters from different nodes in the configuration file
            thresholds_config =optimization_config .get ('thresholds',{})
            optimization_section =optimization_config .get ('optimization',{})
            logging_config =optimization_config .get ('logging',{})

            integration_config_overrides ={
            'health_threshold':thresholds_config .get ('health_threshold',0.4 ),
            'effectiveness_threshold':thresholds_config .get ('effectiveness_threshold',0.3 ),
            'analysis_interval_epochs':optimization_section .get ('analysis_interval_epochs',5 ),
            'min_epochs_before_analysis':optimization_section .get ('min_epochs_before_analysis',3 ),
            'max_optimizations_per_session':optimization_section .get ('max_optimization_attempts',3 ),
            'enable_automatic_optimization':optimization_section .get ('enable_automatic_optimization',False ),
            'enable_detailed_logging':logging_config .get ('enable_detailed_logging',True )
            }

            self .integration_manager =DualTrackIntegrationManager (
            quality_monitor =self .quality_monitor ,
            config =IntegrationConfig (**{k :v for k ,v in integration_config_overrides .items ()
            if hasattr (IntegrationConfig ,k )})
            )



            # Training history
        self .training_history ={
        'train_losses':[],
        'val_losses':[],
        'dual_track_sessions':[],
        'quality_reports':[]
        }

        # Log configuration
        self .logger =logging .getLogger (__name__ )

    def calculate_prediction_metrics (self ,predictions :torch .Tensor ,targets :torch .Tensor )->Dict [str ,float ]:
        """
        Calculate prediction evaluation indicators

        Args:
            predictions: predicted value tensor
            targets: true value tensor

        Returns:
            Dict[str, float]: A dictionary containing various evaluation metrics
        """
        # Convert tonumpyarray
        pred_np =predictions .detach ().cpu ().numpy ().flatten ()
        target_np =targets .detach ().cpu ().numpy ().flatten ()

        # Calculate various indicators
        mse =mean_squared_error (target_np ,pred_np )
        mae =mean_absolute_error (target_np ,pred_np )
        rmse =np .sqrt (mse )

        # calculateR²Fraction，Handle possible exceptions
        try :
            r2 =r2_score (target_np ,pred_np )
        except :
            r2 =0.0 

            # calculateMAPE (Mean Absolute Percentage Error)
            # Avoid divide-by-zero errors
        mask =target_np !=0 
        if mask .sum ()>0 :
            mape =np .mean (np .abs ((target_np [mask ]-pred_np [mask ])/target_np [mask ]))*100 
        else :
            mape =0.0 

        return {
        'mse':mse ,
        'mae':mae ,
        'rmse':rmse ,
        'r2':r2 ,
        'mape':mape 
        }

    def train_epoch (self ,epoch :int )->Dict :
        """
        train a singleepoch
        
        Args:
            epoch: currentepochnumber
            
        Returns:
            Dict: Training results
        """
        self .model .train ()
        train_losses =[]
        all_predictions =[]
        all_targets =[]
        epoch_start_time =time .time ()

        # training loop
        use_amp =getattr (self .args ,'use_amp',False )if self .args is not None else False 
        if use_amp and not hasattr (self ,'scaler'):
            self .scaler =torch .cuda .amp .GradScaler ()
        elif not use_amp :
            self .scaler =None 
        for batch_data in tqdm (self .train_loader ,desc =f'Epoch {epoch } - Training'):
        # Unpack batch data
            if len (batch_data )==5 :
                batch_x ,batch_y ,batch_x_mark ,batch_y_mark ,batch_exog =batch_data 
            else :
                batch_x ,batch_y ,batch_x_mark ,batch_y_mark =batch_data 
                batch_exog =None 

                # Move to device
            batch_x =batch_x .float ().to (self .device )
            batch_y =batch_y .float ().to (self .device )
            batch_x_mark =batch_x_mark .float ().to (self .device )
            batch_y_mark =batch_y_mark .float ().to (self .device )
            if batch_exog is not None :
                batch_exog =batch_exog .float ().to (self .device )

                # Prepare decoder input
            dec_inp =torch .zeros_like (batch_y [:,-self .pred_len :,:]).float ()
            dec_inp =torch .cat ([batch_y [:,:self .label_len ,:],dec_inp ],dim =1 )

            # Forward and Backpropagation（aligned to standard cycle AMP Update logic with optimizer）
            if use_amp :
                self .optimizer .zero_grad ()
                with torch .cuda .amp .autocast ():
                    if hasattr (self .model ,'forecast'):
                        outputs ,gate_vector =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                    else :
                        outputs =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )
                    f_dim =-1 if self .features =='MS'else 0 
                    outputs =outputs [:,-self .pred_len :,f_dim :]
                    batch_y =batch_y [:,-self .pred_len :,f_dim :]
                    loss =self .criterion (outputs ,batch_y )
                    # Collect forecast and target values ​​for metric calculations
                all_predictions .append (outputs .detach ())
                all_targets .append (batch_y .detach ())
                # Backpropagation and Optimizer Stepping（AMP）
                self .scaler .scale (loss ).backward ()
                self .scaler .step (self .optimizer )
                self .scaler .update ()
            else :
                self .optimizer .zero_grad ()
                if hasattr (self .model ,'forecast'):
                    outputs ,gate_vector =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                else :
                    outputs =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )
                f_dim =-1 if self .features =='MS'else 0 
                outputs =outputs [:,-self .pred_len :,f_dim :]
                batch_y =batch_y [:,-self .pred_len :,f_dim :]
                loss =self .criterion (outputs ,batch_y )
                # Collect forecast and target values ​​for metric calculations
                all_predictions .append (outputs .detach ())
                all_targets .append (batch_y .detach ())
                # Backpropagation and Optimizer Stepping（No AMP）
                loss .backward ()
                self .optimizer .step ()

                # against TST strategy for each batch learning rate step（Alignment standard loop）
            if (self .scheduler is not None and self .args is not None and 
            getattr (self .args ,'lradj',None )=='TST'):
                adjust_learning_rate (self .optimizer ,self .scheduler ,epoch +1 ,self .args ,printout =False )
                try :
                    self .scheduler .step ()
                except Exception :
                # Guaranteed，like scheduler Ignore if this step is not supported
                    pass 

            train_losses .append (loss .item ())

            # Rule quality data collection is already done inside the model，Only temporary data needs to be cleaned here
            if (self .enable_rule_optimization and 
            hasattr (self .model ,'clear_quality_data')):
            # Clear temporary data in model，Avoid memory leaks
                self .model .clear_quality_data ()

                # Calculate average training loss
        avg_train_loss =np .mean (train_losses )
        epoch_time =time .time ()-epoch_start_time 

        # debugging information：Check quality monitor status
        if self .enable_rule_optimization and self .quality_monitor :
            self .logger .info (f"📊 Epoch {epoch +1 } Quality control status:")
            self .logger .info (f"   - Total number of batches processed: {self .quality_monitor .total_batches }")
            self .logger .info (f"   - bookepochprocess batch: {len (train_losses )}")
            self .logger .info (f"   - Whether the quality monitor is enabled: {hasattr (self .model ,'rule_quality_monitor')and self .model .rule_quality_monitor is not None }")

            # Compute training predictors
        train_metrics ={}
        if all_predictions and all_targets :
        # Combine predictions and target values ​​for all batches
            all_preds =torch .cat (all_predictions ,dim =0 )
            all_targs =torch .cat (all_targets ,dim =0 )
            train_metrics =self .calculate_prediction_metrics (all_preds ,all_targs )

        result ={
        'avg_train_loss':avg_train_loss ,
        'epoch_time':epoch_time ,
        'num_batches':len (train_losses )
        }
        result .update (train_metrics )

        return result 

    def validate_epoch (self ,epoch :int )->Dict :
        """
        Verify a singleepoch
        
        Args:
            epoch: currentepochnumber
            
        Returns:
            Dict: Verification results
        """
        self .model .eval ()
        val_losses =[]
        all_predictions =[]
        all_targets =[]

        with torch .no_grad ():
            use_amp =getattr (self .args ,'use_amp',False )if self .args is not None else False 
            for batch_data in tqdm (self .val_loader ,desc =f'Epoch {epoch } - Validation'):
            # Unpack batch data
                if len (batch_data )==5 :
                    batch_x ,batch_y ,batch_x_mark ,batch_y_mark ,batch_exog =batch_data 
                else :
                    batch_x ,batch_y ,batch_x_mark ,batch_y_mark =batch_data 
                    batch_exog =None 

                    # Move to device
                batch_x =batch_x .float ().to (self .device )
                batch_y =batch_y .float ().to (self .device )
                batch_x_mark =batch_x_mark .float ().to (self .device )
                batch_y_mark =batch_y_mark .float ().to (self .device )
                if batch_exog is not None :
                    batch_exog =batch_exog .float ().to (self .device )

                    # Prepare decoder input
                dec_inp =torch .zeros_like (batch_y [:,-self .pred_len :,:]).float ()
                dec_inp =torch .cat ([batch_y [:,:self .label_len ,:],dec_inp ],dim =1 )

                # forward propagation（Available during verification phase autocast，but no need scaler）
                if use_amp :
                    with torch .cuda .amp .autocast ():
                        if hasattr (self .model ,'forecast'):
                            outputs ,_ =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                        else :
                            outputs =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )
                        f_dim =-1 if self .features =='MS'else 0 
                        outputs =outputs [:,-self .pred_len :,f_dim :]
                        batch_y =batch_y [:,-self .pred_len :,f_dim :]
                        loss =self .criterion (outputs ,batch_y )
                else :
                    if hasattr (self .model ,'forecast'):
                        outputs ,_ =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                    else :
                        outputs =self .model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )
                    f_dim =-1 if self .features =='MS'else 0 
                    outputs =outputs [:,-self .pred_len :,f_dim :]
                    batch_y =batch_y [:,-self .pred_len :,f_dim :]
                    loss =self .criterion (outputs ,batch_y )

                    # Collect forecast and target values ​​for metric calculations
                all_predictions .append (outputs .detach ())
                all_targets .append (batch_y .detach ())

                val_losses .append (loss .item ())

        avg_val_loss =np .mean (val_losses )

        # Calculate Validation Predictors
        val_metrics ={}
        if all_predictions and all_targets :
        # Combine predictions and target values ​​for all batches
            all_preds =torch .cat (all_predictions ,dim =0 )
            all_targs =torch .cat (all_targets ,dim =0 )
            val_metrics =self .calculate_prediction_metrics (all_preds ,all_targs )

        result ={
        'avg_val_loss':avg_val_loss ,
        'num_batches':len (val_losses )
        }
        result .update (val_metrics )

        return result 

    def train (self ,num_epochs :int ,early_stopping_patience :int =10 )->Dict :
        """
        Complete training cycle，Integrated rule optimization
        
        Args:
            num_epochs: Number of training rounds
            early_stopping_patience: Early stop patience value
            
        Returns:
            Dict: Summary of training results
        """
        best_val_loss =float ('inf')
        patience_counter =0 

        start_msg =f"Start training，common{num_epochs }indivualepoch"
        self .logger .info (start_msg )
        print (start_msg )
        if self .enable_rule_optimization :
            opt_msg ="Rule optimization is enabled"
            self .logger .info (opt_msg )
            print (opt_msg )

        for epoch in range (num_epochs ):
        # 1. training phase
            train_result =self .train_epoch (epoch )

            # 2. Verification phase
            val_result =self .validate_epoch (epoch )

            # 3. Learning rate scheduling（Timing and strategies for aligning standard loops）
            if self .scheduler :
            # No TST：exist epoch Do not step by step or adjust manually
                if getattr (self .args ,'lradj',None )!='TST':
                    if getattr (self .args ,'lradj',None )=='COS':
                        try :
                            self .scheduler .step ()
                        except Exception :
                            pass 
                        try :
                            cur_lr =self .optimizer .param_groups [0 ]['lr']
                            self .logger .info (f"lr = {cur_lr :.10f}")
                            print (f"lr = {cur_lr :.10f}")
                        except Exception :
                            pass 
                    else :
                    # Other strategies pass adjust_learning_rate Manual setting
                        try :
                            if epoch ==0 :
                                self .args .learning_rate =self .optimizer .param_groups [0 ]['lr']
                                print ("lr = {:.10f}".format (self .optimizer .param_groups [0 ]['lr']))
                            adjust_learning_rate (self .optimizer ,self .scheduler ,epoch +1 ,self .args ,printout =True )
                        except Exception :
                            pass 
                else :
                # TST：batch Has stepped inside，Only update information is printed here
                    try :
                        last_lr =self .scheduler .get_last_lr ()[0 ]
                    except Exception :
                        last_lr =self .optimizer .param_groups [0 ]['lr']
                    self .logger .info (f'Updating learning rate to {last_lr }')
                    print (f'Updating learning rate to {last_lr }')

                    # 4. Record training history
            self .training_history ['train_losses'].append (train_result ['avg_train_loss'])
            self .training_history ['val_losses'].append (val_result ['avg_val_loss'])

            # 5. Print progress and forecast metrics
            log_msg =(
            f"Epoch {epoch +1 }/{num_epochs } - "
            f"Train Loss: {train_result ['avg_train_loss']:.6f}, "
            f"Val Loss: {val_result ['avg_val_loss']:.6f}, "
            f"Time: {train_result ['epoch_time']:.2f}s"
            )

            # Add training predictors
            if 'mse'in train_result :
                log_msg +=(
                f"\n  📊 Train Metrics: "
                f"MSE: {train_result ['mse']:.6f}, "
                f"MAE: {train_result ['mae']:.6f}, "
                f"RMSE: {train_result ['rmse']:.6f}, "
                f"R²: {train_result ['r2']:.4f}, "
                f"MAPE: {train_result ['mape']:.2f}%"
                )

                # Add validation predictors
            if 'mse'in val_result :
                log_msg +=(
                f"\n  📈 Val Metrics:   "
                f"MSE: {val_result ['mse']:.6f}, "
                f"MAE: {val_result ['mae']:.6f}, "
                f"RMSE: {val_result ['rmse']:.6f}, "
                f"R²: {val_result ['r2']:.4f}, "
                f"MAPE: {val_result ['mape']:.2f}%"
                )

                # Output to log and console simultaneously
            self .logger .info (log_msg )
            print (log_msg )

            # 6. Dual-track rule optimization inspection and execution
            if self .enable_rule_optimization and hasattr (self ,'integration_manager'):
            # first stage：Check if analysis needs to be run
                should_analyze ,analysis_decision =self .integration_manager .should_trigger_analysis (
                epoch ,val_result ['avg_val_loss']
                )

                # Add debugging information，Shows why analysis is not triggered
                self .logger .info (f"🔍 Dual-track analysis trigger checks - Epoch {epoch +1 }:")
                self .logger .info (f"   - currentepoch: {epoch }")
                self .logger .info (f"   - Quality monitor total number of batches: {self .quality_monitor .total_batches }")
                self .logger .info (f"   - Configuration requires minimum number of batches: {self .integration_manager .config .min_batches_for_analysis }")
                self .logger .info (f"   - Minimal configuration requirementsepochnumber: {self .integration_manager .config .min_epochs_before_analysis }")
                self .logger .info (f"   - Last analysisepoch: {self .integration_manager .last_analysis_epoch }")
                self .logger .info (f"   - Analysis interval requirements: {self .integration_manager .config .analysis_interval_epochs }")
                self .logger .info (f"   - Trigger analysis: {'yes'if should_analyze else 'no'}")
                if not should_analyze :
                    self .logger .info (f"   - Reason not triggered: {analysis_decision .get ('reasons',[])}")

                if should_analyze :
                    self .logger .info (f"🔍 Trigger first stage analysis - Epoch {epoch +1 }")
                    self .logger .info (f"   decision information: {analysis_decision ['reasons']}")

                    try :
                    # Run the first stage analysis
                        optimization_session =self .integration_manager .run_stage1_analysis (epoch )

                        # Get optimization context
                        optimization_contexts =self .integration_manager .get_optimization_contexts_for_api ()

                        if optimization_contexts :
                            self .logger .info (f"🎯 The first stage of analysis is completed，generate {len (optimization_contexts )} optimization context")

                            # Record dual-track session information
                            self .training_history ['dual_track_sessions'].append ({
                            'epoch':epoch ,
                            'session_id':optimization_session .session_id ,
                            'total_candidates':optimization_session .total_candidates ,
                            'high_priority':optimization_session .high_priority_candidates ,
                            'medium_priority':optimization_session .medium_priority_candidates ,
                            'contexts_generated':len (optimization_contexts ),
                            'status':'READY_FOR_STAGE2'
                            })

                            # Phase 2 access point：Handle optimization context
                            # debugging information
                            self .logger .info (f"🔍 debugging information: self.config = {self .config }")
                            self .logger .info (f"🔍 debugging information: enable_stage2_processing = {self .config .get ('enable_stage2_processing',False )}")

                            if self .config .get ('enable_stage2_processing',False ):
                                self .logger .info ("🚀 Start a complete end-to-end optimization process...")
                                try :
                                # Import the complete optimization module（Stages 4 to 7）
                                    from utils .dual_track_optimization .optimization_orchestrator import OptimizationOrchestrator 

                                    # Get the real quality report generated in the first stage
                                    real_quality_report =None 
                                    if hasattr (self .integration_manager ,'current_session')and self .integration_manager .current_session :
                                        real_quality_report =self .integration_manager .current_session .quality_report 
                                        self .logger .info ("✅ Obtain the real quality report generated in the first stage")
                                    else :
                                        self .logger .warning ("⚠️ Unable to obtain phase 1 quality report，Simulated data will be used")

                                        # Initialize the seventh stage optimization orchestrator
                                    orchestrator_config ={
                                    'orchestration':{
                                    'max_concurrent_optimizations':1 ,
                                    'optimization_timeout':300 ,
                                    'enable_performance_validation':True ,
                                    'enable_automatic_rollback':True ,
                                    'batch_size':len (optimization_contexts )
                                    },
                                    'thresholds':{
                                    'health_threshold':0.4 ,
                                    'effectiveness_threshold':0.3 
                                    },
                                    'validation':{
                                    'tolerance_thresholds':{
                                    'mse_degradation':0.05 ,
                                    'mae_degradation':0.03 
                                    }
                                    },
                                    'config_file':self .args .rules_list if hasattr (self .args ,'rules_list')else 'config/rule_patterns.json'
                                    }

                                    orchestrator =OptimizationOrchestrator (orchestrator_config ,self .model )

                                    # Create optimization session
                                    session_id =orchestrator .create_optimization_session (epoch =epoch )
                                    self .logger .info (f"🎯 Create optimization session: {session_id }")

                                    # Prepare model output data（For Phase 6 Performance Verification）
                                    model_outputs ={
                                    'optimization_contexts':optimization_contexts ,
                                    'quality_report':real_quality_report ,
                                    'epoch':epoch ,
                                    'validation_loss':val_result ['avg_val_loss']if 'val_result'in locals ()else 0.0 
                                    }

                                    # Run a complete end-to-end optimization process（Stages 4 to 7）
                                    self .logger .info ("🔄 Execute the complete optimization process（fourth-seven stages）...")
                                    optimization_session =orchestrator .run_full_optimization (session_id ,model_outputs )

                                    # Statistical results
                                    successful_optimizations =sum (1 for r in optimization_session .results if r .success )
                                    total_optimizations =len (optimization_session .results )

                                    self .logger .info ("✅ Complete optimization process completed")
                                    self .logger .info (f"   Total number of optimizations: {total_optimizations }")
                                    self .logger .info (f"   Successful optimization: {successful_optimizations }")
                                    self .logger .info (f"   success rate: {successful_optimizations /total_optimizations :.1%}"if total_optimizations >0 else "   success rate: 0%")
                                    self .logger .info (f"   Session length: {optimization_session .get_duration ():.2f}Second")

                                    # Record complete optimization results
                                    self .training_history ['dual_track_sessions'][-1 ].update ({
                                    'full_optimization_result':{
                                    'session_id':session_id ,
                                    'total_optimizations':total_optimizations ,
                                    'successful_optimizations':successful_optimizations ,
                                    'success_rate':successful_optimizations /total_optimizations if total_optimizations >0 else 0.0 ,
                                    'duration':optimization_session .get_duration (),
                                    'stage_breakdown':{
                                    'stage4_data_processing':True ,
                                    'stage5_rule_management':True ,
                                    'stage6_execution_monitoring':True ,
                                    'stage7_integration_optimization':True 
                                    }
                                    },
                                    'optimization_success':successful_optimizations >0 
                                    })

                                    self .integration_manager .complete_current_session (success =True )

                                except Exception as e :
                                    self .logger .error (f"❌ Complete optimization process failed: {e }")
                                    import traceback 
                                    self .logger .error (f"Error details: {traceback .format_exc ()}")
                                    self .integration_manager .complete_current_session (success =False ,error_message =str (e ))
                            else :
                            # Simulation mode：Only log optimization context
                                self .logger .info ("📌 Optimization context is ready（Simulation mode）")
                                self .integration_manager .complete_current_session (success =True )

                        else :
                            self .logger .info ("ℹ️  The first stage of analysis is completed，But no rules were found that needed optimization")
                            self .integration_manager .complete_current_session (success =True )

                    except Exception as e :
                        self .logger .error (f"❌ First stage analysis failed: {e }")
                        self .integration_manager .complete_current_session (success =False ,error_message =str (e ))



                        # Generate quality reports and session summaries regularly
                if epoch %5 ==0 :
                    if hasattr (self ,'integration_manager'):
                    # Generate integrated manager summary
                        session_summary =self .integration_manager .get_session_summary ()
                        self .training_history ['quality_reports'].append ({
                        'epoch':epoch ,
                        'session_summary':session_summary ,
                        'type':'dual_track_integration'
                        })

                        self .logger .info (f"📊 Session summary - Epoch {epoch +1 }:")
                        self .logger .info (f"   Total sessions: {session_summary ['total_sessions']}")
                        self .logger .info (f"   successful session: {session_summary ['successful_sessions']}")
                        self .logger .info (f"   failed session: {session_summary ['failed_sessions']}")
                        self .logger .info (f"   Total optimization times: {session_summary ['total_optimizations']}")



                        # 7. Early stop inspection
            if val_result ['avg_val_loss']<best_val_loss :
                best_val_loss =val_result ['avg_val_loss']
                patience_counter =0 
                # Save the best model（Aligning checkpoint paths for standard loops）
                try :
                    torch .save (self .model .state_dict (),'best_model.pth')
                except Exception :
                    pass 
                    # Sync save to run_main expected checkpoint path，Easy to load for subsequent evaluation
                if self .checkpoint_dir is not None :
                    try :
                        import os 
                        os .makedirs (self .checkpoint_dir ,exist_ok =True )
                        torch .save (self .model .state_dict (),os .path .join (self .checkpoint_dir ,'checkpoint'))
                    except Exception :
                        pass 
            else :
                patience_counter +=1 

            if patience_counter >=early_stopping_patience :
                early_stop_msg =f"Early stop trigger，existepoch {epoch +1 }"
                self .logger .info (early_stop_msg )
                print (early_stop_msg )
                break 

                # Training completion summary
        training_summary ={
        'total_epochs':epoch +1 ,
        'best_val_loss':best_val_loss ,
        'final_train_loss':self .training_history ['train_losses'][-1 ],
        'final_val_loss':self .training_history ['val_losses'][-1 ],
        'early_stopped':patience_counter >=early_stopping_patience 
        }

        # Dual-track system optimization summary
        if self .enable_rule_optimization and hasattr (self ,'integration_manager'):
            session_summary =self .integration_manager .get_session_summary ()
            training_summary ['dual_track_optimization']=session_summary 

        return training_summary 

    def get_training_report (self )->str :
        """
        Generate detailed training reports
        
        Returns:
            str: Formatted training reports
        """
        report =[]
        report .append ("📊 training report")
        report .append ("="*50 )

        # Basic training information
        if self .training_history ['train_losses']:
            report .append (f"Total number of training rounds: {len (self .training_history ['train_losses'])}")
            report .append (f"final training loss: {self .training_history ['train_losses'][-1 ]:.6f}")
            report .append (f"Final verification loss: {self .training_history ['val_losses'][-1 ]:.6f}")
            report .append (f"Best validation loss: {min (self .training_history ['val_losses']):.6f}")

            # Dual-track system optimization information
        if self .training_history ['dual_track_sessions']:
            report .append ("\n🔧 Dual-track system optimization summary:")
            report .append (f"Total sessions: {len (self .training_history ['dual_track_sessions'])}")
            successful_sessions =sum (1 for session in self .training_history ['dual_track_sessions']
            if session .get ('stage2_success',False ))
            report .append (f"Successfully optimized session: {successful_sessions }")

            total_contexts =sum (
            session .get ('contexts_generated',0 )for session in self .training_history ['dual_track_sessions']
            )
            report .append (f"Total number of generated contexts: {total_contexts }")

            # Latest quality report
        if self .training_history ['quality_reports']:
            latest_report =self .training_history ['quality_reports'][-1 ]
            report .append (f"\n📈 Latest Rules Quality Report (Epoch {latest_report ['epoch']}):")
            report .append (latest_report ['report'])

        return "\n".join (report )
