import argparse 
import os 
import torch 
from torch import nn ,optim 
from torch .optim import lr_scheduler 
from tqdm import tqdm 
import logging 
import json 

import models 

from data_provider .data_factory import data_provider 
import time 
import random 
import numpy as np 

# load .env document
try :
    from dotenv import load_dotenv 
    load_dotenv ()
    print ("✅ successfully loaded .env document")
except ImportError :
    print ("⚠️  python-dotenv Not installed，Try loading manually .env document")
    # Manual loading .env document
    try :
        with open ('.env','r',encoding ='utf-8')as f :
            for line in f :
                line =line .strip ()
                if line and not line .startswith ('#')and '='in line :
                    key ,value =line .split ('=',1 )
                    # Remove possible quotes
                    value =value .strip ().strip ('"').strip ("'")
                    # Only set if the environment variable does not exist（avoid overwritingVSCodeset environment variables）
                    if key .strip ()not in os .environ :
                        os .environ [key .strip ()]=value 
        print ("✅ Manual loading .env File successful")
    except FileNotFoundError :
        print ("⚠️  .env file not found")
    except Exception as e :
        print (f"⚠️  load .env File failed: {e }")

os .environ ['CURL_CA_BUNDLE']=''
os .environ ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:64"

# Configure the logging system - Ensure that the logs in the child process can be output normally
logging .basicConfig (
level =logging .INFO ,
format ='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
datefmt ='%Y-%m-%d %H:%M:%S',
handlers =[
logging .StreamHandler (),# output to console
]
)

from utils .tools import EarlyStopping ,adjust_learning_rate ,vali ,load_content 

parser =argparse .ArgumentParser (description ='Time-LLM')

fix_seed =2021 
random .seed (fix_seed )
torch .manual_seed (fix_seed )
np .random .seed (fix_seed )

# Deterministic training settings
# Set cuDNN deterministic behavior and disable benchmarking to avoid nondeterministic algo selection
torch .backends .cudnn .deterministic =True 
torch .backends .cudnn .benchmark =False 

# Some CUDA kernels require a specific cuBLAS workspace config to be deterministic.
# This must be set before CUDA libraries are used.
os .environ .setdefault ('CUBLAS_WORKSPACE_CONFIG',':4096:8')

try :
# Enforce deterministic algorithms globally. PyTorch will error if a nondeterministic op is used.
    torch .use_deterministic_algorithms (True )
except Exception as _e :
# Fallback: keep training running even if some backends are unavailable; determinism may be partial.
    print (f"⚠️ Unable to enforce full deterministic algorithms: {_e }")

    # basic config
parser .add_argument ('--task_name',type =str ,required =True ,default ='long_term_forecast',
help ='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser .add_argument ('--is_training',type =int ,required =True ,default =1 ,help ='status')
parser .add_argument ('--model_id',type =str ,required =True ,default ='test',help ='model id')
parser .add_argument ('--model_comment',type =str ,required =True ,default ='none',help ='prefix when saving test results')
parser .add_argument ('--model',type =str ,required =True ,default ='logicgate',
choices =['logicgate','RuleGatingTIMELLM','TimeLLM'],
help ='model name, options: [logicgate, TimeLLM] (RuleGatingTIMELLM is legacy alias)')
parser .add_argument ('--seed',type =int ,default =2021 ,help ='random seed')

# rule optimization configuration - new simplified approach
parser .add_argument ('--optimization_mode',type =str ,
choices =['baseline','monitoring','optimization'],
default ='baseline',
help ='''Rule optimization mode:
                   baseline: Pure training without any optimization overhead
                   monitoring: Enable quality monitoring but no rule optimization  
                   optimization: Full rule optimization with API processing''')

# legacy parameters for backward compatibility (deprecated)
parser .add_argument ('--enable_dual_track',action ='store_true',default =False ,
help ='[DEPRECATED] Use --optimization_mode instead')
parser .add_argument ('--enable_stage2_processing',action ='store_true',default =False ,
help ='[DEPRECATED] Use --optimization_mode instead')
parser .add_argument ('--skip_rule_optimization',action ='store_true',default =False ,
help ='[DEPRECATED] Use --optimization_mode instead')

# supporting configuration files
parser .add_argument ('--optimization_config',type =str ,default ='config/optimization_config.json',
help ='path to optimization configuration file')
parser .add_argument ('--rules_list',type =str ,default ='config/rule_patterns.json',
help ='path to rules list file')
parser .add_argument ('--rule_config_path',type =str ,default ='config/rule_patterns.json',
help ='path to rule configuration file for similarity filtering')
parser .add_argument ('--api_key_env',type =str ,default ='OPENAI_API_KEY',
help ='environment variable name for API key')

# data loader
parser .add_argument ('--data',type =str ,required =True ,default ='ETTm1',help ='dataset type')
parser .add_argument ('--root_path',type =str ,default ='./dataset',help ='root path of the data file')
parser .add_argument ('--data_path',type =str ,default ='ETTh1.csv',help ='data file')
parser .add_argument ('--features',type =str ,default ='M',
help ='forecasting task, options:[M, S, MS]; '
'M:multivariate predict multivariate, S: univariate predict univariate, '
'MS:multivariate predict univariate')
parser .add_argument ('--target',type =str ,default ='target',help ='target variable column name')
parser .add_argument ('--loader',type =str ,default ='modal',help ='dataset type')
parser .add_argument ('--freq',type =str ,default ='h',
help ='freq for time features encoding, '
'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
'you can also use more detailed freq like 15min or 3h')
parser .add_argument ('--checkpoints',type =str ,default ='./checkpoints/',help ='location of model checkpoints')

# forecasting task
parser .add_argument ('--seq_len',type =int ,default =96 ,help ='input sequence length')
parser .add_argument ('--label_len',type =int ,default =48 ,help ='start token length')
parser .add_argument ('--pred_len',type =int ,default =96 ,help ='prediction sequence length')
parser .add_argument ('--seasonal_patterns',type =str ,default ='Monthly',help ='subset for M4')

# model define
parser .add_argument ('--enc_in',type =int ,default =7 ,help ='encoder input size')
parser .add_argument ('--dec_in',type =int ,default =7 ,help ='decoder input size')
parser .add_argument ('--c_out',type =int ,default =7 ,help ='output size')
parser .add_argument ('--d_model',type =int ,default =16 ,help ='dimension of model')
parser .add_argument ('--n_heads',type =int ,default =8 ,help ='num of heads')
parser .add_argument ('--e_layers',type =int ,default =2 ,help ='num of encoder layers')
parser .add_argument ('--d_layers',type =int ,default =1 ,help ='num of decoder layers')
parser .add_argument ('--d_ff',type =int ,default =32 ,help ='dimension of fcn')
parser .add_argument ('--moving_avg',type =int ,default =25 ,help ='window size of moving average')
parser .add_argument ('--factor',type =int ,default =1 ,help ='attn factor')
parser .add_argument ('--dropout',type =float ,default =0.1 ,help ='dropout')
parser .add_argument ('--embed',type =str ,default ='timeF',
help ='time features encoding, options:[timeF, fixed, learned]')
parser .add_argument ('--activation',type =str ,default ='gelu',help ='activation')
parser .add_argument ('--output_attention',action ='store_true',help ='whether to output attention in encoder')
parser .add_argument ('--patch_len',type =int ,default =16 ,help ='patch length')
parser .add_argument ('--stride',type =int ,default =8 ,help ='stride')
parser .add_argument ('--prompt_domain',type =int ,default =0 ,help ='')
parser .add_argument ('--llm_model',type =str ,default ='BERT',help ='LLM model')# LLAMA, GPT2, BERT
parser .add_argument ('--llm_dim',type =int ,default ='768',help ='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser .add_argument ('--scale',action ='store_true',default =True ,help ='whether to scale the data')
parser .add_argument ('--num_exog_vars',type =int ,default =0 ,help ='number of exogenous variables')
parser .add_argument ('--rules_content',type =str ,nargs ='+',default =[],help ='list of rules for logicgate')


# Rule filter related parameters
parser .add_argument ('--enable_rule_filtering',action ='store_true',default =True ,help ='enable rule similarity filtering')
parser .add_argument ('--rule_filter_method',type =str ,default ='fast',choices =['basic','fast'],
help ='rule filtering method: basic (accurate) or fast (optimized)')
parser .add_argument ('--rule_use_cache',action ='store_true',default =True ,help ='use cache in rule filtering')
parser .add_argument ('--rule_simplified',action ='store_true',default =False ,help ='use simplified mode (only first batch)')
parser .add_argument ('--rule_percentile',type =int ,default =75 ,help ='percentile for dynamic threshold')
parser .add_argument ('--rule_window_size',type =int ,default =24 ,help ='window size for trend analysis')
parser .add_argument ('--rule_debug',action ='store_true',default =False ,help ='enable debug output for rule filtering')

# optimization
parser .add_argument ('--num_workers',type =int ,default =0 ,help ='data loader num workers')
parser .add_argument ('--itr',type =int ,default =1 ,help ='experiments times')
parser .add_argument ('--train_epochs',type =int ,default =10 ,help ='train epochs')
parser .add_argument ('--align_epochs',type =int ,default =10 ,help ='alignment epochs')
parser .add_argument ('--batch_size',type =int ,default =32 ,help ='batch size of train input data')
parser .add_argument ('--eval_batch_size',type =int ,default =8 ,help ='batch size of model evaluation')
parser .add_argument ('--patience',type =int ,default =10 ,help ='early stopping patience')
parser .add_argument ('--learning_rate',type =float ,default =0.0001 ,help ='optimizer learning rate')
parser .add_argument ('--des',type =str ,default ='test',help ='exp description')
parser .add_argument ('--loss',type =str ,default ='MSE',help ='loss function')
parser .add_argument ('--lradj',type =str ,default ='type1',help ='adjust learning rate')
parser .add_argument ('--pct_start',type =float ,default =0.2 ,help ='pct_start')
parser .add_argument ('--use_amp',action ='store_true',help ='use automatic mixed precision training',default =False )
parser .add_argument ('--llm_layers',type =int ,default =6 )
parser .add_argument ('--percent',type =int ,default =100 )

args =parser .parse_args ()

if args .model =='RuleGatingTIMELLM':
    print ("⚠️  Model name RuleGatingTIMELLM Renamed to logicgate，Automatic compatibility conversion")
    args .model ='logicgate'


def is_logicgate_model (model_name :str )->bool :
    return model_name in ('logicgate','RuleGatingTIMELLM')

    # Parameter validation and legacy conversion
def validate_and_convert_optimization_params (args ):
    """Validate optimization parameters and convert legacy parameters"""

    # Check for legacy parameter usage and auto-convert
    legacy_used =any ([
    args .enable_dual_track ,
    args .enable_stage2_processing ,
    args .skip_rule_optimization 
    ])

    if legacy_used :
        print ("⚠️  WARNING: You are using deprecated parameters. Please use --optimization_mode instead.")
        print ("   Deprecated: --enable_dual_track, --enable_stage2_processing, --skip_rule_optimization")
        print ("   New: --optimization_mode [baseline|monitoring|optimization]")

        # Auto-convert legacy parameters to new mode
        if args .skip_rule_optimization :
            args .optimization_mode ='baseline'
            print (f"   Auto-converted to: --optimization_mode {args .optimization_mode }")
        elif args .enable_dual_track and args .enable_stage2_processing :
            args .optimization_mode ='optimization'
            print (f"   Auto-converted to: --optimization_mode {args .optimization_mode }")
        elif args .enable_dual_track and not args .enable_stage2_processing :
            args .optimization_mode ='monitoring'
            print (f"   Auto-converted to: --optimization_mode {args .optimization_mode }")
        else :
            args .optimization_mode ='baseline'
            print (f"   Auto-converted to: --optimization_mode {args .optimization_mode }")
        print ()

        # Validate API key for optimization mode
    if args .optimization_mode =='optimization':
        import os 
        api_key =os .getenv (args .api_key_env )
        if not api_key or api_key =='your_qwen_api_key_here':
            print (f"❌ ERROR: optimization mode requires a valid API key in {args .api_key_env }")
            print ("   Please set your API key or use 'monitoring' mode instead.")
            exit (1 )

            # Set internal flags based on mode for backward compatibility
    if args .optimization_mode =='baseline':
        args .enable_dual_track =False 
        args .enable_stage2_processing =False 
        args .skip_rule_optimization =True 
    elif args .optimization_mode =='monitoring':
        args .enable_dual_track =True 
        args .enable_stage2_processing =False 
        args .skip_rule_optimization =False 
    elif args .optimization_mode =='optimization':
        args .enable_dual_track =True 
        args .enable_stage2_processing =True 
        args .skip_rule_optimization =False 

    return args 

args =validate_and_convert_optimization_params (args )

# Replaced with simple device settings
device =torch .device ('cuda'if torch .cuda .is_available ()else 'cpu')

if __name__ =='__main__':
    for ii in range (args .itr ):
    # setting record of experiments
        setting ='{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format (
        args .task_name ,
        args .model_id ,
        args .model ,
        args .data ,
        args .features ,
        args .seq_len ,
        args .label_len ,
        args .pred_len ,
        args .d_model ,
        args .n_heads ,
        args .e_layers ,
        args .d_layers ,
        args .d_ff ,
        args .factor ,
        args .embed ,
        args .des ,ii )

        train_data ,train_loader =data_provider (args ,'train')
        vali_data ,vali_loader =data_provider (args ,'val')
        test_data ,test_loader =data_provider (args ,'test')

        if is_logicgate_model (args .model ):
        # Load rules from file (unified approach)
            rules_for_model =[]
            if args .rules_list and os .path .exists (args .rules_list ):
                try :
                    with open (args .rules_list ,'r',encoding ='utf-8')as f :
                        rules_config =json .load (f )
                        # Extract rule text list
                    for rule_id ,rule_data in rules_config .get ('rule_patterns',{}).items ():
                    # Prefer description field as rule text
                        rule_text =rule_data .get ('description',rule_data .get ('rule_text',''))
                        if rule_text :
                            rules_for_model .append (rule_text )
                    print(f"✅ Loaded {len(rules_for_model)} rules from {args.rules_list}")
                except Exception as e :
                    print(f"❌ Failed to load rules file: {e}")
                    rules_for_model =args .rules_content if args .rules_content else []
            else :
            # Fallback to direct rule content
                rules_for_model =args .rules_content if args .rules_content else []

            model =models .logicgate (args ,rules_for_model ).float ()
        else :# TimeLLM
            model =models .TimeLLM (args ).float ()

        path =os .path .join (args .checkpoints ,
        setting +'-'+args .model_comment )# unique checkpoint saving path
        args .content =load_content (args )
        if not os .path .exists (path ):
            os .makedirs (path )

        time_now =time .time ()

        train_steps =len (train_loader )
        early_stopping =EarlyStopping (patience =args .patience )

        trained_parameters =[]
        for p in model .parameters ():
            if p .requires_grad is True :
                trained_parameters .append (p )

        model_optim =optim .Adam (trained_parameters ,lr =args .learning_rate )

        if args .lradj =='COS':
            scheduler =torch .optim .lr_scheduler .CosineAnnealingLR (model_optim ,T_max =20 ,eta_min =1e-8 )
        else :
            scheduler =lr_scheduler .OneCycleLR (optimizer =model_optim ,
            steps_per_epoch =train_steps ,
            pct_start =args .pct_start ,
            epochs =args .train_epochs ,
            max_lr =args .learning_rate )

        criterion =nn .MSELoss ()
        mae_metric =nn .L1Loss ()

        model =model .to (device )

        # Rule optimization mode selection - simplified logic
        enhanced_trainer =None 

        if is_logicgate_model (args .model ):
            print (f"🎯 Rule optimization mode: {args .optimization_mode }")

            if args .optimization_mode =='baseline':
                print ("📊 baseline mode: pure training，No optimization overhead")
                print ("   - Quality control: Disabled")
                print ("   - training loop: Standard cycle")
                print ("   - Execution path: Standard training loop will be used（No.415start of line）")
                # enhanced_trainer remains None, will use standard training loop

            elif args .optimization_mode =='monitoring':
                print ("📈 Monitor mode: Enable quality monitoring，Optimization without rules")
                print ("   - Quality control: Enabled")
                print ("   - training loop: Standard cycle")
                print ("   - Data cleaning: Everybatchimplement")
                print ("   - quality report: Every5epochsgenerate")
                print ("   - Execution path: Standard training loop will be used（No.415start of line）")
                # Use standard training loop with quality monitoring
                # Enhanced trainer not needed for monitoring

            elif args .optimization_mode =='optimization':
                print ("🚀 Optimization mode: Complete rule optimization process")

                # Import required modules
                from utils .enhanced_training_loop import EnhancedTrainingLoop 
                # Load optimization configuration
                optimization_config ={}
                if args .optimization_config and os .path .exists (args .optimization_config ):
                    try :
                        with open (args .optimization_config ,'r',encoding ='utf-8')as f :
                            optimization_config =json .load (f )
                        print (f"✅ from {args .optimization_config } Load optimization configuration")
                    except Exception as e :
                        print (f"⚠️ Failed to load optimization configuration: {e }")
                        optimization_config ={}

                        # Create enhanced trainer for full optimization
                enhanced_trainer =EnhancedTrainingLoop (
                model =model ,
                train_loader =train_loader ,
                val_loader =vali_loader ,
                optimizer =model_optim ,
                criterion =criterion ,
                device =device ,
                scheduler =scheduler ,
                enable_rule_optimization =True ,
                optimization_config =optimization_config ,
                config ={
                'enable_stage2_processing':True ,# Always True for optimization mode
                'stage2_config':{
                'api_key_env':args .api_key_env ,
                'max_retries':3 ,
                'timeout':30 
                }
                },
                args =args ,
                checkpoint_dir =path 
                )

                print ("✅ Complete optimization system initialization completed")
                print ("   - Quality control: enable")
                print ("   - first stage analysis: enable")
                print ("   - second stageAPIdeal with: enable")
                print (f"   - Configuration file: {args .optimization_config }")
                print (f"   - rules file: {args .rules_list }")
        else :
            print ("📌 NologicgateModel，Skip rule optimization settings")

            # Choose training method
        if enhanced_trainer is not None :
            print ("🔄 Use plyometric training loops（Integrated dual-track optimization）")
            # Use plyometric training loops
            training_result =enhanced_trainer .train (
            num_epochs =args .train_epochs ,
            early_stopping_patience =args .patience 
            )

            print ("📊 Training completion summary:")
            print (f"   - total rounds: {training_result ['total_epochs']}")
            print (f"   - Best validation loss: {training_result ['best_val_loss']:.6f}")
            print (f"   - final training loss: {training_result ['final_train_loss']:.6f}")
            print (f"   - Stop early: {'yes'if training_result ['early_stopped']else 'no'}")

            if 'rule_optimizations'in enhanced_trainer .training_history :
                opt_count =len (enhanced_trainer .training_history ['rule_optimizations'])
                print (f"   - Rule optimization times: {opt_count }")

                # After the plyometric training cycle is completed，Skip standard training loop
            continue 

            # Use standard training loop
        print ("🔄 Use standard training loop")

        # Configure quality monitoring for monitoring mode
        if args .optimization_mode =='monitoring'and is_logicgate_model (args .model ):
            print ("📊 Enable quality monitoring data collection")
            print ("🔄 will proceed {} indivualepochtraining（Includes quality control）".format (args .train_epochs ))
        elif args .optimization_mode =='baseline':
            print ("🎯 Baseline mode training")
            print ("🔄 will proceed {} indivualepochpure training".format (args .train_epochs ))
        else :
            print ("🔄 will proceed {} indivualepochtraining".format (args .train_epochs ))

        if args .use_amp :
            scaler =torch .cuda .amp .GradScaler ()

        for epoch in range (args .train_epochs ):
            iter_count =0 
            train_loss =[]

            model .train ()
            epoch_time =time .time ()
            # Use a unified loop structure
            for i ,batch_data in tqdm (enumerate (train_loader )):
            # Handle different data loader outputs based on model type
                if is_logicgate_model (args .model )and len (batch_data )==5 :
                    batch_x ,batch_y ,batch_x_mark ,batch_y_mark ,batch_exog =batch_data 
                    batch_exog =batch_exog .float ().to (device )
                else :
                    batch_x ,batch_y ,batch_x_mark ,batch_y_mark =batch_data 
                    batch_exog =None # No exogenous variables for standard models

                iter_count +=1 
                model_optim .zero_grad ()

                batch_x =batch_x .float ().to (device )
                batch_y =batch_y .float ().to (device )
                batch_x_mark =batch_x_mark .float ().to (device )
                batch_y_mark =batch_y_mark .float ().to (device )

                # decoder input
                dec_inp =torch .zeros_like (batch_y [:,-args .pred_len :,:]).float ().to (
                device )
                dec_inp =torch .cat ([batch_y [:,:args .label_len ,:],dec_inp ],dim =1 ).float ().to (
                device )

                # encoder - decoder
                if args .use_amp :
                    with torch .cuda .amp .autocast ():
                        if args .output_attention :
                            if is_logicgate_model (args .model ):
                                outputs ,gate_vector =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                            else :
                                outputs =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )
                        else :
                            outputs =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )

                        f_dim =-1 if args .features =='MS'else 0 
                        outputs =outputs [:,-args .pred_len :,f_dim :]
                        batch_y =batch_y [:,-args .pred_len :,f_dim :].to (device )
                        loss =criterion (outputs ,batch_y )
                        train_loss .append (loss .item ())
                else :
                    if args .output_attention :
                        if is_logicgate_model (args .model ):
                            outputs ,gate_vector =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                        else :
                            outputs =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )
                    else :
                        if is_logicgate_model (args .model ):
                            outputs ,gate_vector =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                        else :
                            outputs =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )

                    f_dim =-1 if args .features =='MS'else 0 
                    outputs =outputs [:,-args .pred_len :,f_dim :]
                    batch_y =batch_y [:,-args .pred_len :,f_dim :]
                    loss =criterion (outputs ,batch_y )
                    train_loss .append (loss .item ())

                if (i +1 )%100 ==0 :
                    print (
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format (i +1 ,epoch +1 ,loss .item ()))
                    speed =(time .time ()-time_now )/iter_count 
                    left_time =speed *((args .train_epochs -epoch )*train_steps -i )
                    print ('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format (speed ,left_time ))
                    iter_count =0 
                    time_now =time .time ()

                if args .use_amp :
                    scaler .scale (loss ).backward ()
                    scaler .step (model_optim )
                    scaler .update ()
                else :
                    loss .backward ()
                    model_optim .step ()

                    # Quality monitoring data cleanup for monitoring mode
                if (args .optimization_mode =='monitoring'and 
                is_logicgate_model (args .model )and 
                hasattr (model ,'clear_quality_data')):
                    model .clear_quality_data ()

                if args .lradj =='TST':
                    adjust_learning_rate (model_optim ,scheduler ,epoch +1 ,args ,printout =False )
                    scheduler .step ()

            print ("Epoch: {} cost time: {}".format (epoch +1 ,time .time ()-epoch_time ))
            train_loss =np .average (train_loss )
            vali_loss ,vali_mae_loss =vali (args ,model ,vali_data ,vali_loader ,criterion ,mae_metric )
            test_loss ,test_mae_loss =vali (args ,model ,test_data ,test_loader ,criterion ,mae_metric )
            print (
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format (
            epoch +1 ,train_loss ,vali_loss ,test_loss ,test_mae_loss ))

            # Generate quality monitoring report for monitoring mode
            if (args .optimization_mode =='monitoring'and 
            is_logicgate_model (args .model )and 
            hasattr (model ,'rule_quality_monitor')and 
            model .rule_quality_monitor is not None ):
                try :
                # Generate basic quality report every 5 epochs
                    if (epoch +1 )%5 ==0 :
                        total_batches =model .rule_quality_monitor .total_batches 
                        print (f"📊 Epoch {epoch +1 } Quality Control Summary:")
                        print (f"   - Total number of batches processed: {total_batches }")
                        if total_batches >0 :
                            library_health =model .rule_quality_monitor .calculate_library_health ()
                            avg_health =library_health .mean ().item ()
                            print (f"   - average rule health: {avg_health :.3f}")
                except Exception :
                # Silent error handling to avoid disrupting training
                    pass 

            early_stopping (vali_loss ,model ,path )
            if early_stopping .early_stop :
                print ("Early stopping")
                break 

            if args .lradj !='TST':
                if args .lradj =='COS':
                    scheduler .step ()
                    print ("lr = {:.10f}".format (model_optim .param_groups [0 ]['lr']))
                else :
                    if epoch ==0 :
                        args .learning_rate =model_optim .param_groups [0 ]['lr']
                        print ("lr = {:.10f}".format (model_optim .param_groups [0 ]['lr']))
                    adjust_learning_rate (model_optim ,scheduler ,epoch +1 ,args ,printout =True )

            else :
                print ('Updating learning rate to {}'.format (scheduler .get_last_lr ()[0 ]))

                # Load the best model for testing
    print ('Load the best model for testing...')
    best_model_path =path +'/'+'checkpoint'
    model .load_state_dict (torch .load (best_model_path ))
    model .eval ()

    print ('Evaluate on the test set...')
    from utils .metrics import metric 
    test_pred =[]
    test_true =[]

    with torch .no_grad ():
        for i ,batch_data in enumerate (test_loader ):
        # Handle different data loader outputs based on model type
            if is_logicgate_model (args .model )and len (batch_data )==5 :
                batch_x ,batch_y ,batch_x_mark ,batch_y_mark ,batch_exog =batch_data 
                batch_exog =batch_exog .float ().to (device )
            else :
                batch_x ,batch_y ,batch_x_mark ,batch_y_mark =batch_data 
                batch_exog =None # No exogenous variables for standard models

            batch_x =batch_x .float ().to (device )
            batch_y =batch_y .float ().to (device )
            batch_x_mark =batch_x_mark .float ().to (device )
            batch_y_mark =batch_y_mark .float ().to (device )

            # decoder input
            dec_inp =torch .zeros_like (batch_y [:,-args .pred_len :,:]).float ()
            dec_inp =torch .cat ([batch_y [:,:args .label_len ,:],dec_inp ],dim =1 ).float ().to (device )

            # encoder - decoder
            if args .output_attention :
                if is_logicgate_model (args .model ):
                    outputs ,gate_vector =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                else :
                    outputs =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )[0 ]
            else :
                if is_logicgate_model (args .model ):
                    outputs ,gate_vector =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark ,batch_exog )
                else :
                    outputs =model (batch_x ,batch_x_mark ,dec_inp ,batch_y_mark )

            f_dim =-1 if args .features =='MS'else 0 
            outputs =outputs [:,-args .pred_len :,f_dim :]
            batch_y =batch_y [:,-args .pred_len :,f_dim :]

            pred =outputs .detach ().cpu ().numpy ()
            true =batch_y .detach ().cpu ().numpy ()

            test_pred .append (pred )
            test_true .append (true )

    test_pred =np .concatenate (test_pred ,axis =0 )
    test_true =np .concatenate (test_true ,axis =0 )

    # Calculate various indicators
    mae ,mse ,rmse ,mape ,mspe =metric (test_pred ,test_true )

    print ('Test set evaluation results：')
    print (f'MAE:\t{mae :.4f}')
    print (f'MSE:\t{mse :.4f}')
    print (f'RMSE:\t{rmse :.4f}')
    print (f'MAPE:\t{mape :.4f}')
    print (f'MSPE:\t{mspe :.4f}')

    # path = './checkpoints'  # unique checkpoint saving path
    # del_files(path)  # delete checkpoint files
    # print('success delete checkpoints')
