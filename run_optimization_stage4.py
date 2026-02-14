 #!/usr/bin/env python3
"""
=============================================================================
No.4stage：Complete dual-track optimization（IncludeAPIcall）run script - PythonVersion

Function：
- Enable dual-track rule optimization system
- Perform first stage analysis + second stageAPIdeal with
- Complete end-to-end rule optimization process
- Supports debugging and error handling
=============================================================================
"""

import sys 
import subprocess 
import logging 
from datetime import datetime 
from pathlib import Path 
import argparse 


def setup_logging (log_file :str ,debug :bool =False )->logging .Logger :
    """Set up logging system"""
    # Configure rootloggerto capture the logs of all submodules
    root_logger =logging .getLogger ()
    root_logger .setLevel (logging .DEBUG if debug else logging .INFO )

    # clear existinghandlers
    for handler in root_logger .handlers [:]:
        root_logger .removeHandler (handler )

        # createformatter
    formatter =logging .Formatter (
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt ='%Y-%m-%d %H:%M:%S'
    )

    # consolehandler
    console_handler =logging .StreamHandler ()
    console_handler .setFormatter (formatter )
    root_logger .addHandler (console_handler )

    # documenthandler
    file_handler =logging .FileHandler (log_file ,encoding ='utf-8')
    file_handler .setFormatter (formatter )
    root_logger .addHandler (file_handler )

    # return specificloggerfor main program
    main_logger =logging .getLogger ('stage4_runner')
    return main_logger 


def build_training_command ()->list :
    """Build training command"""
    command =[
    sys .executable ,"train.py",
    "--task_name","long_term_forecast",
    "--is_training","1",
    "--model_id","elec_dt_s4",
    "--model_comment","dt_s4_api",
    "--model","logicgate",
    "--data","ETT_exog",
    "--root_path","./dataset/electricity-price",
    "--data_path","NP.csv",
    "--features","MS",
    "--target","OT",
    "--seq_len","168",
    "--label_len","48",
    "--pred_len","24",
    "--enc_in","3",
    "--dec_in","3",
    "--c_out","1",
    "--d_model","32",
    "--n_heads","8",
    "--e_layers","2",
    "--d_layers","1",
    "--d_ff","128",
    "--dropout","0.05",
    "--embed","fixed",
    "--freq","h",
    "--train_epochs","10",
    "--batch_size","16",
    "--learning_rate","0.001",
    "--rules_list","config/rule_patterns.json",
    "--rule_config_path","config/rule_patterns.json",
    "--optimization_mode","optimization",# new parameters：Use full optimization mode
    "--optimization_config","config/optimization_config_stage4.json",
    "--rule_filter_method","basic",
    "--enable_rule_filtering",
    "--rule_percentile","75",
    "--rule_window_size","24",
    "--api_key_env","OPENAI_API_KEY",
    "--num_exog_vars","2"
    ]
    return command 


def run_training (logger :logging .Logger ,log_file :str ,debug :bool =False )->bool :
    """Run the training process"""
    logger .info ("🔄 Start complete dual-track training...")

    command =build_training_command ()

    if debug :
        logger .debug (f"execute command: {' '.join (command )}")

    try :
    # Run training command
        with open (log_file ,'a',encoding ='utf-8')as log_f :
            process =subprocess .Popen (
            command ,
            stdout =subprocess .PIPE ,
            stderr =subprocess .STDOUT ,# Willstderrredirect tostdout，Make sure to capture all logs
            universal_newlines =True ,
            bufsize =1 
            )

            # Real-time output log
            for line in iter (process .stdout .readline ,''):
                print (line ,end ='')# output to console
                log_f .write (line )# Write to log file
                log_f .flush ()# Make sure to write immediately

            process .wait ()
            return_code =process .returncode 

        if return_code ==0 :
            logger .info ("")
            logger .info ("✅ No.4Phase complete training completed！")
            logger .info (f"📊 View log: {log_file }")
            logger .info ("📈 Checkpoints are saved in: checkpoints/")
            logger .info ("")
            logger .info ("🔍 Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know.：")
            logger .info ("   - Quality control data collection ✅")
            logger .info ("   - Dual-track analysis logic ✅")
            logger .info ("   - Optimize context generation ✅")
            logger .info ("   - Candidate rule identification ✅")
            logger .info ("   - APICall and rule optimization ✅")
            logger .info ("   - End-to-end optimization closed loop ✅")
            logger .info ("")
            logger .info ("📊 View optimization statistics：")
            logger .info (f"grep -E 'Successfully replaced|Successful enhancement|failed replacement|failure enhancement' {log_file }")
            return True 
        else :
            logger .error ("")
            logger .error ("❌ No.4Phase training failed！")
            logger .error (f"📋 Please check the log file: {log_file }")
            logger .error ("🔧 Troubleshooting common problems：")
            logger .error ("   1. examineAPIIs the key valid?")
            logger .error ("   2. Check whether the network connection is normal")
            logger .error ("   3. examineAPIIs the service available?")
            logger .error ("   4. Check whether the configuration file format is correct")
            return False 

    except Exception as e :
        logger .error (f"❌ An error occurred while running training: {e }")
        if debug :
            logger .exception ("Detailed error message：")
        return False 


def main ():
    """main function"""
    parser =argparse .ArgumentParser (
    description ="No.4stage：Complete dual-track optimization（IncludeAPIcall）run script",
    formatter_class =argparse .RawDescriptionHelpFormatter 
    )
    parser .add_argument (
    "--debug",
    action ="store_true",
    help ="Enable debug mode，Show details"
    )
    parser .add_argument (
    "--log-dir",
    default ="logs/stage4",
    help ="Log directory (default: logs/stage4)"
    )

    args =parser .parse_args ()

    print ("🚀 Start the first4stage：Complete dual-track optimization（IncludeAPIcall）")
    print ("==================================================")

    # Create log directory
    log_dir =Path (args .log_dir )
    log_dir .mkdir (parents =True ,exist_ok =True )

    # Generate timestamp and log file name
    timestamp =datetime .now ().strftime ("%Y%m%d_%H%M%S")
    log_file =log_dir /f"full_dual_track_stage4_{timestamp }.log"

    # Setup log
    logger =setup_logging (str (log_file ),args .debug )

    logger .info (f"📝 log file: {log_file }")
    logger .info ("⚙️  Configuration file: config/optimization_config_stage4.json")
    logger .info ("📋 rules file: config/rule_patterns.json")

    try :
    # Run training
        success =run_training (logger ,str (log_file ),args .debug )

        if not success :
            sys .exit (1 )

    except KeyboardInterrupt :
        logger .info ("\n⏹️  User interrupted program execution")
        sys .exit (0 )
    except Exception as e :
        logger .error (f"❌ An unexpected error occurred during program execution: {e }")
        if args .debug :
            logger .exception ("Detailed error message：")
        sys .exit (1 )


if __name__ =="__main__":
    main ()
