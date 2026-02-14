"""
Rule version control system

Management of all historical versions of the rule configuration，Provide Backup、Roll back、Functions such as completeness verification。
Ensuring the security and traceability of rule replacement operations。
"""

import json 
import shutil 
import hashlib 
import logging 
from datetime import datetime 
from pathlib import Path 
from typing import Dict ,List ,Optional ,Any 
from dataclasses import dataclass ,asdict 

from .data_structures import ValidationError 


@dataclass 
class VersionInfo :
    """Version Data Class"""
    version_id :str 
    timestamp :str 
    operation_type :str # manual, replacement, enhancement, rollback
    rule_idx :Optional [int ]=None 
    backup_path :str =""
    file_hash :str =""
    description :str =""
    metadata :Dict [str ,Any ]=None 

    def __post_init__ (self ):
        if self .metadata is None :
            self .metadata ={}


@dataclass 
class RollbackResult :
    """Backroll Result Data Class"""
    success :bool 
    rollback_to :str 
    current_backup :str 
    reason :str 
    timestamp :str 
    errors :List [str ]=None 

    def __post_init__ (self ):
        if self .errors is None :
            self .errors =[]


class RuleVersionManager :
    """
    Rule version control system
    
    Manage all historical versions of rule configuration，Provide Backup、Roll back、Functions such as completeness verification。
    Ensuring the security and traceability of rule replacement operations。
    """

    def __init__ (self ,backup_dir :str ="config/backups",config_file :str ="config/rule_patterns.json"):
        self .backup_dir =Path (backup_dir )
        self .config_file =Path (config_file )
        self .version_index_path =self .backup_dir /"version_index.json"
        self .logger =logging .getLogger (__name__ )

        # Create Backup Directory
        self .backup_dir .mkdir (parents =True ,exist_ok =True )

        # Loading Version Index
        self .version_index =self ._load_version_index ()

    def create_backup (self ,operation_type :str ="manual",rule_idx :Optional [int ]=None ,
    description :str ="")->str :
        """
        Create Profile Backup
        
        Args:
            operation_type: Operation Type
            rule_idx: Rule Index（If applicable）
            description: Backup Description
            
        Returns:
            str: Backup file path
            
        Raises:
            ValidationError: When backup creation failed
        """
        try :
        # Check that profile exists
            if not self .config_file .exists ():
                raise ValidationError (f"Configuration file does not exist: {self .config_file }")

                # Generate backup filenames
            timestamp =datetime .now ().strftime ('%Y%m%d_%H%M%S_%f')[:-3 ]# Include milliseconds
            version_id =f"v_{timestamp }"
            backup_filename =f"rule_patterns_{version_id }_{operation_type }"
            if rule_idx is not None :
                backup_filename +=f"_rule{rule_idx }"
            backup_filename +=".json"

            backup_path =self .backup_dir /backup_filename 

            # Copy Profile
            shutil .copy2 (self .config_file ,backup_path )

            # Calculating Hash Files
            file_hash =self ._calculate_file_hash (backup_path )

            # Create Version Information
            version_info =VersionInfo (
            version_id =version_id ,
            timestamp =datetime .now ().isoformat (),
            operation_type =operation_type ,
            rule_idx =rule_idx ,
            backup_path =str (backup_path ),
            file_hash =file_hash ,
            description =description or f"{operation_type }Operational backup",
            metadata ={
            'original_file_size':self .config_file .stat ().st_size ,
            'backup_file_size':backup_path .stat ().st_size 
            }
            )

            # Update Index
            self .version_index .append (asdict (version_info ))
            self ._save_version_index ()

            self .logger .info (f"Backup created successfully: {backup_path }")
            return str (backup_path )

        except Exception as e :
            self .logger .error (f"Failed to create backup: {e }")
            raise ValidationError (f"Backup creation failed: {str (e )}")

    def rollback_to_version (self ,backup_path :str ,reason :str ="performance_degradation")->RollbackResult :
        """
        Roll back to the specified version
        
        Args:
            backup_path: Backup file path
            reason: Reason for Rollback
            
        Returns:
            RollbackResult: Rollback Results
        """
        try :
            backup_file =Path (backup_path )
            if not backup_file .exists ():
                return RollbackResult (
                success =False ,
                rollback_to =backup_path ,
                current_backup ="",
                reason =reason ,
                timestamp =datetime .now ().isoformat (),
                errors =[f"Backup file does not exist: {backup_path }"]
                )

                # Verify Backup File
            if not self ._validate_backup_file (backup_file ):
                return RollbackResult (
                success =False ,
                rollback_to =backup_path ,
                current_backup ="",
                reason =reason ,
                timestamp =datetime .now ().isoformat (),
                errors =[f"Backup file verification failed: {backup_path }"]
                )

                # Create backup for current status
            current_backup =self .create_backup ("pre_rollback",description =f"Backup before rollback: {reason }")

            try :
            # Execute Rollback
                shutil .copy2 (backup_file ,self .config_file )

                # Record rollback operations
                self ._log_rollback_operation (backup_path ,reason )

                self .logger .info (f"Rollback successful: {backup_path }")
                return RollbackResult (
                success =True ,
                rollback_to =backup_path ,
                current_backup =current_backup ,
                reason =reason ,
                timestamp =datetime .now ().isoformat ()
                )

            except Exception as e :
            # Rollback Failed，Try restitution.
                try :
                    shutil .copy2 (current_backup ,self .config_file )
                except :
                    pass # And if it fails,，There's been an error in the record, but there's no anomaly.

                return RollbackResult (
                success =False ,
                rollback_to =backup_path ,
                current_backup =current_backup ,
                reason =reason ,
                timestamp =datetime .now ().isoformat (),
                errors =[f"Rollback execution failed: {str (e )}"]
                )

        except Exception as e :
            self .logger .error (f"Error during rollback: {e }")
            return RollbackResult (
            success =False ,
            rollback_to =backup_path ,
            current_backup ="",
            reason =reason ,
            timestamp =datetime .now ().isoformat (),
            errors =[f"Exception during rollback process: {str (e )}"]
            )

    def get_version_history (self ,limit :int =10 )->List [Dict ]:
        """
        Get Version History
        
        Args:
            limit: Number of returned versions limited
            
        Returns:
            List[Dict]: Version History List
        """
        # Order in chronological order
        sorted_versions =sorted (
        self .version_index ,
        key =lambda x :x ['timestamp'],
        reverse =True 
        )
        return sorted_versions [:limit ]

    def find_version_by_operation (self ,operation_type :str ,rule_idx :Optional [int ]=None )->List [Dict ]:
        """
        Find version by operation type
        
        Args:
            operation_type: Operation Type
            rule_idx: Rule Index（Optional）
            
        Returns:
            List[Dict]: A list of matching versions
        """
        matches =[]
        for version in self .version_index :
            if version ['operation_type']==operation_type :
                if rule_idx is None or version .get ('rule_idx')==rule_idx :
                    matches .append (version )

                    # Order in chronological order
        return sorted (matches ,key =lambda x :x ['timestamp'],reverse =True )

    def cleanup_old_backups (self ,retention_days :int =30 )->int :
        """
        Clear old backup files
        
        Args:
            retention_days: Number of days retained
            
        Returns:
            int: Number of files cleared
        """
        from datetime import timedelta 

        cutoff_date =datetime .now ()-timedelta (days =retention_days )
        cleaned_count =0 

        # Found version to clean up
        versions_to_remove =[]
        for i ,version in enumerate (self .version_index ):
            version_date =datetime .fromisoformat (version ['timestamp'])
            if version_date <cutoff_date :
            # Remove Backup File
                backup_path =Path (version ['backup_path'])
                if backup_path .exists ():
                    try :
                        backup_path .unlink ()
                        cleaned_count +=1 
                    except Exception as e :
                        self .logger .warning (f"Failed to delete backup file: {backup_path }, {e }")

                versions_to_remove .append (i )

                # Remove from Index
        for i in reversed (versions_to_remove ):
            del self .version_index [i ]

            # Save updated index
        self ._save_version_index ()

        self .logger .info (f"Cleaned up {cleaned_count } old backup files")
        return cleaned_count 

    def _load_version_index (self )->List [Dict ]:
        """Loading Version Index"""
        if self .version_index_path .exists ():
            try :
                with open (self .version_index_path ,'r',encoding ='utf-8')as f :
                    return json .load (f )
            except Exception as e :
                self .logger .warning (f"Failed to load version index: {e }")
        return []

    def _save_version_index (self ):
        """Save Version Index"""
        try :
            with open (self .version_index_path ,'w',encoding ='utf-8')as f :
                json .dump (self .version_index ,f ,indent =2 ,ensure_ascii =False )
        except Exception as e :
            self .logger .error (f"Failed to save version index: {e }")

    def _calculate_file_hash (self ,file_path :Path )->str :
        """Calculating Hash Files"""
        hash_md5 =hashlib .md5 ()
        with open (file_path ,"rb")as f :
            for chunk in iter (lambda :f .read (4096 ),b""):
                hash_md5 .update (chunk )
        return hash_md5 .hexdigest ()

    def _validate_backup_file (self ,backup_file :Path )->bool :
        """Verify Backup File"""
        try :
        # Check if the file exists
            if not backup_file .exists ():
                return False 

                # Check if the file is validJSON
            with open (backup_file ,'r',encoding ='utf-8')as f :
                json .load (f )

                # Find corresponding version information
            backup_path_str =str (backup_file )
            version_info =None 
            for version in self .version_index :
                if version ['backup_path']==backup_path_str :
                    version_info =version 
                    break 

                    # If version information is found，Verify Document Hash
            if version_info and version_info .get ('file_hash'):
                current_hash =self ._calculate_file_hash (backup_file )
                return current_hash ==version_info ['file_hash']

            return True # Without Hashi,，Consider the document valid

        except Exception as e :
            self .logger .error (f"Verification of backup file failed: {e }")
            return False 

    def _log_rollback_operation (self ,backup_path :str ,reason :str ):
        """Record rollback operations"""
        rollback_info ={
        'timestamp':datetime .now ().isoformat (),
        'operation':'rollback',
        'backup_path':backup_path ,
        'reason':reason 
        }

        # Could be expanded to write a specific rollback log file
        self .logger .info (f"Rollback operation record: {rollback_info }")

    def get_stats (self )->Dict [str ,Any ]:
        """Access to version management statistical information"""
        total_versions =len (self .version_index )
        operation_counts ={}

        for version in self .version_index :
            op_type =version ['operation_type']
            operation_counts [op_type ]=operation_counts .get (op_type ,0 )+1 

            # Calculate Backup Directory Size
        total_size =0 
        for backup_file in self .backup_dir .glob ("*.json"):
            total_size +=backup_file .stat ().st_size 

        return {
        'total_versions':total_versions ,
        'operation_counts':operation_counts ,
        'backup_directory_size':total_size ,
        'backup_directory':str (self .backup_dir ),
        'latest_version':self .version_index [-1 ]if self .version_index else None 
        }
