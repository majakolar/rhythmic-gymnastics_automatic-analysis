"""
Label consolidation utilities for handling rare subcategories in hierarchical classification.
"""
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import os
import pickle
from pathlib import Path
from rg_ai.utils.utils_annotations import get_label_from_filename


class LabelConsolidator:
    """Handles consolidation of rare sublabels into more general categories."""
    
    def __init__(self, min_samples_threshold: int = 10):
        """
        Initialize the label consolidator.
        
        Args:
            min_samples_threshold: Minimum number of samples required to keep a sublabel separate
        """
        self.min_samples_threshold = min_samples_threshold
        
    def analyze_sublabel_distribution(self, data_folder: str, joined_label_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
        """
        Analyze the distribution of sublabels in the dataset.
        
        Args:
            data_folder: Path to folder containing embedding files
            joined_label_dict: Current sublabel structure
            
        Returns:
            Dict mapping main_label -> sublabel -> count
        """
        distribution = defaultdict(lambda: defaultdict(int))
        
        # get all pickle files
        files = [f for f in os.listdir(data_folder) if f.endswith(".pkl")]
        
        for file in files:
            main_label = file.split("_")[0]
            if main_label not in joined_label_dict:
                continue
                
            # find which sublabel this file belongs to
            sublabel = self._extract_sublabel_from_filename(file, main_label, joined_label_dict[main_label])
            if sublabel:
                distribution[main_label][sublabel] += 1
                
        return dict(distribution)
    
    def _extract_sublabel_from_filename(self, filename: str, main_label: str, sublabels: List[str]) -> str:
        """Extract sublabel from filename with improved matching."""
        # result = get_label_from_filename(
        #     filename=filename,
        #     main_label_index=0,
        #     use_sublabels=True,
        #     joined_label_dict={main_label: sublabels},
        #     fallback_general=True
        # )
        # return result

        # sublabels by length (longest first) to avoid partial matches
        sorted_sublabels = sorted(sublabels, key=len, reverse=True)
        
        for sublabel in sorted_sublabels:
            if sublabel == "GENERAL":
                continue  
            if f"{main_label}_{sublabel}_person" in filename:
                return sublabel
        
        return "GENERAL" 

    
    def create_consolidation_mapping(self, 
                                   distribution: Dict[str, Dict[str, int]], 
                                   consolidation_rules: Dict[str, Dict[str, List[str]]] = None) -> Dict[str, Dict[str, str]]:
        """
        Create mapping for consolidating rare sublabels.
        
        Args:
            distribution: Sublabel distribution from analyze_sublabel_distribution
            consolidation_rules: Optional manual rules for consolidation
                Format: {main_label: {target_sublabel: [source_sublabels]}}
                Example: {"Jump": {"GENERAL": ["Stag_Leap", "Split_Leap"]}}
                
        Returns:
            Dict mapping main_label -> original_sublabel -> consolidated_sublabel
        """
        consolidation_map = defaultdict(dict)
        
        for main_label, sublabel_counts in distribution.items():
            if consolidation_rules and main_label in consolidation_rules:
                for target_sublabel, source_sublabels in consolidation_rules[main_label].items():
                    for source_sublabel in source_sublabels:
                        consolidation_map[main_label][source_sublabel] = target_sublabel
            
            for sublabel, count in sublabel_counts.items():
                if sublabel not in consolidation_map[main_label]:  
                    if count < self.min_samples_threshold:
                        consolidation_map[main_label][sublabel] = "GENERAL"
                    else:
                        consolidation_map[main_label][sublabel] = sublabel  
                        
        return dict(consolidation_map)
    
    def apply_consolidation(self, 
                          original_joined_dict: Dict[str, List[str]], 
                          consolidation_map: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Apply consolidation mapping to create new joined_label_dict.
        
        Args:
            original_joined_dict: Original sublabel structure
            consolidation_map: Mapping from create_consolidation_mapping
            
        Returns:
            New joined_label_dict with consolidated sublabels
        """
        consolidated_dict = {}
        
        for main_label, original_sublabels in original_joined_dict.items():
            if main_label in consolidation_map:
                consolidated_sublabels = set()
                for original_sublabel in original_sublabels:
                    consolidated_sublabel = consolidation_map[main_label].get(original_sublabel, original_sublabel)
                    consolidated_sublabels.add(consolidated_sublabel)
                
                consolidated_dict[main_label] = sorted(list(consolidated_sublabels))
            else:
                consolidated_dict[main_label] = original_sublabels
                
        return consolidated_dict
    
    def create_sublabel_dict(self, joined_label_dict: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Create flat dictionary mapping each sublabel to unique index.
        Improved version of the existing function.
        """
        sublabel_dict = {}
        idx = 0
        
        for main_label, sublabels in joined_label_dict.items():
            if not sublabels:
                sublabel_dict[f"{main_label}"] = idx
                idx += 1
            else:
                for sublabel in sublabels:
                    if sublabel == "GENERAL":
                        # Use main label for general category
                        #sublabel_dict[main_label] = idx
                        sublabel_dict[f"{main_label}"] = idx #_{sublabel}

                    else:
                            # Use full sublabel name
                        sublabel_dict[f"{main_label}_{sublabel}"] = idx
                    idx += 1
                    
        return sublabel_dict
    
    def generate_consolidation_report(self, 
                                    original_distribution: Dict[str, Dict[str, int]], 
                                    consolidation_map: Dict[str, Dict[str, str]]) -> str:
        """Generate a human-readable report of the consolidation."""
        report = ["Label Consolidation Report", "=" * 50, ""]
        
        for main_label in sorted(original_distribution.keys()):
            report.append(f"{main_label}:")
            
            if main_label in consolidation_map:
                # group by target sublabel
                target_groups = defaultdict(list)
                for orig, target in consolidation_map[main_label].items():
                    count = original_distribution[main_label].get(orig, 0)
                    target_groups[target].append((orig, count))
                
                for target, orig_list in sorted(target_groups.items()):
                    total_count = sum(count for _, count in orig_list)
                    if len(orig_list) == 1 and orig_list[0][0] == target:
                        # no consolidation
                        report.append(f"  {target}: {total_count} samples")
                    else:
                        # consolidation happened
                        orig_str = ", ".join([f"{orig}({count})" for orig, count in orig_list])
                        report.append(f"  {target}: {total_count} samples <- {orig_str}")
            else:
                # no consolidation for this main label
                for sublabel, count in sorted(original_distribution[main_label].items()):
                    report.append(f"  {sublabel}: {count} samples")
            
            report.append("")
            
        return "\n".join(report)


def consolidate_labels(data_folder: str, 
                      original_joined_dict: Dict[str, List[str]], 
                      consolidation_rules: Dict[str, Dict[str, List[str]]] = None,
                      min_samples_threshold: int = 10) -> Tuple[Dict[str, List[str]], Dict[str, int], str]:
    """
    Convenience function to perform complete label consolidation.
    
    Args:
        data_folder: Path to folder containing embedding files
        original_joined_dict: Original sublabel structure
        consolidation_rules: Manual consolidation rules (optional)
        min_samples_threshold: Minimum samples to keep sublabel separate
        
    Returns:
        Tuple of (consolidated_joined_dict, sublabel_dict, consolidation_report)
    """
    consolidator = LabelConsolidator(min_samples_threshold)
    
    distribution = consolidator.analyze_sublabel_distribution(data_folder, original_joined_dict)
    consolidation_map = consolidator.create_consolidation_mapping(distribution, consolidation_rules)
    consolidated_joined_dict = consolidator.apply_consolidation(original_joined_dict, consolidation_map)
    
    sublabel_dict = consolidator.create_sublabel_dict(consolidated_joined_dict)
    
    report = consolidator.generate_consolidation_report(distribution, consolidation_map)
    
    return consolidated_joined_dict, sublabel_dict, report, consolidation_map


def find_missing_sublabels(data_folder: str, joined_label_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Find sublabels that exist in the data but are missing from joined_label_dict.
    
    Args:
        data_folder: Path to folder containing embedding files
        joined_label_dict: Current sublabel structure
        
    Returns:
        Dict mapping main_label -> list of missing sublabels
    """
    missing_sublabels = defaultdict(set)
    
    files = [f for f in os.listdir(data_folder) if f.endswith(".pkl")]
    
    for file in files:
        main_label = file.split("_")[0]
        if main_label not in joined_label_dict:
            continue
            
        # extract all possible sublabels from filename
        filename_parts = file.split("_")
        for i, part in enumerate(filename_parts):
            if part == main_label and i + 1 < len(filename_parts):
                # look for sublabel pattern after main label
                remaining_parts = filename_parts[i+1:]
                
                # try to reconstruct sublabel (handle multi-word sublabels)
                for j in range(1, len(remaining_parts)):
                    potential_sublabel = "_".join(remaining_parts[:j])
                    if potential_sublabel not in joined_label_dict[main_label] and potential_sublabel != "person":
                        # if this looks like a valid sublabel (not video name parts)
                        if not any(c.isdigit() for c in potential_sublabel) and len(potential_sublabel) > 2:
                            missing_sublabels[main_label].add(potential_sublabel)
    
    return {k: sorted(list(v)) for k, v in missing_sublabels.items()}