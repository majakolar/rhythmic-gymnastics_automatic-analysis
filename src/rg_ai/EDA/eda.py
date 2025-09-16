#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) Script for Annotation Data
===========================================================

This script provides comprehensive statistical analysis and interactive visualizations 
of annotation data, focusing on movement types, subtypes, transitions, and temporal patterns.

"""

import os
import json
import glob
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import itertools

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


@dataclass
class AnnotationData:
    """Container for annotation data with metadata."""
    movement_type: str
    subtype: str
    start_time: float
    end_time: float
    duration: float
    video: str
    annotation_id: str
    source_file: str
    score: Optional[float] = None
    incorrect_execution: Optional[bool] = None
    comments: Optional[str] = None


@dataclass
class TransitionData:
    """Container for transition analysis data."""
    from_type: str
    to_type: str
    from_subtype: str
    to_subtype: str
    wait_time: float
    from_annotation: AnnotationData
    to_annotation: AnnotationData


class AnnotationEDA:
    def __init__(self, annotations_folder: str, output_folder: str = "eda_results", hide_titles: bool = False, 
                 tick_font_size: int = 12, axis_font_size: int = 14):
        """
        Initialize EDA with annotation data.
        
        Args:
            annotations_folder: Path to folder containing annotation JSON files
            output_folder: Path to save output visualizations
            hide_titles: If True, plots will not include titles (useful for scientific articles)
            tick_font_size: Font size for axis tick labels
            axis_font_size: Font size for axis titles
        """
        self.annotations_folder = annotations_folder
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.hide_titles = hide_titles
        self.tick_font_size = tick_font_size
        self.axis_font_size = axis_font_size
        self.color_palette = px.colors.qualitative.Set3
        
        print("Loading annotation data...")
        self.annotations = self._load_all_annotations()
        self.df = self._create_dataframe()
        
        print(f"Loaded {len(self.annotations)} annotations from {len(self.df['video'].unique())} videos")
        
        print("Calculating transitions...")
        self.transitions = self._calculate_transitions()
        
    def _load_all_annotations(self) -> List[AnnotationData]:
        """Load all annotations from JSON files."""
        if not os.path.exists(self.annotations_folder):
            raise ValueError(f"Annotations folder not found: {self.annotations_folder}")
        
        json_files = glob.glob(os.path.join(self.annotations_folder, "**/*.json"), recursive=True)
        
        if not json_files:
            raise ValueError(f"No annotation files found in {self.annotations_folder}")
        
        annotations = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for video_name, video_annotations in data.items():
                    for ann_key, ann_data in video_annotations.items():
                        if ann_key.startswith('annotation_'):
                            try:
                                annotation = AnnotationData(
                                    movement_type=str(ann_data.get("movement_type", "Unknown")),
                                    subtype=str(ann_data.get("subtype", "Unknown")),
                                    start_time=float(ann_data.get("start_time", 0)),
                                    end_time=float(ann_data.get("end_time", 0)),
                                    duration=float(ann_data.get("end_time", 0)) - float(ann_data.get("start_time", 0)),
                                    video=video_name,
                                    annotation_id=ann_key,
                                    source_file=json_file,
                                    score=ann_data.get("score"),
                                    incorrect_execution=ann_data.get("incorrect_execution"),
                                    comments=ann_data.get("comments")
                                )
                                annotations.append(annotation)
                            except (ValueError, KeyError) as e:
                                warnings.warn(f"Skipping invalid annotation {ann_key} in {json_file}: {e}")
                                
            except (json.JSONDecodeError, IOError) as e:
                warnings.warn(f"Error reading JSON file {json_file}: {e}")
                continue
        
        annotations.sort(key=lambda x: (x.video, x.start_time))
        return annotations
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from annotations for easier analysis."""
        data = []
        for ann in self.annotations:
            data.append({
                'movement_type': ann.movement_type,
                'subtype': ann.subtype,
                'start_time': ann.start_time,
                'end_time': ann.end_time,
                'duration': ann.duration,
                'video': ann.video,
                'annotation_id': ann.annotation_id,
                'source_file': ann.source_file,
                'score': ann.score,
                'incorrect_execution': ann.incorrect_execution,
                'comments': ann.comments,
                'combined_label': f"{ann.movement_type}_{ann.subtype}"
            })
        
        df = pd.DataFrame(data)
        
        if len(df) > 0:
            df['duration_bins'] = pd.cut(df['duration'], bins=10, labels=False)
            df['video_order'] = df.groupby('video').cumcount()  
            
        return df
    
    def _calculate_transitions(self) -> List[TransitionData]:
        """Calculate transitions between consecutive annotations."""
        transitions = []
        
        for video in self.df['video'].unique():
            video_annotations = [ann for ann in self.annotations if ann.video == video]
            video_annotations.sort(key=lambda x: x.start_time)
            
            for i in range(len(video_annotations) - 1):
                current = video_annotations[i]
                next_ann = video_annotations[i + 1]
                
                wait_time = next_ann.start_time - current.end_time
                
                transition = TransitionData(
                    from_type=current.movement_type,
                    to_type=next_ann.movement_type,
                    from_subtype=current.subtype,
                    to_subtype=next_ann.subtype,
                    wait_time=wait_time,
                    from_annotation=current,
                    to_annotation=next_ann
                )
                transitions.append(transition)
        
        return transitions
    
    def analyze_type_distributions(self):
        """Analyze and visualize movement type and subtype distributions."""
        print("\n=== MOVEMENT TYPE AND SUBTYPE ANALYSIS ===")
        
        type_counts = Counter([ann.movement_type for ann in self.annotations])
        subtype_counts = Counter([ann.subtype for ann in self.annotations])
        combined_counts = Counter([f"{ann.movement_type}_{ann.subtype}" for ann in self.annotations])
        
        print(f"Total annotations: {len(self.annotations)}")
        print(f"Unique movement types: {len(type_counts)}")
        print(f"Unique subtypes: {len(subtype_counts)}")
        print(f"Unique type-subtype combinations: {len(combined_counts)}")
        
        print(f"\nTop 10 Movement Types:")
        for movement_type, count in type_counts.most_common(10):
            percentage = (count / len(self.annotations)) * 100
            print(f"  {movement_type}: {count} ({percentage:.1f}%)")
        
        print(f"\nTop 10 Subtypes:")
        for subtype, count in subtype_counts.most_common(10):
            percentage = (count / len(self.annotations)) * 100
            print(f"  {subtype}: {count} ({percentage:.1f}%)")
        
        self._plot_type_distributions(type_counts, subtype_counts, combined_counts)
    
    def _plot_type_distributions(self, type_counts: Counter, subtype_counts: Counter, combined_counts: Counter):
        """Create interactive bar plots for type distributions."""
        self._plot_treemap(combined_counts)
        self._plot_movement_subtype_bars(combined_counts)
    
    def _plot_treemap(self, combined_counts: Counter):
        """Create treemap visualization for type-subtype combinations."""
        
        labels = []
        parents = []
        values = []
        colors = []
        
        labels.append("All Actions")
        parents.append("")
        values.append(len(self.annotations))
        colors.append(0)
        
        type_groups = defaultdict(list)
        for combo, count in combined_counts.items():
            if '_' in combo:
                movement_type, subtype = combo.split('_', 1)
                type_groups[movement_type].append((subtype, count))
            else:
                type_groups[combo].append(("", count))
        
        color_idx = 1
        for movement_type, subtypes in type_groups.items():
            total_count = sum(count for _, count in subtypes)
            labels.append(movement_type)
            parents.append("All Actions")
            values.append(total_count)
            colors.append(color_idx)
            
            for subtype, count in subtypes:
                if subtype: 
                    labels.append(f"{movement_type}_{subtype}") #
                    parents.append(movement_type)
                    values.append(count)
                    colors.append(color_idx)
            
            color_idx += 1
        
        fig_treemap = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=3,
        ))
        
        layout_params = {
            'width': 1200,
            'height': 700,
            'xaxis': dict(
                title=dict(text='Movement Type - Subtype', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='Count', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Hierarchical View: Movement Types → Subtypes',
                x=0.5,
                font=dict(size=18)
            )
        
        fig_treemap.update_layout(**layout_params)
        
        html_path = self.output_folder / "type_subtype_treemap.html"
        png_path = self.output_folder / "type_subtype_treemap.png"
        fig_treemap.write_html(html_path)
        fig_treemap.write_image(png_path, width=1200, height=700, scale=2)
        print(f"Treemap plot saved to: {html_path}")
    
    def _get_movement_type_color_mapping(self) -> Dict[str, str]:
        """Create consistent color mapping for movement types across all plots."""
        unique_types = sorted(set(ann.movement_type for ann in self.annotations))
        return {mtype: self.color_palette[i % len(self.color_palette)] 
                for i, mtype in enumerate(unique_types)}

    def _plot_movement_subtype_bars(self, combined_counts: Counter):
        """Create grouped bar plot showing movement types and their subtypes."""
        
        color_mapping = self._get_movement_type_color_mapping()
        
        type_groups = defaultdict(list)
        for combo, count in combined_counts.items():
            if '_' in combo:
                movement_type, subtype = combo.split('_', 1)
                type_groups[movement_type].append((subtype, count))
            else:
                type_groups[combo].append(("Unknown", count))
        
        all_labels = []
        all_counts = []
                
        for movement_type, subtype_data in sorted(type_groups.items()):
            subtype_data.sort(key=lambda x: x[1], reverse=True)
            
            for subtype, count in subtype_data:
                unique_label = f"{movement_type} - {subtype}" if subtype != "Unknown" else movement_type
                all_labels.append(unique_label)
                all_counts.append(count)
        
        fig_bars = go.Figure()
        
        for movement_type in sorted(type_groups.keys()):
            type_labels = []
            type_counts = []
            
            for label, count in zip(all_labels, all_counts):
                if label.startswith(movement_type):
                    type_labels.append(label)
                    type_counts.append(count)
            
            if type_labels:  
                total_count = sum(type_counts)
                
                fig_bars.add_trace(go.Bar(
                    name=f"{movement_type} (n = {total_count})",
                    y=type_labels,
                    x=type_counts,
                    orientation='h',
                    hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata:.1f}%<extra></extra>',
                    customdata=[(count/len(self.annotations))*100 for count in type_counts],
                    textposition='auto',
                    marker=dict(
                        color=color_mapping[movement_type],
                        line=dict(color='black', width=1)
                    )
                ))
        
        layout_params = {
            'width': 1400,
            'height': max(700, len(all_labels) * 25), 
            'bargap': 0.1,  #gap between bars
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=self.tick_font_size)
            ),
            'xaxis': dict(
                title=dict(text='Count', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='Movement Type - Subtype', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Movement Types and Subtypes Distribution',
                x=0.5,
                font=dict(size=18)
            )
        
        fig_bars.update_layout(**layout_params)
        
        html_path = self.output_folder / "movement_subtypes_bars.html"
        png_path = self.output_folder / "movement_subtypes_bars.png"
        fig_bars.write_html(html_path)
        fig_bars.write_image(png_path, width=1400, height=700, scale=2)
        print(f"Movement subtypes bar plot saved to: {html_path}")
    
    def analyze_wait_times(self, threshold_seconds: float = 2.0):
        """Analyze wait times between consecutive annotations."""
        print(f"\n=== WAIT TIME ANALYSIS (Threshold: {threshold_seconds}s) ===")
        
        if not self.transitions:
            print("No transitions found.")
            return
        
        wait_times = [t.wait_time for t in self.transitions]

        # get all transitions where the wait time is less than 0, should be impossible so we have to check
        # if there are any, print them
        for t in self.transitions:
            if t.wait_time < 0:
                print(f"Transition with wait time < 0: {t.from_type}→{t.to_type} ({t.wait_time}s)")

                print(t)


        short_waits = [t for t in self.transitions if t.wait_time < threshold_seconds]
        
        print(f"Total transitions: {len(self.transitions)}")
        print(f"Transitions with wait time < {threshold_seconds}s: {len(short_waits)} ({len(short_waits)/len(self.transitions)*100:.1f}%)")
        
        print(f"\nWait Time Statistics:")
        print(f"  Mean: {np.mean(wait_times):.2f}s")
        print(f"  Median: {np.median(wait_times):.2f}s")
        print(f"  Std Dev: {np.std(wait_times):.2f}s")
        print(f"  Min: {np.min(wait_times):.2f}s")
        print(f"  Max: {np.max(wait_times):.2f}s")
        
        if short_waits:
            same_type_count = sum(1 for t in short_waits if t.from_type == t.to_type)
            same_subtype_count = sum(1 for t in short_waits if t.from_subtype == t.to_subtype)
            
            print(f"\nFor transitions with wait time < {threshold_seconds}s:")
            print(f"  Same movement type: {same_type_count} ({same_type_count/len(short_waits)*100:.1f}%)")
            print(f"  Same subtype: {same_subtype_count} ({same_subtype_count/len(short_waits)*100:.1f}%)")
            
            short_transitions = Counter([f"{t.from_type}→{t.to_type}" for t in short_waits])
            print(f"\nMost common transitions with short wait times:")
            for transition, count in short_transitions.most_common(10):
                percentage = (count / len(short_waits)) * 100
                print(f"  {transition}: {count} ({percentage:.1f}%)")
        
        self._plot_wait_time_analysis(wait_times, short_waits, threshold_seconds)
    
    def _plot_wait_time_analysis(self, wait_times: List[float], short_waits: List[TransitionData], threshold: float):
        """Create interactive visualizations for wait time analysis."""
        
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=wait_times,
            nbinsx=50,
            name='All Wait Times',
            opacity=0.9,
            marker_color='lightblue',
            hovertemplate='Wait Time: %{x:.2f}s<br>Count: %{y}<extra></extra>'
        ))
        
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {threshold}s",
            annotation_position="top right",
            annotation_font=dict(size=self.tick_font_size)
        )
        
        layout_params = {
            'width': 1200,
            'height': 600,
            'showlegend': False,
            'xaxis': dict(
                title=dict(text='Wait Time (seconds)', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='Frequency', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Distribution of Wait Times Between Annotations',
                x=0.5,
                font=dict(size=18)
            )
        
        fig_hist.update_layout(**layout_params)
        
        html_path = self.output_folder / "wait_times_distribution.html"
        png_path = self.output_folder / "wait_times_distribution.png"
        fig_hist.write_html(html_path)
        fig_hist.write_image(png_path, width=1200, height=600, scale=2)
        print(f"Wait times histogram saved to: {html_path}")
        
        if short_waits:
            self._plot_wait_times_by_transition_type(short_waits, threshold)
    
    def _plot_wait_times_by_transition_type(self, short_waits: List[TransitionData], threshold: float):
        """Create box plot of wait times grouped by transition characteristics."""
        
        transition_categories = []
        wait_time_values = []
        
        for t in short_waits:
            if t.from_type == t.to_type and t.from_subtype == t.to_subtype:
                category = "Same Type & Subtype"
            elif t.from_type == t.to_type:
                category = "Same Type, Different Subtype"
            #elif t.from_subtype == t.to_subtype:
            #    category = "Different Type, Same Subtype"
            else:
                category = "Different Type & Subtype"
            
            transition_categories.append(category)
            wait_time_values.append(t.wait_time)
        
        fig_box = go.Figure()
        
        categories = [
            "Same Type & Subtype", 
            "Same Type, Different Subtype", 
            #"Different Type, Same Subtype", 
            "Different Type & Subtype"
            ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, category in enumerate(categories):
            category_waits = [wt for cat, wt in zip(transition_categories, wait_time_values) if cat == category]
            n_waits = len(category_waits)
            if category_waits:
                fig_box.add_trace(go.Box(
                    y=category_waits,
                    name=f"{category} (n = {n_waits})",
                    #marker_color=self.color_palette[(i % len(self.color_palette)) + 3],
                    fillcolor=self.color_palette[(i % len(self.color_palette)) + 3],
                    opacity=0.7,
                    marker_color='black', #colors[i % len(colors)],
                    hovertemplate='<b>%{fullData.name}</b><br>Wait Time: %{y:.2f}s<br>Number of transitions: %{n_waits}<extra></extra>'
                ))
        
        layout_params = {
            'width': 1200,
            'height': 600,
            'showlegend': False,
            'xaxis': dict(
                title=dict(text='Transition Category', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='Wait Time (seconds)', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text=f'Wait Times by Transition Type (< {threshold}s)',
                x=0.5,
                font=dict(size=18)
            )
        
        fig_box.update_layout(**layout_params)
        
        html_path = self.output_folder / "wait_times_by_transition_type.html"
        png_path = self.output_folder / "wait_times_by_transition_type.png"
        fig_box.write_html(html_path)
        fig_box.write_image(png_path, width=1200, height=600, scale=2)
        print(f"Wait times by transition type saved to: {html_path}")
    
    def analyze_transitions(self, threshold_seconds: float = 2.0):
        """Comprehensive analysis of action transitions."""
        print(f"\n=== TRANSITION ANALYSIS ===")
        
        if not self.transitions:
            print("No transitions found.")
            return
        
        all_transitions = Counter([f"{t.from_type}→{t.to_type}" for t in self.transitions])
        short_transitions = Counter([f"{t.from_type}→{t.to_type}" for t in self.transitions if t.wait_time < threshold_seconds])
        
        print(f"Total unique transition patterns: {len(all_transitions)}")
        print(f"Short-wait transition patterns: {len(short_transitions)}")
        
        print(f"\nMost common transitions (all wait times):")
        for transition, count in all_transitions.most_common(15):
            percentage = (count / len(self.transitions)) * 100
            print(f"  {transition}: {count} ({percentage:.1f}%)")
        
        self._plot_transition_matrix(threshold_seconds)

    
    def _plot_transition_matrix(self, threshold: float):
        """Create interactive heatmap of transition patterns."""
        
        all_types = sorted(set([t.from_type for t in self.transitions] + [t.to_type for t in self.transitions]))
        
        short_transitions = [t for t in self.transitions if t.wait_time < threshold]
        
        matrix = np.zeros((len(all_types), len(all_types)))
        for t in short_transitions:
            from_idx = all_types.index(t.from_type)
            to_idx = all_types.index(t.to_type)
            matrix[from_idx][to_idx] += 1
        
        text_matrix = []
        for row in matrix:
            text_row = []
            for val in row:
                if val > 0:
                    text_row.append(str(int(val)))
                else:
                    text_row.append("")  
            text_matrix.append(text_row)
        
        max_val = np.max(matrix) if np.max(matrix) > 0 else 1
        threshold_val = max_val * 0.5  
        
        text_colors = []
        for row in matrix:
            color_row = []
            for val in row:
                if val > threshold_val:
                    color_row.append('white') 
                else:
                    color_row.append('black') 
            text_colors.append(color_row)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_types,
            y=all_types,
            colorscale='Blues', # color scale could be greens
            hovertemplate='From: %{y}<br>To: %{x}<br>Count: %{z}<extra></extra>',
            colorbar=dict(title="Transition Count"),
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 38},
            showscale=True
        ))
     
        
        layout_params = {
            'width': 1000,
            'height': 800,
            'xaxis': dict(
                title=dict(text='To Movement Type', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='From Movement Type', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text=f'Transition Matrix (Wait Time < {threshold}s)',
                x=0.5,
                font=dict(size=18)
            )
        
        fig_heatmap.update_layout(**layout_params)
        
        html_path = self.output_folder / f"transition_matrix_{threshold}s.html"
        png_path = self.output_folder / f"transition_matrix_{threshold}s.png"
        fig_heatmap.write_html(html_path)
        fig_heatmap.write_image(png_path, width=1000, height=800, scale=2)
        print(f"Transition matrix saved to: {html_path}")
    
    def analyze_temporal_patterns(self, violin_plot: bool = False):
        """Analyze temporal patterns in the data."""
        print(f"\n=== TEMPORAL PATTERNS ANALYSIS ===")
        
        durations = [ann.duration for ann in self.annotations]
        
        print(f"Duration Statistics:")
        print(f"  Mean: {np.mean(durations):.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")
        print(f"  Std Dev: {np.std(durations):.2f}s")
        print(f"  Min: {np.min(durations):.2f}s")
        print(f"  Max: {np.max(durations):.2f}s")
        
        type_durations = defaultdict(list)
        for ann in self.annotations:
            type_durations[ann.movement_type].append(ann.duration)
        
        print(f"\nAverage duration by movement type (top 10):")
        avg_durations = [(mtype, np.mean(durs)) for mtype, durs in type_durations.items()]
        avg_durations.sort(key=lambda x: x[1], reverse=True)
        
        for mtype, avg_dur in avg_durations[:10]:
            print(f"  {mtype}: {avg_dur:.2f}s")
        
        self._plot_duration_analysis(type_durations, violin_plot)
    
    def _plot_duration_analysis(self, type_durations: Dict[str, List[float]], violin_plot: bool = False):
        """Create visualizations for duration analysis."""
        
        color_mapping = self._get_movement_type_color_mapping()
        
        fig = go.Figure()
        
        min_samples = 5
        types_to_plot = [(mtype, durs) for mtype, durs in type_durations.items() if len(durs) >= min_samples]
        types_to_plot.sort(key=lambda x: np.median(x[1]), reverse=True)
        
        for mtype, durs in types_to_plot[:15]:  # top 15
            if violin_plot:
                fig.add_trace(go.Violin(
                    y=durs,
                    name=mtype,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=color_mapping[mtype],
                    opacity=0.7,
                    hovertemplate='<b>%{fullData.name}</b><br>Duration: %{y:.2f}s<extra></extra>'
                ))
            else:
                fig.add_trace(go.Box(
                    y=durs,
                    name=mtype,
                    boxmean=True,
                    fillcolor=color_mapping[mtype],
                    marker_color='black',
                    hovertemplate='<b>%{fullData.name}</b><br>Duration: %{y:.2f}s<extra></extra>'
                ))
        
        layout_params = {
            'width': 1400,
            'height': 700,
            'showlegend': False,
            'xaxis': dict(
                title=dict(text='Movement Type', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='Duration (seconds)', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Duration Distribution by Movement Type',
                x=0.5,
                font=dict(size=18)
            )
        
        fig.update_layout(**layout_params)
        
        fig.update_xaxes(tickangle=0)
        
        plot_type = "violin" if violin_plot else "box"
        html_path = self.output_folder / f"duration_by_type_{plot_type}.html"
        png_path = self.output_folder / f"duration_by_type_{plot_type}.png"
        fig.write_html(html_path)
        fig.write_image(png_path, width=1400, height=700, scale=2)
        print(f"Duration {plot_type} plot saved to: {html_path}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print(f"\n=== SUMMARY REPORT ===")
        
        report_path = self.output_folder / "eda_summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("ANNOTATION DATA EDA SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset Overview:\n")
            f.write(f"  Total annotations: {len(self.annotations)}\n")
            f.write(f"  Unique videos: {len(self.df['video'].unique())}\n")
            f.write(f"  Unique movement types: {len(self.df['movement_type'].unique())}\n")
            f.write(f"  Unique subtypes: {len(self.df['subtype'].unique())}\n")
            f.write(f"  Total transitions: {len(self.transitions)}\n\n")
            
            durations = [ann.duration for ann in self.annotations]
            f.write(f"Duration Statistics:\n")
            f.write(f"  Mean: {np.mean(durations):.2f}s\n")
            f.write(f"  Median: {np.median(durations):.2f}s\n")
            f.write(f"  Std Dev: {np.std(durations):.2f}s\n")
            f.write(f"  Range: {np.min(durations):.2f}s - {np.max(durations):.2f}s\n\n")
            
            if self.transitions:
                wait_times = [t.wait_time for t in self.transitions]
                f.write(f"Wait Time Statistics:\n")
                f.write(f"  Mean: {np.mean(wait_times):.2f}s\n")
                f.write(f"  Median: {np.median(wait_times):.2f}s\n")
                f.write(f"  Std Dev: {np.std(wait_times):.2f}s\n")
                f.write(f"  Range: {np.min(wait_times):.2f}s - {np.max(wait_times):.2f}s\n\n")
            
            type_counts = Counter([ann.movement_type for ann in self.annotations])
            f.write(f"Top 10 Movement Types:\n")
            for mtype, count in type_counts.most_common(10):
                percentage = (count / len(self.annotations)) * 100
                f.write(f"  {mtype}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nGenerated files:\n")
            for file_path in sorted(self.output_folder.glob("*.html")):
                f.write(f"  - {file_path.name}\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def analyze_annotations_per_video(self):
        """Analyze and plot the number of annotations per video."""
        print(f"\n=== ANNOTATIONS PER VIDEO ANALYSIS ===")
        
        video_counts = self.df['video'].value_counts().sort_index()
        
        print(f"Total videos: {len(video_counts)}")
        print(f"Annotations per video - Min: {video_counts.min()}, Max: {video_counts.max()}, Mean: {video_counts.mean():.1f}")
        
        print(f"Mean number of annotations per video: {video_counts.mean():.1f}, standard deviation: {video_counts.std():.1f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=video_counts.index,
            y=video_counts.values,
            marker_color='steelblue',
            hovertemplate='<b>%{x}</b><br>Annotations: %{y}<extra></extra>'
        ))
        
        layout_params = {
            'width': 1200,
            'height': 600,
            'showlegend': False,
            'xaxis': dict(
                title=dict(text='Video', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            ),
            'yaxis': dict(
                title=dict(text='Number of Annotations', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(text='Number of Annotations per Video', x=0.5, font=dict(size=18))
        
        fig.update_layout(**layout_params)
        fig.update_xaxes(tickangle=45)
        
        html_path = self.output_folder / "annotations_per_video.html"
        png_path = self.output_folder / "annotations_per_video.png"
        fig.write_html(html_path)
        fig.write_image(png_path, width=1200, height=600, scale=2)
        print(f"Annotations per video plot saved to: {html_path}")

    def analyze_subtype_transitions(self, threshold_seconds: float = 2.0):
        """Analyze transitions between specific action subtypes (type + subtype combinations)."""
        print(f"\n=== SUBTYPE TRANSITION ANALYSIS ===")
        
        if not self.transitions:
            print("No transitions found.")
            return
        
        # combined labels for analysis
        subtype_transitions = Counter([f"{t.from_type}_{t.from_subtype}→{t.to_type}_{t.to_subtype}" for t in self.transitions])
        short_subtype_transitions = Counter([f"{t.from_type}_{t.from_subtype}→{t.to_type}_{t.to_subtype}" 
                                       for t in self.transitions if t.wait_time < threshold_seconds])
        
        # same action repetitions
        same_action_transitions = [t for t in self.transitions 
                             if t.from_type == t.to_type and t.from_subtype == t.to_subtype]
        same_action_short = [t for t in same_action_transitions if t.wait_time < threshold_seconds]
        
        print(f"Total unique subtype transition patterns: {len(subtype_transitions)}")
        print(f"Short-wait subtype transition patterns: {len(short_subtype_transitions)}")
        print(f"\nSame action repetitions:")
        print(f"  All wait times: {len(same_action_transitions)} ({len(same_action_transitions)/len(self.transitions)*100:.1f}%)")
        print(f"  Wait time < {threshold_seconds}s: {len(same_action_short)} ({len(same_action_short)/len(self.transitions)*100:.1f}%)")
        
        if same_action_short:
            same_action_counts = Counter([f"{t.from_type}_{t.from_subtype}" for t in same_action_short])
            print(f"\nMost common repeated actions (wait time < {threshold_seconds}s):")
            for action, count in same_action_counts.most_common(10):
                percentage = (count / len(same_action_short)) * 100
                print(f"  {action}: {count} ({percentage:.1f}%)")
        
        print(f"\nMost common subtype transitions (wait time < {threshold_seconds}s):")
        for transition, count in short_subtype_transitions.most_common(15):
            percentage = (count / len([t for t in self.transitions if t.wait_time < threshold_seconds])) * 100
            print(f"  {transition}: {count} ({percentage:.1f}%)")
        
        self._plot_subtype_transition_matrix(threshold_seconds)

    def _plot_subtype_transition_matrix(self, threshold: float):
        """Create interactive heatmap of subtype transition patterns."""
        
        # all unique subtype combinations
        all_subtypes = sorted(set([f"{t.from_type}_{t.from_subtype}" for t in self.transitions] + 
                             [f"{t.to_type}_{t.to_subtype}" for t in self.transitions]))
        
        short_transitions = [t for t in self.transitions if t.wait_time < threshold]
        
        # transition matrix
        matrix = np.zeros((len(all_subtypes), len(all_subtypes)))
        for t in short_transitions:
            from_subtype = f"{t.from_type}_{t.from_subtype}"
            to_subtype = f"{t.to_type}_{t.to_subtype}"
            from_idx = all_subtypes.index(from_subtype)
            to_idx = all_subtypes.index(to_subtype)
            matrix[from_idx][to_idx] += 1
        
        # text matrix for display
        text_matrix = []
        for row in matrix:
            text_row = []
            for val in row:
                if val > 0:
                    text_row.append(str(int(val)))
                else:
                    text_row.append("")  
            text_matrix.append(text_row)
        
        # diagonal (same action repetitions) for highlighting
        diagonal_sum = np.trace(matrix)
        total_transitions = np.sum(matrix)
        diagonal_percentage = (diagonal_sum / total_transitions * 100) if total_transitions > 0 else 0
        
        print(f"Same action repetitions in matrix: {int(diagonal_sum)} ({diagonal_percentage:.1f}% of short transitions)")
        
        # custom colorscale to highlight diagonal
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_subtypes,
            y=all_subtypes,
            colorscale='Greens',
            hovertemplate='From: %{y}<br>To: %{x}<br>Count: %{z}<extra></extra>',
            colorbar=dict(title="Transition Count"),
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 10},  # font size due to more labels
            showscale=True
        ))
        
        # diagonal line to highlight same action repetitions
        fig_heatmap.add_shape(
            type="line",
            x0=-0.5, y0=-0.5,
            x1=len(all_subtypes)-0.5, y1=len(all_subtypes)-0.5,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        layout_params = {
            'width': max(1200, len(all_subtypes) * 20),  # width based on number of subtypes
            'height': max(1000, len(all_subtypes) * 20),  # height
            'xaxis': dict(
                title=dict(text='To Action (Type_Subtype)', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=8),  # font for readability
                tickangle=45
            ),
            'yaxis': dict(
                title=dict(text='From Action (Type_Subtype)', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=8)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text=f'Subtype Transition Matrix (Wait Time < {threshold}s)<br>Red diagonal shows same action repetitions',
                x=0.5,
                font=dict(size=16)
            )
        
        fig_heatmap.update_layout(**layout_params)
        
        html_path = self.output_folder / f"subtype_transition_matrix_{threshold}s.html"
        png_path = self.output_folder / f"subtype_transition_matrix_{threshold}s.png"
        fig_heatmap.write_html(html_path)
        fig_heatmap.write_image(png_path, width=layout_params['width'], height=layout_params['height'], scale=2)
        print(f"Subtype transition matrix saved to: {html_path}")

    def analyze_grade_distributions(self):
        """Analyze and visualize grade distributions for each movement type-subtype combination."""
        print(f"\n=== GRADE DISTRIBUTION ANALYSIS ===")
        
        # annotations that have valid scores
        scored_annotations = [ann for ann in self.annotations if ann.score is not None]
        
        if not scored_annotations:
            print("No annotations with scores found.")
            return
        
        print(f"Total annotations with scores: {len(scored_annotations)} out of {len(self.annotations)}")
        
        # scores by type-subtype combination
        type_subtype_scores = defaultdict(list)
        for ann in scored_annotations:
            combined_label = f"{ann.movement_type}_{ann.subtype}"
            type_subtype_scores[combined_label].append(ann.score)
        
        # statistics for each type-subtype combination
        print(f"\nGrade statistics by Type-Subtype combination:")
        print(f"{'Type-Subtype':<40} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 80)
        
        stats_data = []
        for combination, scores in sorted(type_subtype_scores.items()):
            if len(scores) >= 3:  # include combinations with at least 3 scores
                stats_data.append({
                    'combination': combination,
                    'count': len(scores),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                })
                print(f"{combination:<40} {len(scores):<8} {np.mean(scores):<8.2f} {np.std(scores):<8.2f} {np.min(scores):<8.2f} {np.max(scores):<8.2f}")
        
        # mean score for better visualization
        stats_data.sort(key=lambda x: x['mean'], reverse=True)
        
        if not stats_data:
            print("No type-subtype combinations with sufficient data (≥3 scores) found.")
            return
        
        self._plot_grade_distributions(stats_data)
        self._plot_grade_statistics_summary(stats_data)
    
    def _plot_grade_distributions(self, stats_data: List[Dict]):
        """Create box plots showing grade distributions for each type-subtype combination."""
        
        # top combinations by sample size to avoid overcrowding
        top_combinations = sorted(stats_data, key=lambda x: x['count'], reverse=True)[:20]
        
        fig = go.Figure()
        
        color_mapping = self._get_movement_type_color_mapping()
        
        for i, data in enumerate(top_combinations):
            combination = data['combination']
            scores = data['scores']
            count = data['count']
            
            # movement type for color mapping
            movement_type = combination.split('_')[0]
            color = color_mapping.get(movement_type, self.color_palette[i % len(self.color_palette)])
            
            fig.add_trace(go.Box(
                y=scores,
                name=f"{combination} (n={count})",
                boxmean=True,
                fillcolor=color,
                marker_color='black',
                opacity=0.7,
                hovertemplate=f'<b>{combination}</b><br>Score: %{{y}}<br>Count: {count}<extra></extra>'
            ))
        
        layout_params = {
            'width': max(1400, len(top_combinations) * 60),
            'height': 700,
            'showlegend': False,
            'xaxis': dict(
                title=dict(text='Movement Type - Subtype', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size),
                tickangle=45
            ),
            'yaxis': dict(
                title=dict(text='Grade/Score', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Grade Distribution by Movement Type - Subtype',
                x=0.5,
                font=dict(size=18)
            )
        
        fig.update_layout(**layout_params)
        
        html_path = self.output_folder / "grade_distributions_by_type_subtype.html"
        png_path = self.output_folder / "grade_distributions_by_type_subtype.png"
        fig.write_html(html_path)
        fig.write_image(png_path, width=layout_params['width'], height=700, scale=2)
        print(f"Grade distributions plot saved to: {html_path}")
    
    def _plot_grade_statistics_summary(self, stats_data: List[Dict]):
        """Create summary visualization of grade statistics."""
        
        # subplot with mean scores and sample sizes
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(['Mean Grades by Type-Subtype', 'Sample Sizes by Type-Subtype'] if not self.hide_titles else ['', '']),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        combinations = [data['combination'] for data in stats_data]
        means = [data['mean'] for data in stats_data]
        stds = [data['std'] for data in stats_data]
        counts = [data['count'] for data in stats_data]
        
        color_mapping = self._get_movement_type_color_mapping()
        
        # mean scores with error bars (std dev)
        colors = []
        for combination in combinations:
            movement_type = combination.split('_')[0]
            colors.append(color_mapping.get(movement_type, '#1f77b4'))
        
        fig.add_trace(
            go.Bar(
                x=combinations,
                y=means,
                error_y=dict(type='data', array=stds, visible=True),
                marker_color=colors,
                name='Mean Grade',
                hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f}<br>Std: %{error_y.array:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # sample sizes
        fig.add_trace(
            go.Bar(
                x=combinations,
                y=counts,
                marker_color='lightblue',
                name='Sample Size',
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(
            title_text='Movement Type - Subtype',
            tickangle=45,
            tickfont=dict(size=self.tick_font_size),
            title_font=dict(size=self.axis_font_size),
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text='Mean Grade',
            tickfont=dict(size=self.tick_font_size),
            title_font=dict(size=self.axis_font_size),
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text='Number of Annotations',
            tickfont=dict(size=self.tick_font_size),
            title_font=dict(size=self.axis_font_size),
            row=2, col=1
        )
        
        layout_params = {
            'width': max(1400, len(combinations) * 60),
            'height': 900,
            'showlegend': False
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Grade Statistics Summary by Type-Subtype',
                x=0.5,
                font=dict(size=18)
            )
        
        fig.update_layout(**layout_params)
        
        html_path = self.output_folder / "grade_statistics_summary.html"
        png_path = self.output_folder / "grade_statistics_summary.png"
        fig.write_html(html_path)
        fig.write_image(png_path, width=layout_params['width'], height=900, scale=2)
        print(f"Grade statistics summary plot saved to: {html_path}")

    def analyze_grade_distributions_by_type(self):
        """Analyze and visualize grade distributions for each movement type (ignoring subtypes)."""
        print(f"\n=== GRADE DISTRIBUTION BY MOVEMENT TYPE ANALYSIS ===")
        
        # annotations that have valid scores
        scored_annotations = [ann for ann in self.annotations if ann.score is not None]
        
        if not scored_annotations:
            print("No annotations with scores found.")
            return
        
        print(f"Total annotations with scores: {len(scored_annotations)} out of {len(self.annotations)}")
        
        # scores by movement type only
        type_scores = defaultdict(list)
        for ann in scored_annotations:
            type_scores[ann.movement_type].append(ann.score)
        
        # statistics for each movement type
        print(f"\nGrade statistics by Movement Type:")
        print(f"{'Movement Type':<30} {'Count':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
        print("-" * 70)
        
        stats_data = []
        for movement_type, scores in sorted(type_scores.items()):
            if len(scores) >= 3:  # include types with at least 3 scores
                stats_data.append({
                    'type': movement_type,
                    'count': len(scores),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                })
                print(f"{movement_type:<30} {len(scores):<8} {np.mean(scores):<8.2f} {np.std(scores):<8.2f} {np.min(scores):<8.2f} {np.max(scores):<8.2f}")
        
        # sorting by mean score for better visualization
        stats_data.sort(key=lambda x: x['mean'], reverse=True)
        
        if not stats_data:
            print("No movement types with sufficient data (≥3 scores) found.")
            return
        
        self._plot_grade_distributions_by_type(stats_data)
    
    def _plot_grade_distributions_by_type(self, stats_data: List[Dict]):
        """Create box plots showing grade distributions for each movement type."""
        
        fig = go.Figure()
        
        color_mapping = self._get_movement_type_color_mapping()
        
        for i, data in enumerate(stats_data):
            movement_type = data['type']
            scores = data['scores']
            count = data['count']
            
            color = color_mapping.get(movement_type, self.color_palette[i % len(self.color_palette)])
            
            fig.add_trace(go.Box(
                y=scores,
                name=f"{movement_type} (n={count})",
                boxmean=True,
                fillcolor=color,
                marker_color='black',
                opacity=0.7,
                hovertemplate=f'<b>{movement_type}</b><br>Score: %{{y}}<br>Count: {count}<extra></extra>'
            ))
        
        layout_params = {
            'width': max(1200, len(stats_data) * 80),
            'height': 700,
            'showlegend': False,
            'xaxis': dict(
                title=dict(text='Movement Type', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size),
                tickangle=0
            ),
            'yaxis': dict(
                title=dict(text='Grade/Score', font=dict(size=self.axis_font_size)),
                tickfont=dict(size=self.tick_font_size)
            )
        }
        
        if not self.hide_titles:
            layout_params['title'] = dict(
                text='Grade Distribution by Movement Type',
                x=0.5,
                font=dict(size=18)
            )
        
        fig.update_layout(**layout_params)
        
        html_path = self.output_folder / "grade_distributions_by_type.html"
        png_path = self.output_folder / "grade_distributions_by_type.png"
        fig.write_html(html_path)
        fig.write_image(png_path, width=layout_params['width'], height=700, scale=2)
        print(f"Grade distributions by type plot saved to: {html_path}")

    def run_full_analysis(self, wait_time_threshold: float = 2.0, violin_plot: bool = False):
        """Run complete EDA analysis with all visualizations."""
        print("Starting comprehensive EDA analysis...")
        print("=" * 60)
        
        self.analyze_annotations_per_video()
        self.analyze_type_distributions()
        self.analyze_grade_distributions()  
        self.analyze_grade_distributions_by_type()
        self.analyze_wait_times(wait_time_threshold)
        self.analyze_transitions(wait_time_threshold)
        self.analyze_subtype_transitions(wait_time_threshold)
        self.analyze_temporal_patterns(violin_plot)
        self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("EDA Analysis Complete!")
        print(f"All results saved to: {self.output_folder}")
        print(f"Open the HTML files for interactive visualizations.")


def main():
    """Main function to run EDA analysis."""
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for Annotation Data")
    
    parser.add_argument(
        "--annotations_folder",
        type=str,
        default="data/raw/annotations",
        help="Path to folder containing annotation JSON files"
    )
    
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/results/visualizations/EDA/Annotations Tokyo Q pack 1 2 and four of 5_no_titles - subtypes transitions",
        help="Path to save output visualizations and reports"
    )
    
    parser.add_argument(
        "--wait_time_threshold",
        type=float,
        default=2.0,
        help="Threshold in seconds for analyzing short wait times between annotations"
    )
    
    parser.add_argument(
        "--hide_titles",
        action="store_true",
        default=True,
        help="Hide plot titles (useful for scientific articles)"
    )
    
    parser.add_argument(
        "--violin_plot",
        action="store_true",
        default=False,
        help="Use violin plots instead of box plots for duration analysis"
    )
    
    parser.add_argument(
        "--tick_font_size",
        type=int,
        default=20,
        help="Font size for axis tick labels"
    )
    
    parser.add_argument(
        "--axis_font_size",
        type=int,
        default=24,
        help="Font size for axis titles"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.annotations_folder):
        print(f"Error: Annotations folder not found: {args.annotations_folder}")
        return
    
    try:
        eda = AnnotationEDA(args.annotations_folder, args.output_folder, args.hide_titles,
                           args.tick_font_size, args.axis_font_size)
        eda.run_full_analysis(args.wait_time_threshold, args.violin_plot)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main() 