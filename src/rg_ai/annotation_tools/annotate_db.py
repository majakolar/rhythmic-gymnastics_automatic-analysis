import sys
import json
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QLineEdit, QFileDialog, QMessageBox, QCheckBox,
                            QTextEdit, QFrame, QSizePolicy, QTableWidget,
                            QTableWidgetItem, QHeaderView, QDialog, QSpinBox,
                            QSlider, QSplitter, QDialogButtonBox, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, QUrl, QEvent
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtGui import QAction, QShortcut, QKeySequence, QPainter, QPen, QBrush, QColor

class TimeEditDialog(QDialog):
    def __init__(self, current_time, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Time")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Minutes and seconds input
        time_layout = QHBoxLayout()
        self.minutes_spin = QSpinBox()
        self.minutes_spin.setRange(0, 59)
        self.minutes_spin.setValue(int(current_time) // 60)
        time_layout.addWidget(QLabel("Minutes:"))
        time_layout.addWidget(self.minutes_spin)
        
        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(0, 59)
        self.seconds_spin.setValue(int(current_time) % 60)
        time_layout.addWidget(QLabel("Seconds:"))
        time_layout.addWidget(self.seconds_spin)
        
        layout.addLayout(time_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def get_time(self):
        return self.minutes_spin.value() * 60 + self.seconds_spin.value()

class AnnotationOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.annotations = []
        self.current_position = 0
        self.current_annotation = None
        
        # Make sure the overlay is visible
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet("background: transparent;")
        self.raise_()  # Make sure it's on top
        self.show()

    def set_annotations(self, annotations):
        self.annotations = annotations
        self.update()

    def update_position(self, position=None):
        if position is not None:
            self.current_position = position / 1000.0  # Convert to seconds
            # Find current annotation
            self.current_annotation = None
            for annotation in self.annotations:
                if annotation["start_time"] <= self.current_position <= annotation["end_time"]:
                    self.current_annotation = annotation
                    break
        self.update()

    def paintEvent(self, event):
        if not self.current_annotation:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set up text style
        font = painter.font()
        font.setPointSize(20)
        font.setBold(True)
        painter.setFont(font)

        # Draw semi-transparent background for text
        text = f"{self.current_annotation['movement_type']} - {self.current_annotation['score']}"
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(text)
        text_height = metrics.height()
        
        # Position text in the middle of the video
        x = (self.width() - text_width) // 2
        y = self.height() // 2

        # Draw background
        painter.setPen(QPen(QColor(0, 0, 0, 0)))
        painter.setBrush(QBrush(QColor(0, 0, 0, 128)))
        padding = 10
        painter.drawRect(x - padding, y - text_height - padding, 
                        text_width + 2 * padding, text_height + 2 * padding)

        # Draw text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(x, y, text)

    def resizeEvent(self, event):
        self.setGeometry(self.parent().geometry())
        super().resizeEvent(event)

    def showEvent(self, event):
        self.setGeometry(self.parent().geometry())
        super().showEvent(event)

class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rhythmic Gymnastics Annotation Tool")
        self.setGeometry(100, 100, 1000, 500)
        
        self.video_path = None
        self.annotations = []
        self.current_annotation = None
        self.selected_annotation_index = -1
        self.playback_speed = 1.0
        self.annotation_overlay = None
        self.active_movement_type = None 
        self.open_video_overlay_button = None
        
        #try:
            # #with open('src/rg_ai/annotation_tools/db_config.json', 'r') as f:
            #     self.config = json.load(f)
            #     self.movement_types = list(self.config["movement_types"].keys())
            #     if self.movement_types:
            #         self.active_movement_type = self.movement_types[0]
        #except FileNotFoundError:
            #QMessageBox.warning(self, "Warning", "Could not find db_config.json. Using default configuration.")
        self.config = {
            "movement_types": {
                "Jump": {
                    "GENERAL": 0.5,
                    "Split Leap": 0.4, # 0.3, 0.4, 0.5
                    "Stag Leap": 0.3, # 0.2, 0.3, 0.4
                    "Turning Split Leap": 0.5, # 0.4, 0.5, 0.6
                    "Turning Stag Leap": 0.4, # 0.3, 0.4, 0.5
                    "Butterfly": 0.5 # 0.5
                },
                "Balance": {
                    "GENERAL": 0.5,
                    "Front Split - Trunk Backward": 0.5, # 0.4, 0.5
                    "Front Split - Trunk Backward - Below Horizontal": 0.5, #
                    "Side Split - Trunk Horizontal": 0.5, # 0.4, 0.5
                    "Back Split - Trunk Horizontal": 0.5, # 0.4, 0.5
                    "Fouetté": 0.5, # 0.3, 0.5
                    
                },
                "Rotation": {
                    "GENERAL": 0.4,
                    "Spiral turn": 0.3, #0.1, 0.3
                    "Side Split - Trunk Horizontal": 0.5, # 0.4, 0.5
                    "Back Split": 0.4, # 0.3, 0.4
                    "Back Split - Trunk Horizontal": 0.5, # 0.4, 0.5, 0.6
                    "Attitude": 0.5, # 0.3, 0.5
                    "Ring with help": 0.3,
                    "Fouetté": 0.2, # 0.1, 0.2, 0.3
                    "Penché": 0.4,
                }
            }
        }
        self.movement_types = list(self.config["movement_types"].keys())
        if self.movement_types:
            self.active_movement_type = self.movement_types[0]
        
        # Create UI first
        self.create_ui()
        
        # Set Cursor
        self.setCursor(Qt.CursorShape.ArrowCursor)

        # Then setup media player
        self.setup_media_player()
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()

        # Set initial UI state for movement type
        if self.active_movement_type:
            self.set_movement_type(self.active_movement_type)

    def setup_media_player(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.errorOccurred.connect(self.handle_media_error)
        self.media_player.durationChanged.connect(self.setup_seek_bar)
        
        # Setup timer for position updates
        self.timer = QTimer()
        self.timer.setInterval(100)  # Update every 100ms
        self.timer.timeout.connect(self.update_time_display)
        self.timer.start()

    def handle_media_error(self, error, error_string):
        QMessageBox.warning(self, "Media Error", f"Error playing video: {error_string}")

    def create_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(10)
        splitter.setChildrenCollapsible(False)
        
        # Left side: Video and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video info display
        video_info_layout = QHBoxLayout()
        self.video_name_label = QLabel("No video loaded")
        self.video_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        video_info_layout.addWidget(self.video_name_label)
        left_layout.addLayout(video_info_layout)
        
        # Video container with overlay
        video_container = QFrame()
        video_container.setFrameShape(QFrame.Shape.StyledPanel)
        video_container.setMinimumSize(320, 240)
        video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a black frame to act as a stable background for the video widget
        self.video_display_container_frame = QFrame()
        self.video_display_container_frame.setStyleSheet("QFrame { background-color: black; }")
        self.video_display_container_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_display_container_frame.setMinimumSize(320, 180)
        self.video_display_container_frame.installEventFilter(self) # For open_video_overlay_button positioning

        # Layout for the black frame to hold the video widget
        video_display_frame_layout = QVBoxLayout(self.video_display_container_frame)
        video_display_frame_layout.setContentsMargins(0,0,0,0)

        # Create video widget (will be child of video_display_container_frame's layout)
        self.video_widget = QVideoWidget()
        # self.video_widget.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent) # May not be needed now
        # self.video_widget.setAutoFillBackground(True) # May not be needed now
        # self.video_widget.setStyleSheet("QVideoWidget { background-color: black; }") # Frame handles black bg
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        video_display_frame_layout.addWidget(self.video_widget) # Add video widget to the black frame
        video_container_layout.addWidget(self.video_display_container_frame) # Add black frame to main video area
        left_layout.addWidget(video_container)
        
        # Create annotation overlay (parented to video_widget as before)
        self.annotation_overlay = AnnotationOverlay(self.video_widget)
        
        # Create the "Open Video" button that overlays the video_display_container_frame
        self.open_video_overlay_button = QPushButton("Open Video", self.video_display_container_frame) # Parent to the frame
        self.open_video_overlay_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_video_overlay_button.clicked.connect(self.open_video)
        self.open_video_overlay_button.setStyleSheet("""
            QPushButton {
                font-size: 22px; 
                font-weight: bold; 
                padding: 18px; 
                background-color: rgba(80, 80, 80, 0.85); 
                color: white; 
                border: 1px solid rgba(200, 200, 200, 0.9);
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: rgba(100, 100, 100, 0.9);
            }
        """)
        # Initial state for the button
        if not self.video_path:
            self.open_video_overlay_button.show()
            self.open_video_overlay_button.raise_()
            # Position after UI is shown and sized
            QTimer.singleShot(100, self.position_open_video_button) 
        else:
            self.open_video_overlay_button.hide()
        
        # Movement type buttons
        movement_buttons_layout = QHBoxLayout()
        movement_buttons_layout.setSpacing(5)
        
        self.jump_button = QPushButton("Jump (J)")
        self.jump_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;                       
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.jump_button.clicked.connect(lambda: self.set_movement_type("Jump"))
        movement_buttons_layout.addWidget(self.jump_button)
        
        self.balance_button = QPushButton("Balance (B)")
        self.balance_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        self.balance_button.clicked.connect(lambda: self.set_movement_type("Balance"))
        movement_buttons_layout.addWidget(self.balance_button)
        
        self.rotation_button = QPushButton("Rotation (R)")
        self.rotation_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.rotation_button.clicked.connect(lambda: self.set_movement_type("Rotation"))
        movement_buttons_layout.addWidget(self.rotation_button)
        
        movement_buttons_layout.addStretch() # Add stretch after main movement buttons
        left_layout.addLayout(movement_buttons_layout) # Add this row of buttons to left_layout

        # Type selection label (e.g., "Jump Type:", "Rotation Type:")
        self.type_selection_label = QLabel("Type:") # Initial text, updated by set_movement_type
        left_layout.addWidget(self.type_selection_label) # Add label as a new row in left_layout

        # Container for all subtype buttons
        self.subtype_buttons_container_layout = QHBoxLayout()
        self.subtype_buttons_container_layout.setSpacing(5)
        
        # Create buttons for each type
        self.jump_type_buttons = []
        # Dynamically create buttons from config, ensuring "GENERAL" is first if it exists
        jump_subtypes = list(self.config["movement_types"]["Jump"].keys())
        if "GENERAL" in jump_subtypes:
            jump_subtypes.insert(0, jump_subtypes.pop(jump_subtypes.index("GENERAL")))

        for jump_type_original in jump_subtypes:
            jump_type_display = jump_type_original.replace(" - ", " -\n")
            btn = QPushButton(jump_type_display)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-height: 50px; /* Allow text wrapping */
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:checked {
                    background-color: #2E7D32;
                }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, t=jump_type_original: self.set_jump_type(t))
            self.jump_type_buttons.append(btn)
            self.subtype_buttons_container_layout.addWidget(btn) # Add to new container
            btn.hide()  # Initially hide the button
        
        self.balance_type_buttons = []
        # Dynamically create buttons from config, ensuring "GENERAL" is first if it exists
        balance_subtypes = list(self.config["movement_types"]["Balance"].keys())
        if "GENERAL" in balance_subtypes:
            balance_subtypes.insert(0, balance_subtypes.pop(balance_subtypes.index("GENERAL")))

        for balance_type_original in balance_subtypes:
            balance_type_display = balance_type_original.replace(" - ", " -\n")
            btn = QPushButton(balance_type_display)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-height: 50px; /* Allow text wrapping */
                }
                QPushButton:hover {
                    background-color: #0b7dda;
                }
                QPushButton:checked {
                    background-color: #1565C0;
                }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, t=balance_type_original: self.set_balance_type(t))
            self.balance_type_buttons.append(btn)
            self.subtype_buttons_container_layout.addWidget(btn) # Add to new container
            btn.hide()  # Initially hide the button
        
        self.rotation_subtype_buttons = []
        rotation_subtypes_keys = list(self.config["movement_types"]["Rotation"].keys())
        if "GENERAL" in rotation_subtypes_keys:
            rotation_subtypes_keys.insert(0, rotation_subtypes_keys.pop(rotation_subtypes_keys.index("GENERAL")))

        for rotation_subtype_original in rotation_subtypes_keys:
            rotation_subtype_display = rotation_subtype_original.replace(" - ", " -\n")
            btn = QPushButton(rotation_subtype_display)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-height: 50px; /* Allow text wrapping */
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:checked {
                    background-color: #b71c1c;
                }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, t=rotation_subtype_original: self.set_rotation_subtype(t))
            self.rotation_subtype_buttons.append(btn)
            self.subtype_buttons_container_layout.addWidget(btn) # Add to new container
            btn.hide()  # Initially hide the button
        
        self.subtype_buttons_container_layout.addStretch() # Add stretch after subtype buttons
        left_layout.addLayout(self.subtype_buttons_container_layout) # Add subtype buttons row to left_layout
        
        # Scoring buttons
        scoring_buttons_layout = QHBoxLayout()
        scoring_buttons_layout.setSpacing(5)
        
        self.score_buttons = []
        for score in [round(x * 0.1, 1) for x in range(0, 16)]:
            score_btn = QPushButton(f"{score}")
            score_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FFC107;
                    color: black;
                    border: none;
                    padding: 8px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #FFB300;
                }
                QPushButton:checked {
                    background-color: #FFA000;
                }
            """)
            score_btn.setCheckable(True)
            score_btn.clicked.connect(lambda checked, s=score: self.set_score(s))
            self.score_buttons.append(score_btn)
            scoring_buttons_layout.addWidget(score_btn)
        
        left_layout.addLayout(scoring_buttons_layout)
        
        # Video controls
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)
        
        # Seek bar with improved functionality
        seek_layout = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(0)
        self.seek_slider.setTracking(True)  # Update position while dragging
        self.seek_slider.sliderPressed.connect(self.seek_slider_pressed)
        self.seek_slider.sliderReleased.connect(self.seek_slider_released)
        self.seek_slider.sliderMoved.connect(self.seek_video)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #0b7dda;
            }
            QSlider::sub-page:horizontal {
                background: #2196F3;
                border-radius: 4px;
            }
        """)
        seek_layout.addWidget(self.seek_slider)
        controls_layout.addLayout(seek_layout)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        # Playback speed controls
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Playback Speed:"))
        
        self.speed_label = QLabel("Speed: 1.0x")
        speed_layout.addWidget(self.speed_label)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(2)
        self.speed_slider.setMaximum(20)
        self.speed_slider.setValue(10)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.valueChanged.connect(lambda value: self.set_playback_speed(value / 10))
        speed_layout.addWidget(self.speed_slider)
        
        playback_layout.addLayout(speed_layout)
        playback_layout.addStretch() # Stretch after speed controls

        # Define all playback/action buttons before adding them to the layout

        self.save_annotation_button = QPushButton("Save Annotation")
        self.save_annotation_button.setFixedHeight(40)
        self.save_annotation_button.clicked.connect(self.save_annotation)
        self.save_annotation_button.setStyleSheet("""
            QPushButton {
                background-color: #B90BD4; /* A shade of purple */
                    color: white;
                border: none;
                padding: 8px 12px; /* Adjust padding as needed */
                border-radius: 4px;
                    font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2E7D32; /* Darker green on hover */
            }
            QPushButton:pressed {
                background-color: #1B5E20; /* Even darker green when pressed */
            }
        """)

        self.seek_backward_button = QPushButton("« 1s")
        self.seek_backward_button.setToolTip("Seek backward 1 second (Left Arrow)")
        self.seek_backward_button.clicked.connect(self.seek_backward)
        self.seek_backward_button.setFixedHeight(40)

        self.play_button = QPushButton("Play")
        self.play_button.setFixedHeight(40)
        self.play_button.clicked.connect(self.toggle_media_playback)

        self.seek_forward_button = QPushButton("1s »")
        self.seek_forward_button.setToolTip("Seek forward 1 second (Right Arrow)")
        self.seek_forward_button.clicked.connect(self.seek_forward)
        self.seek_forward_button.setFixedHeight(40)

        

        # Add buttons in the desired order: SeekBack, Play, SeekForward, then SaveAnnotation
        playback_layout.addWidget(self.save_annotation_button)
        playback_layout.addWidget(self.seek_backward_button)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(self.seek_forward_button)
        
        # Time display (add a stretch before it to push it to the far right of buttons)
        playback_layout.addStretch() 
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(150)
        playback_layout.addWidget(self.time_label)
        
        controls_layout.addLayout(playback_layout)
        left_layout.addLayout(controls_layout)

        # Annotation controls
        #annotation_layout = QHBoxLayout()
        #annotation_layout.setSpacing(10)
        
        # Movement type selection
        self.subtype_combo = QComboBox()
        # annotation_layout.addWidget(QLabel("Subtype:"))
        # annotation_layout.addWidget(self.subtype_combo)
        
        self.score_input = QLineEdit()
        # self.score_input.setPlaceholderText("Enter score (e.g., 0.5)")
        # self.score_input.setFixedWidth(200)
        # self.score_input.returnPressed.connect(self.score_input.clearFocus)  # Add Enter key handling
        # annotation_layout.addWidget(QLabel("Score:"))
        # annotation_layout.addWidget(self.score_input)
        
        self.incorrect_checkbox = QCheckBox("Incorrect Execution")
        self.incorrect_checkbox.stateChanged.connect(self.handle_incorrect_execution)
        # annotation_layout.addWidget(self.incorrect_checkbox)
        
        # self.start_button = QPushButton("Mark Start (S)")
        # self.start_button.setFixedWidth(100)
        # self.start_button.clicked.connect(self.mark_start)
        # #annotation_layout.addWidget(self.start_button)
        
        # self.end_button = QPushButton("Mark End (E)")
        # self.end_button.setFixedWidth(100)
        # self.end_button.clicked.connect(self.mark_end)
        #annotation_layout.addWidget(self.end_button)
        
        #left_layout.addLayout(annotation_layout)
        
        # Comment section
        comment_layout = QVBoxLayout()
        comment_layout.setSpacing(5)
        comment_layout.addWidget(QLabel("Comments:"))
        self.comment_text = QTextEdit()
        self.comment_text.setPlaceholderText("Enter any additional comments about the movement...")
        self.comment_text.setMaximumHeight(100)
        comment_layout.addWidget(self.comment_text)
        left_layout.addLayout(comment_layout)

        # Add Save Annotation button
        # self.save_annotation_button = QPushButton("Save Annotation")
        # self.save_annotation_button.setFixedWidth(150)
        # self.save_annotation_button.clicked.connect(self.save_annotation)
        # left_layout.addWidget(self.save_annotation_button)
        
        # Right side: Annotation list
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create tab widget for annotations
        self.annotation_tabs = QTabWidget()
        self.annotation_tabs.setTabsClosable(True)
        self.annotation_tabs.tabCloseRequested.connect(self.close_annotation_tab)
        right_layout.addWidget(self.annotation_tabs)
        
        # Annotation list controls
        list_controls = QHBoxLayout()
        self.edit_button = QPushButton("Edit Selected")
        self.edit_button.clicked.connect(self.edit_selected_annotation)
        self.edit_button.setEnabled(False)
        list_controls.addWidget(self.edit_button)
        
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self.delete_selected_annotation)
        self.delete_button.setEnabled(False)
        list_controls.addWidget(self.delete_button)
        
        right_layout.addLayout(list_controls)
        
        # Add Finished & Export button
        self.finished_button = QPushButton("Finished and Export file")
        self.finished_button.setToolTip("Export all annotations to a JSON file")
        self.finished_button.clicked.connect(self.export_annotations)
        self.finished_button.setStyleSheet("""
            QPushButton {
                background-color: #4A148C; /* Darker Purple */
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #6A1B9A; /* Medium Purple */
            }
            QPushButton:pressed {
                background-color: #38006B; /* Very Dark Purple */
            }
        """)
        self.finished_button.setFixedHeight(40) # Consistent height
        right_layout.addWidget(self.finished_button)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial sizes (video section gets more space)
        total_width = self.width()
        splitter.setSizes([int(total_width * 0.8), int(total_width * 0.2)])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Create menu bar
        self.create_menu_bar()
        
        # Connect signals after all UI elements are created
        self.subtype_combo.currentTextChanged.connect(self.update_score)

    def handle_incorrect_execution(self, state):
        if state == Qt.CheckState.Checked.value:
            self.score_input.setText("0.0")
            self.score_input.setEnabled(False)
        else:
            self.score_input.setEnabled(True)

    def mark_start(self):
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first")
            return False

        position = self.media_player.position() / 1000.0  # Convert to seconds
        self.current_annotation = {
            "video": self.video_path.split("/")[-1],
            "start_time": round(position, 2),
            "end_time": None,
            "movement_type": None, # To be filled on save
            "subtype": None,       # To be filled on save
            "score": None,         # To be filled on save
            "incorrect_execution": False, # Default, to be filled on save
            "comments": ""         # To be filled on save
        }
        self.statusBar().showMessage(f"Annotation started at {position:.2f}s. Press Space again to mark end.")
        return True

    def mark_end(self):
        if not self.current_annotation or "start_time" not in self.current_annotation or self.current_annotation.get("start_time") is None:
            QMessageBox.warning(self, "Warning", "Please mark start time first (Press Space).")
            return False

        position = self.media_player.position() / 1000.0  # Convert to seconds
        
        if round(position, 2) <= self.current_annotation["start_time"]:
            QMessageBox.warning(self, "Warning", f"End time ({position:.2f}s) must be after start time ({self.current_annotation['start_time']:.2f}s).")
            return False
            
        self.current_annotation["end_time"] = round(position, 2)
        self.statusBar().showMessage(f"Annotation end marked at {position:.2f}s. Fill details and click 'Save Annotation'.")
        # Video is already paused by handle_spacebar_press
        return True

    def open_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", 
                                                 "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.video_path = file_name
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
            self.video_name_label.setText(f"Current Video: {file_name.split('/')[-1]}")
            self.statusBar().showMessage(f"Loaded: {file_name}")
            
            if self.open_video_overlay_button: # Ensure button exists
                self.open_video_overlay_button.hide() # Hide the overlay button
            
            # Set seek bar range to video duration
            self.media_player.durationChanged.connect(self.setup_seek_bar)
            
            # Start playing automatically
            self.media_player.play()
            self.play_button.setText("Pause")

    def setup_seek_bar(self, duration):
        """Set up the seek bar with the video duration"""
        self.seek_slider.setMaximum(duration)
        self.seek_slider.setValue(0)
        self.update_time_display()
        # Set video duration in overlay
        if self.annotation_overlay:
            self.annotation_overlay.set_annotations(self.annotations)

    def update_position(self, position):
        """Update the seek slider position when video is playing"""
        if not self.seek_slider.isSliderDown():  # Only update if user isn't dragging
            self.seek_slider.setValue(position)
        self.update_time_display()
        
        # Show current annotation if any
        current_time = position / 1000.0  # Convert to seconds
        current_annotation = None
        for annotation in self.annotations:
            if annotation["video"] == self.video_path.split("/")[-1]:
                if annotation["start_time"] <= current_time <= annotation["end_time"]:
                    current_annotation = annotation
                    break
        
        if current_annotation:
            self.video_widget.setStyleSheet(f"""
                QVideoWidget {{
                    background-color: transparent;
                }}
                QVideoWidget::after {{
                    content: "{current_annotation['movement_type']} - {current_annotation['score']}";
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    background-color: rgba(0, 0, 0, 0.5);
                    padding: 10px;
                    border-radius: 5px;
                }}
            """)
        else:
            self.video_widget.setStyleSheet("QVideoWidget { background-color: transparent; }")

    def update_time_display(self):
        """Update the time display with current position and duration"""
        duration = self.media_player.duration()
        position = self.media_player.position()
        
        # Convert to minutes and seconds
        duration_min = duration // 60000
        duration_sec = (duration % 60000) // 1000
        position_min = position // 60000
        position_sec = (position % 60000) // 1000
        
        self.time_label.setText(f"{position_min:02d}:{position_sec:02d} / {duration_min:02d}:{duration_sec:02d}")

    def export_annotations(self):
        if not self.annotations:
            QMessageBox.warning(self, "Warning", "No annotations to export")
            return

        # Group annotations by video
        video_groups = {}
        for annotation in self.annotations:
            video_name = annotation["video"]
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(annotation)

        # Convert to nested dictionary format
        export_data = {}
        for video_name, annotations in video_groups.items():
            export_data[video_name] = {}
            for i, annotation in enumerate(annotations):
                # Remove video name from individual annotations since it's now the key
                annotation_copy = annotation.copy()
                del annotation_copy["video"]
                export_data[video_name][f"annotation_{i+1}"] = annotation_copy

        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"annotations_{timestamp}.json"
        
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Annotations", default_filename, 
                                                 "JSON Files (*.json)")
        if file_name:
            with open(file_name, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.statusBar().showMessage(f"Annotations saved to {file_name}")

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        export_action = QAction("Export Annotations", self)
        export_action.triggered.connect(self.export_annotations)
        file_menu.addAction(export_action)

        clear_action = QAction("Clear All Annotations", self)
        clear_action.triggered.connect(self.clear_annotations)
        file_menu.addAction(clear_action)

    def setup_shortcuts(self):
        # Movement type shortcuts
        self.jump_shortcut = QShortcut(QKeySequence("J"), self)
        self.jump_shortcut.activated.connect(lambda: self.set_movement_type("Jump"))
        
        self.balance_shortcut = QShortcut(QKeySequence("B"), self)
        self.balance_shortcut.activated.connect(lambda: self.set_movement_type("Balance"))
        
        self.rotation_shortcut = QShortcut(QKeySequence("R"), self)
        self.rotation_shortcut.activated.connect(lambda: self.set_movement_type("Rotation"))
        
        # Playback and Annotation shortcuts
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.space_shortcut.activated.connect(self.handle_spacebar_press)
        
        self.left_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self)
        self.left_shortcut.activated.connect(self.seek_backward)
        
        self.right_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self)
        self.right_shortcut.activated.connect(self.seek_forward)

    def seek_backward(self):
        if self.video_path:
            current_pos = self.media_player.position()
            self.media_player.setPosition(max(0, current_pos - 1000))  # 1 second back

    def seek_forward(self):
        if self.video_path:
            current_pos = self.media_player.position()
            duration = self.media_player.duration()
            self.media_player.setPosition(min(duration, current_pos + 1000))  # 1 second forward

    def set_playback_speed(self, speed):
        self.playback_speed = speed
        self.media_player.setPlaybackRate(speed)
        self.speed_label.setText(f"Speed: {speed}x")

    def set_movement_type(self, movement_type):
        # Debug print to see what movement type is being set
        print(f"Setting movement type to: {movement_type}")
        self.active_movement_type = movement_type # Set the active type
        self.update_subtypes() # Update subtypes based on the new active type
        
        # Hide all specific type buttons first (from all categories)
        # and ensure they are unchecked.
        all_button_lists = [self.jump_type_buttons, self.balance_type_buttons, self.rotation_subtype_buttons]
        for button_list in all_button_lists:
            for btn in button_list:
                btn.hide()
                btn.setChecked(False)

        # Update label and show appropriate buttons
        if movement_type == "Jump":
            print("Configuring for Jump type buttons") # Consistent debug print
            self.type_selection_label.setText("Jump Type:")
            for btn in self.jump_type_buttons:
                btn.show()
        elif movement_type == "Balance":
            print("Configuring for Balance type buttons") # Consistent debug print
            self.type_selection_label.setText("Balance Type:")
            for btn in self.balance_type_buttons:
                btn.show()
        elif movement_type == "Rotation":
            print("Configuring for Rotation type buttons") # Consistent debug print
            self.type_selection_label.setText("Rotation Type:")
            for btn in self.rotation_subtype_buttons:
                btn.show()
        else:
            # If movement_type is not recognized, reset the label
            self.type_selection_label.setText("Type:")
            # All buttons are already hidden by the loop above

        # Ensure the label is visible (it's always part of the layout)
        self.type_selection_label.show()

    def set_jump_type(self, jump_type_original):
        # Uncheck all other buttons and check the selected one
        expected_display_text = jump_type_original.replace(" - ", " -\n")
        for btn in self.jump_type_buttons:
            if btn.text() == expected_display_text:
                btn.setChecked(True)
            else:
                btn.setChecked(False)
        
        # Set the subtype in the combo box
        index = self.subtype_combo.findText(jump_type_original)
        if index >= 0:
            self.subtype_combo.setCurrentIndex(index)

    def set_balance_type(self, balance_type_original):
        # Uncheck all other buttons and check the selected one
        expected_display_text = balance_type_original.replace(" - ", " -\n")
        for btn in self.balance_type_buttons:
            if btn.text() == expected_display_text:
                btn.setChecked(True)
            else:
                btn.setChecked(False)
        
        # Set the subtype in the combo box
        index = self.subtype_combo.findText(balance_type_original)
        if index >= 0:
            self.subtype_combo.setCurrentIndex(index)

    def set_rotation_subtype(self, rotation_subtype_original):
        # Uncheck all other buttons and check the selected one
        expected_display_text = rotation_subtype_original.replace(" - ", " -\n")
        for btn in self.rotation_subtype_buttons:
            if btn.text() == expected_display_text:
                btn.setChecked(True)
            else:
                btn.setChecked(False)
        
        # Set the subtype in the combo box
        index = self.subtype_combo.findText(rotation_subtype_original)
        if index >= 0:
            self.subtype_combo.setCurrentIndex(index)

    def update_subtypes(self):
        current_type = self.active_movement_type # Use active_movement_type
        self.subtype_combo.clear()
        
        if not current_type: # Add guard for None
            return

        if current_type in self.config["movement_types"]:
            subtypes = list(self.config["movement_types"][current_type].keys())
            # For Jump type, ensure our quick selection options are at the top
            # if current_type == "Jump":
            #     # First add our quick selection options
            #     quick_options = ["General", "Split Leap", "Ring Leap"]  # Set "General" as the first option
            #     for option in quick_options:
            #         if option in subtypes:
            #             self.subtype_combo.addItem(option)
            #             subtypes.remove(option)
            #     # Then add any remaining subtypes
            #     for subtype in subtypes:
            #         self.subtype_combo.addItem(subtype)
            #     self.subtype_combo.setCurrentText("General")  # Set "General" as the default
            #else:
                # For other types, just add all subtypes
            self.subtype_combo.addItems(subtypes) # Add all subtypes
            # Set "GENERAL" as the default if it exists, otherwise the first item.
            if "GENERAL" in subtypes:
                self.subtype_combo.setCurrentText("GENERAL")
            elif subtypes: # if subtypes is not empty
                self.subtype_combo.setCurrentIndex(0)

            self.update_score()  # Update score when movement type changes

    def update_score(self):
        current_type = self.active_movement_type # Use active_movement_type
        current_subtype = self.subtype_combo.currentText()
        
        if current_type in self.config["movement_types"] and current_subtype in self.config["movement_types"][current_type]:
            try:
                prescribed_score = self.config["movement_types"][current_type][current_subtype]
                self.score_input.setText(str(prescribed_score))
            except KeyError:
                print(f"Warning: No prescribed score found for {current_type} - {current_subtype}")
                self.score_input.clear()

    def update_annotation_table(self):
        # Store current tab index
        current_tab_index = self.annotation_tabs.currentIndex()
        current_video = self.annotation_tabs.tabText(current_tab_index) if current_tab_index >= 0 else None
        
        # Clear all existing tabs
        self.annotation_tabs.clear()
        
        # Group annotations by video
        video_groups = {}
        for annotation in self.annotations:
            video_name = annotation["video"]
            if video_name not in video_groups:
                video_groups[video_name] = []
            video_groups[video_name].append(annotation)
        
        # Sort annotations by start time for each video
        for video_name in video_groups:
            video_groups[video_name] = sorted(video_groups[video_name], key=lambda x: x["start_time"])
        
        # Update video overlay with current video's annotations
        if self.video_path:
            current_video_name = self.video_path.split("/")[-1]
            if current_video_name in video_groups:
                self.annotation_overlay.set_annotations(video_groups[current_video_name])
        
        # Create a tab for each video
        for video_name, annotations in video_groups.items():
            # Create a new table widget for this video
            table = QTableWidget()
            table.setColumnCount(7)
            table.setHorizontalHeaderLabels([
                "Type", "Subtype", "Start", "End", "Score", "Incorrect", "Comments"
            ])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
            
            # Add rows for annotations
            table.setRowCount(len(annotations))
            for row, annotation in enumerate(annotations):
                for col, key in enumerate(["movement_type", "subtype", "start_time", "end_time", "score", "incorrect_execution", "comments"]):
                    if key in ["start_time", "end_time"]:
                        value = f"{annotation[key]:.2f}s"
                    elif key == "incorrect_execution":
                        value = "Yes" if annotation[key] else "No"
                    elif key == "score":
                        try:
                            value = f"{annotation[key]:.2f}"
                        except:
                            value = "N/A"
                    else:
                        value = str(annotation[key])
                    
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                    item.setData(Qt.ItemDataRole.UserRole, annotation)
                    table.setItem(row, col, item)
            
            # Connect signals
            table.cellChanged.connect(lambda row, col, t=table: self.handle_cell_changed(row, col, t))
            table.itemSelectionChanged.connect(lambda t=table: self.annotation_selected(t))
            
            # Add the table to a new tab
            self.annotation_tabs.addTab(table, video_name)
        
        # Restore the current tab if it exists
        if current_video:
            for i in range(self.annotation_tabs.count()):
                if self.annotation_tabs.tabText(i) == current_video:
                    self.annotation_tabs.setCurrentIndex(i)
                    break

    def annotation_selected(self, table):
        selected_items = table.selectedItems()
        if selected_items:
            # Get the first selected row
            row = selected_items[0].row()
            # Get the item from the first column to get the annotation
            item = table.item(row, 0)  # Always use first column for annotation data
            if item:
                # Get the annotation data
                annotation = item.data(Qt.ItemDataRole.UserRole)
                if annotation:
                    # Find the annotation index in the main list
                    self.selected_annotation_index = self.annotations.index(annotation)
                    self.edit_button.setEnabled(True)
                    self.delete_button.setEnabled(True)
                    return

        self.selected_annotation_index = -1
        self.edit_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def handle_cell_changed(self, row, column, table):
        # Get the item that was changed
        item = table.item(row, column)
        if not item:
            return

        # Get the annotation data
        annotation = item.data(Qt.ItemDataRole.UserRole)
        if not annotation:
            return

        # Get the column name
        column_names = ["movement_type", "subtype", "start_time", "end_time", "score", "incorrect_execution", "comments"]
        key = column_names[column]

        # Update the annotation based on the column
        if key in ["start_time", "end_time"]:
            try:
                # Remove 's' suffix and convert to float
                value = float(item.text().rstrip('s'))
                annotation[key] = value
                # If it's start_time, update the video position
                if key == "start_time":
                    self.media_player.setPosition(int(value * 1000))
                # Update the display format
                item.setText(f"{value:.2f}s")
            except ValueError:
                return
        elif key == "incorrect_execution":
            annotation[key] = item.text() == "Yes"
            # Update the display format
            item.setText("Yes" if annotation[key] else "No")
        elif key == "score":
            try:
                value = float(item.text())
                annotation[key] = value
                # Update the display format
                item.setText(f"{value:.2f}")
            except ValueError:
                return
        else:
            annotation[key] = item.text()

        # Update the annotation in the main list
        for i, ann in enumerate(self.annotations):
            if ann == annotation:
                self.annotations[i] = annotation
                break

    def edit_selected_annotation(self):
        if self.selected_annotation_index == -1:
            return

        # Get the current table widget
        table = self.annotation_tabs.currentWidget()
        if not table:
            return

        # Get the selected row
        selected_items = table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        # Get the annotation data
        item = table.item(row, 0)
        if not item:
            return

        annotation = item.data(Qt.ItemDataRole.UserRole)
        if not annotation:
            return

        # Create a dialog with all fields
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation")
        layout = QVBoxLayout(dialog)

        # Create input widgets for each field
        fields = {}
        for col, key in enumerate(["movement_type", "subtype", "start_time", "end_time", "score", "incorrect_execution", "comments"]):
            field_layout = QHBoxLayout()
            field_layout.addWidget(QLabel(key.replace("_", " ").title() + ":"))
            
            if key in ["start_time", "end_time"]:
                input_widget = QLineEdit(f"{annotation[key]:.2f}")
            elif key == "incorrect_execution":
                input_widget = QCheckBox()
                input_widget.setChecked(annotation[key])
            elif key == "score":
                input_widget = QLineEdit(str(annotation[key]))
            else:
                input_widget = QLineEdit(str(annotation[key]))
            
            fields[key] = input_widget
            field_layout.addWidget(input_widget)
            layout.addLayout(field_layout)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Update all fields
            for key, widget in fields.items():
                if key in ["start_time", "end_time"]:
                    try:
                        value = float(widget.text())
                        annotation[key] = value
                        if key == "start_time":
                            self.media_player.setPosition(int(value * 1000))
                    except ValueError:
                        continue
                elif key == "incorrect_execution":
                    annotation[key] = widget.isChecked()
                elif key == "score":
                    try:
                        value = float(widget.text())
                        annotation[key] = value
                    except ValueError:
                        continue
                else:
                    annotation[key] = widget.text()

            # Update the table
            for col, key in enumerate(["movement_type", "subtype", "start_time", "end_time", "score", "incorrect_execution", "comments"]):
                if key in ["start_time", "end_time"]:
                    value = f"{annotation[key]:.2f}s"
                elif key == "incorrect_execution":
                    value = "Yes" if annotation[key] else "No"
                elif key == "score":
                    value = f"{annotation[key]:.2f}"
                else:
                    value = str(annotation[key])
                
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                item.setData(Qt.ItemDataRole.UserRole, annotation)
                table.setItem(row, col, item)

            # Update the annotation in the main list
            self.annotations[self.selected_annotation_index] = annotation

    def delete_selected_annotation(self):
        if self.selected_annotation_index == -1:
            return

        reply = QMessageBox.question(self, "Delete Annotation",
                                   "Are you sure you want to delete this annotation?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            del self.annotations[self.selected_annotation_index]
            self.update_annotation_table()
            self.statusBar().showMessage(f"Annotation deleted. Total: {len(self.annotations)}")

    def seek_slider_pressed(self):
        """Pause video when user starts dragging the seek slider"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")

    def seek_slider_released(self):
        """Resume video playback if it was playing before seeking"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.media_player.play()
            self.play_button.setText("Pause")

    def seek_video(self, position):
        """Seek to the specified position in the video"""
        self.media_player.setPosition(position)
        self.update_time_display()

    def toggle_media_playback(self):
        """Purely toggles media playback state and updates the play button text."""
        if not self.video_path:
            return
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.play_button.setText("Play")
        else:
            self.media_player.play()
            self.play_button.setText("Pause")

    def handle_spacebar_press(self):
        """Handles annotation start/end marking via spacebar."""
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first.")
            return

        if self.current_annotation is None or self.current_annotation.get("start_time") is None:
            # Start a new annotation
            self.mark_start()
            # if self.mark_start():
            #     if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            #         self.media_player.pause()
            #         self.play_button.setText("Play")
        elif self.current_annotation.get("end_time") is None:
            # End the current annotation
            if self.mark_end():
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                    self.media_player.pause()
                    self.play_button.setText("Play")
            # Video remains paused as it was paused on mark_start
        else:
            self.mark_start()
            # if self.mark_start(): # Start a new one, overwriting unsaved current_annotation
            #     if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            #         self.media_player.pause()
            #         self.play_button.setText("Play")

    def set_score(self, score):
        # Set the score in the input field
        self.score_input.setText(str(score))
        self.statusBar().showMessage(f"Score set to {score}")

        # Update check state for all score buttons to ensure exclusivity
        for btn in self.score_buttons:
            # QPushButton stores text as str, score is float, compare carefully
            if btn.text() == f"{score:.1f}" or btn.text() == str(score): # Handle cases like 1.0 vs 1
                btn.setChecked(True)
            else:
                btn.setChecked(False)

    def save_annotation(self):
        if not self.current_annotation or self.current_annotation.get("end_time") is None:
            QMessageBox.warning(self, "Warning", "Annotation not fully marked (start and end times required). Press Space to mark start/end.")
            return

        try:
            score_text = self.score_input.text()
            score_value = float(score_text) if score_text.strip() else None # Handle empty score_input
            
            if self.incorrect_checkbox.isChecked():
                score_value = 0.0 # Override score if incorrect execution
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid score (number).")
            return

        # Populate details from UI into the current_annotation
        self.current_annotation["movement_type"] = self.active_movement_type # Use active_movement_type
        self.current_annotation["subtype"] = self.subtype_combo.currentText()
        self.current_annotation["score"] = score_value
        self.current_annotation["incorrect_execution"] = self.incorrect_checkbox.isChecked()
        self.current_annotation["comments"] = self.comment_text.toPlainText()
        
        # Find the correct position to insert the new annotation to keep the list sorted
        insert_index = 0
        for i, annotation in enumerate(self.annotations):
            try:
                if self.current_annotation["start_time"] < annotation["start_time"]:
                    insert_index = i
                    break
                else:
                    insert_index = i + 1
            except KeyError:
                # Fallback if an existing annotation is malformed (should not happen with new logic)
                print(f"Warning: Malformed existing annotation found: {annotation}")
        
        self.annotations.insert(insert_index, dict(self.current_annotation)) # Store a copy
        
        self.current_annotation = None 
        
        self.statusBar().showMessage(f"Annotation saved. Total: {len(self.annotations)}")
        
        # Reset UI fields for the next annotation
        self.score_input.clear()
        self.incorrect_checkbox.setChecked(False)
        self.comment_text.clear()
        # Consider resetting movement type and subtype or leaving them as is for batch annotations
        # self.movement_type.setCurrentIndex(0) 
        # self.update_subtypes()

        self.update_annotation_table()

        # Resume video playback after saving, if not already playing
        if self.media_player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.play()
            self.play_button.setText("Pause")

    def close_annotation_tab(self, index):
        # Get the video name from the tab
        video_name = self.annotation_tabs.tabText(index)
        # Just remove the tab, don't delete annotations
        self.annotation_tabs.removeTab(index)
        self.statusBar().showMessage(f"Tab for {video_name} closed")

    def clear_annotations(self):
        if not self.annotations:
            return

        reply = QMessageBox.question(self, "Clear Annotations",
                                   "Are you sure you want to clear all annotations? This cannot be undone.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.annotations = []
            self.update_annotation_table()
            self.statusBar().showMessage("All annotations cleared")

    def position_open_video_button(self):
        if not self.open_video_overlay_button or not self.video_display_container_frame: # Check frame
            return
        
        if self.open_video_overlay_button.isVisible():
            button_hint = self.open_video_overlay_button.sizeHint()
            
            btn_w = max(180, button_hint.width()) 
            btn_h = max(70, button_hint.height()) 

            max_w = self.video_display_container_frame.width() - 40 # Use frame width
            max_h = self.video_display_container_frame.height() - 40 # Use frame height
            
            actual_w = min(btn_w, max_w)
            actual_h = min(btn_h, max_h)

            if actual_w <= 0 or actual_h <=0: 
                return

            self.open_video_overlay_button.resize(actual_w, actual_h)
            
            x = (self.video_display_container_frame.width() - actual_w) / 2 # Use frame width
            y = (self.video_display_container_frame.height() - actual_h) / 2 # Use frame height
            self.open_video_overlay_button.move(int(x), int(y))
            self.open_video_overlay_button.raise_()

    def eventFilter(self, source, event):
        if source == self.video_display_container_frame and event.type() == QEvent.Type.Resize: # Monitor frame
            QTimer.singleShot(0, self.position_open_video_button)
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        """
        Handles the event when the user tries to close the window.
        Prompts for confirmation to prevent accidental data loss.
        """
        if not self.annotations: # If no annotations, close without prompt or if already saved.
                               # For simplicity now, prompt if any annotations exist.
                               # A more advanced check might see if annotations are "dirty" (unsaved changes).
            event.accept()
            return

        reply = QMessageBox.question(self, "Confirm Exit",
                                   "Are you sure you want to close the application?\n"
                                   "Any unsaved annotations will be lost. \n"
                                   "Consider using 'File > Export Annotations' or the 'Finished & Export' button.",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No) # Default to No

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()  # User confirmed, so close the window
        else:
            event.ignore()  # User cancelled, so do nothing and keep the window open

def main():
    app = QApplication(sys.argv)
    window = AnnotationTool()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()