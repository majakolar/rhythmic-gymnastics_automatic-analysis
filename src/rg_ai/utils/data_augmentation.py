import os
from datetime import datetime

import albumentations as A
import matplotlib.pyplot as plt


class DataAugmentor:
    def __init__(self):
        self.first_frame_data = None

    def augment_data(self, frame, current_frame, currently_plotting, base_save_dir, video_name):
        if current_frame == 0:
            transform = A.ReplayCompose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=10, p=0.3),
                    A.Perspective(scale=(0, 0.05), p=0.6),
                ]
            )
            self.first_frame_data = transform(image=frame)  # Store first frame transformation
            frame = self.first_frame_data["image"]
        else:
            frame = A.ReplayCompose.replay(self.first_frame_data["replay"], image=frame)["image"]

        if currently_plotting and current_frame==0:
            os.makedirs(base_save_dir, exist_ok=True)
            self.plot_frame(frame, base_save_dir, video_name)

        return frame

    def plot_frame(self, frame, base_save_dir, video_name):
        # Plot frame for sanity check
        # if currently_plotting and current_frame % 30 == 0:
        plt.figure(figsize=(10, 6))
        plt.imshow(frame)
        plt.axis("off")
        plt.savefig(
            os.path.join(
                base_save_dir,
                "fig",
                f"augmented_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{video_name}.png",
            )
        )