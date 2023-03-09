import torch
import pandas as pd
import datetime as dt
from pathlib import Path
from PIL import Image
from torchvision import transforms

class SDOBenchmark():
    def __init__(self, csv_file, root_folder, transform ):
        metadata = pd.read_csv(csv_file, parse_dates=["start", "end"])
        self.root_folder = root_folder
        self.transform = transform
        self.time_steps_values = [0, 7 * 60, 10 * 60 + 30, 11 * 60 + 50]
        self.time_steps = [0, 1, 2, 3]

        self.setup(metadata)

    def target_transform(self, flux):
        if flux >= 1e-6:
            return 1
        return 0

    def setup(self, metadata):
        ls = []
        for i in range(len(metadata)):
            sample_metadata = metadata.iloc[i]
            target = sample_metadata["peak_flux"]

            sample_active_region, sample_date = sample_metadata["id"].split("_", maxsplit = 1)

            path_check = []

            for time_step in self.time_steps:
                image_date = sample_metadata["start"] + dt.timedelta(minutes = self.time_steps_values[time_step])
                image_date_str = dt.datetime.strftime(image_date, "%Y-%m-%dT%H%M%S")
                image_name = f"{image_date_str}__magnetogram.jpg"
                path_check.append(Path(sample_active_region) / sample_date / image_name)

            if not all((self.root_folder / path).exists() for path in path_check):

                continue

            ls.append((path_check, target))

        self.ls = ls

    def __len__(self):
        return len(self.ls)

    def __getitem__(self, index):
        metadata = self.ls[index]
        target = metadata[1]
        images = [Image.open(self.root_folder / path) for path in metadata[0]]
        to_tensor = transforms.ToTensor()

        images = [to_tensor(image) for image in images]
        if self.transform :
            images = [self.transform(image) for image in images]



        image = torch.cat(images, 0)

        target = self.target_transform(target)

        return image, target

    def y(self, indices = None):
        ls = self.ls
        if indices is not None:
            ls = (self.ls[i] for i in indices)

        return [self.target_transform(y[1]) for y in ls]




