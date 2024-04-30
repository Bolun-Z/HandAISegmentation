from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
import os
import torch
import numpy as np

class CreatDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, dataset_path, data_arg=True, mode='Train'):
        super(CreatDataset, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.mode = mode
        self.data_arg = data_arg

    def __getitem__(self, index):
        filename = self.annotation_lines[index].split()[0]
        image_path = os.path.join(self.dataset_path, "Image", filename)
        label_path = os.path.join(self.dataset_path, "Label", filename)

        image = sitk.ReadImage(image_path)
        image = self.resize_image(image, [self.input_shape[1], self.input_shape[2], self.input_shape[0]], is_label=False)
        image_array = sitk.GetArrayFromImage(image)
        image_array = torch.FloatTensor(image_array).unsqueeze(0)

        label = sitk.ReadImage(label_path, sitk.sitkUInt8)
        label = self.resize_image(label, new_size=[self.input_shape[1], self.input_shape[2], self.input_shape[0]], is_label=True)
        label_array = sitk.GetArrayFromImage(label)

        return image_array, label_array

    def __len__(self):
        return self.length

    def load_data(self):
        return self

    def resize_image(self, sitk_image, new_size=[256, 512, 1], is_label=False):
        '''
        sitk_image:
        new_spacing: x,y,z
        is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
        '''
        size = np.array(sitk_image.GetSize())
        spacing = np.array(sitk_image.GetSpacing())
        new_size = np.array(new_size)
        new_spacing_refine = size * spacing / new_size
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(s) for s in new_size]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(sitk_image.GetDirection())
        resample.SetOutputOrigin(sitk_image.GetOrigin())
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing_refine)

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)

        itk_image = resample.Execute(sitk_image)
        return itk_image


