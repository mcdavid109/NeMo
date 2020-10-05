# Copyright (c) 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Anh Tuan Nguyen"


from os import makedirs
from os.path import expanduser, join, exists, isdir

import json
import cv2
from PIL import Image

import torch
from torchvision.transforms import transforms
from torchvision.io import read_video
from torchvision.datasets.utils import download_and_extract_archive, check_md5

from typing import Any, Optional
from dataclasses import dataclass

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

from nemo.utils import logging
from nemo.core.classes import Dataset

# from nemo.core.neural_types import CategoricalValuesType, ChannelType, MaskType, NeuralType, RegressionValuesType

from nemo.utils.configuration_parsing import get_value_from_dictionary, get_value_list_from_dictionary
from nemo.utils.configuration_error import ConfigurationError

# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class CLEVRERConfig:
	"""
	Structured config for the CLEVRER dataset.

	For more details please refer to:
	http://clevrer.csail.mit.edu/

	Args:
		_target_: Specification of dataset class
		root: Folder where task will store data (DEFAULT: "~/data/clevrer")
		split: Defines the set (split) that will be used (Options: train | val | test ) (DEFAULT: train)
		stream_frames: Flag indicating whether the task will load and return frames in the video (DEFAULT: True)
		transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
		download: downloads the data if not present (DEFAULT: True)
	"""

	# Dataset target class name.
	_target_: str = "nemo.collections.vis.datasets.CLEVRER"
	root: str = "~/data/clevrer"
	split: str = "train"
	stream_frames: bool = True
	# transform: Optional[Any] = None # Provided manually?
	download: bool = True


# Register the config.
cs.store(
	group="nemo.collections.vis.datasets",
	name="CLEVRER",
	node=ObjectConf(target="nemo.collections.vis.datasets.CLEVRER", params=CLEVRERConfig()),
)


class CLEVRER(Dataset):
	"""
	Class fetching data from the CLEVRER (Video Question Answering for Temporal and Causal Reasoning) dataset.

	The CLEVRER dataset consists of the followings:

		- 20,000 videos, separated into train (index 0 - 9999), validation (index 10000 - 14999), and test (index 15000 - 19999) splits.
		- Questions which are categorized into descriptives, explanatory, predictive and counterfactual
		- Annotation files which contain object properties, motion trajectories and collision events

	For more details please refer to the associated _website or _paper.

	After downloading and extracting, we will have the following directory

	data/clevrer
	videos/
		train/
			video_00000-01000
			...
			video_09000-10000
		val/
			video_10000-11000
			...
			video_14000-15000
		test/
			video_15000-16000
			...
			video_19000-20000
	annotations/
		train/
			annotation_00000-01000
			...
			annotation_09000-10000
		val/
			annotation_11000-12000
			...
			annotation_14000-15000
	questions/
		train.json
		validation.json
		test.json

	.. _website: http://clevrer.csail.mit.edu/

	.._paper: https://arxiv.org/pdf/1910.01442

	"""
	download_url_prefix_videos = "http://data.csail.mit.edu/clevrer/videos"
	download_url_prefix_annotations = "http://data.csail.mit.edu/clevrer/annotations"
	download_url_prefix_questions = "http://data.csail.mit.edu/clevrer/questions"

	videos_names = {"train": "video_train.zip", "dev": "video_validation.zip", "test": "video_test.zip"}
	videos_md5s = {"train": "8bcd8cda154e813ce009b2ee226abf7c", "dev": "948abe8ec22083de11796919cbee36eb", "test": "3cfa74a01e8527026a589343a4d1fd9e"}
	annotations_names = {"train": "annotation_train.zip", "dev": "annotation_validation.zip"}
	annotations_md5s = {"train": "41656fca05e59a673e46763162375b6d", "dev": "e64f0eb54e37e6f96f27139c7182f497"}
	question_names = {"train": "train.json", "dev": "validation.json", "test": "test.json"}

	def __init__(
		self,
		root: str = "~/data/clevrer",
		split: str = "train",
		stream_frames: bool = True,
		transform: Optional[Any] = None,
		download: bool = True,
	):
		"""
		Initializes dataset object. Calls base constructor.
		Downloads the dataset if not present and loads the adequate files depending on the mode.

		Args:
		root: Folder where task will store data (DEFAULT: "~/data/clevrer")
			split: Defines the set (split) that will be used (Options: train | val | test) (DEFAULT: train)
			stream_images: Flag indicating whether the task will return frames from the video (DEFAULT: True)
			transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
			download: downloads the data if not present (DEFAULT: True)
		"""

		# Call constructors of parent class.
		super().__init__()

		# Get the absolute path.
		self._root = expanduser(root)

		# Process split.
		self._split = get_value_from_dictionary(
			split,
			"train | val | test".split(
				" | "
			),
		)

		# Download dataset when required.
		if download:
			self.download()

		# Get flag informing whether we want to stream frames back to user or not.
		self._stream_frames = stream_frames

		# Set original image dimensions.
		self._height = 320
		self._width = 480
		self._depth = 3

		# Save image transform(s).
		self._image_transform = transform

		# Check presence of Resize transform.
		if self._image_transform is not None:
			resize = None
			# Check single transform.
			if isinstance(self._image_transform, transforms.Resize):
				resize = self._image_transform
			# Check transform composition.
			elif isinstance(self._image_transform, transforms.Compose):
				# Iterate throught transforms.
				for trans in self._image_transform.transforms:
					if isinstance(trans, transforms.Resize):
						resize = trans
			# Update the image dimensions [H,W].
			if resize is not None:
				self._height = resize.size[0]
				self._width = resize.size[1]

		logging.info("Setting image size to [D  x H x W]: {} x {} x {}".format(self._depth, self._height, self._width))

		if self._split == 'train':
			data_file = join(self._root, "questions", 'train.json')
		elif self._split == 'val':
			data_file = join(self._root, "questions", 'validation.json')
		elif self._split == 'test':
			data_file = join(self._root, "questions", 'test.json')
		else:
			raise ConfigurationError("Split `{}` not supported yet".format(self._split))

		# Load data from file.
		self.data = self.load_data(data_file)

		# Display exemplary sample.
		i = 0
		sample = self.data[i]
		# Check if this is a test set.
		if "answer" not in sample.keys():
			sample["answer"] = "<UNK>"
		logging.info(
			"Exemplary sample number {}\n  question_type: {}\n  question_subtype: {}\n  question_id: {}\n question: {}\n  answer: {}".format(
				i,
				sample["question_type"],
				sample["question_subtype"],
				sample["question_id"],
				sample["question"],
				sample["answer"]
			)
		)

	def _check_integrity(self) -> bool:
		dataset_split = ["train", "val", "test"]
		ret = False
		for split in dataset_split:
			# Check video files
			videofile = join(self._root, self.videos_names[split])
			videochecksum = self.videos_md5s[split]
			if not exists(videofile):
				logging.info("Cannot find video files")
				return False
			ret = ret | check_md5(fpath=videofile, md5=videochecksum)

			# Check annotations files
			annotationfile = join(self._root, self.annotations_names[split])
			annotationchecksum = self.annotations_md5s[split]
			if not exists(annotationfile):
				logging.info("Cannot find annotation files")
			return False
			ret = ret | check_md5(fpath=annotationfile, md5=annotationchecksum)
		logging.info('Files already downloaded, checking integrity...')
		# Check md5 and return the result.
		return ret

	def download(self) -> None:
		if self._check_integrity():
			logging.info('Files verified')
			return
		# Else: download (once again).
		logging.info('Downloading and extracting archive')

		# Download videos, annotations, questions
		dataset_split = ["train", "val", "test"]
		for split in dataset_split:
			# Download video files
			videofile = self.videos_names[split]
			videourl = self.download_url_prefix_videos + "videos" + "/" + self.video_names[split]
			videochecksum = self.videos_md5s[split]
			videodir = join(self._root, "videos", split)
			if not isdir(videodir):
				makedirs(videodir)
			download_and_extract_archive(videourl, download_root=videodir, filename=videofile, md5=videochecksum)

			# Download annotation files
			annotationfile = self.annotations_names[split]
			annotationurl = self.download_url_prefix_annotations + "annotations" + "/" + self.annotations_names[split]
			annotationchecksum = self.annotations_md5s[split]
			# create annotation dir and extracy
			annotationdir = join(self._root, "annotation", split)
			if not isdir(annotationdir):
				makedirs(annotationdir)
			download_and_extract_archive(annotationurl, download_root=annotationdir, filename=annotationfile, md5=annotationchecksum)

			# Download questions files
			questionfile = self.question_names[split]
			questionurl = self.download_url_prefix_questions + "questions" + "/" + self.question_names[split]
			questiondir = join(self._root, "question")
			if not isdir(questiondir):
				makedirs(questiondir)
			download_url(questionurl, root=questiondir, filename=questionfile)


	def load_data(self, source_data_file):
		"""
		Loads the dataset from source file.

		"""
		dataset = []

		with open(source_data_file) as f:
			logging.info("Loading samples from '{}'...".format(source_data_file[i]))
			data = json.load(f)
			for questions in data:
				for question in questions['questions']:
					question_data = question
					question_data['scene_index'] = questions['scene_index']
					question_data['video_filename'] = questions['video_filename']
					dataset.append(question_data)
		logging.info("Loaded dataset consisting of {} samples".format(len(dataset)))
		return dataset

	def __len__(self):
		"""
		Returns:
			The size of the loaded dataset split.
		"""
		return len(self.data)

	def get_frames(self, video_filename):
		"""
		Function loads and returns video frames along with its size.
		Additionally, it performs all the required transformations.

		Args:
			img_id: Identifier of the images.

		Returns:
			image (PIL Image / Tensor, depending on the applied transforms)
		"""

		# Load the image and convert to RGB.
		img = Image.open(join(self._split_image_folder, img_id)).convert('RGB')

		if self._image_transform is not None:
			# Apply transformation(s).
			img = self._image_transform(img)

		# Return image.
		return img

	def __getitem__(self, index):
		"""
		Getter method to access the dataset and return a single sample.

		Args:
			index: index of the sample to return.

		Returns:
			indices, images_ids, images, questions, answers, question_types, spatial_features, object_features, object_normalized_bbox, scene_graph 
		"""
		# Get item.
		item = self.data[index]

		# Load and stream the image ids.
		img_id = item["imageId"]

		# Load the adequate image - only when required.
		if self._stream_images:
			img = self.get_image(img_id)
		else:
			img = None

		# Return question.
		question = item["question"]

		# Return answer.
		if "answer" in item.keys():
			answer = item["answer"]
		else:
			answer = "<UNK>"

		# Question type related variables.
		if "types" in item.keys():
			question_type = item["types"]
		else:
			question_type = "<UNK>"

		# Load images features and scene graphs
		if self._extract_features:
			# Spatial features
			if self._load_spatial_features:
				spatial_features = self._spatial_features_loader.load_feature(img_id)
			else:
				spatial_features = None
			# Object features
			if self._load_object_features:
				object_features, object_normalized_bbox, _ = self._object_features_loader.load_feature_normalized_bbox(img_id)
			else:
				object_features = None
				object_normalized_bbox = None
			# Scene graph
			if self._load_scene_graph:
				scene_graph_features, _ , _ =  self._scene_graph_loader.load_feature_normalized_bbox(img_id)
			else:
				scene_graph_features = None

		# Return sample.
		return index, img_id, img, question, answer, question_type, spatial_features, object_features, object_normalized_bbox, scene_graph_features

	def collate_fn(self, batch):
		"""
		Combines a list of samples (retrieved with :py:func:`__getitem__`) into a batch.

		Args:
			batch: list of individual samples to combine

		Returns:
			Batch of: indices, images_ids, images, questions, answers, question_types, spatial_features, object_features, object_normalized_bbox, scene_graph 

		"""
		# Collate indices.
		indices_batch = [sample[0] for sample in batch]

		# Stack images_ids and images.
		img_ids_batch = [sample[1] for sample in batch]

		if self._stream_images:
			imgs_batch = torch.stack([sample[2] for sample in batch]).type(torch.FloatTensor)
		else:
			imgs_batch = None

		# Collate questions and answers
		questions_batch = [sample[3] for sample in batch]
		answers_batch = [sample[4] for sample in batch]

		# Collate question_types 
		question_type_batch = [sample[5] for sample in batch]

		# Collate images features
		if self._extract_features:
			# Spatial features
			if self._load_spatial_features:
				spatial_features_batch = [sample[6] for sample in batch]
			else:
				spatial_features_batch = None
			# Object features
			if self._load_object_features:
				object_features_batch = [sample[7] for sample in batch]
			else:
				object_features_batch = None
			# Scene graph
			if self._load_scene_graph:
				scene_graph_batch = [sample[8] for sample in batch]
			else:
				scene_graph_batch = None

		# Return collated dict.
		return indices_batch, img_ids_batch, imgs_batch, questions_batch, answers_batch, question_type_batch, spatial_features_batch, object_features_batch, scene_graph_batch


