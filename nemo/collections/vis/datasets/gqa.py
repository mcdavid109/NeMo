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
from os.path import expanduser, join, exists

import json
from PIL import Image

from glob import glob

import torch
from torchvision.transforms import transforms

from typing import Any, Optional
from dataclasses import dataclass

from hydra.types import ObjectConf
from hydra.core.config_store import ConfigStore

from nemo.utils import logging
from nemo.core.classes import Dataset

from nemo.collections.vis.data_utils import SpatialFeatureLoader, ObjectsFeatureLoader, SceneGraphFeatureLoader
# Create the config store instance.
cs = ConfigStore.instance()


@dataclass
class GQAConfig:
	"""
	Structured config for the GQA dataset.

	For more details please refer to:
	https://cs.stanford.edu/people/dorarad/gqa/

	Args:
		_target_: Specification of dataset class
		root: Folder where task will store data (DEFAULT: "~/data/gqa")
		split: Defines the set (split) that will be used (Options: train | val | test ) (DEFAULT: train)
		stream_images: Flag indicating whether the task will load and return images (DEFAULT: True)
		transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
		dataset_type: Type of dataset to use for training, can be either all or balanced (DEFAULT: all)
		extract_features: Whether we will pre-extracted features from the images (DEFAULT: False)
		load_spatial_features: Whether we will load spatial features from pretrained Resnet (DEFAULT: False)
		load_object_features: Whether we will load object features from pretrained fast-RCNN (DEFAULT: False)
		load_scene_graph: Whether we will load the scene graph (DEFAULT: False)
		download: downloads the data if not present (DEFAULT: True)
	"""

	# Dataset target class name.
	_target_: str = "nemo.collections.vis.datasets.GQA"
	root: str = "~/data/gqa"
	split: str = "train"
	stream_images: bool = True
	dataset_type: str = "all"
	extract_features: bool = False
	load_spatial_features: bool = False
	load_object_features: bool = False
	load_scene_graph: bool = False
	# transform: Optional[Any] = None # Provided manually?
	download: bool = True


# Register the config.
cs.store(
	group="nemo.collections.vis.datasets",
	name="GQA",
	node=ObjectConf(target="nemo.collections.vis.datasets.GQA", params=GQAConfig()),
)


class GQA(Dataset):
	"""
	Class fetching data from the GQA (Visual Reasoning in the Real World) dataset.

	The GQA dataset consists of the followings:

		- Real world images with scene graph annotation (113,018 images)
		- Questions with functional program annotation (22M questions)
		- Vocabulary size of 3097 and 1878 possible answers
		- Questions can be categorized into either structural and semantics type

	For more details please refer to the associated _website or _paper.

	After downloading and extracting, we will have the following directory

	data/gqa
	questions/
		train_all_questions/
			train_all_questions_0.json
			...
			train_all_questions_9.json
		train_balanced_questions.json
		val_all_questions.json
		val_balanced_questions.json
		submission_all_questions.json
		test_all_questions.json
		test_balanced_questions.json
	spatial/
		gqa_spatial_info.json
		gqa_spatial_0.h5
		...
		gqa_spatial_15.h5
	objects/
		gqa_objects_info.json
		gqa_objects_0.h5
		...
		gqa_objects_15.h5
	sceneGraphs/
		train_sceneGraphs.json
		val_sceneGraphs.json
	images/
		...

	.. _website: https://cs.stanford.edu/people/dorarad/gqa/

	.._paper: https://arxiv.org/pdf/1902.09506

	"""

	download_url_prefix = "https://nlp.stanford.edu/data/gqa/"
	zip_names = {"scene": "sceneGraphs.zip", "questions": "questions1.2.zip", "images": "images.zip"}
	features_names = {"spatial": "spatialFeatures.zip", "object": "objectFeatures.zip"}
	zip_md5s = {"scene": "d93e1360598166e3222715e81c950de6", "questions": "e28f02144d5086871827574db4752720", "images": "ce0e89c03830722434d7f20a41b05342"}
	features_md5s = {"spatial": "87d183dc2438f22a831f9dd6deea7773", "object": "eb7a2580989a5384433bf562e6ebffc8"}

	def __init__(
		self,
		root: str = "~/data/gqa",
		split: str = "train",
		stream_images: bool = True,
		dataset_type: str = "all",
		extract_features: bool = False,
		load_spatial_features: bool = False,
		load_object_features: bool = False,
		load_scene_graph: bool = False,
		transform: Optional[Any] = None,
		download: bool = True,
	):
		"""
		Initializes dataset object. Calls base constructor.
		Downloads the dataset if not present and loads the adequate files depending on the mode.

		Args:
		root: Folder where task will store data (DEFAULT: "~/data/gqa")
			split: Defines the set (split) that will be used (Options: train | val | test) (DEFAULT: train)
			stream_images: Flag indicating whether the task will load and return images (DEFAULT: True)
			dataset_type: Flag indicating which type of dataset we will use (all or balanced) (DEFAULT: "all")
			extract_features: Flag indicating whether we want raw image or features extracted from pre-trained model (DEFAULT: False)
			load_spatial_features: Flag indicating whether we will load spatial features from Resnet (DEFAULT: False)
			load_object_features: Flag indicating whether we will load object features from Fast-RCNN (DEFAULT: False)
			load_scene_graph: Flag indicating whether we will load scene graph (DEFAULT: False)
			transform: TorchVision image preprocessing/augmentations to apply (DEFAULT: None)
			download: downloads the data if not present (DEFAULT: True)
		"""
		# Call constructors of parent class.
		super().__init__()

		# Get the absolute path.
		self._root = expanduser(root)

		# if don't have the root, create it
		if not exists(self._root):
			makedirs(self._root)

		# Process split.
		self._split = split

		# Download dataset when required.
		if download:
			self.download()

		# Get flag informing whether we want to stream images or not.
		self._stream_images = stream_images

		self._dataset_type = dataset_type
		self._extract_features = extract_features
		self._load_spatial_features = load_spatial_features
		self._load_object_features = load_object_features
		self._load_scene_graph = load_scene_graph

		# Set original image dimensions.
		self._height = 480
		self._width = 640
		self._depth = 3

		# Number of objects
		self._num_objects = 100

		# Features and scene graph loader
		self._spatial_features_loader = None
		self._object_features_loader = None
		self._scene_graph_loader = None

		self._vocab_object_file = './data_utils/gqa/vocab_files/objects.txt'
		self._vocab_attributes_file = './data_utils/gqa/vocab_files/attributes.txt'
		self._vocab_question_file = './data_utils/gqa/vocab_files/questions.txt'
		self._vocab_answers_file = './data_utils/gqa/vocab_files/answers.txt'

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

		self._split_image_folder = join(self._root, "images")
		# instantiate feature loader object
		if self._extract_features:
			if self._load_spatial_features:
				spatial_features_dirs = join(self._root, "spatial")
				self._spatial_features_loader = SpatialFeatureLoader(spatial_features_dirs)
			if self._load_object_features:
				object_features_dirs = join(self._root, "objects")
				self._object_features_loader = ObjectsFeatureLoader(object_features_dirs)
		if self._load_scene_graph:
			if self._split == 'train':
				scene_graph_dirs = join(self._root, "sceneGraphs", 'train_sceneGraphs.json')
				self._scene_graph_loader = SceneGraphFeatureLoader(scene_graph_dirs, self._vocab_object_file, self._vocab_attributes_file, self._num_objects)
			elif self._split == 'validation':
				scene_graph_dirs = join(self._root, "sceneGraphs", 'val_sceneGraphs.json')
				self._scene_graph_loader = SceneGraphFeatureLoader(scene_graph_dirs, self._vocab_object_file, self._vocab_attributes_file, self._num_objects)

		# Set split-dependent data.
		if self._split == 'train':
			# Training split folder and file with data question.
			if self._dataset_type == "all":
				num_files = len(glob(join(self._root, "questions", "train_all_questions", 'train_all_questions_*.json')))
				data_file = [join(self._root, "questions", "train_all_questions", 'train_all_questions_%d.json' % n) for n in range(num_files)]
			elif self._dataset_type == "balanced":
				data_file = join(self._root, "questions", 'train_balanced_questions.json')
			else:
				raise ValueError("Dataset type `{}` not supported yet".format(self._dataset_type))

		elif self._split == 'validation':
			# Validation split folder and file with data question.
			if self._dataset_type == "all":
				data_file = join(self._root, "questions", 'val_all_questions.json')
			elif self._dataset_type == "balanced":
				data_file = join(self._root, "questions", 'val_balanced_questions.json')
			else:
				raise ValueError("Dataset type `{}` not supported yet".format(self._dataset_type))

		elif self._split == 'test':
			# Test split folder and file with data question.
			if self._dataset_type == "all":
				data_file = join(self._root, "questions", 'test_all_questions.json')
			elif self._dataset_type == "balanced":
				data_file = join(self._root, "questions", 'test_balanced_questions.json')
			else:
				raise ValueError("Dataset type `{}` not supported yet".format(self._dataset_type))
		else:
			raise ValueError("Split `{}` not supported yet".format(self._split))

		# Load data from file.
		self.data = self.load_data(data_file)

		# Display exemplary sample.
		i = 0
		sample = self.data[i]
		# Check if this is a test set.
		if "answer" not in sample.keys():
			sample["answer"] = "<UNK>"
			sample["types"] = "<UNK>"
		logging.info(
			"Exemplary sample number {}\n  question_type: {}\n  image_ids: {}\n  question: {}\n  answer: {}".format(
				i,
				sample["types"],
				sample["imageId"],
				sample["question"],
				sample["answer"],
			)
		)

	def _check_integrity(self) -> bool:
		# We always return questions
		questionfile = join(self._root, "questions", self.zip_names["questions"])
		questionchecksum = self.zip_md5s["questions"]
		if not exists(questionfile):
			logging.info("Cannot find question files")
			return False
		ret = check_md5(fpath=questionfile, md5=questionchecksum)

		# In case we want to return ground truth scene graph
		if self._load_scene_graph:
			scenefile = join(self._root, "sceneGraphs", self.zip_names["scene"])
			scenechecksum = self.zip_md5s["scene"]
			if not exists(scenefile):
				logging.info("Cannot find scene graph files")
			return False
			ret = ret & check_md5(fpath=scenefile, md5=scenechecksum)

		# In case we want to return features
		if self._extract_features:
			# spatial features (from ResNet101)
			if self._load_spatial_features:
				spatialfile = join(self._root, "spatial", self.features_names["spatial"])
				spatialchecksum = self.features_md5s["spatial"]
				if not exists(spatialfile):
					logging.info("Cannot find spatial features files")
					return False
				ret = ret & check_md5(fpath=spatialfile, md5=spatialchecksum)
			# object features (from Faster-RNN)
			elif self._load_object_features:
				objectfile = join(self._root, self.features_names["object"])
				objectchecksum = self.features_md5s["object"]
				if not exists(objectfile):
					logging.info("Cannot find object features files")
					return False
				ret = ret & check_md5(fpath=objectfile, md5=objectchecksum)
		# In case we want to return images
		if self._stream_images:
			imagefile = join(self._root, "images", self.zip_names["images"])
			imagechecksum = self.zip_md5s["images"]
			if not exists(imagefile):
				logging.info("Cannot find image files")
				return False
			ret = ret & check_md5(fpath=imagefile, md5=imagechecksum)

		logging.info('Files already downloaded, checking integrity...')
		# Check md5 and return the result.
		return ret

	def download(self) -> None:
		if self._check_integrity():
			logging.info('Files verified')
			return
		# Else: download (once again).
		logging.info('Downloading and extracting archive')

		# We always return questions
		questionfile = self.zip_names["questions"]
		questionurl = self.download_url_prefix + self.zip_names["questions"]
		questionchecksum = self.zip_md5s["questions"]
		questiondir = join(self._root, "questions")
		if not exists(questiondir):
			makedirs(questiondir)
		download_and_extract_archive(questionurl, download_root=questiondir, filename=questionfile, md5=questionchecksum)

		# In case we want to return scene graph
		if self._load_scene_graph:
			scenefile = self.zip_names["scene"]
			sceneurl = self.download_url_prefix + self.zip_names["scene"]
			scenechecksum = self.zip_md5s["scene"]
			scenedir = join(self._root, "sceneGraphs")
			if not exists(scenedir):
				makedirs(scenedir)
			download_and_extract_archive(sceneurl, download_root=scenedir, filename=scenefile, md5=scenechecksum)

		# In case we want to return features
		if self._extract_features:
			# spatial features (from ResNet-101)
			if self._load_spatial_features:
				spatialfile = self.features_names["spatial"]
				spatialurl = self.download_url_prefix + self.features_names["spatial"]
				download_and_extract_archive(spatialurl, download_root=self._root, filename=spatialfile, md5=spatialchecksum)

			# object features (from Faster-RNN)
			elif self._load_object_features:
				objectfile = self.features_names["object"]
				objecturl = self.download_url_prefix + self.features_names["object"]
				objectchecksum = self.features_md5s["object"]
				download_and_extract_archive(objecturl, download_root=self._root, filename=objectfile, md5=objectchecksum)
		# In case we want to return images
		if self._stream_images:
			imagefile = self.zip_names["images"]
			imageurl = self.download_url_prefix + self.zip_names["images"]
			imagechecksum = self.zip_md5s["images"]
			download_and_extract_archive(imageurl, download_root=self._root, filename=imagefile, md5=imagechecksum)


	def load_data(self, source_data_file):
		"""
		Loads the dataset from source file.

		"""
		dataset = []

		# train_all case
		if isinstance(source_data_file, list):
			for i in range(len(source_data_file)):
				with open(source_data_file[i]) as f:
					logging.info("Loading samples from '{}'...".format(source_data_file[i]))
					data = json.load(f)
					for _ , dataitem in data.items():
						question_data = {}
						# We only select a subset of question related info
						question_data['groups'] = dataitem['groups']
						question_data['answer'] = dataitem['answer']
						question_data['types'] = dataitem['types']
						question_data['semanticStr'] = dataitem['semanticStr']
						question_data['fullAnswer'] = dataitem['fullAnswer']
						question_data['question'] = dataitem['question']
						question_data['semantic'] = dataitem['semantic']
						question_data['imageId'] = dataitem['imageId']
						dataset.append(question_data)
		else:
			if self._split == 'test':
				with open(source_data_file) as f:
					logging.info("Loading samples from '{}'...".format(source_data_file))
					data = json.load(f)
					for _ , dataitem in data.items():
						question_data = {}
						# Test does not have much info like train, dev
						question_data['question'] = dataitem['question']
						question_data['isBalanced'] = dataitem['isBalanced']
						question_data['imageId'] = dataitem['imageId']
						dataset.append(question_data)
			else:
				with open(source_data_file) as f:
					logging.info("Loading samples from '{}'...".format(source_data_file))
					data = json.load(f)
					for _ , dataitem in data.items():
						question_data = {}
						# We only select a subset of question related info
						question_data['groups'] = dataitem['groups']
						question_data['answer'] = dataitem['answer']
						question_data['types'] = dataitem['types']
						question_data['semanticStr'] = dataitem['semanticStr']
						question_data['fullAnswer'] = dataitem['fullAnswer']
						question_data['question'] = dataitem['question']
						question_data['semantic'] = dataitem['semantic']
						question_data['imageId'] = dataitem['imageId']
						dataset.append(question_data)

		logging.info("Loaded dataset consisting of {} samples".format(len(dataset)))
		return dataset

	def __len__(self):
		"""
		Returns:
			The size of the loaded dataset split.
		"""
		return len(self.data)

	def get_image(self, img_id):
		"""
		Function loads and returns image along with its size.
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
				object_normalized_bbox_batch = [sample[8] for sample in batch]
			else:
				object_features_batch = None
				object_normalized_bbox_batch = None
		# Scene graph
		if self._load_scene_graph:
			scene_graph_batch = [sample[9] for sample in batch]
		else:
			scene_graph_batch = None

		# Return collated dict.
		return indices_batch, img_ids_batch, imgs_batch, questions_batch, answers_batch, question_type_batch, spatial_features_batch, object_features_batch, object_normalized_bbox_batch, scene_graph_batch
